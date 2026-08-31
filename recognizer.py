import cv2
import numpy as np
import pickle
import threading
import os
from database import get_all_encodings, mark_attendance

# ── Detection backend priority ────────────────────────────────────────────────
USE_DEEPFACE        = False
USE_FACE_RECOGNITION = False

try:
    from deepface import DeepFace
    USE_DEEPFACE = True
    print("[AttendAI] Using DeepFace for recognition")
except ImportError:
    pass

if not USE_DEEPFACE:
    try:
        import face_recognition
        USE_FACE_RECOGNITION = True
        print("[AttendAI] Using face_recognition for recognition")
    except ImportError:
        pass

if not USE_DEEPFACE and not USE_FACE_RECOGNITION:
    print("[AttendAI] Using OpenCV fallback for recognition")

# ── OpenCV cascade (used for detection in all modes) ─────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Settings ──────────────────────────────────────────────────────────────────
TOLERANCE       = 0.5
MIN_FACE_SIZE   = 30
SCALE_FACTOR    = 0.5
CONFIRM_FRAMES  = 3
MAX_FACES       = 50
DEEPFACE_MODEL  = "Facenet"      # Options: Facenet, VGG-Face, ArcFace
DEEPFACE_DIST   = "cosine"       # Options: cosine, euclidean

# ── Stored face images folder for DeepFace ────────────────────────────────────
FACES_DIR = "known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

# ── Confirmation tracker ──────────────────────────────────────────────────────
_seen_counts = {}
_seen_lock   = threading.Lock()

def reset_confirmation():
    with _seen_lock:
        _seen_counts.clear()

# ── Encode face from uploaded photo ──────────────────────────────────────────

def encode_face_from_image(img_array):
    """Encode uploaded photo → bytes or None."""
    if USE_DEEPFACE:
        try:
            # Save temp image for DeepFace
            tmp = "tmp_upload.jpg"
            cv2.imwrite(tmp, img_array)
            # Try to detect face — if no face, DeepFace raises exception
            result = DeepFace.represent(
                img_path=tmp,
                model_name=DEEPFACE_MODEL,
                enforce_detection=True,
                detector_backend="opencv"
            )
            os.remove(tmp)
            if result:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
                return pickle.dumps(embedding)
            return None
        except Exception as e:
            if os.path.exists("tmp_upload.jpg"):
                os.remove("tmp_upload.jpg")
            print(f"[DeepFace] encode error: {e}")
            return None

    elif USE_FACE_RECOGNITION:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            return pickle.dumps(encodings[0])
        # Try with equalised image
        gray = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
        rgb2 = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb2)
        return pickle.dumps(encodings[0]) if encodings else None

    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detect_all_faces(gray)
        if not faces:
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
        return pickle.dumps(roi.flatten().astype(np.float32))

# ── Face detection ────────────────────────────────────────────────────────────

def detect_all_faces(gray_img):
    """Detect ALL faces — returns list of (x,y,w,h)."""
    try:
        faces = face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_img, scaleFactor=1.05, minNeighbors=3,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
        return list(faces) if len(faces) > 0 else []
    except Exception:
        return []

def detect_largest_face(gray_img):
    faces = detect_all_faces(gray_img)
    if not faces:
        return None
    return sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

# ── Load known faces ──────────────────────────────────────────────────────────

def load_known_faces():
    rows = get_all_encodings()
    known = []
    for row in rows:
        enc = pickle.loads(row["encoding"])
        known.append({
            "id":       row["id"],
            "name":     row["name"],
            "roll_no":  row["roll_no"],
            "encoding": enc
        })
    return known

# ── Match face ────────────────────────────────────────────────────────────────

def match_face(unknown_encoding, known_faces):
    """Return best matching student or None."""
    if not known_faces:
        return None

    if USE_DEEPFACE or USE_FACE_RECOGNITION:
        # Cosine similarity for DeepFace embeddings & euclidean for face_recognition
        best_dist, best_match = float("inf"), None
        for k in known_faces:
            if USE_DEEPFACE:
                # Cosine distance
                a = unknown_encoding / (np.linalg.norm(unknown_encoding) + 1e-10)
                b = k["encoding"]   / (np.linalg.norm(k["encoding"])    + 1e-10)
                dist = 1.0 - float(np.dot(a, b))
            else:
                dist = float(face_recognition.face_distance(
                    [k["encoding"]], unknown_encoding)[0])
            if dist < best_dist:
                best_dist  = dist
                best_match = k

        threshold = 0.35 if USE_DEEPFACE else TOLERANCE
        return best_match if best_dist <= threshold else None

    else:
        # OpenCV fallback
        best_dist, best_match = float("inf"), None
        for k in known_faces:
            dist = np.linalg.norm(unknown_encoding - k["encoding"])
            if dist < best_dist:
                best_dist  = dist
                best_match = k
        return best_match if best_dist < 8000 else None

# ── DeepFace encode from cropped face region ──────────────────────────────────

def _deepface_encode_roi(roi_bgr):
    """Encode a cropped face region using DeepFace."""
    try:
        tmp = "tmp_roi.jpg"
        cv2.imwrite(tmp, roi_bgr)
        result = DeepFace.represent(
            img_path=tmp,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,
            detector_backend="skip"
        )
        os.remove(tmp)
        if result:
            return np.array(result[0]["embedding"], dtype=np.float32)
        return None
    except Exception:
        if os.path.exists("tmp_roi.jpg"):
            os.remove("tmp_roi.jpg")
        return None

# ── Main frame processor ──────────────────────────────────────────────────────

def process_frame(frame, known_faces, marked_today):
    """
    Detect ALL faces in frame simultaneously.
    Returns (annotated_frame, newly_marked_list).
    """
    annotated    = frame.copy()
    newly_marked = []
    faces_found  = 0

    if USE_FACE_RECOGNITION:
        small     = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small, model="hog")
        locations = locations[:MAX_FACES]
        faces_found = len(locations)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        scale = 1.0 / SCALE_FACTOR

        for (top, right, bottom, left), enc in zip(locations, encodings):
            top, right, bottom, left = (int(top*scale), int(right*scale),
                                        int(bottom*scale), int(left*scale))
            match = match_face(enc, known_faces)
            _draw_box(annotated, top, right, bottom, left, match)
            _confirm_and_mark(match, marked_today, newly_marked)

    else:
        # Use OpenCV to detect face locations (works for both DeepFace & fallback)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = detect_all_faces(gray)
        faces_found = len(faces)

        for (x, y, w, h) in faces[:MAX_FACES]:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            if USE_DEEPFACE:
                enc = _deepface_encode_roi(roi)
                match = match_face(enc, known_faces) if enc is not None else None
            else:
                gray_roi = cv2.resize(
                    cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
                enc = gray_roi.flatten().astype(np.float32)
                match = match_face(enc, known_faces)

            _draw_box(annotated, y, x+w, y+h, x, match)
            _confirm_and_mark(match, marked_today, newly_marked)

    _draw_stats(annotated, faces_found, len(marked_today))
    return annotated, newly_marked

def _confirm_and_mark(match, marked_today, newly_marked):
    if not match:
        return
    sid = match["id"]
    with _seen_lock:
        _seen_counts[sid] = _seen_counts.get(sid, 0) + 1
        count = _seen_counts[sid]
    if count >= CONFIRM_FRAMES and sid not in marked_today:
        if mark_attendance(sid):
            marked_today.add(sid)
            newly_marked.append(match)

# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_box(img, top, right, bottom, left, match):
    if match:
        color = (0, 220, 120)
        label = f"{match['name']}  {match['roll_no']}"
    else:
        color = (0, 100, 220)
        label = "Unknown"

    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    label_y = bottom + 22 if bottom + 30 < img.shape[0] else top - 8
    cv2.rectangle(img,
                  (left, label_y - 18),
                  (left + len(label)*9 + 8, label_y + 4),
                  color, cv2.FILLED)
    cv2.putText(img, label, (left + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    if match:
        _draw_corners(img, top, right, bottom, left, color)

def _draw_corners(img, top, right, bottom, left, color, length=18, thickness=3):
    for corner, dx, dy in [((left,top),1,1),((right,top),-1,1),
                            ((left,bottom),1,-1),((right,bottom),-1,-1)]:
        cv2.line(img, corner, (corner[0]+dx*length, corner[1]), color, thickness)
        cv2.line(img, corner, (corner[0], corner[1]+dy*length), color, thickness)

def _draw_stats(img, faces_count, marked_count):
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (260,70), (10,12,16), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, f"Faces detected : {faces_count}",
                (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,229,160), 1, cv2.LINE_AA)
    cv2.putText(img, f"Marked present : {marked_count}",
                (8,48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,170,255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Engine: {'DeepFace' if USE_DEEPFACE else 'face_recognition' if USE_FACE_RECOGNITION else 'OpenCV'}",
                (8,66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90,96,112), 1, cv2.LINE_AA)
