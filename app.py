import cv2, numpy as np, io, csv, json, pickle, threading, time, os, base64
from datetime import date
from flask import (Flask, render_template, Response, request,
                   redirect, url_for, flash, jsonify, send_file)
from database import (init_db, add_student, save_encoding, get_all_students,
                      delete_student, get_attendance_report, get_today_stats, mark_attendance)
from recognizer import encode_face_from_image, load_known_faces, process_frame

app = Flask(__name__)
app.secret_key = "attendai_luxury_2024"

# ── Dirs ──────────────────────────────────────────────────────────────────────
UNKNOWN_DIR  = "unknown_faces"
SETTINGS_FILE = "camera_settings.json"
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ── Global state ──────────────────────────────────────────────────────────────
camera_lock    = threading.Lock()
camera_active  = False
latest_frame   = None
known_faces_cache = []
marked_today   = set()
recently_marked = []
camera_error   = None
unknown_count  = 0
_unknown_save_interval = 30   # save unknown face every N seconds
_last_unknown_save     = {}   # track last save time per position

# ── Camera settings ───────────────────────────────────────────────────────────
def load_camera_settings():
    try:
        with open(SETTINGS_FILE) as f: return json.load(f)
    except:
        return {"mode":"webcam","webcam_index":0,"brand":"","ip":"","port":"554",
                "username":"","password":"","stream_path":"/stream",
                "network_type":"wifi","custom_url":"","connected":False}

def save_camera_settings(s):
    with open(SETTINGS_FILE,"w") as f: json.dump(s, f, indent=2)

def build_rtsp_url(s):
    if s.get("custom_url"): return s["custom_url"]
    brand = s.get("brand","").lower()
    ip,port,user,pwd = s.get("ip",""),s.get("port","554"),s.get("username",""),s.get("password","")
    auth  = f"{user}:{pwd}@" if user else ""
    paths = {
        "hikvision": f"rtsp://{auth}{ip}:{port}/Streaming/Channels/101",
        "dahua":     f"rtsp://{auth}{ip}:{port}/cam/realmonitor?channel=1&subtype=0",
        "cp plus":   f"rtsp://{auth}{ip}:{port}/stream1",
        "cpplus":    f"rtsp://{auth}{ip}:{port}/stream1",
        "axis":      f"rtsp://{auth}{ip}:{port}/axis-media/media.amp",
        "reolink":   f"rtsp://{auth}{ip}:{port}/h264Preview_01_main",
        "generic":   f"rtsp://{auth}{ip}:{port}{s.get('stream_path','/stream')}",
    }
    return paths.get(brand, f"rtsp://{auth}{ip}:{port}{s.get('stream_path','/stream')}")

def test_camera_connection(source):
    try:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        if not cap.isOpened(): return False, "Could not open camera stream."
        ret, frame = cap.read()
        cap.release()
        return (True,"Connection successful!") if (ret and frame is not None) else (False,"No frames received.")
    except Exception as e:
        return False, str(e)

# ── Save unknown face to disk ─────────────────────────────────────────────────
def save_unknown_face(frame, box):
    try:
        top, right, bottom, left = box
        pad = 20
        h, w = frame.shape[:2]
        y1,y2 = max(0,top-pad),   min(h, bottom+pad)
        x1,x2 = max(0,left-pad),  min(w, right+pad)
        roi  = frame[y1:y2, x1:x2]
        if roi.size == 0: return
        ts   = time.strftime("%Y%m%d_%H%M%S")
        fname= f"unknown_{ts}_{left}.jpg"
        cv2.imwrite(os.path.join(UNKNOWN_DIR, fname), roi)
    except Exception as e:
        print(f"[Unknown] save error: {e}")

# ── Camera thread ─────────────────────────────────────────────────────────────
def camera_thread(source):
    global latest_frame, camera_active, recently_marked, camera_error, unknown_count
    camera_error = None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        camera_active = False
        camera_error  = "Failed to open camera."
        return

    last_unknown_time = 0

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            camera_error = "Lost camera connection."
            camera_active = False
            break

        annotated, newly, unknown_boxes = process_frame_with_unknown(frame, known_faces_cache, marked_today)
        recently_marked.extend(newly)
        if len(recently_marked) > 20:
            recently_marked = recently_marked[-20:]

        unknown_count = len(unknown_boxes)

        # Save unknown faces every 30 seconds
        now = time.time()
        if unknown_boxes and (now - last_unknown_time) > _unknown_save_interval:
            for box in unknown_boxes[:3]:
                save_unknown_face(frame, box)
            last_unknown_time = now

        with camera_lock:
            latest_frame = annotated
        time.sleep(0.03)

    cap.release()
    with camera_lock:
        latest_frame = None

def process_frame_with_unknown(frame, known_faces, marked_today):
    """Wrapper that also returns unknown face boxes."""
    from recognizer import (detect_all_faces, match_face, _draw_box,
                             _draw_corners, _draw_stats,
                             USE_DEEPFACE, USE_FACE_RECOGNITION,
                             SCALE_FACTOR, MAX_FACES, CONFIRM_FRAMES,
                             _seen_counts, _seen_lock, _deepface_encode_roi)
    annotated    = frame.copy()
    newly_marked = []
    unknown_boxes = []
    faces_found  = 0

    if USE_FACE_RECOGNITION:
        try:
            import face_recognition as fr2
        except ImportError:
            fr2 = None
        small     = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = fr2.face_locations(rgb_small, model="hog")[:MAX_FACES]
        faces_found = len(locations)
        encodings = fr2.face_encodings(rgb_small, locations)
        scale = 1.0 / SCALE_FACTOR
        for (top, right, bottom, left), enc in zip(locations, encodings):
            top,right,bottom,left = int(top*scale),int(right*scale),int(bottom*scale),int(left*scale)
            match = match_face(enc, known_faces)
            _draw_box(annotated, top, right, bottom, left, match)
            if match:
                _confirm_mark(match, marked_today, newly_marked)
            else:
                unknown_boxes.append((top, right, bottom, left))
    else:
        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = detect_all_faces(gray)
        faces_found = len(faces)
        for (x,y,w,h) in faces[:MAX_FACES]:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue
            if USE_DEEPFACE:
                enc   = _deepface_encode_roi(roi)
                match = match_face(enc, known_faces) if enc is not None else None
            else:
                gr    = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),(64,64))
                enc   = gr.flatten().astype(np.float32)
                match = match_face(enc, known_faces)
            _draw_box(annotated, y, x+w, y+h, x, match)
            if match:
                _confirm_mark(match, marked_today, newly_marked)
            else:
                unknown_boxes.append((y, x+w, y+h, x))

    _draw_stats(annotated, faces_found, len(marked_today))
    return annotated, newly_marked, unknown_boxes

def _confirm_mark(match, marked_today, newly_marked):
    from recognizer import _seen_counts, _seen_lock, CONFIRM_FRAMES
    sid = match["id"]
    with _seen_lock:
        _seen_counts[sid] = _seen_counts.get(sid, 0) + 1
        count = _seen_counts[sid]
    if count >= CONFIRM_FRAMES and sid not in marked_today:
        if mark_attendance(sid):
            marked_today.add(sid)
            newly_marked.append(match)

def gen_frames():
    while camera_active:
        with camera_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05); continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)

# ── Helper ────────────────────────────────────────────────────────────────────
def get_unknown_face_count():
    try:
        return len([f for f in os.listdir(UNKNOWN_DIR) if f.endswith('.jpg')])
    except:
        return 0

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    settings = load_camera_settings()
    return render_template("index.html",
        stats=get_today_stats(), students=get_all_students(),
        today_records=get_attendance_report(filter_date=date.today().isoformat()),
        camera_active=camera_active, camera_error=camera_error,
        settings=settings, unknown_face_count=get_unknown_face_count())

@app.route("/camera/settings", methods=["GET","POST"])
def camera_settings():
    settings = load_camera_settings()
    if request.method == "POST":
        for k in ["mode","brand","ip","port","username","password","stream_path","network_type","custom_url"]:
            settings[k] = request.form.get(k,"").strip()
        settings["webcam_index"] = int(request.form.get("webcam_index",0))
        settings["connected"] = False
        save_camera_settings(settings)
        flash("Settings saved!", "success")
        return redirect(url_for("camera_settings"))
    return render_template("camera_settings.html", settings=settings, camera_active=camera_active,
                           unknown_face_count=get_unknown_face_count())

@app.route("/camera/test", methods=["POST"])
def test_camera():
    settings = load_camera_settings()
    source   = settings.get("webcam_index",0) if settings["mode"]=="webcam" else build_rtsp_url(settings)
    success, message = test_camera_connection(source)
    settings["connected"] = success
    save_camera_settings(settings)
    return jsonify({"success":success,"message":message,
                    "url": f"Webcam #{source}" if settings["mode"]=="webcam" else source})

@app.route("/camera/start", methods=["POST"])
def start_camera():
    global camera_active, known_faces_cache, marked_today, recently_marked, camera_error
    if not camera_active:
        settings = load_camera_settings()
        source   = settings.get("webcam_index",0) if settings["mode"]=="webcam" else build_rtsp_url(settings)
        known_faces_cache = load_known_faces()
        marked_today = set(); recently_marked = []; camera_error = None
        camera_active = True
        threading.Thread(target=camera_thread, args=(source,), daemon=True).start()
        flash("Camera started!", "success")
    return redirect(url_for("live"))

@app.route("/camera/stop", methods=["POST"])
def stop_camera():
    global camera_active
    camera_active = False
    time.sleep(0.3)
    flash("Camera stopped.", "info")
    return redirect(url_for("index"))

@app.route("/live")
def live():
    settings = load_camera_settings()
    return render_template("live.html", camera_active=camera_active,
                           camera_error=camera_error, settings=settings,
                           today=date.today().isoformat(),
                           unknown_face_count=get_unknown_face_count())

@app.route("/video_feed")
def video_feed():
    if not camera_active: return "", 204
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/recent")
def api_recent():
    return jsonify(recently_marked[-5:])

@app.route("/api/stats")
def api_stats():
    return jsonify(get_today_stats())

@app.route("/api/camera_status")
def api_camera_status():
    return jsonify({"active":camera_active,"error":camera_error})

@app.route("/api/unknown_count")
def api_unknown_count():
    return jsonify({"count": get_unknown_face_count()})

# ── Students ──────────────────────────────────────────────────────────────────

@app.route("/students")
def students():
    return render_template("students.html", students=get_all_students(),
                           camera_active=camera_active,
                           unknown_face_count=get_unknown_face_count())

@app.route("/students/add", methods=["GET","POST"])
def add_student_route():
    if request.method == "POST":
        name    = request.form.get("name","").strip()
        roll_no = request.form.get("roll_no","").strip()
        dept    = request.form.get("dept","").strip()
        photo_data = request.form.get("photo_data","")

        if not name or not roll_no:
            flash("Name and Roll No are required.", "error")
            return redirect(url_for("add_student_route"))

        sid = add_student(name, roll_no, dept)
        if sid is None:
            flash(f"Roll No '{roll_no}' already exists.", "error")
            return redirect(url_for("add_student_route"))

        # Encode face from captured photo
        if photo_data:
            try:
                img_data = base64.b64decode(photo_data.split(",")[1])
                img_arr  = np.frombuffer(img_data, np.uint8)
                img      = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                encoding = encode_face_from_image(img) if img is not None else None
                if encoding:
                    save_encoding(sid, encoding)
                    flash(f"Student '{name}' added with face registered!", "success")
                else:
                    flash(f"Student '{name}' added but no face detected. Please capture again.", "info")
            except Exception as e:
                flash(f"Student added but face encoding failed: {e}", "info")
        else:
            flash(f"Student '{name}' added! Please capture their face.", "info")

        return redirect(url_for("students"))

    return render_template("add_student.html", camera_active=camera_active,
                           unknown_face_count=get_unknown_face_count())

@app.route("/students/<int:student_id>/capture_face", methods=["GET","POST"])
def capture_face(student_id):
    students_list = get_all_students()
    student = next((s for s in students_list if s["id"] == student_id), None)

    if request.method == "POST":
        data = request.get_json()
        photo_data = data.get("photo_data","") if data else ""
        try:
            img_data = base64.b64decode(photo_data.split(",")[1])
            img_arr  = np.frombuffer(img_data, np.uint8)
            img      = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"success":False,"error":"Invalid image"})
            encoding = encode_face_from_image(img)
            if encoding is None:
                return jsonify({"success":False,"error":"No face detected. Please try again in better lighting."})
            save_encoding(student_id, encoding)
            # Reload known faces cache
            global known_faces_cache
            known_faces_cache = load_known_faces()
            return jsonify({"success":True})
        except Exception as e:
            return jsonify({"success":False,"error":str(e)})

    return render_template("capture_face.html", student=student,
                           camera_active=camera_active,
                           unknown_face_count=get_unknown_face_count())

@app.route("/students/<int:student_id>/delete", methods=["POST"])
def delete_student_route(student_id):
    delete_student(student_id)
    flash("Student deleted.", "info")
    return redirect(url_for("students"))

# ── Unknown faces ─────────────────────────────────────────────────────────────

@app.route("/unknown_faces")
def unknown_faces():
    faces = []
    try:
        for fname in sorted(os.listdir(UNKNOWN_DIR), reverse=True):
            if not fname.endswith('.jpg'): continue
            parts = fname.replace('.jpg','').split('_')
            faces.append({
                "filename": fname,
                "date": f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:]}" if len(parts)>1 else "—",
                "time": f"{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:]}" if len(parts)>2 else "—",
            })
    except: pass
    return render_template("unknown_faces.html", faces=faces,
                           camera_active=camera_active,
                           unknown_face_count=len(faces))

@app.route("/unknown_faces/image/<filename>")
def unknown_face_image(filename):
    path = os.path.join(UNKNOWN_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "", 404

@app.route("/unknown_faces/delete/<filename>", methods=["POST"])
def delete_unknown_face(filename):
    try:
        os.remove(os.path.join(UNKNOWN_DIR, filename))
        flash("Unknown face deleted.", "info")
    except: pass
    return redirect(url_for("unknown_faces"))

@app.route("/unknown_faces/clear", methods=["POST"])
def clear_unknown_faces():
    try:
        for f in os.listdir(UNKNOWN_DIR):
            if f.endswith('.jpg'): os.remove(os.path.join(UNKNOWN_DIR, f))
        flash("All unknown faces cleared.", "info")
    except: pass
    return redirect(url_for("unknown_faces"))

@app.route("/unknown_faces/register/<filename>", methods=["GET","POST"])
def register_unknown(filename):
    if request.method == "POST":
        name    = request.form.get("name","").strip()
        roll_no = request.form.get("roll_no","").strip()
        dept    = request.form.get("dept","").strip()
        if not name or not roll_no:
            flash("Name and Roll No required.", "error")
            return redirect(request.url)
        sid = add_student(name, roll_no, dept)
        if sid is None:
            flash("Roll No already exists.", "error")
            return redirect(request.url)
        # Encode from unknown face image
        img_path = os.path.join(UNKNOWN_DIR, filename)
        img      = cv2.imread(img_path)
        if img is not None:
            encoding = encode_face_from_image(img)
            if encoding:
                save_encoding(sid, encoding)
                global known_faces_cache
                known_faces_cache = load_known_faces()
        # Delete from unknown folder
        try: os.remove(img_path)
        except: pass
        flash(f"'{name}' registered successfully!", "success")
        return redirect(url_for("students"))

    return render_template("register_unknown.html", filename=filename,
                           camera_active=camera_active,
                           unknown_face_count=get_unknown_face_count())

# ── Reports ───────────────────────────────────────────────────────────────────

@app.route("/reports")
def reports():
    fd = request.args.get("date","")
    fs = request.args.get("student","")
    return render_template("reports.html",
        records=get_attendance_report(filter_date=fd or None, filter_student=fs or None),
        filter_date=fd, filter_student=fs,
        camera_active=camera_active,
        unknown_face_count=get_unknown_face_count())

@app.route("/reports/export")
def export_csv():
    fd = request.args.get("date",""); fs = request.args.get("student","")
    records = get_attendance_report(filter_date=fd or None, filter_student=fs or None)
    output  = io.StringIO()
    writer  = csv.DictWriter(output, fieldnames=["name","roll_no","dept","date","time","status"])
    writer.writeheader(); writer.writerows(records); output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name=f"attendance_{fd or 'all'}.csv")

@app.route("/manual_mark", methods=["POST"])
def manual_mark():
    sid = request.form.get("student_id")
    if sid:
        mark_attendance(int(sid))
        flash("Attendance marked manually.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, threaded=True)
