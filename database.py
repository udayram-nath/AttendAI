import sqlite3
from datetime import date, datetime

DB_PATH = "attendance.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            dept TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'present',
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            UNIQUE(student_id, date)
        );
    """)
    # Add dept column if missing (migration)
    try:
        conn.execute("ALTER TABLE students ADD COLUMN dept TEXT DEFAULT ''")
        conn.commit()
    except:
        pass
    conn.commit()
    conn.close()

def add_student(name, roll_no, dept=''):
    conn = get_db()
    try:
        conn.execute("INSERT INTO students (name, roll_no, dept) VALUES (?, ?, ?)", (name, roll_no, dept))
        conn.commit()
        sid = conn.execute("SELECT id FROM students WHERE roll_no=?", (roll_no,)).fetchone()["id"]
        return sid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def save_encoding(student_id, encoding_bytes):
    conn = get_db()
    conn.execute("DELETE FROM face_encodings WHERE student_id=?", (student_id,))
    conn.execute("INSERT INTO face_encodings (student_id, encoding) VALUES (?, ?)", (student_id, encoding_bytes))
    conn.commit()
    conn.close()

def get_all_encodings():
    conn = get_db()
    rows = conn.execute("""
        SELECT fe.encoding, s.id, s.name, s.roll_no
        FROM face_encodings fe JOIN students s ON fe.student_id = s.id
    """).fetchall()
    conn.close()
    return rows

def mark_attendance(student_id):
    today = date.today().isoformat()
    now   = datetime.now().strftime("%H:%M:%S")
    conn  = get_db()
    try:
        conn.execute("INSERT INTO attendance (student_id, date, time) VALUES (?,?,?)", (student_id, today, now))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def get_attendance_report(filter_date=None, filter_student=None):
    conn  = get_db()
    query = "SELECT s.name, s.roll_no, s.dept, a.date, a.time, a.status FROM attendance a JOIN students s ON a.student_id=s.id WHERE 1=1"
    params = []
    if filter_date:
        query += " AND a.date=?"; params.append(filter_date)
    if filter_student:
        query += " AND (s.name LIKE ? OR s.roll_no LIKE ?)"; params.extend([f"%{filter_student}%"]*2)
    query += " ORDER BY a.date DESC, a.time DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_students():
    conn = get_db()
    rows = conn.execute("""
        SELECT s.*, COUNT(fe.id) as has_face
        FROM students s LEFT JOIN face_encodings fe ON s.id=fe.student_id
        GROUP BY s.id ORDER BY s.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_student(student_id):
    conn = get_db()
    conn.execute("DELETE FROM students WHERE id=?", (student_id,))
    conn.commit()
    conn.close()

def get_today_stats():
    today   = date.today().isoformat()
    conn    = get_db()
    total   = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    present = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
    conn.close()
    return {"total": total, "present": present, "absent": total - present, "date": today}
