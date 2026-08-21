import sqlite3

DB_NAME = "emotions.db"

def create_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)

    # Videos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            title TEXT
        )
    """)

    # Emotions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            video_id TEXT,
            emotion TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
    """)

    conn.commit()
    conn.close()

def insert_student(student_id, name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO students (id, name) VALUES (?, ?)", (student_id, name))
    conn.commit()
    conn.close()

def insert_video(video_id, title):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO videos (id, title) VALUES (?, ?)", (video_id, title))
    conn.commit()
    conn.close()

def insert_emotion(student_id, video_id, emotion):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotions (student_id, video_id, emotion) VALUES (?, ?, ?)", 
                   (student_id, video_id, emotion))
    conn.commit()
    conn.close()
