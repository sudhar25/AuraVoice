import sqlite3
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
88
# Define the path to the database file
db_path = os.path.join(script_dir, 'pronunciation.db')

def create_database():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            file_path TEXT NOT NULL,
            dtw_score REAL,
            lev_score INTEGER,
            result TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_attempt(word, timestamp, file_path, dtw_score, lev_score, result):
    """Save a pronunciation attempt to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attempts (word, timestamp, file_path, dtw_score, lev_score, result)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (word, timestamp, file_path, dtw_score, lev_score, result))
    conn.commit()
    conn.close()

def view_attempts():
    """View all pronunciation attempts from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT word, timestamp, file_path, result FROM attempts ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        print(f"Word: {row[0]} | Time: {row[1]} | File: {row[2]} | Result: {row[3]}")