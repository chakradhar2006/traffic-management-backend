import sqlite3
import os
import io
import hashlib
import boto3
from botocore.client import Config
from datetime import datetime

# ── S3 client ─────────────────────────────────────────────────────────────────
BUCKET_NAME     = os.environ.get("BUCKET_NAME", "")
BUCKET_REGION   = os.environ.get("BUCKET_REGION", "")
BUCKET_ENDPOINT = os.environ.get("BUCKET_ENDPOINT", "")
BUCKET_ACCESS_KEY = os.environ.get("BUCKET_ACCESS_KEY", "")
BUCKET_SECRET_KEY = os.environ.get("BUCKET_SECRET_KEY", "")

def _get_s3_client():
    return boto3.client(
        "s3",
        region_name=BUCKET_REGION,
        endpoint_url=BUCKET_ENDPOINT,
        aws_access_key_id=BUCKET_ACCESS_KEY,
        aws_secret_access_key=BUCKET_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

# Get the absolute path to the database directory based on the location of this script
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
DB_PATH = os.path.join(DB_DIR, "traffic.db")

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Vehicles log (for historical density tracking)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane_number INTEGER NOT NULL,
            vehicle_count INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Emergency events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane_number INTEGER NOT NULL,
            vehicle_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Rule breakers (Zebra crossing violations)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rule_breakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane_number INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Users (Admin vs Normal Users)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            user_id TEXT UNIQUE NOT NULL,
            region TEXT NOT NULL,
            phone TEXT NOT NULL,
            experience TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')


    
    conn.commit()
    conn.close()

def log_vehicle_count(lane, count):
    """Log the raw count of vehicles for a lane."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO vehicles_log (lane_number, vehicle_count) VALUES (?, ?)", (lane, count))
    conn.commit()
    conn.close()

def log_emergency(lane, vehicle_type):
    """Log an emergency vehicle event."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emergency_events (lane_number, vehicle_type) VALUES (?, ?)", (lane, vehicle_type))
    conn.commit()
    conn.close()

def log_rule_breaker(lane, filename, image_bytes):
    """Upload violation image to S3 and log the S3 URL in the database."""
    s3_key = f"violations/{filename}"
    s3_url = filename  # fallback: store filename if upload fails
    try:
        s3 = _get_s3_client()
        s3.upload_fileobj(
            io.BytesIO(image_bytes),
            BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "image/jpeg"},
        )
        s3_url = s3_key
    except Exception as e:
        print(f"S3 upload failed for {filename}: {e}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO rule_breakers (lane_number, image_path) VALUES (?, ?)", (lane, s3_url))
    conn.commit()
    conn.close()

def get_recent_rule_breakers(limit=20):
    """Fetch the latest zebra crossing violations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, lane_number, image_path, timestamp FROM rule_breakers ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "lane": r[1], "image_path": r[2], "timestamp": r[3]} for r in rows]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(first_name, last_name, user_id, region, phone, experience, password, role='user'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (first_name, last_name, user_id, region, phone, experience, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (first_name, last_name, user_id, region, phone, experience, hash_password(password), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT first_name, last_name, user_id, region, phone, experience, password_hash, role FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "first_name": row[0],
            "last_name": row[1],
            "user_id": row[2],
            "region": row[3],
            "phone": row[4],
            "experience": row[5],
            "password_hash": row[6],
            "role": row[7]
        }
    return None


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
