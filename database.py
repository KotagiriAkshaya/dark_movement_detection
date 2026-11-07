
import sqlite3

DB = "results.db"


def init_db():
    with sqlite3.connect(DB) as c:

        # Existing video analysis history
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT,
                result_video TEXT,
                simulated INTEGER,
                duration REAL,
                human_seconds INTEGER,
                created TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # New live webcam detection history
        c.execute("""
            CREATE TABLE IF NOT EXISTS live_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_count INTEGER,
                movement INTEGER,
                screenshot TEXT,
                created TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_run(original_name, result_video, simulated, duration, human_seconds):
    with sqlite3.connect(DB) as c:
        c.execute(
            """INSERT INTO runs
            (original_name, result_video, simulated, duration, human_seconds)
            VALUES (?, ?, ?, ?, ?)""",
            (original_name, result_video, int(simulated), duration, human_seconds),
        )


def get_runs():
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        return c.execute(
            "SELECT * FROM runs ORDER BY id DESC"
        ).fetchall()


# Save live webcam detection
def add_live_detection(person_count, movement, screenshot=None):
    with sqlite3.connect(DB) as c:
        c.execute(
            """INSERT INTO live_detections
            (person_count, movement, screenshot)
            VALUES (?, ?, ?)""",
            (person_count, int(movement), screenshot),
        )


# Retrieve live webcam history
def get_live_detections():
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        return c.execute(
            "SELECT * FROM live_detections ORDER BY id DESC"
        ).fetchall()
