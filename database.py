import json
import sqlite3
import os

DB_PATH = os.environ.get("GRADE_NINJA_DB", "data/grade_ninja.db")

_connection: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA foreign_keys=ON")
    return _connection


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Configured',
            grades TEXT NOT NULL DEFAULT '[]',
            grade_count INTEGER NOT NULL DEFAULT 5,
            target_column TEXT,
            feature_columns TEXT NOT NULL DEFAULT '[]',
            dataset_filename TEXT,
            row_count INTEGER NOT NULL DEFAULT 0,
            labeled_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dataset_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            target_column TEXT NOT NULL DEFAULT '',
            data TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_dataset_rows_session
            ON dataset_rows(session_id);
    """)
    db.commit()

    # Seed with mock data if tables are empty
    count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    if count == 0:
        _seed(db)


def _seed(db: sqlite3.Connection):
    mock_sessions = [
        {
            "name": "JBS Navirai — Wringer 1",
            "date": "2026-02-11",
            "status": "Configured",
            "grades": [
                {"id": 1, "name": "A", "color": "#00b894"},
                {"id": 2, "name": "B", "color": "#6c5ce7"},
                {"id": 3, "name": "C", "color": "#fdcb6e"},
                {"id": 4, "name": "D", "color": "#e17055"},
                {"id": 5, "name": "E", "color": "#d63031"},
            ],
            "target_column": "grade",
            "feature_columns": ["count_br", "count_ct", "count_d2", "count_dc", "count_he", "count_hs", "count_rw", "count_w"],
            "dataset_filename": "jbs-navirai-metrics.csv",
            "rows": [
                {"grade": "B", "area_sqft": 44.5, "count_br": 14, "count_ct": 86, "count_d2": 5, "count_dc": 0, "count_he": 0, "count_hs": 23, "count_rw": 4, "count_w": 114, "area_br": 0.002, "imageSrc": "/hide1.webp"},
                {"grade": "E", "area_sqft": 51.0, "count_br": 11, "count_ct": 129, "count_d2": 36, "count_dc": 14, "count_he": 0, "count_hs": 24, "count_rw": 20, "count_w": 34, "area_br": 0.003, "imageSrc": "/hide2.webp"},
                {"grade": "C", "area_sqft": 38.2, "count_br": 3, "count_ct": 42, "count_d2": 0, "count_dc": 0, "count_he": 1, "count_hs": 8, "count_rw": 1, "count_w": 88, "area_br": 0.001, "imageSrc": "/hide1.webp"},
                {"grade": "C", "area_sqft": 46.1, "count_br": 22, "count_ct": 97, "count_d2": 8, "count_dc": 2, "count_he": 0, "count_hs": 19, "count_rw": 7, "count_w": 65, "area_br": 0.004, "imageSrc": "/hide2.webp"},
                {"grade": "D", "area_sqft": 52.3, "count_br": 8, "count_ct": 110, "count_d2": 18, "count_dc": 5, "count_he": 2, "count_hs": 31, "count_rw": 12, "count_w": 42, "area_br": 0.002, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 41.7, "count_br": 28, "count_ct": 145, "count_d2": 22, "count_dc": 9, "count_he": 0, "count_hs": 27, "count_rw": 15, "count_w": 28, "area_br": 0.006, "imageSrc": "/hide2.webp"},
                {"grade": "", "area_sqft": 49.8, "count_br": 5, "count_ct": 63, "count_d2": 2, "count_dc": 0, "count_he": 0, "count_hs": 12, "count_rw": 3, "count_w": 95, "area_br": 0.001, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 43.0, "count_br": 1, "count_ct": 18, "count_d2": 0, "count_dc": 0, "count_he": 0, "count_hs": 3, "count_rw": 0, "count_w": 150, "area_br": 0.000, "imageSrc": "/hide2.webp"},
                {"grade": "", "area_sqft": 47.5, "count_br": 16, "count_ct": 102, "count_d2": 11, "count_dc": 3, "count_he": 1, "count_hs": 20, "count_rw": 9, "count_w": 55, "area_br": 0.003, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 45.2, "count_br": 9, "count_ct": 71, "count_d2": 4, "count_dc": 1, "count_he": 0, "count_hs": 15, "count_rw": 5, "count_w": 82, "area_br": 0.002, "imageSrc": "/hide2.webp"},
            ],
        },
        {
            "name": "JBS Leon — Wringer 2",
            "date": "2026-02-08",
            "status": "Configured",
            "grades": [
                {"id": 1, "name": "A", "color": "#00b894"},
                {"id": 2, "name": "B", "color": "#6c5ce7"},
                {"id": 3, "name": "C", "color": "#fdcb6e"},
                {"id": 4, "name": "D", "color": "#e17055"},
                {"id": 5, "name": "E", "color": "#d63031"},
            ],
            "target_column": "grade",
            "feature_columns": ["count_br", "count_ct", "count_d2", "count_he", "count_hs", "count_rw", "count_w"],
            "dataset_filename": "jbs-leon-metrics.csv",
            "rows": [
                {"grade": "C", "area_sqft": 40.1, "count_br": 18, "count_ct": 88, "count_d2": 7, "count_he": 1, "count_hs": 21, "count_rw": 8, "count_w": 70, "imageSrc": "/hide1.webp"},
                {"grade": "B", "area_sqft": 48.3, "count_br": 6, "count_ct": 55, "count_d2": 2, "count_he": 0, "count_hs": 10, "count_rw": 2, "count_w": 105, "imageSrc": "/hide2.webp"},
                {"grade": "D", "area_sqft": 39.4, "count_br": 30, "count_ct": 155, "count_d2": 25, "count_he": 3, "count_hs": 28, "count_rw": 18, "count_w": 22, "imageSrc": "/hide1.webp"},
                {"grade": "A", "area_sqft": 45.6, "count_br": 2, "count_ct": 22, "count_d2": 0, "count_he": 0, "count_hs": 4, "count_rw": 0, "count_w": 140, "imageSrc": "/hide2.webp"},
                {"grade": "E", "area_sqft": 50.2, "count_br": 35, "count_ct": 170, "count_d2": 40, "count_he": 5, "count_hs": 32, "count_rw": 22, "count_w": 15, "imageSrc": "/hide1.webp"},
                {"grade": "B", "area_sqft": 44.8, "count_br": 7, "count_ct": 60, "count_d2": 3, "count_he": 0, "count_hs": 11, "count_rw": 3, "count_w": 98, "imageSrc": "/hide2.webp"},
                {"grade": "C", "area_sqft": 42.5, "count_br": 20, "count_ct": 95, "count_d2": 10, "count_he": 2, "count_hs": 18, "count_rw": 9, "count_w": 60, "imageSrc": "/hide1.webp"},
                {"grade": "D", "area_sqft": 37.9, "count_br": 26, "count_ct": 138, "count_d2": 19, "count_he": 1, "count_hs": 25, "count_rw": 14, "count_w": 30, "imageSrc": "/hide2.webp"},
                {"grade": "A", "area_sqft": 46.7, "count_br": 1, "count_ct": 15, "count_d2": 0, "count_he": 0, "count_hs": 2, "count_rw": 0, "count_w": 155, "imageSrc": "/hide1.webp"},
                {"grade": "C", "area_sqft": 43.3, "count_br": 15, "count_ct": 82, "count_d2": 6, "count_he": 1, "count_hs": 16, "count_rw": 6, "count_w": 72, "imageSrc": "/hide2.webp"},
            ],
        },
        {
            "name": "Bader Leon — Wringer 1",
            "date": "2026-02-01",
            "status": "Configured",
            "grades": [
                {"id": 1, "name": "TR-1", "color": "#00b894"},
                {"id": 2, "name": "TR-2", "color": "#6c5ce7"},
                {"id": 3, "name": "TR-3", "color": "#e17055"},
            ],
            "target_column": "grade",
            "feature_columns": ["count_br", "count_ct", "count_d2", "count_he", "count_w"],
            "dataset_filename": "bader-leon-metrics.csv",
            "rows": [
                {"grade": "TR-1", "area_sqft": 42.0, "count_br": 10, "count_ct": 75, "count_d2": 4, "count_he": 0, "count_w": 85, "imageSrc": "/hide1.webp"},
                {"grade": "TR-2", "area_sqft": 47.1, "count_br": 25, "count_ct": 130, "count_d2": 15, "count_he": 2, "count_w": 35, "imageSrc": "/hide2.webp"},
                {"grade": "TR-1", "area_sqft": 44.9, "count_br": 3, "count_ct": 30, "count_d2": 1, "count_he": 0, "count_w": 120, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 50.5, "count_br": 18, "count_ct": 100, "count_d2": 9, "count_he": 1, "count_w": 55, "imageSrc": "/hide2.webp"},
                {"grade": "", "area_sqft": 39.8, "count_br": 32, "count_ct": 160, "count_d2": 28, "count_he": 4, "count_w": 18, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 46.3, "count_br": 12, "count_ct": 85, "count_d2": 6, "count_he": 1, "count_w": 75, "imageSrc": "/hide2.webp"},
                {"grade": "", "area_sqft": 41.5, "count_br": 8, "count_ct": 58, "count_d2": 3, "count_he": 0, "count_w": 92, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 48.7, "count_br": 22, "count_ct": 120, "count_d2": 14, "count_he": 3, "count_w": 40, "imageSrc": "/hide2.webp"},
                {"grade": "", "area_sqft": 37.2, "count_br": 38, "count_ct": 180, "count_d2": 35, "count_he": 5, "count_w": 10, "imageSrc": "/hide1.webp"},
                {"grade": "", "area_sqft": 43.8, "count_br": 14, "count_ct": 90, "count_d2": 7, "count_he": 1, "count_w": 68, "imageSrc": "/hide2.webp"},
            ],
        },
    ]

    for session_data in mock_sessions:
        rows = session_data.pop("rows")
        grades = session_data["grades"]
        target_col = session_data["target_column"]
        labeled_count = sum(1 for r in rows if r.get(target_col))

        cursor = db.execute(
            """INSERT INTO sessions
               (name, date, status, grades, grade_count, target_column,
                feature_columns, dataset_filename, row_count, labeled_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_data["name"],
                session_data["date"],
                session_data["status"],
                json.dumps(grades),
                len(grades),
                target_col,
                json.dumps(session_data["feature_columns"]),
                session_data["dataset_filename"],
                len(rows),
                labeled_count,
                session_data["date"],
            ),
        )
        session_id = cursor.lastrowid

        db.executemany(
            """INSERT INTO dataset_rows (session_id, target_column, data)
               VALUES (?, ?, ?)""",
            [
                (session_id, row.get(target_col, ""), json.dumps(row))
                for row in rows
            ],
        )

    db.commit()
