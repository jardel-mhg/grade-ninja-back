import json
from fastapi import APIRouter, HTTPException

from database import get_db
from schemas import SessionCreate, SessionUpdate, SessionResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

DEFAULT_GRADES = [
    {"id": 1, "name": "A", "color": "#00b894"},
    {"id": 2, "name": "B", "color": "#6c5ce7"},
    {"id": 3, "name": "C", "color": "#fdcb6e"},
    {"id": 4, "name": "D", "color": "#e17055"},
    {"id": 5, "name": "E", "color": "#d63031"},
]


def _row_to_session(row) -> dict:
    train_result_raw = row["train_result"]
    train_result = json.loads(train_result_raw) if train_result_raw else None

    return SessionResponse(
        id=row["id"],
        name=row["name"],
        date=row["date"],
        status=row["status"],
        grades=json.loads(row["grades"]),
        gradeCount=row["grade_count"],
        targetColumn=row["target_column"],
        featureColumns=json.loads(row["feature_columns"]),
        datasetFilename=row["dataset_filename"],
        rowCount=row["row_count"],
        labeledCount=row["labeled_count"],
        trainResult=train_result,
        createdAt=row["created_at"],
    ).model_dump()


@router.get("")
def list_sessions():
    db = get_db()
    rows = db.execute("SELECT * FROM sessions ORDER BY id").fetchall()
    return [_row_to_session(r) for r in rows]


@router.get("/{session_id}")
def get_session(session_id: int):
    db = get_db()
    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return _row_to_session(row)


@router.post("", status_code=201)
def create_session(body: SessionCreate):
    db = get_db()
    now = body.date or __import__("datetime").date.today().isoformat()
    grades = [g.model_dump() for g in body.grades] if body.grades else DEFAULT_GRADES
    grade_count = body.gradeCount if body.gradeCount is not None else len(grades)

    cursor = db.execute(
        """INSERT INTO sessions
           (name, date, status, grades, grade_count, target_column,
            feature_columns, dataset_filename, row_count, labeled_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)""",
        (
            body.name,
            now,
            body.status,
            json.dumps(grades),
            grade_count,
            body.targetColumn,
            json.dumps(body.featureColumns),
            body.datasetFilename,
            now,
        ),
    )
    db.commit()

    row = db.execute("SELECT * FROM sessions WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _row_to_session(row)


@router.put("/{session_id}")
def update_session(session_id: int, body: SessionUpdate):
    db = get_db()
    existing = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Session not found")

    updates = body.model_dump(exclude_none=True)
    if not updates:
        return _row_to_session(existing)

    # Map camelCase fields to snake_case columns
    column_map = {
        "name": "name",
        "date": "date",
        "status": "status",
        "grades": "grades",
        "gradeCount": "grade_count",
        "targetColumn": "target_column",
        "featureColumns": "feature_columns",
        "datasetFilename": "dataset_filename",
        "rowCount": "row_count",
        "labeledCount": "labeled_count",
        "trainResult": "train_result",
    }

    set_parts = []
    values = []
    for camel, snake in column_map.items():
        if camel in updates:
            val = updates[camel]
            if camel == "grades":
                val = json.dumps([g.model_dump() if hasattr(g, "model_dump") else g for g in val])
            elif camel == "featureColumns":
                val = json.dumps(val)
            elif camel == "trainResult" and val is not None:
                val = json.dumps(val)
            set_parts.append(f"{snake} = ?")
            values.append(val)

    if set_parts:
        values.append(session_id)
        db.execute(
            f"UPDATE sessions SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        db.commit()

    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_session(row)


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: int):
    db = get_db()
    existing = db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Session not found")
    db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    db.commit()
