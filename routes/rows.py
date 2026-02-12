import json
from fastapi import APIRouter, HTTPException

from database import get_db
from schemas import RowsBulkCreate, RowUpdate

router = APIRouter(prefix="/api/sessions/{session_id}/rows", tags=["rows"])


def _row_to_dict(row) -> dict:
    data = json.loads(row["data"])
    return {
        "id": row["id"],
        "sessionId": row["session_id"],
        "targetColumn": row["target_column"],
        **data,
    }


def _assert_session(session_id: int):
    db = get_db()
    s = db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")


def _update_session_counts(db, session_id: int):
    row_count = db.execute(
        "SELECT COUNT(*) FROM dataset_rows WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    labeled_count = db.execute(
        "SELECT COUNT(*) FROM dataset_rows WHERE session_id = ? AND target_column != ''",
        (session_id,),
    ).fetchone()[0]
    db.execute(
        "UPDATE sessions SET row_count = ?, labeled_count = ? WHERE id = ?",
        (row_count, labeled_count, session_id),
    )


@router.get("")
def list_rows(session_id: int):
    _assert_session(session_id)
    db = get_db()
    rows = db.execute(
        "SELECT * FROM dataset_rows WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.post("/bulk", status_code=201)
def bulk_create_rows(session_id: int, body: RowsBulkCreate):
    _assert_session(session_id)
    db = get_db()

    # Get session target column for extracting label
    session = db.execute("SELECT target_column FROM sessions WHERE id = ?", (session_id,)).fetchone()
    target_col = session["target_column"] or ""

    params = []
    for row in body.rows:
        target_value = row.get("targetColumn", row.get(target_col, "")) if target_col else row.get("targetColumn", "")
        params.append((session_id, target_value or "", json.dumps(row)))

    db.executemany(
        "INSERT INTO dataset_rows (session_id, target_column, data) VALUES (?, ?, ?)",
        params,
    )
    _update_session_counts(db, session_id)
    db.commit()
    return {"inserted": len(body.rows)}


@router.put("/{row_id}")
def update_row(session_id: int, row_id: int, body: RowUpdate):
    _assert_session(session_id)
    db = get_db()

    existing = db.execute(
        "SELECT * FROM dataset_rows WHERE id = ? AND session_id = ?",
        (row_id, session_id),
    ).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Row not found")

    current_data = json.loads(existing["data"])
    new_target = body.targetColumn if body.targetColumn is not None else existing["target_column"]

    if body.data:
        current_data.update(body.data)

    # Also update the target column key inside the data JSON
    session = db.execute("SELECT target_column FROM sessions WHERE id = ?", (session_id,)).fetchone()
    target_col_name = session["target_column"]
    if target_col_name and body.targetColumn is not None:
        current_data[target_col_name] = body.targetColumn

    db.execute(
        "UPDATE dataset_rows SET target_column = ?, data = ? WHERE id = ?",
        (new_target, json.dumps(current_data), row_id),
    )
    _update_session_counts(db, session_id)
    db.commit()

    updated = db.execute("SELECT * FROM dataset_rows WHERE id = ?", (row_id,)).fetchone()
    return _row_to_dict(updated)


@router.delete("")
def delete_rows(session_id: int):
    _assert_session(session_id)
    db = get_db()
    db.execute("DELETE FROM dataset_rows WHERE session_id = ?", (session_id,))
    _update_session_counts(db, session_id)
    db.commit()
    return {"deleted": True}
