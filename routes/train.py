import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from schemas import TrainRequest, TrainResponse, TrainStatus
from ml.rf import train_random_forest

router = APIRouter(prefix="/api/train", tags=["Training"])

MOCK_FILE = Path(__file__).parent.parent / "mocks" / "train.json"


def _load_mock():
    with open(MOCK_FILE) as f:
        return json.load(f)


@router.post("", response_model=TrainResponse, summary="Start a training job")
def start_training(body: TrainRequest):
    df = pd.DataFrame(body.rows)

    now = datetime.now(timezone.utc)
    job_id = f"train_{body.sessionId}_{int(now.timestamp())}"

    try:
        result = train_random_forest(df, body.targetColumn, body.featureColumns)
    except Exception as e:
        print(f"Training failed: {type(e).__name__}: {e}")
        return TrainResponse(
            job_id=job_id,
            status="failed",
            sessionId=body.sessionId,
            created_at=now.isoformat(),
            message=str(e),
        )

    print(f"Training done — accuracy: {result['accuracy']}, "
          f"train: {result['train_size']}, test: {result['test_size']}")

    return TrainResponse(
        job_id=job_id,
        status="completed",
        sessionId=body.sessionId,
        created_at=now.isoformat(),
        message=f"Training completed — accuracy: {result['accuracy']}",
    )


@router.get("/{job_id}", response_model=TrainStatus, summary="Get training job status")
def get_training_status(job_id: str):
    data = _load_mock()
    result = data["train_status"].copy()
    result["job_id"] = job_id
    return result
