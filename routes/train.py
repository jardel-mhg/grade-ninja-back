import json
from pathlib import Path

from fastapi import APIRouter

from schemas import TrainRequest, TrainResponse, TrainStatus

router = APIRouter(prefix="/api/train", tags=["Training"])

MOCK_FILE = Path(__file__).parent.parent / "mocks" / "train.json"


def _load_mock():
    with open(MOCK_FILE) as f:
        return json.load(f)


@router.post("", response_model=TrainResponse, summary="Start a training job")
def start_training(body: TrainRequest):
    """Submits a training job for the given project and dataset. Returns the job ID and initial status."""
    data = _load_mock()
    return data["train_response"]


@router.get("/{job_id}", response_model=TrainStatus, summary="Get training job status")
def get_training_status(job_id: str):
    """Returns the current status and metrics of a training job. Once completed, includes accuracy, precision, recall, F1 score, and confusion matrix."""
    data = _load_mock()
    result = data["train_status"].copy()
    result["job_id"] = job_id
    return result
