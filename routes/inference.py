import json
from pathlib import Path

from fastapi import APIRouter

from schemas import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/api/inference", tags=["Inference"])

MOCK_FILE = Path(__file__).parent.parent / "mocks" / "inference.json"


def _load_mock():
    with open(MOCK_FILE) as f:
        return json.load(f)


@router.post("", response_model=InferenceResponse, summary="Predict hide grade")
def run_inference(body: InferenceRequest):
    """Submits hide defect features and returns the predicted grade (B/C/D/E/EN) with confidence scores and per-grade probabilities."""
    data = _load_mock()
    return data["inference_response"]
