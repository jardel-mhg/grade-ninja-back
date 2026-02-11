import json
from pathlib import Path

from fastapi import APIRouter

from schemas import Dataset, DatasetCreate

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

MOCK_FILE = Path(__file__).parent.parent / "mocks" / "datasets.json"


def _load_mock():
    with open(MOCK_FILE) as f:
        return json.load(f)


@router.get("", response_model=list[Dataset], summary="List all datasets")
def list_datasets():
    """Returns all datasets with their hides and defect data."""
    data = _load_mock()
    return data["datasets"]


@router.get("/{dataset_id}", response_model=Dataset, summary="Get dataset by ID")
def get_dataset(dataset_id: str):
    """Returns a single dataset with all its hides and defect details."""
    data = _load_mock()
    for dataset in data["datasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    return {"error": "Dataset not found"}, 404


@router.post("", response_model=Dataset, summary="Create a new dataset", status_code=201)
def create_dataset(body: DatasetCreate):
    """Creates a new empty dataset linked to a project."""
    data = _load_mock()
    dataset = data["datasets"][1].copy()
    dataset["id"] = "ds_new"
    dataset["name"] = body.name
    dataset["project_id"] = body.project_id
    return dataset
