from __future__ import annotations

from pydantic import BaseModel, Field


# --- Projects ---

class GradeRule(BaseModel):
    grade: str = Field(example="B")
    description: str = Field(example="Premium quality, minimal defects")
    max_total_defects: int | None = Field(example=2)
    forbidden_defects: list[str] = Field(example=["hole", "loose_grain"])


class ProjectRules(BaseModel):
    grades: list[str] = Field(example=["B", "C", "D", "E", "EN"])
    defect_features: list[str] = Field(example=["brand_mark", "scratch", "hole"])
    grade_rules: list[GradeRule]


class Project(BaseModel):
    id: str = Field(example="proj_001")
    name: str = Field(example="Standard Automotive Leather")
    description: str = Field(example="Grading rules for automotive upholstery leather")
    created_at: str = Field(example="2026-01-15T10:30:00Z")
    updated_at: str = Field(example="2026-02-01T14:20:00Z")
    rules: ProjectRules


class ProjectCreate(BaseModel):
    name: str = Field(example="New Leather Project")
    description: str = Field(example="Custom grading rules")
    rules: ProjectRules | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, example="Updated Project Name")
    description: str | None = Field(default=None, example="Updated description")
    rules: ProjectRules | None = None


# --- Datasets ---

class Defect(BaseModel):
    type: str = Field(example="scratch")
    x: int = Field(example=120)
    y: int = Field(example=340)
    severity: str = Field(example="minor")


class Hide(BaseModel):
    id: str = Field(example="hide_001")
    serial_number: str = Field(example="LH-2026-00142")
    area_sqft: float = Field(example=48.5)
    defects: list[Defect]
    grade: str = Field(example="B")
    graded_at: str = Field(example="2026-01-20T09:15:00Z")


class Dataset(BaseModel):
    id: str = Field(example="ds_001")
    name: str = Field(example="January 2026 Batch")
    project_id: str = Field(example="proj_001")
    created_at: str = Field(example="2026-01-20T08:00:00Z")
    hide_count: int = Field(example=3)
    hides: list[Hide]


class DatasetCreate(BaseModel):
    name: str = Field(example="March 2026 Batch")
    project_id: str = Field(example="proj_001")


# --- Train ---

class TrainRequest(BaseModel):
    project_id: str = Field(example="proj_001")
    dataset_id: str = Field(example="ds_001")


class TrainResponse(BaseModel):
    job_id: str = Field(example="train_job_001")
    status: str = Field(example="running")
    project_id: str = Field(example="proj_001")
    dataset_id: str = Field(example="ds_001")
    created_at: str = Field(example="2026-02-10T12:00:00Z")
    message: str = Field(example="Training job started successfully")


class ConfusionMatrix(BaseModel):
    B: dict[str, int] = Field(example={"B": 42, "C": 3, "D": 0, "E": 0, "EN": 0})
    C: dict[str, int] = Field(example={"B": 2, "C": 38, "D": 4, "E": 1, "EN": 0})
    D: dict[str, int] = Field(example={"B": 0, "C": 3, "D": 35, "E": 5, "EN": 2})
    E: dict[str, int] = Field(example={"B": 0, "C": 0, "D": 4, "E": 30, "EN": 6})
    EN: dict[str, int] = Field(example={"B": 0, "C": 0, "D": 1, "E": 3, "EN": 21})


class TrainMetrics(BaseModel):
    accuracy: float = Field(example=0.87)
    precision: float = Field(example=0.85)
    recall: float = Field(example=0.83)
    f1_score: float = Field(example=0.84)
    confusion_matrix: ConfusionMatrix


class TrainStatus(BaseModel):
    job_id: str = Field(example="train_job_001")
    status: str = Field(example="completed")
    project_id: str = Field(example="proj_001")
    dataset_id: str = Field(example="ds_001")
    created_at: str = Field(example="2026-02-10T12:00:00Z")
    completed_at: str = Field(example="2026-02-10T12:05:30Z")
    metrics: TrainMetrics
    model_version: str = Field(example="v0.1.0")


# --- Inference ---

class InferenceRequest(BaseModel):
    project_id: str = Field(example="proj_001")
    serial_number: str = Field(example="LH-2026-00250")
    defects: list[Defect] = Field(example=[
        {"type": "scratch", "x": 120, "y": 340, "severity": "minor"},
        {"type": "tick_mark", "x": 450, "y": 210, "severity": "minor"},
    ])


class InferenceResponse(BaseModel):
    hide_id: str = Field(example="hide_new_001")
    serial_number: str = Field(example="LH-2026-00250")
    predicted_grade: str = Field(example="C")
    confidence: float = Field(example=0.82)
    grade_probabilities: dict[str, float] = Field(example={
        "B": 0.08, "C": 0.82, "D": 0.07, "E": 0.02, "EN": 0.01,
    })
    defects_analyzed: list[Defect]
    model_version: str = Field(example="v0.1.0")
    inferred_at: str = Field(example="2026-02-10T14:30:00Z")
