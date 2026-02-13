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

class GradeConfig(BaseModel):
    id: int = Field(example=1)
    name: str = Field(example="A")
    color: str = Field(example="#00b894")


class TrainRequest(BaseModel):
    sessionId: int = Field(example=1)
    targetColumn: str = Field(example="grade")
    featureColumns: list[str] = Field(example=["count_br", "count_ct"])
    grades: list[GradeConfig]
    rows: list[dict]


class ClassMetrics(BaseModel):
    precision: float = Field(example=0.85)
    recall: float = Field(example=0.83)
    f1_score: float = Field(example=0.84)
    support: int = Field(example=42)


class TrainResultMetrics(BaseModel):
    accuracy: float = Field(example=0.87)
    precision: float = Field(example=0.85)
    recall: float = Field(example=0.83)
    f1_score: float = Field(example=0.84)
    confusionMatrix: dict[str, dict[str, int]]
    classificationReport: dict[str, ClassMetrics]
    featureImportances: dict[str, float]
    targetDistribution: dict[str, int]
    trainSize: int = Field(example=80)
    testSize: int = Field(example=20)


class TrainResponse(BaseModel):
    job_id: str = Field(example="train_job_001")
    status: str = Field(example="running")
    sessionId: int = Field(example=1)
    created_at: str = Field(example="2026-02-10T12:00:00Z")
    message: str = Field(example="Training job started successfully")
    metrics: TrainResultMetrics | None = None


class PredictRequest(BaseModel):
    featureColumns: list[str]
    rows: list[dict]


class PredictResponse(BaseModel):
    predictions: list[str]


# --- Sessions ---

class SessionCreate(BaseModel):
    name: str
    date: str | None = None
    status: str = "Configured"
    grades: list[GradeConfig] | None = None
    gradeCount: int | None = None
    targetColumn: str | None = None
    featureColumns: list[str] = []
    datasetFilename: str | None = None


class SessionUpdate(BaseModel):
    name: str | None = None
    date: str | None = None
    status: str | None = None
    grades: list[GradeConfig] | None = None
    gradeCount: int | None = None
    targetColumn: str | None = None
    featureColumns: list[str] | None = None
    datasetFilename: str | None = None
    rowCount: int | None = None
    labeledCount: int | None = None
    trainResult: dict | None = None


class SessionResponse(BaseModel):
    id: int
    name: str
    date: str
    status: str
    grades: list[GradeConfig]
    gradeCount: int
    targetColumn: str | None
    featureColumns: list[str]
    datasetFilename: str | None
    rowCount: int
    labeledCount: int
    trainResult: dict | None = None
    createdAt: str


class RowsBulkCreate(BaseModel):
    rows: list[dict]


class RowUpdate(BaseModel):
    targetColumn: str | None = None
    data: dict | None = None


class RowResponse(BaseModel):
    id: int
    sessionId: int
    targetColumn: str
    # remaining CSV columns are merged at top level via dict


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
