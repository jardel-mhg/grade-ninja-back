from __future__ import annotations

from pydantic import BaseModel, Field


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
