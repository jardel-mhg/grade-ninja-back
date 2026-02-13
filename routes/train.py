import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from database import get_db
from schemas import (
    TrainRequest,
    TrainResponse,
    TrainResultMetrics,
    PredictRequest,
    PredictResponse,
)
from ml.rf import train_random_forest

router = APIRouter(prefix="/api", tags=["Training"])

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/train", response_model=TrainResponse, summary="Start a training job")
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

    # Save model to disk
    model = result.pop("model")
    model_path = MODELS_DIR / f"session_{body.sessionId}.joblib"
    joblib.dump(
        {"model": model, "feature_columns": body.featureColumns},
        model_path,
    )

    # Build metrics for response
    metrics = TrainResultMetrics(
        accuracy=result["accuracy"],
        precision=result["precision"],
        recall=result["recall"],
        f1_score=result["f1_score"],
        confusionMatrix=result["confusion_matrix"],
        classificationReport=result["classification_report"],
        featureImportances=result["feature_importances"],
        targetDistribution=result["target_distribution"],
        trainSize=result["train_size"],
        testSize=result["test_size"],
    )

    # Persist metrics to session in SQLite
    db = get_db()
    db.execute(
        "UPDATE sessions SET train_result = ? WHERE id = ?",
        (json.dumps(metrics.model_dump()), body.sessionId),
    )
    db.commit()

    print(
        f"Training done — accuracy: {result['accuracy']}, "
        f"train: {result['train_size']}, test: {result['test_size']}"
    )

    return TrainResponse(
        job_id=job_id,
        status="completed",
        sessionId=body.sessionId,
        created_at=now.isoformat(),
        message=f"Training completed — accuracy: {result['accuracy']}",
        metrics=metrics,
    )


@router.post(
    "/sessions/{session_id}/predict",
    response_model=PredictResponse,
    summary="Predict grades using saved model",
)
def predict(session_id: int, body: PredictRequest):
    model_path = MODELS_DIR / f"session_{session_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model for this session")

    saved = joblib.load(model_path)
    model = saved["model"]

    df = pd.DataFrame(body.rows)
    X = df[body.featureColumns].apply(pd.to_numeric, errors="coerce")
    predictions = model.predict(X).tolist()

    return PredictResponse(predictions=predictions)
