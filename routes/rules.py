import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ml.rule_extractor import extract_rules, save_rules, load_rules
from ml.rule_engine import predict_rules_batch

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

router = APIRouter(prefix="/api/sessions", tags=["Rules"])


class RuleCondition(BaseModel):
    feature: str
    op: str   # "<=" or ">"
    value: float


class RulesConfig(BaseModel):
    grade_order: list[str]
    fallback: str
    rules: dict[str, list[RuleCondition]]


class RulesPredictRequest(BaseModel):
    featureColumns: list[str]
    rows: list[dict]


@router.post("/{session_id}/rules/extract", summary="Extract rules from trained model")
def extract(session_id: int):
    model_path = MODELS_DIR / f"session_{session_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model for this session")

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    label_encoder = saved.get("label_encoder")

    # Build a dummy X from zeros — we only need the feature structure;
    # the model will be run on its training-like data passed via the store.
    # Since we don't persist training rows, we generate rules from the model
    # by creating a synthetic grid across feature ranges.
    # Better: caller passes rows. For now, raise helpful error.
    raise HTTPException(
        status_code=400,
        detail="Pass rows via POST body — use /rules/extract-from-rows instead",
    )


@router.post("/{session_id}/rules/extract-from-rows", summary="Extract rules from model using provided rows")
def extract_from_rows(session_id: int, body: dict):
    model_path = MODELS_DIR / f"session_{session_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model for this session")

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    label_encoder = saved.get("label_encoder")
    model_type = saved.get("model_type", "random_forest")

    rows = body.get("rows", [])
    if not rows:
        raise HTTPException(status_code=400, detail="rows is required")

    # Derive grade order from actual predictions on the provided rows
    # (most reliable — always matches what the model actually predicts)
    grade_order = body.get("gradeOrder") or []

    df = pd.DataFrame(rows)
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    # For GradientBoosting with label encoder, wrap to decode predictions back to grade names
    class DecodingModel:
        def __init__(self, m, le):
            self.m = m
            self.le = le
        def predict(self, X):
            return self.le.inverse_transform(self.m.predict(X))

    if label_encoder is not None:
        effective_model = DecodingModel(model, label_encoder)
    else:
        effective_model = model

    # Always derive grade order from actual predictions — ignores stale model metadata
    import numpy as np
    y_pred = effective_model.predict(X)
    seen = []
    for g in y_pred:
        s = str(g)
        if s not in seen:
            seen.append(s)
    # Respect client-supplied order if provided, otherwise use prediction order
    if grade_order:
        actual_grade_order = [g for g in grade_order if g in seen] + [g for g in seen if g not in grade_order]
    else:
        actual_grade_order = seen

    print(f"Rule extraction — grade order from predictions: {actual_grade_order}")

    rules = extract_rules(effective_model, X, feature_columns, grade_order=actual_grade_order)
    save_rules(session_id, rules)

    return rules


@router.get("/{session_id}/rules", summary="Get current rules config")
def get_rules(session_id: int):
    rules = load_rules(session_id)
    if rules is None:
        raise HTTPException(status_code=404, detail="No rules found for this session")
    return rules


@router.put("/{session_id}/rules", summary="Save edited rules config")
def update_rules(session_id: int, body: dict):
    # Validate required fields
    if "grade_order" not in body or "rules" not in body:
        raise HTTPException(status_code=400, detail="grade_order and rules are required")
    save_rules(session_id, body)
    return body


@router.post("/{session_id}/rules/predict", summary="Predict using rule engine (no ML)")
def predict_with_rules(session_id: int, body: RulesPredictRequest):
    rules = load_rules(session_id)
    if rules is None:
        raise HTTPException(status_code=404, detail="No rules found — extract rules first")

    predictions = predict_rules_batch(body.rows, rules)
    return {"predictions": predictions}
