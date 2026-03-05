import numpy as np
import pandas as pd
import shap
from sklearn.utils import resample


def compute_shap(
    model,
    X_test: pd.DataFrame,
    y_test,
    feature_columns: list[str],
    class_names: list[str],
    sample_size: int = 100,
) -> dict:
    """Compute SHAP values and return per-class importance + signature features.

    Returns:
        perClassImportance: {grade: {feature: mean_abs_shap, ...}, ...} sorted desc
        signatures: {grade: {feature: uniqueness_score, ...}, ...} top-5 per grade
    """
    # Stratified sample for speed
    n = min(sample_size, len(X_test))
    if n < len(X_test):
        X_shap, y_shap = resample(
            X_test, y_test, n_samples=n, stratify=y_test, random_state=42
        )
    else:
        X_shap = X_test
        y_shap = np.asarray(y_test)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_shap)

    # Normalise to list-of-2D-arrays shape: [n_classes][n_samples, n_features]
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        # XGBoost multiclass: (n_samples, n_features, n_classes)
        shap_per_class = [raw[:, :, i] for i in range(raw.shape[2])]
    elif isinstance(raw, list):
        # RandomForest: list of (n_samples, n_features) per class
        shap_per_class = raw
    else:
        # Binary or single output — wrap in list
        shap_per_class = [raw]

    # Map class index → class name using class_names order
    # class_names must match the order the model uses internally
    n_classes = len(shap_per_class)
    names = class_names[:n_classes]

    # Mean absolute SHAP per class per feature
    per_class_mean: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        arr = shap_per_class[idx]  # (n_samples, n_features)
        mean_abs = np.abs(arr).mean(axis=0)
        sorted_feats = sorted(
            zip(feature_columns, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        per_class_mean[str(name)] = {
            feat: round(val, 6) for feat, val in sorted_feats
        }

    # Signature: per-grade top-5 uniqueness = grade_importance - avg_others
    signatures: dict[str, dict[str, float]] = {}
    for name in names:
        others = [v for k, v in per_class_mean.items() if k != str(name)]
        uniqueness: dict[str, float] = {}
        for feat in feature_columns:
            own = per_class_mean[str(name)].get(feat, 0.0)
            avg_other = (
                sum(o.get(feat, 0.0) for o in others) / len(others) if others else 0.0
            )
            uniqueness[feat] = round(own - avg_other, 6)

        top5 = dict(
            sorted(uniqueness.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        signatures[str(name)] = top5

    return {
        "perClassImportance": per_class_mean,
        "signatures": signatures,
    }
