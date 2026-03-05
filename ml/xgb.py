import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_xgboost(
    df: pd.DataFrame, target_column: str, feature_columns: list[str]
) -> dict:
    """Train a Gradient Boosting classifier and return model + full metrics."""
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df[target_column]

    # Drop rows where target is NaN or empty
    valid = y.notna() & (y != "")
    X = X.loc[valid]
    y = y.loc[valid]

    print(f"Dataset received: {len(X)} rows, {y.nunique()} classes, features: {feature_columns}")
    print(f"Class distribution:\n{y.value_counts().to_string()}")

    # Drop classes with fewer than 2 samples (can't stratify-split them)
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X = X.loc[mask]
    y = y.loc[mask]

    if len(y) < 5 or y.nunique() < 2:
        raise ValueError(
            f"Not enough data to train: {len(y)} rows, {y.nunique()} classes"
        )

    # Encode string labels to ints for sample_weight computation
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Per-class sample weights for imbalanced data
    classes, counts = np.unique(y_train_enc, return_counts=True)
    class_weight = len(y_train_enc) / (len(classes) * counts)
    weight_map = dict(zip(classes, class_weight))
    sample_weight = np.array([weight_map[c] for c in y_train_enc])

    gb = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=3,
        random_state=42,
        verbose=1,
    )
    gb.fit(X_train, y_train_enc, sample_weight=sample_weight)

    y_pred_enc = gb.predict(X_test)

    # Decode back to original labels
    y_test_orig = le.inverse_transform(y_test_enc)
    y_pred_orig = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test_orig, y_pred_orig)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test_orig, y_pred_orig, labels=labels)
    cm_dict = {
        str(actual): {str(labels[j]): int(cm[i][j]) for j in range(len(labels))}
        for i, actual in enumerate(labels)
    }

    report = classification_report(
        y_test_orig, y_pred_orig, labels=labels, output_dict=True
    )

    print("Computing permutation importances...")
    perm = permutation_importance(gb, X_test, y_test_enc, n_repeats=3, random_state=42, n_jobs=-1)
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_columns, perm.importances_mean)
    }

    dist = y.value_counts().to_dict()
    target_dist = {str(k): int(v) for k, v in dist.items()}

    weighted = report.get("weighted avg", {})

    print("\n" + "=" * 50)
    print("HIST GRADIENT BOOSTING RESULTS")
    print("=" * 50)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test_orig, y_pred_orig, labels=labels))

    return {
        "model": gb,
        "label_encoder": le,
        "X_test": X_test,
        "y_test": y_test_orig,
        "accuracy": round(acc, 4),
        "precision": round(weighted.get("precision", 0), 4),
        "recall": round(weighted.get("recall", 0), 4),
        "f1_score": round(weighted.get("f1-score", 0), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "confusion_matrix": cm_dict,
        "classification_report": {
            k: {
                "precision": round(v.get("precision", 0), 4),
                "recall": round(v.get("recall", 0), 4),
                "f1_score": round(v.get("f1-score", 0), 4),
                "support": int(v.get("support", 0)),
            }
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "feature_importances": importances,
        "target_distribution": target_dist,
    }
