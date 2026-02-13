import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_random_forest(
    df: pd.DataFrame, target_column: str, feature_columns: list[str]
) -> dict:
    """Train a Random Forest classifier and return model + full metrics.

    Returns dict with: model, accuracy, train_size, test_size,
    confusion_matrix, classification_report, feature_importances,
    target_distribution
    """
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df[target_column]

    # Drop rows where target is NaN or empty
    valid = y.notna() & (y != "")
    X = X.loc[valid]
    y = y.loc[valid]

    # Need at least 2 samples per class for stratified split
    if len(y) < 5 or y.nunique() < 2:
        raise ValueError(
            f"Not enough data to train: {len(y)} rows, {y.nunique()} classes"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbose=1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Confusion matrix as {actual: {predicted: count}}
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_dict = {}
    for i, actual in enumerate(labels):
        cm_dict[str(actual)] = {
            str(labels[j]): int(cm[i][j]) for j in range(len(labels))
        }

    # Classification report as per-class dict
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

    # Feature importances
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_columns, rf.feature_importances_)
    }

    # Target distribution in training data
    dist = y.value_counts().to_dict()
    target_dist = {str(k): int(v) for k, v in dist.items()}

    # Weighted averages from report
    weighted = report.get("weighted avg", {})

    # Evaluation
    print("\n" + "=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels))

    return {
        "model": rf,
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
