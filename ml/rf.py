import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_random_forest(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> dict:
    """Train a Random Forest classifier and return accuracy.

    Returns dict with: accuracy, train_size, test_size
    """
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df[target_column]

    # Drop rows where target is NaN or empty
    valid = y.notna() & (y != "")
    X = X.loc[valid]
    y = y.loc[valid]

    # Need at least 2 samples per class for stratified split
    min_split = max(2, int(len(y) * 0.2))
    if len(y) < 5 or y.nunique() < 2:
        raise ValueError(f"Not enough data to train: {len(y)} rows, {y.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=1000, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "accuracy": round(acc, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
