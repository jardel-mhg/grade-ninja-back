"""
Extract human-readable threshold rules from a trained model.

Strategy: for each grade, train a shallow binary decision tree
(grade vs all others) on the model's own predictions, then extract
the positive decision paths as conjunctive (AND) rule groups.

Rule format per grade: list of paths (OR between paths, AND within each path)
  [ [cond, cond, ...], [cond, cond, ...], ... ]
"""

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

GRADE_ORDER = ["B", "C", "D", "E", "EN"]  # default fallback only
RULES_DIR = Path(__file__).parent.parent / "data" / "rules"
RULES_DIR.mkdir(parents=True, exist_ok=True)


INF = math.inf


def _path_to_intervals(path: list[dict]) -> dict[str, tuple[float, float]]:
    """Convert path conditions to per-feature (lower_exclusive, upper_inclusive) intervals."""
    intervals: dict[str, tuple[float, float]] = {}
    for cond in path:
        f = cond["feature"]
        lo, hi = intervals.get(f, (-INF, INF))
        if cond["op"] == ">":
            lo = max(lo, cond["value"])
        else:
            hi = min(hi, cond["value"])
        intervals[f] = (lo, hi)
    return intervals


def _intervals_to_path(intervals: dict[str, tuple[float, float]]) -> list[dict]:
    """Convert per-feature intervals back to a list of conditions."""
    conds = []
    for f, (lo, hi) in intervals.items():
        if lo != -INF:
            conds.append({"feature": f, "op": ">", "value": lo})
        if hi != INF:
            conds.append({"feature": f, "op": "<=", "value": hi})
    return conds


def _try_merge_paths(path_a: list[dict], path_b: list[dict]) -> list[dict] | None:
    """Merge two paths via interval union if exactly one feature has differing intervals
    and those intervals are adjacent or overlapping (no gap between them).
    """
    iv_a = _path_to_intervals(path_a)
    iv_b = _path_to_intervals(path_b)
    all_features = set(iv_a) | set(iv_b)

    differing = [
        f for f in all_features
        if iv_a.get(f, (-INF, INF)) != iv_b.get(f, (-INF, INF))
    ]

    if len(differing) != 1:
        return None

    diff_feat = differing[0]
    a_lo, a_hi = iv_a.get(diff_feat, (-INF, INF))
    b_lo, b_hi = iv_b.get(diff_feat, (-INF, INF))

    # Intervals are adjacent or overlapping if they share or touch a boundary
    if max(a_lo, b_lo) > min(a_hi, b_hi) and a_hi != b_lo and b_hi != a_lo:
        return None  # Gap between intervals — cannot merge

    union_lo = min(a_lo, b_lo)
    union_hi = max(a_hi, b_hi)

    merged_intervals = {
        f: (iv_a.get(f, (-INF, INF)) if f != diff_feat else (union_lo, union_hi))
        for f in all_features
    }
    return _intervals_to_path(merged_intervals)


def _simplify_paths(paths: list[list[dict]]) -> list[list[dict]]:
    """Repeatedly merge adjacent/overlapping path pairs until stable, then deduplicate."""
    paths = [p for p in paths if p]

    changed = True
    while changed:
        changed = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                merged = _try_merge_paths(paths[i], paths[j])
                if merged is not None:
                    paths = [p for k, p in enumerate(paths) if k not in (i, j)]
                    paths.append(merged)
                    changed = True
                    break
            if changed:
                break

    # Remove exact duplicates
    seen: list[frozenset] = []
    unique = []
    for p in paths:
        key = frozenset((c["feature"], c["op"], c["value"]) for c in p)
        if key not in seen:
            seen.append(key)
            unique.append(p)
    return unique


def _extract_positive_paths(tree, feature_names: list[str]) -> list[list[dict]]:
    """Return all root→leaf paths that predict class 1 (positive grade).
    Each path is a list of AND-ed conditions.
    """
    t = tree.tree_
    paths = []

    def walk(node, conditions):
        if t.children_left[node] == -1:
            if t.value[node][0].argmax() == 1:
                paths.append(list(conditions))
            return
        feat = feature_names[t.feature[node]]
        threshold = round(float(t.threshold[node]), 4)
        walk(t.children_left[node],  conditions + [{"feature": feat, "op": "<=", "value": threshold}])
        walk(t.children_right[node], conditions + [{"feature": feat, "op": ">",  "value": threshold}])

    walk(0, [])
    return paths


def extract_rules(
    model,
    X: pd.DataFrame,
    feature_columns: list[str],
    grade_order: list[str] = GRADE_ORDER,
    max_depth: int = 4,
    max_paths: int = 5,
) -> dict:
    """
    Given a trained model and dataset X, produce a rules config dict.

    Rules per grade are stored as list-of-paths (OR of ANDs):
      rules[grade] = [ [cond, ...], [cond, ...], ... ]
    """
    print("Extracting rules from model predictions...")
    y_pred = model.predict(X)

    rules_per_grade: dict[str, list[list[dict]]] = {}

    for grade in grade_order:
        y_binary = (y_pred == grade).astype(int)
        if y_binary.sum() == 0:
            print(f"  Grade {grade}: no predictions — skipping")
            rules_per_grade[grade] = []
            continue

        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X, y_binary)

        paths = _extract_positive_paths(dt, feature_columns)
        paths = _simplify_paths(paths)

        # Sort by path length (simpler paths first), cap at max_paths
        paths.sort(key=lambda p: len(p))
        paths = paths[:max_paths]

        print(f"  Grade {grade}: {len(paths)} rule group(s), "
              f"{sum(len(p) for p in paths)} total conditions")
        rules_per_grade[grade] = paths

    return {
        "grade_order": grade_order,
        "fallback": grade_order[-1],
        "rules": rules_per_grade,
    }


def save_rules(session_id: int, rules: dict) -> Path:
    path = RULES_DIR / f"session_{session_id}.json"
    path.write_text(json.dumps(rules, indent=2))
    print(f"Rules saved to {path}")
    return path


def load_rules(session_id: int) -> dict | None:
    path = RULES_DIR / f"session_{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
