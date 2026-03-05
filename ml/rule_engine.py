"""
Evaluate a row against the rules config and return the highest matching grade.

Rule structure per grade: list of paths (OR of ANDs)
  rules[grade] = [ [cond, cond, ...], [cond, cond, ...], ... ]
A row matches a grade if ANY path matches (all conditions in that path are met).
"""


def _matches_path(row: dict, path: list[dict]) -> bool:
    """Return True if the row satisfies ALL conditions in this path (AND logic)."""
    for cond in path:
        val = row.get(cond["feature"])
        if val is None:
            return False
        try:
            val = float(val)
        except (TypeError, ValueError):
            return False
        threshold = cond["value"]
        if cond["op"] == "<=" and not (val <= threshold):
            return False
        if cond["op"] == ">" and not (val > threshold):
            return False
    return True


def predict_rule(row: dict, rules_config: dict) -> str:
    """
    Evaluate a single row against all grade rules.
    Returns the highest-priority grade whose ANY path matches, or fallback.
    """
    grade_order = rules_config["grade_order"]
    fallback = rules_config["fallback"]
    rules = rules_config["rules"]

    for grade in grade_order:
        paths = rules.get(grade, [])
        if not paths:
            continue
        # OR logic: match if any path's AND conditions are all satisfied
        if any(_matches_path(row, path) for path in paths):
            return grade

    return fallback


def predict_rules_batch(rows: list[dict], rules_config: dict) -> list[str]:
    return [predict_rule(row, rules_config) for row in rows]
