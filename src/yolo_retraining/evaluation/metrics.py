from __future__ import annotations

from typing import Any, Mapping, Sequence


def summarize_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix: list[dict[str, float]] = []
    for row in rows:
        tests = row.get("last_metrics", {})
        matrix.append({group: float(metrics["map50_95"]) for group, metrics in tests.items()})
    seen_mean = [sum(row.values()) / len(row) if row else 0.0 for row in matrix]
    forgetting: dict[str, float] = {}
    if matrix:
        final = matrix[-1]
        for group, final_value in final.items():
            history = [row[group] for row in matrix if group in row]
            forgetting[group] = max(history) - final_value
    return {
        "matrix": matrix,
        "seen_mean": seen_mean,
        "final_mean_forgetting": sum(forgetting.values()) / len(forgetting) if forgetting else 0.0,
        "final_forgetting": forgetting,
    }

