from __future__ import annotations

from typing import Any, Mapping, Sequence


def summarize_matrix(rows: Sequence[Mapping[str, Any]], *, stage_ids: Sequence[str] | None = None, epsilon: float = 1e-12) -> dict[str, Any]:
    matrix: list[dict[str, float]] = []
    for row in rows:
        tests = row.get("metrics", row.get("last_metrics", {}))
        matrix.append({group: float(metrics["map50_95"]) for group, metrics in tests.items()})
    stage_ids = list(stage_ids or [])
    seen_mean = []
    for index, row in enumerate(matrix):
        seen_groups = [group for group in stage_ids[: index + 1] if group in row]
        values = [row[group] for group in seen_groups] if seen_groups else list(row.values())
        seen_mean.append(sum(values) / len(values) if values else 0.0)
    forgetting: dict[str, float] = {}
    forgetting_ratio: dict[str, float] = {}
    retention_ratio: dict[str, float] = {}
    global_regression: dict[str, float] = {}
    if matrix:
        final = matrix[-1]
        for group, final_value in final.items():
            history = [row[group] for row in matrix if group in row]
            regression = max(history) - final_value
            global_regression[group] = regression
            if group in stage_ids:
                forgetting[group] = regression
                forgetting_ratio[group] = forgetting[group] / max(max(history), epsilon)
                retention_ratio[group] = final_value / max(max(history), epsilon)
    acquisition_abs: dict[str, float] = {}
    acquisition_ratio: dict[str, float] = {}
    for index, group in enumerate(stage_ids):
        if index == 0 or index >= len(matrix) or group not in matrix[index] or group not in matrix[index - 1]:
            continue
        before = matrix[index - 1][group]
        learned = matrix[index][group] - before
        acquisition_abs[group] = learned
        acquisition_ratio[group] = learned / max(1.0 - before, epsilon)
    final_mean = sum(matrix[-1].values()) / len(matrix[-1]) if matrix and matrix[-1] else 0.0
    initial_overlap = set(stage_ids) & set(matrix[0]) & set(matrix[-1]) if matrix else set()
    bwt_values = [matrix[-1][group] - matrix[0][group] for group in initial_overlap]
    return {
        "checkpoint": "best",
        "matrix": matrix,
        "seen_mean": seen_mean,
        "final_mean_map50_95": final_mean,
        "global_test_regression": global_regression,
        "final_mean_forgetting": sum(forgetting.values()) / len(forgetting) if forgetting else 0.0,
        "final_forgetting": forgetting,
        "final_forgetting_ratio": forgetting_ratio,
        "final_retention_ratio": retention_ratio,
        "acquisition_abs": acquisition_abs,
        "acquisition_headroom_ratio": acquisition_ratio,
        "backward_transfer": sum(bwt_values) / len(bwt_values) if bwt_values else None,
        "stage_specific_metrics_available": bool(acquisition_abs),
    }
