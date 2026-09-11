from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


IOU_THRESHOLDS = (0.10, 0.20, 0.30, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PRIMARY_IOU_INDEX = IOU_THRESHOLDS.index(0.30)
MAP50_INDEX = IOU_THRESHOLDS.index(0.50)


@dataclass(frozen=True)
class IouEvaluation:
    results: tuple[float, ...]
    per_class_map50_95: np.ndarray
    timing: Any
    metrics: dict[str, Any]
    raw_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_with_low_iou_metrics(validation_module: Any, runner: Callable[..., Any], *args: Any, **kwargs: Any) -> IouEvaluation:
    """Run YOLOv5 validation once and expose AP at IoU 0.1, 0.2 and 0.3.

    The fixed vendor source remains untouched. During this call only, the
    validation IoU vector is extended; returned standard mAP@0.5 and
    mAP@0.5:0.95 are then reconstructed from their original ten thresholds.
    """

    original_linspace = validation_module.torch.linspace
    original_ap_per_class = validation_module.ap_per_class
    captured: dict[str, Any] = {}

    def extended_linspace(start: Any, end: Any, steps: Any, *extra: Any, **options: Any) -> Any:
        if float(start) == 0.5 and float(end) == 0.95 and int(steps) == 10:
            return validation_module.torch.tensor(IOU_THRESHOLDS, *extra, **options)
        return original_linspace(start, end, steps, *extra, **options)

    def capture_ap_per_class(tp: Any, conf: Any, pred_cls: Any, target_cls: Any, *extra: Any, **options: Any) -> Any:
        captured["raw_stats"] = tuple(
            np.asarray(value).copy() for value in (tp, conf, pred_cls, target_cls)
        )
        result = original_ap_per_class(tp, conf, pred_cls, target_cls, *extra, **options)
        captured["ap_result"] = result
        names = options.get("names", {})
        captured["primary_result"] = original_ap_per_class(
            np.asarray(tp)[:, PRIMARY_IOU_INDEX : PRIMARY_IOU_INDEX + 1],
            conf,
            pred_cls,
            target_cls,
            plot=False,
            names=names,
        )
        return result

    validation_module.torch.linspace = extended_linspace
    validation_module.ap_per_class = capture_ap_per_class
    try:
        raw_results, raw_maps, timing = runner(*args, **kwargs)
    finally:
        validation_module.ap_per_class = original_ap_per_class
        validation_module.torch.linspace = original_linspace

    class_count = len(raw_maps)
    ap_result = captured.get("ap_result")
    if ap_result is None:
        ap = np.zeros((0, len(IOU_THRESHOLDS)), dtype=float)
        class_ids = np.zeros(0, dtype=int)
        primary_precision = primary_recall = 0.0
        raw_stats = (
            np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
        )
    else:
        ap = np.asarray(ap_result[5], dtype=float)
        class_ids = np.asarray(ap_result[6], dtype=int)
        primary_result = captured["primary_result"]
        primary_precision = float(np.asarray(primary_result[2], dtype=float).mean())
        primary_recall = float(np.asarray(primary_result[3], dtype=float).mean())
        raw_stats = captured["raw_stats"]

    def mean_at(index: int) -> float:
        return float(ap[:, index].mean()) if ap.size else 0.0

    map10 = mean_at(IOU_THRESHOLDS.index(0.10))
    map20 = mean_at(IOU_THRESHOLDS.index(0.20))
    map30 = mean_at(PRIMARY_IOU_INDEX)
    map50 = mean_at(MAP50_INDEX)
    map50_95 = float(ap[:, MAP50_INDEX:].mean()) if ap.size else 0.0

    def per_class_at(index: int, default: float) -> np.ndarray:
        values = np.zeros(class_count, dtype=float) + default
        for row, class_id in enumerate(class_ids):
            values[int(class_id)] = float(ap[row, index])
        return values

    per_class_map50_95 = np.zeros(class_count, dtype=float) + map50_95
    for row, class_id in enumerate(class_ids):
        per_class_map50_95[int(class_id)] = float(ap[row, MAP50_INDEX:].mean())

    metrics = {
        "iou_thresholds": list(IOU_THRESHOLDS),
        "precision_recall_iou": 0.30,
        "precision": primary_precision,
        "recall": primary_recall,
        "map10": map10,
        "map20": map20,
        "map30": map30,
        "map50": map50,
        "map50_95": map50_95,
        "per_class_ap10": per_class_at(IOU_THRESHOLDS.index(0.10), map10).tolist(),
        "per_class_ap20": per_class_at(IOU_THRESHOLDS.index(0.20), map20).tolist(),
        "per_class_ap30": per_class_at(PRIMARY_IOU_INDEX, map30).tolist(),
        "per_class_ap50": per_class_at(MAP50_INDEX, map50).tolist(),
        "per_class_ap": per_class_map50_95.tolist(),
    }
    results = (
        primary_precision,
        primary_recall,
        map50,
        map50_95,
        *tuple(float(value) for value in raw_results[4:]),
    )
    return IouEvaluation(results, per_class_map50_95, timing, metrics, raw_stats)
