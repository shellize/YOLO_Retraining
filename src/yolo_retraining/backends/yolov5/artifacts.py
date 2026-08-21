from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _smooth(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    filter_size = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(filter_size // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]), 0)
    return np.convolve(padded, np.ones(filter_size) / filter_size, mode="valid")


def confidence_sweep(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    *,
    names: dict[int, str] | list[str] | tuple[str, ...],
    points: int | None = None,
    conf_floor: float = 0.001,
    thresholds: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Build fixed-confidence P/R/F1 curves from one YOLOv5 validation pass."""
    conf_floor = float(conf_floor)
    if not 0.0 <= conf_floor <= 1.0:
        raise ValueError("confidence sweep floor must be between 0 and 1")

    if thresholds is None:
        points = int(points or 101)
        if points < 2:
            raise ValueError("confidence sweep requires at least two points")
        threshold_values = np.linspace(conf_floor, 1.0, points)
    else:
        threshold_values = np.asarray([float(value) for value in thresholds], dtype=float)
        if threshold_values.size == 0:
            raise ValueError("confidence sweep requires at least one threshold")
        if np.any(threshold_values < 0.0) or np.any(threshold_values > 1.0):
            raise ValueError("confidence sweep thresholds must be between 0 and 1")
        if np.any(np.diff(threshold_values) <= 0.0):
            raise ValueError("confidence sweep thresholds must be strictly increasing")
    points = int(threshold_values.size)

    confidence = np.asarray(conf, dtype=float).reshape(-1)
    predicted_class = np.asarray(pred_cls, dtype=int).reshape(-1)
    target_class = np.asarray(target_cls, dtype=int).reshape(-1)
    correct = np.asarray(tp, dtype=bool)
    correct50 = correct[:, 0] if correct.ndim == 2 and correct.shape[1] else np.zeros(confidence.size, dtype=bool)

    if isinstance(names, dict):
        class_names = {int(class_id): str(class_name) for class_id, class_name in names.items()}
    else:
        class_names = {class_id: str(class_name) for class_id, class_name in enumerate(names)}
    observed_classes = list(target_class) + list(predicted_class)
    max_observed_class = max((int(class_id) for class_id in observed_classes), default=-1)
    class_count = max(max(class_names.keys(), default=-1) + 1, max_observed_class + 1)
    class_names.update({class_id: str(class_id) for class_id in range(class_count) if class_id not in class_names})
    ground_truth_counts = np.bincount(target_class, minlength=class_count) if class_count else np.zeros(0, dtype=int)
    has_ground_truth = ground_truth_counts > 0

    overall_precision = np.zeros(points, dtype=float)
    overall_recall = np.zeros(points, dtype=float)
    overall_f1 = np.zeros(points, dtype=float)
    micro_precision = np.zeros(points, dtype=float)
    micro_recall = np.zeros(points, dtype=float)
    micro_f1 = np.zeros(points, dtype=float)
    total_tp = np.zeros(points, dtype=int)
    total_fp = np.zeros(points, dtype=int)
    total_fn = np.zeros(points, dtype=int)
    per_class_precision = np.zeros((class_count, points), dtype=float)
    per_class_recall = np.zeros((class_count, points), dtype=float)
    per_class_f1 = np.zeros((class_count, points), dtype=float)
    per_class_tp = np.zeros((class_count, points), dtype=int)
    per_class_fp = np.zeros((class_count, points), dtype=int)
    per_class_fn = np.zeros((class_count, points), dtype=int)

    for index, threshold in enumerate(threshold_values):
        included = confidence >= threshold
        true_positive = included & correct50
        false_positive = included & ~correct50
        tp_counts = np.bincount(predicted_class[true_positive], minlength=class_count)[:class_count] if class_count else np.zeros(0, dtype=int)
        fp_counts = np.bincount(predicted_class[false_positive], minlength=class_count)[:class_count] if class_count else np.zeros(0, dtype=int)
        fn_counts = np.maximum(ground_truth_counts - tp_counts, 0)
        precision = np.divide(tp_counts, tp_counts + fp_counts, out=np.zeros(class_count, dtype=float), where=(tp_counts + fp_counts) > 0)
        recall = np.divide(tp_counts, ground_truth_counts, out=np.zeros(class_count, dtype=float), where=ground_truth_counts > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(class_count, dtype=float), where=(precision + recall) > 0)

        per_class_precision[:, index] = precision
        per_class_recall[:, index] = recall
        per_class_f1[:, index] = f1
        per_class_tp[:, index] = tp_counts
        per_class_fp[:, index] = fp_counts
        per_class_fn[:, index] = fn_counts
        total_tp[index] = int(tp_counts.sum())
        total_fp[index] = int(fp_counts.sum())
        total_fn[index] = int(fn_counts.sum())
        if has_ground_truth.any():
            overall_precision[index] = float(precision[has_ground_truth].mean())
            overall_recall[index] = float(recall[has_ground_truth].mean())
            overall_f1[index] = float(f1[has_ground_truth].mean())
        micro_precision[index] = float(total_tp[index] / (total_tp[index] + total_fp[index])) if total_tp[index] + total_fp[index] else 0.0
        micro_recall[index] = float(total_tp[index] / (total_tp[index] + total_fn[index])) if total_tp[index] + total_fn[index] else 0.0
        micro_f1[index] = (
            2 * micro_precision[index] * micro_recall[index] / (micro_precision[index] + micro_recall[index])
            if micro_precision[index] + micro_recall[index]
            else 0.0
        )

    best_index = int(_smooth(overall_f1, 0.1).argmax()) if has_ground_truth.any() else 0

    def float_values(values: np.ndarray) -> list[float]:
        return [float(value) for value in values]

    def int_values(values: np.ndarray) -> list[int]:
        return [int(value) for value in values]

    per_class = []
    for class_id in range(class_count):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "ground_truth_count": int(ground_truth_counts[class_id]),
                "precision": float_values(per_class_precision[class_id]),
                "recall": float_values(per_class_recall[class_id]),
                "f1": float_values(per_class_f1[class_id]),
                "tp": int_values(per_class_tp[class_id]),
                "fp": int_values(per_class_fp[class_id]),
                "fn": int_values(per_class_fn[class_id]),
            }
        )

    return {
        "schema_version": 1,
        "confidence_floor": conf_floor,
        "iou_threshold": 0.5,
        "thresholds": float_values(threshold_values),
        "prediction_count_after_conf_floor": int(confidence.size),
        "ground_truth_count": int(target_class.size),
        "overall": {
            "precision": float_values(overall_precision),
            "recall": float_values(overall_recall),
            "f1": float_values(overall_f1),
            "micro_precision": float_values(micro_precision),
            "micro_recall": float_values(micro_recall),
            "micro_f1": float_values(micro_f1),
            "tp": int_values(total_tp),
            "fp": int_values(total_fp),
            "fn": int_values(total_fn),
        },
        "best_f1": {
            "index": best_index,
            "confidence": float(threshold_values[best_index]),
            "precision": float(overall_precision[best_index]),
            "recall": float(overall_recall[best_index]),
            "f1": float(overall_f1[best_index]),
        },
        "per_class": per_class,
    }


class PredictionArtifactCollector:
    """Collect per-image records through YOLOv5 callbacks without editing its source."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", buffering=1024 * 1024)
        self._pending_correct: list[np.ndarray] = []
        self._correct_rows: list[list[bool]] = []
        self._confidences: list[float] = []
        self._predicted_classes: list[int] = []
        self._target_classes: list[int] = []
        self._image_index = 0
        self._iou_points = 10

    def record_correct(self, correct: Any) -> None:
        array = correct.detach().cpu().numpy().astype(bool)
        self._iou_points = int(array.shape[1]) if array.ndim == 2 and array.shape[1] else self._iou_points
        self._pending_correct.append(array)

    def on_val_batch_end(self, _batch_i: int, _images: Any, targets: Any, paths: Any, _shapes: Any, predictions: Any) -> None:
        targets_cpu = targets.detach().cpu()
        for sample_index, (path, prediction) in enumerate(zip(paths, predictions)):
            target_rows = targets_cpu[targets_cpu[:, 0] == sample_index]
            target_class_ids = [int(value) for value in target_rows[:, 1].tolist()]
            prediction_rows = prediction.detach().cpu().tolist()
            if target_class_ids and prediction_rows:
                if not self._pending_correct:
                    raise RuntimeError("YOLOv5 prediction artifact collector lost a process_batch result")
                correct = self._pending_correct.pop(0)
                if correct.shape[0] != len(prediction_rows):
                    raise RuntimeError("YOLOv5 prediction artifact collector saw mismatched prediction counts")
            else:
                correct = np.zeros((len(prediction_rows), self._iou_points), dtype=bool)

            record_predictions = []
            for row, correct_row in zip(prediction_rows, correct.tolist()):
                confidence = round(float(row[4]), 8)
                class_id = int(row[5])
                record_predictions.append(
                    {
                        "confidence": confidence,
                        "predicted_class_id": class_id,
                        "correct_iou": [bool(value) for value in correct_row],
                    }
                )
                self._confidences.append(confidence)
                self._predicted_classes.append(class_id)
                self._correct_rows.append([bool(value) for value in correct_row])
            self._target_classes.extend(target_class_ids)
            self._handle.write(
                json.dumps(
                    {
                        "image_index": self._image_index,
                        "image_path": str(path),
                        "target_class_ids": target_class_ids,
                        "predictions": record_predictions,
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._image_index += 1

        if self._pending_correct:
            raise RuntimeError("YOLOv5 prediction artifact collector has unmatched process_batch results")

    def raw_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        correct = np.asarray(self._correct_rows, dtype=bool)
        if correct.size == 0:
            correct = np.zeros((0, self._iou_points), dtype=bool)
        return (
            correct,
            np.asarray(self._confidences, dtype=float),
            np.asarray(self._predicted_classes, dtype=int),
            np.asarray(self._target_classes, dtype=int),
        )

    def close(self) -> None:
        self._handle.close()
