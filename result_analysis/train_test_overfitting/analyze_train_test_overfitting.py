"""Reusable YOLOv5 result analysis for staged retraining runs.

The default invocation discovers the three completed sequence baselines in
``runs/sequences`` and analyzes stages 1--3 with ``best.pt``.  Each invocation
writes one self-contained analysis directory containing:

* fixed-confidence and YOLOv5 best-F1 summary metrics;
* train/test comparison tables;
* one row per image with TP/FP/FN and per-image precision/recall;
* one row per prediction and per ground-truth box;
* the exact manifests, data YAMLs, weights, and metric parameters used.

The inference and AP implementation intentionally reuses the pinned YOLOv5
v7.0 evaluation primitives so that mAP50 remains comparable with the existing
``evaluation.json`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_ROOT = PROJECT_ROOT / "runs" / "sequences"
DEFAULT_DATA_LAYOUT = PROJECT_ROOT / "configs" / "data" / "self_improving.yaml"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "result_analysis" / "train_test_overfitting"
DEFAULT_METHODS = ("full-warm", "random-replay", "full-cold")
DEFAULT_FIXED_CONFIDENCE = 0.25
STAGE_PATTERN = re.compile(r"(?:^|__)stage(?P<stage>\d+)(?:__|$)")


def _ensure_project_imports() -> None:
    source = str(PROJECT_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _parse_integer_spec(spec: str) -> list[int]:
    """Parse ``1,2,5-8`` into a sorted, de-duplicated integer list."""

    if not spec.strip():
        raise ValueError("integer specification cannot be empty")
    values: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ValueError(f"invalid integer token {token!r}; expected values like '1' or '1-3'")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            raise ValueError(f"integer range must be ascending: {token!r}")
        values.update(range(start, end + 1))
    return sorted(values)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _method_from_run(run_dir: Path) -> str:
    name = run_dir.name
    marker = "__seq-"
    if marker in name:
        remainder = name.split(marker, 1)[1]
        return remainder.split("__stage", 1)[0]
    return name


def _stage_from_task(task_dir: Path) -> int | None:
    match = STAGE_PATTERN.search(task_dir.name)
    return int(match.group("stage")) if match else None


def _completed_sequence_run(path: Path) -> bool:
    status_path = path / "sequence_status.json"
    if not status_path.is_file():
        return False
    status = _read_yaml(status_path) if status_path.suffix in {".yaml", ".yml"} else json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "completed"


def discover_run_dirs(sequence_root: Path, explicit: Sequence[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(value).expanduser().resolve() for value in explicit]
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"sequence run directory does not exist: {missing}")
        return paths

    discovered: list[Path] = []
    for method in DEFAULT_METHODS:
        candidates = sorted(
            (
                path
                for path in sequence_root.expanduser().resolve().iterdir()
                if path.is_dir() and f"__seq-{method}__" in path.name and _completed_sequence_run(path)
            ),
            key=lambda path: path.name,
        )
        if not candidates:
            raise FileNotFoundError(f"no completed {method!r} sequence found below {sequence_root}")
        discovered.append(candidates[-1])
    return discovered


def _resolve_checkpoint(task_dir: Path, selector: str) -> Path:
    candidate = Path(selector).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    checkpoint = (task_dir / "checkpoints" / candidate).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return checkpoint


def _task_test_ids(task_dir: Path, task_payload: Mapping[str, Any], registry: Mapping[str, Any]) -> list[str]:
    files = sorted((task_dir / "data").glob("test_*_ids.txt"))
    ids: list[str] = []
    for path in files:
        ids.extend(_read_lines(path))
    if not ids:
        data_payload = task_payload.get("data", {})
        groups = data_payload.get("test", []) if isinstance(data_payload, Mapping) else []
        for group_id in groups:
            ids.extend(registry.get("groups", {}).get(str(group_id), {}).get("test", []))
    return list(dict.fromkeys(ids))


def collect_tasks(
    run_dirs: Sequence[Path],
    stages: Sequence[int],
    checkpoint_selector: str,
    registry: Mapping[str, Any],
    skip_missing_ids: bool = False,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    wanted_stages = set(stages)
    for run_dir in run_dirs:
        tasks_dir = run_dir / "tasks"
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"sequence run has no tasks directory: {tasks_dir}")
        method = _method_from_run(run_dir)
        for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
            stage = _stage_from_task(task_dir)
            if stage is None or stage not in wanted_stages:
                continue
            task_payload = _read_yaml(task_dir / "task.yaml") if (task_dir / "task.yaml").is_file() else {}
            result_payload = json.loads((task_dir / "task_result.json").read_text(encoding="utf-8")) if (task_dir / "task_result.json").is_file() else {}
            if result_payload and result_payload.get("status") not in {None, "completed"}:
                raise RuntimeError(f"task is not completed: {task_dir}")
            train_ids_path = task_dir / "data" / "selected_ids.txt"
            if not train_ids_path.is_file():
                raise FileNotFoundError(f"task has no selected_ids.txt: {train_ids_path}")
            train_ids = _read_lines(train_ids_path)
            test_ids = _task_test_ids(task_dir, task_payload, registry)
            if not test_ids:
                raise ValueError(f"task has no test IDs: {task_dir}")
            checkpoint = _resolve_checkpoint(task_dir, checkpoint_selector)
            missing_train_ids = [sample_id for sample_id in train_ids if sample_id not in registry["records"]]
            missing_test_ids = [sample_id for sample_id in test_ids if sample_id not in registry["records"]]
            missing_ids = [*missing_train_ids, *missing_test_ids]
            if missing_ids:
                if not skip_missing_ids:
                    raise KeyError(f"{task_dir}: data layout cannot resolve sample IDs {missing_ids[:5]}; use --skip-missing-ids only if the local data version is known to differ")
                print(f"WARNING: {task_dir} skips {len(missing_ids)} missing local sample IDs", flush=True)
                train_ids = [sample_id for sample_id in train_ids if sample_id in registry["records"]]
                test_ids = [sample_id for sample_id in test_ids if sample_id in registry["records"]]
            if not train_ids or not test_ids:
                raise ValueError(f"{task_dir}: no evaluable train/test IDs remain after missing-ID filtering")
            tasks.append(
                {
                    "method": method,
                    "stage": stage,
                    "run_dir": run_dir,
                    "task_dir": task_dir,
                    "task_name": task_dir.name,
                    "task_config": task_payload,
                    "task_result": result_payload,
                    "checkpoint": checkpoint,
                    "train_ids": train_ids,
                    "test_ids": test_ids,
                    "missing_train_ids": missing_train_ids,
                    "missing_test_ids": missing_test_ids,
                }
            )
    tasks.sort(key=lambda item: (str(item["method"]), int(item["stage"]), str(item["task_dir"])))
    if not tasks:
        raise ValueError(f"no tasks matched stages {list(stages)}")
    return tasks


def _write_split_dataset_files(
    split_dir: Path,
    split: str,
    sample_ids: Sequence[str],
    registry: Mapping[str, Any],
) -> tuple[Path, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / "images.txt"
    image_paths = [str(Path(registry["records"][sample_id]["image_path"]).resolve()) for sample_id in sample_ids]
    manifest.write_text("".join(f"{path}\n" for path in image_paths), encoding="utf-8")
    data_yaml = split_dir / "data.yaml"
    payload = {
        "train": str(manifest.resolve()),
        "val": str(manifest.resolve()),
        "nc": len(registry["names"]),
        "names": {index: name for index, name in enumerate(registry["names"])},
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return manifest, data_yaml


def _match_details(labels: Any, detections: Any, iou_threshold: float, box_iou: Any) -> list[tuple[int, int, float]]:
    """Return YOLOv5-compatible one-to-one matches at one IoU threshold."""

    torch = __import__("torch")
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return []
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    label_indices, detection_indices = torch.where((iou >= iou_threshold) & correct_class)
    if label_indices.shape[0] == 0:
        return []
    matches = torch.cat(
        (torch.stack((label_indices, detection_indices), 1), iou[label_indices, detection_indices][:, None]),
        1,
    ).cpu().numpy()
    if matches.shape[0] > 1:
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return [(int(label), int(detection), float(overlap)) for label, detection, overlap in matches]


def _best_f1_confidence(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray) -> tuple[float, float, float, float]:
    """Reproduce YOLOv5's confidence sweep and smoothed mean-F1 choice."""

    if target_cls.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, target_counts = np.unique(target_cls, return_counts=True)
    confidence_axis = np.linspace(0, 1, 1000)
    precision = np.zeros((len(unique_classes), 1000))
    recall = np.zeros((len(unique_classes), 1000))
    for class_index, class_id in enumerate(unique_classes):
        mask = pred_cls == class_id
        if not mask.any() or target_counts[class_index] == 0:
            continue
        false_positive = (1 - tp[mask]).cumsum(0)
        true_positive = tp[mask].cumsum(0)
        recall_curve = true_positive / target_counts[class_index]
        precision_curve = true_positive / (true_positive + false_positive)
        recall[class_index] = np.interp(-confidence_axis, -conf[mask], recall_curve[:, 0], left=0)
        precision[class_index] = np.interp(-confidence_axis, -conf[mask], precision_curve[:, 0], left=1)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    filter_size = round(len(f1.mean(0)) * 0.1 * 2) // 2 + 1
    padding = np.ones(filter_size // 2)
    smoothed = np.convolve(
        np.concatenate((padding * f1.mean(0)[0], f1.mean(0), padding * f1.mean(0)[-1]), 0),
        np.ones(filter_size) / filter_size,
        mode="valid",
    )
    index = int(smoothed.argmax())
    return (
        float(confidence_axis[index]),
        float(precision[:, index].mean()),
        float(recall[:, index].mean()),
        float(f1[:, index].mean()),
    )


def _fixed_class_metrics(
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    confidence_threshold: float,
    class_names: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    correct, confidence, predicted_class, target_class = stats
    correct50 = correct[:, 0].astype(bool) if correct.ndim == 2 and correct.shape[1] else np.zeros(len(confidence), dtype=bool)
    included = confidence >= confidence_threshold
    target_ids = sorted(set(int(value) for value in target_class.tolist()))
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        target_count = int((target_class == class_id).sum())
        class_predictions = predicted_class == class_id
        tp = int((included & class_predictions & correct50).sum())
        fp = int((included & class_predictions & ~correct50).sum())
        fn = target_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / target_count if target_count else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_count": target_count,
                "prediction_count_at_conf": int((included & class_predictions).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision_at_conf": precision,
                "recall_at_conf": recall,
                "f1_at_conf": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "has_ground_truth": class_id in target_ids,
            }
        )
    evaluated = [row for row in per_class if row["has_ground_truth"]]
    tp_total = int(sum(row["tp"] for row in per_class))
    fp_total = int(sum(row["fp"] for row in per_class))
    fn_total = int(sum(row["fn"] for row in per_class))
    return per_class, {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": float(np.mean([row["precision_at_conf"] for row in evaluated])) if evaluated else 0.0,
        "recall": float(np.mean([row["recall_at_conf"] for row in evaluated])) if evaluated else 0.0,
        "f1": float(np.mean([row["f1_at_conf"] for row in evaluated])) if evaluated else 0.0,
        "micro_precision": tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0,
        "micro_recall": tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0,
    }


def summarize_stats(
    stats_raw: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    class_names: Sequence[str],
    confidence_threshold: float,
    ap_per_class: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stats_raw:
        stats = tuple(np.concatenate([item[index] for item in stats_raw], axis=0) for index in range(4))
    else:
        stats = (
            np.zeros((0, 10), dtype=bool),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )
    correct, confidence, predicted_class, target_class = stats
    fixed_per_class, fixed = _fixed_class_metrics(stats, confidence_threshold, class_names)

    standard_precision = standard_recall = map50 = map50_95 = 0.0
    best_confidence = best_precision = best_recall = best_f1 = 0.0
    ap50_by_class = {class_id: 0.0 for class_id in range(len(class_names))}
    if target_class.size:
        _, _, p, r, _, ap, ap_classes = ap_per_class(correct, confidence, predicted_class, target_class, plot=False, names={i: name for i, name in enumerate(class_names)})
        ap50 = ap[:, 0] if ap.ndim == 2 and ap.shape[1] else np.zeros(len(ap))
        map50 = float(ap50.mean()) if ap50.size else 0.0
        map50_95 = float(ap.mean(1).mean()) if ap.size else 0.0
        standard_precision = float(p.mean()) if p.size else 0.0
        standard_recall = float(r.mean()) if r.size else 0.0
        for class_id, value in zip(ap_classes.tolist(), ap50.tolist()):
            ap50_by_class[int(class_id)] = float(value)
        if correct.size and correct.any():
            best_confidence, best_precision, best_recall, best_f1 = _best_f1_confidence(correct, confidence, predicted_class, target_class)
    for row in fixed_per_class:
        row["ap50"] = ap50_by_class[row["class_id"]]
    metrics = {
        "image_count": len(stats_raw),
        "ground_truth_count": int(target_class.size),
        "prediction_count_after_conf_floor": int(confidence.size),
        "confidence_threshold": confidence_threshold,
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "map50": map50,
        "precision_at_conf": fixed["precision"],
        "recall_at_conf": fixed["recall"],
        "f1_at_conf": fixed["f1"],
        "micro_precision_at_conf": fixed["micro_precision"],
        "micro_recall_at_conf": fixed["micro_recall"],
        "tp_at_conf": fixed["tp"],
        "fp_at_conf": fixed["fp"],
        "fn_at_conf": fixed["fn"],
        "precision_best_f1": standard_precision,
        "recall_best_f1": standard_recall,
        "f1_best_f1": 2 * standard_precision * standard_recall / (standard_precision + standard_recall) if standard_precision + standard_recall else 0.0,
        "best_f1_confidence": best_confidence,
        "map50_95_for_reference": map50_95,
    }
    return metrics, fixed_per_class


class YoloV5Runtime:
    """Lazy adapter around the pinned YOLOv5 v7.0 source."""

    def __init__(self, yolov5_root: Path | None, device_name: str, batch: int, workers: int, imgsz: int, half: bool, nms_iou: float, conf_floor: float, max_det: int) -> None:
        _ensure_project_imports()
        from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier
        from yolo_retraining.backends.yolov5.source import validate_yolov5_source

        source_info = validate_yolov5_source(yolov5_root)
        self.source_root = Path(source_info["root"])
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        import utils.dataloaders as dataloaders

        install_read_only_verifier(dataloaders)
        from models.common import DetectMultiBackend
        from utils.dataloaders import create_dataloader
        from utils.general import check_img_size, non_max_suppression, scale_boxes, xywh2xyxy
        from utils.metrics import ap_per_class, box_iou
        from utils.torch_utils import select_device
        from val import process_batch

        self.DetectMultiBackend = DetectMultiBackend
        self.create_dataloader = create_dataloader
        self.check_img_size = check_img_size
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.xywh2xyxy = xywh2xyxy
        self.ap_per_class = ap_per_class
        self.box_iou = box_iou
        self.process_batch = process_batch
        self.device = select_device(device_name, batch_size=batch)
        self.batch = batch
        self.workers = workers
        self.imgsz = imgsz
        self.half = half
        self.nms_iou = nms_iou
        self.conf_floor = conf_floor
        self.max_det = max_det
        self.torch = __import__("torch")

    def load_model(self, checkpoint: Path, data_yaml: Path) -> Any:
        previous_cwd = os.getcwd()
        os.chdir(self.source_root)
        try:
            model = self.DetectMultiBackend(str(checkpoint), device=self.device, dnn=False, data=str(data_yaml), fp16=self.half)
            model.eval()
            model.imgsz = self.check_img_size(self.imgsz, s=model.stride)
            model.warmup(imgsz=(1 if model.pt else self.batch, 3, model.imgsz, model.imgsz))
            return model
        finally:
            os.chdir(previous_cwd)

    def evaluate_split(
        self,
        model: Any,
        manifest: Path,
        sample_ids: Sequence[str],
        registry: Mapping[str, Any],
        context: Mapping[str, Any],
        output_dir: Path,
        match_iou: float,
        confidence_threshold: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        path_to_id = {
            str(Path(registry["records"][sample_id]["image_path"]).resolve()).casefold(): sample_id
            for sample_id in sample_ids
        }
        dataloader = self.create_dataloader(
            str(manifest),
            model.imgsz,
            self.batch,
            model.stride,
            False,
            pad=0.5,
            rect=True,
            workers=self.workers,
            prefix=f"{context['split']}: ",
        )[0]
        iouv = self.torch.linspace(0.5, 0.95, 10, device=self.device)
        stats_raw: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        image_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        ground_truth_rows: list[dict[str, Any]] = []
        class_names = list(registry["names"])
        with self.torch.inference_mode():
            for batch_i, (images, targets, paths, shapes) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                images = images.half() if model.fp16 else images.float()
                images /= 255
                _, _, height, width = images.shape
                targets[:, 2:] *= self.torch.tensor((width, height, width, height), device=self.device)
                predictions = model(images, augment=False)
                predictions = self.non_max_suppression(
                    predictions,
                    conf_thres=self.conf_floor,
                    iou_thres=self.nms_iou,
                    labels=[],
                    multi_label=True,
                    agnostic=False,
                    max_det=self.max_det,
                )
                for image_index, prediction in enumerate(predictions):
                    labels = targets[targets[:, 0] == image_index, 1:]
                    label_count = int(labels.shape[0])
                    image_path = Path(str(paths[image_index])).resolve()
                    sample_id = path_to_id.get(str(image_path).casefold(), str(image_path))
                    native_prediction = prediction.clone()
                    if native_prediction.shape[0]:
                        self.scale_boxes(images[image_index].shape[1:], native_prediction[:, :4], shapes[image_index][0], shapes[image_index][1])
                    native_labels = None
                    if label_count:
                        target_boxes = self.xywh2xyxy(labels[:, 1:5])
                        self.scale_boxes(images[image_index].shape[1:], target_boxes, shapes[image_index][0], shapes[image_index][1])
                        native_labels = self.torch.cat((labels[:, 0:1], target_boxes), 1)
                    else:
                        native_labels = self.torch.zeros((0, 5), device=self.device)
                    correct = self.torch.zeros(native_prediction.shape[0], iouv.shape[0], dtype=self.torch.bool, device=self.device)
                    if label_count and native_prediction.shape[0]:
                        correct = self.process_batch(native_prediction, native_labels, iouv)
                    correct_numpy = correct.detach().cpu().numpy().astype(bool)
                    confidence_numpy = prediction[:, 4].detach().cpu().numpy() if prediction.shape[0] else np.zeros(0)
                    predicted_class_numpy = prediction[:, 5].detach().cpu().numpy() if prediction.shape[0] else np.zeros(0)
                    target_class_numpy = labels[:, 0].detach().cpu().numpy() if label_count else np.zeros(0)
                    stats_raw.append((correct_numpy, confidence_numpy, predicted_class_numpy, target_class_numpy))

                    matches = _match_details(native_labels, native_prediction, match_iou, self.box_iou)
                    prediction_to_match = {prediction_index: (label_index, overlap) for label_index, prediction_index, overlap in matches}
                    included = confidence_numpy >= confidence_threshold
                    matched_prediction_indices = {prediction_index for _, prediction_index, _ in matches}
                    tp_mask = correct_numpy[:, 0] if correct_numpy.shape[1] else np.zeros(len(confidence_numpy), dtype=bool)
                    tp_at_conf = included & tp_mask
                    fp_at_conf = included & ~tp_mask
                    fn_at_conf = label_count - int(tp_at_conf.sum())
                    class_tp: dict[str, int] = {}
                    class_fp: dict[str, int] = {}
                    class_fn: dict[str, int] = {}
                    for class_id, class_name in enumerate(class_names):
                        gt_count = int((target_class_numpy == class_id).sum())
                        tp_count = int((tp_at_conf & (predicted_class_numpy == class_id)).sum())
                        fp_count = int((fp_at_conf & (predicted_class_numpy == class_id)).sum())
                        if gt_count or tp_count or fp_count:
                            class_tp[class_name] = tp_count
                            class_fp[class_name] = fp_count
                            class_fn[class_name] = gt_count - tp_count
                    precision = int(tp_at_conf.sum()) / int(included.sum()) if included.sum() else 0.0
                    recall = int(tp_at_conf.sum()) / label_count if label_count else 0.0
                    image_rows.append(
                        {
                            **context,
                            "batch_index": batch_i,
                            "image_index": len(image_rows),
                            "sample_id": sample_id,
                            "image_path": str(image_path),
                            "ground_truth_count": label_count,
                            "prediction_count_after_conf_floor": int(len(confidence_numpy)),
                            "prediction_count_at_conf": int(included.sum()),
                            "tp_at_conf": int(tp_at_conf.sum()),
                            "fp_at_conf": int(fp_at_conf.sum()),
                            "fn_at_conf": int(fn_at_conf),
                            "precision_at_conf": precision,
                            "recall_at_conf": recall,
                            "f1_at_conf": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "tp_by_class": json.dumps(class_tp, ensure_ascii=False, sort_keys=True),
                            "fp_by_class": json.dumps(class_fp, ensure_ascii=False, sort_keys=True),
                            "fn_by_class": json.dumps(class_fn, ensure_ascii=False, sort_keys=True),
                        }
                    )
                    for prediction_index, row in enumerate(native_prediction.detach().cpu().tolist()):
                        x1, y1, x2, y2, score, class_id = row
                        matched_label, matched_iou = prediction_to_match.get(prediction_index, (-1, 0.0))
                        prediction_rows.append(
                            {
                                **context,
                                "image_index": len(image_rows) - 1,
                                "sample_id": sample_id,
                                "image_path": str(image_path),
                                "prediction_index": prediction_index,
                                "class_id": int(class_id),
                                "class_name": class_names[int(class_id)] if 0 <= int(class_id) < len(class_names) else str(int(class_id)),
                                "confidence": float(score),
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "matched_gt_index": matched_label,
                                "matched_iou": matched_iou,
                                "is_matched_iou50": prediction_index in matched_prediction_indices,
                                "included_at_conf": bool(included[prediction_index]),
                                "is_tp_at_conf": bool(tp_at_conf[prediction_index]),
                                "is_fp_at_conf": bool(fp_at_conf[prediction_index]),
                            }
                        )
                    matched_gt_indices = {label_index: (prediction_index, overlap) for label_index, prediction_index, overlap in matches}
                    for label_index, row in enumerate(native_labels.detach().cpu().tolist()):
                        class_id, x1, y1, x2, y2 = row
                        matched_prediction, matched_iou = matched_gt_indices.get(label_index, (-1, 0.0))
                        ground_truth_rows.append(
                            {
                                **context,
                                "image_index": len(image_rows) - 1,
                                "sample_id": sample_id,
                                "image_path": str(image_path),
                                "gt_index": label_index,
                                "class_id": int(class_id),
                                "class_name": class_names[int(class_id)] if 0 <= int(class_id) < len(class_names) else str(int(class_id)),
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "matched_prediction_index": matched_prediction,
                                "matched_iou": matched_iou,
                                "detected_at_conf": matched_prediction >= 0 and bool(included[matched_prediction]),
                            }
                        )
        metrics, per_class_rows = summarize_stats(stats_raw, class_names, confidence_threshold, self.ap_per_class)
        for row in per_class_rows:
            row.update(context)
        metrics.update(context)
        _write_json(output_dir / "metrics.json", metrics)
        _write_csv(output_dir / "per_image_metrics.csv", image_rows)
        _write_csv(output_dir / "predictions.csv", prediction_rows)
        _write_csv(output_dir / "ground_truths.csv", ground_truth_rows)
        _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
        return metrics, image_rows, prediction_rows, ground_truth_rows, per_class_rows


def _load_registry(data_layout: Path) -> dict[str, Any]:
    _ensure_project_imports()
    from yolo_retraining.data import build_registry

    payload = _read_yaml(data_layout)
    groups = payload.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"result analysis currently expects a grouped data layout: {data_layout}")
    catalog = {str(group_id): str(data_layout.resolve()) for group_id in groups}
    return build_registry(catalog)


def _comparison_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in summary_rows:
        key = (str(row["method"]), int(row["stage"]), str(row["task_name"]))
        output = grouped.setdefault(key, {"method": key[0], "stage": key[1], "task_name": key[2]})
        split = str(row["split"])
        for field in ("precision", "recall", "map50", "precision_best_f1", "recall_best_f1", "best_f1_confidence", "tp_at_conf", "fp_at_conf", "fn_at_conf"):
            output[f"{split}_{field}"] = row[field]
    return [grouped[key] for key in sorted(grouped)]


def _fixed_comparison_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("precision", "recall", "map50", "tp_at_conf", "fp_at_conf", "fn_at_conf")
    return [
        {
            "method": row["method"],
            "stage": row["stage"],
            **{f"train_{field}": row.get(f"train_{field}", "") for field in fields},
            **{f"test_{field}": row.get(f"test_{field}", "") for field in fields},
        }
        for row in comparison_rows
    ]


def _best_f1_comparison_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("precision_best_f1", "recall_best_f1", "best_f1_confidence", "map50")
    return [
        {
            "method": row["method"],
            "stage": row["stage"],
            **{f"train_{field}": row.get(f"train_{field}", "") for field in fields},
            **{f"test_{field}": row.get(f"test_{field}", "") for field in fields},
        }
        for row in comparison_rows
    ]


def _write_markdown(path: Path, summary_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    fixed_rows = _fixed_comparison_rows(comparison_rows)
    best_f1_rows = _best_f1_comparison_rows(comparison_rows)
    lines = [
        "# YOLOv5 结果分析",
        "",
        f"- 权重选择：`{config['checkpoint_selector']}`",
        f"- 固定置信度阈值：`{config['confidence_threshold']}`",
        f"- PR 扫描下限：`{config['confidence_floor']}`",
        f"- 指标匹配 IoU：`{config['match_iou_threshold']}`",
        f"- NMS IoU：`{config['nms_iou_threshold']}`",
        "",
        "## 统一固定置信度下的 Train/Test 对比",
        "",
        "每行一个 method-stage；Train 和 Test 使用同一个固定置信度阈值。",
        "",
        "| method | stage | train P | train R | train mAP50 | test P | test R | test mAP50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_rows:
        lines.append(
            f"| {row['method']} | {row['stage']} | {float(row.get('train_precision', 0)):.6f} | {float(row.get('train_recall', 0)):.6f} | {float(row.get('train_map50', 0)):.6f} | {float(row.get('test_precision', 0)):.6f} | {float(row.get('test_recall', 0)):.6f} | {float(row.get('test_map50', 0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 各自最佳 F1 点的 Train/Test 对比",
            "",
            "每个 split 独立寻找最佳 F1 置信度，因此 Train 和 Test 的 best confidence 可以不同。",
            "",
            "| method | stage | train P | train R | train best conf | train mAP50 | test P | test R | test best conf | test mAP50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_f1_rows:
        lines.append(
            f"| {row['method']} | {row['stage']} | {float(row.get('train_precision_best_f1', 0)):.6f} | {float(row.get('train_recall_best_f1', 0)):.6f} | {float(row.get('train_best_f1_confidence', 0)):.6f} | {float(row.get('train_map50', 0)):.6f} | {float(row.get('test_precision_best_f1', 0)):.6f} | {float(row.get('test_recall_best_f1', 0)):.6f} | {float(row.get('test_best_f1_confidence', 0)):.6f} | {float(row.get('test_map50', 0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "说明：逐图 TP/FP/FN 和每个 split 的详细指标仍保存在 `per_image_metrics.csv`、`summary.csv` 及各 task 的明细目录中。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=None, help="sequence run directory; repeat three times, or omit to auto-discover the three completed baselines")
    parser.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT, help=f"root used for auto-discovery (default: {DEFAULT_SEQUENCE_ROOT})")
    parser.add_argument("--data-layout", type=Path, default=DEFAULT_DATA_LAYOUT, help=f"grouped data layout YAML (default: {DEFAULT_DATA_LAYOUT})")
    parser.add_argument("--stages", default="1,2,3", help="stages to analyze, e.g. '1,2,3' or '0-3' (default: 1,2,3)")
    parser.add_argument("--checkpoint", default="best.pt", help="checkpoint file name under each task/checkpoints, or one explicit checkpoint path (default: best.pt)")
    parser.add_argument("--output-dir", type=Path, default=None, help="full output directory for this analysis; omitted creates a timestamped child under result_analysis/train_test_overfitting")
    parser.add_argument("--analysis-name", default=None, help="name for the auto-created analysis child directory")
    parser.add_argument("--device", default="0", help="YOLOv5 device argument, e.g. 0 or cpu")
    parser.add_argument("--batch", type=int, default=16, help="inference batch size; independent of training batch")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--half", action="store_true", help="use FP16 inference on CUDA")
    parser.add_argument("--limit-images", type=int, default=0, help="optional per-split image limit for a smoke check; 0 analyzes every image")
    parser.add_argument("--skip-missing-ids", action="store_true", help="skip IDs absent from the local data layout and record them; default is to fail loudly")
    parser.add_argument("--conf-thres", "--fixed-conf-thres", dest="conf_thres", type=float, default=DEFAULT_FIXED_CONFIDENCE, help=f"one shared confidence threshold for fixed P/R and per-image TP/FP/FN (default: {DEFAULT_FIXED_CONFIDENCE}; override when the test protocol specifies another value)")
    parser.add_argument("--conf-floor", type=float, default=0.001, help="lowest confidence retained for PR/AP computation")
    parser.add_argument("--match-iou-thres", type=float, default=0.5, help="IoU threshold for TP/FP/FN and mAP50")
    parser.add_argument("--nms-iou-thres", type=float, default=0.6, help="NMS IoU threshold; defaults to YOLOv5 val.py's 0.6")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--yolov5-root", type=Path, default=None, help="optional pinned YOLOv5 source root")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing output directory")
    return parser


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        path = args.output_dir.expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        name = args.analysis_name or f"{timestamp}__best__stages-{args.stages.replace(',', '_')}"
        path = (DEFAULT_ANALYSIS_ROOT / name).resolve()
    if path.exists() and any(path.iterdir()) and not args.overwrite:
        raise FileExistsError(f"analysis output directory is not empty; choose another path or use --overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.conf_floor <= 1 or not 0 <= args.conf_thres <= 1:
        raise ValueError("conf-floor and conf-thres must be in [0, 1]")
    if args.conf_thres < args.conf_floor:
        raise ValueError("conf-thres cannot be lower than conf-floor")
    if abs(args.match_iou_thres - 0.5) > 1e-12:
        raise ValueError("this analysis reports mAP50, so --match-iou-thres must remain 0.5")
    if not 0 < args.match_iou_thres <= 1 or not 0 < args.nms_iou_thres <= 1:
        raise ValueError("IoU thresholds must be in (0, 1]")
    if args.batch <= 0 or args.imgsz <= 0 or args.workers < 0 or args.max_det <= 0 or args.limit_images < 0:
        raise ValueError("batch/imgsz/max-det must be positive, workers non-negative, and limit-images non-negative")

    stages = _parse_integer_spec(args.stages)
    data_layout = args.data_layout.expanduser().resolve()
    if not data_layout.is_file():
        raise FileNotFoundError(f"data layout does not exist: {data_layout}")
    run_dirs = discover_run_dirs(args.sequence_root, args.run_dir)
    registry = _load_registry(data_layout)
    tasks = collect_tasks(run_dirs, stages, args.checkpoint, registry, args.skip_missing_ids)
    output_dir = _output_dir(args)
    runtime = YoloV5Runtime(
        args.yolov5_root,
        args.device,
        args.batch,
        args.workers,
        args.imgsz,
        args.half,
        args.nms_iou_thres,
        args.conf_floor,
        args.max_det,
    )

    analysis_config: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "checkpoint_selector": args.checkpoint,
        "stages": stages,
        "data_layout": str(data_layout),
        "sequence_runs": [str(path) for path in run_dirs],
        "confidence_threshold": args.conf_thres,
        "confidence_floor": args.conf_floor,
        "match_iou_threshold": args.match_iou_thres,
        "nms_iou_threshold": args.nms_iou_thres,
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "half": args.half,
        "limit_images": args.limit_images,
        "skip_missing_ids": args.skip_missing_ids,
        "max_det": args.max_det,
        "tasks": [
            {
                "method": task["method"],
                "stage": task["stage"],
                "task_name": task["task_name"],
                "task_dir": str(task["task_dir"]),
                "checkpoint": str(task["checkpoint"]),
                "train_count": len(task["train_ids"]),
                "test_count": len(task["test_ids"]),
                "missing_train_ids": task["missing_train_ids"],
                "missing_test_ids": task["missing_test_ids"],
            }
            for task in tasks
        ],
    }
    _write_json(output_dir / "analysis_config.json", analysis_config)
    _write_json(output_dir / "label_schema.json", {"names": registry["names"], "data_layout": str(data_layout)})
    _write_csv(
        output_dir / "missing_ids.csv",
        [
            {"method": task["method"], "stage": task["stage"], "task_name": task["task_name"], "split": "train", "sample_id": sample_id}
            for task in tasks
            for sample_id in task["missing_train_ids"]
        ]
        + [
            {"method": task["method"], "stage": task["stage"], "task_name": task["task_name"], "split": "test", "sample_id": sample_id}
            for task in tasks
            for sample_id in task["missing_test_ids"]
        ],
    )

    all_summary_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    all_image_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    all_ground_truth_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        task_key = f"{_safe_name(task['method'])}__stage{task['stage']}__{_safe_name(task['task_name'])}"
        task_output = output_dir / "evaluations" / task_key
        task_output.mkdir(parents=True, exist_ok=True)
        context_base = {
            "method": task["method"],
            "stage": task["stage"],
            "task_name": task["task_name"],
            "checkpoint": str(task["checkpoint"]),
        }
        split_manifests: dict[str, tuple[Path, Path]] = {}
        for split, original_ids in (("train", task["train_ids"]), ("test", task["test_ids"])):
            ids = original_ids[: args.limit_images] if args.limit_images else original_ids
            split_manifests[split] = _write_split_dataset_files(task_output / "datasets" / split, split, ids, registry)
            (task_output / "datasets" / split / "sample_ids.txt").write_text("".join(f"{sample_id}\n" for sample_id in ids), encoding="utf-8")
        model = runtime.load_model(task["checkpoint"], split_manifests["train"][1])
        for split, original_ids in (("train", task["train_ids"]), ("test", task["test_ids"])):
            ids = original_ids[: args.limit_images] if args.limit_images else original_ids
            context = {**context_base, "split": split}
            print(f"[{task_index}/{len(tasks)}] {task['method']} stage{task['stage']} {split}: {len(ids)} images", flush=True)
            split_output = task_output / split
            metrics, image_rows, prediction_rows, ground_truth_rows, class_rows = runtime.evaluate_split(
                model,
                split_manifests[split][0],
                ids,
                registry,
                context,
                split_output,
                args.match_iou_thres,
                args.conf_thres,
            )
            all_summary_rows.append(metrics)
            all_class_rows.extend(class_rows)
            all_image_rows.extend(image_rows)
            all_prediction_rows.extend(prediction_rows)
            all_ground_truth_rows.extend(ground_truth_rows)

    comparison = _comparison_rows(all_summary_rows)
    fixed_comparison = _fixed_comparison_rows(comparison)
    best_f1_comparison = _best_f1_comparison_rows(comparison)
    _write_csv(output_dir / "summary.csv", all_summary_rows)
    _write_csv(output_dir / "comparison.csv", comparison)
    _write_csv(output_dir / "comparison_fixed_threshold.csv", fixed_comparison)
    _write_csv(output_dir / "comparison_best_f1.csv", best_f1_comparison)
    _write_csv(output_dir / "per_class_metrics.csv", all_class_rows)
    _write_csv(output_dir / "per_image_metrics.csv", all_image_rows)
    _write_csv(output_dir / "predictions.csv", all_prediction_rows)
    _write_csv(output_dir / "ground_truths.csv", all_ground_truth_rows)
    _write_markdown(output_dir / "comparison.md", all_summary_rows, comparison, analysis_config)
    print(f"analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
