"""Build an AnyLabeling-friendly FP/FN review directory for one task result.

The input is one completed task directory. The script evaluates the requested
split with ``best.pt`` (by default), then creates image links and matching
X-AnyLabeling JSON files in separate ``false_positives`` and
``false_negatives`` directories. Each JSON overlays all ground-truth boxes in
white, true-positive predictions in green, and false-positive predictions in
red. A white ground-truth box without a matching green prediction is a false
negative. Source images are never moved or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = Path(__file__).resolve().parent
OVERFITTING_METHOD_ROOT = PROJECT_ROOT / "result_analysis" / "train_test_overfitting"
DEFAULT_CONFIDENCE = 0.25
ANYLABELING_JSON_VERSION = "4.0.2"
REVIEW_STATUS_COLORS = {
    "GT": [255, 255, 255],
    "TP": [0, 200, 0],
    "FP": [230, 25, 25],
}

if str(OVERFITTING_METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERFITTING_METHOD_ROOT))

from analyze_train_test_overfitting import (  # noqa: E402
    YoloV5Runtime,
    _load_registry,
    _write_split_dataset_files,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _validate_torch_numpy_compatibility() -> None:
    """Fail early for the known Torch 2.2 + NumPy 2 ABI mismatch."""

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("PyTorch is not installed in the active environment") from error
    torch_numbers = tuple(
        int(value) for value in re.match(r"(\d+)\.(\d+)", torch_version).groups()
    )
    numpy_major = int(np.__version__.split(".", 1)[0])
    if torch_numbers <= (2, 2) and numpy_major >= 2:
        raise RuntimeError(
            "the active environment is incompatible: "
            f"torch {torch_version} cannot use NumPy {np.__version__}. "
            "Install a NumPy 1.x release (for example: python -m pip install 'numpy<2') "
            "before running YOLOv5 inference."
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "error_type",
        "review_file",
        "sample_id",
        "image_path",
        "ground_truth_count",
        "prediction_count_at_conf",
        "tp_at_conf",
        "fp_at_conf",
        "fn_at_conf",
        "fp_by_class",
        "fn_by_class",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip(" ._") or "item"


def _read_sample_ids(task_dir: Path, split: str) -> list[str]:
    data_dir = task_dir / "data"
    if split == "test":
        files = sorted(data_dir.glob("test_*_ids.txt"))
    elif split == "validation":
        files = [data_dir / "validation_ids.txt"]
    else:
        files = [data_dir / "selected_ids.txt"]

    existing = [path for path in files if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"task has no sample-ID file for split {split!r}: {data_dir}")

    sample_ids: list[str] = []
    for path in existing:
        sample_ids.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    unique_ids = list(dict.fromkeys(sample_ids))
    if not unique_ids:
        raise ValueError(f"sample-ID files are empty for split {split!r}: {existing}")
    return unique_ids


def _resolve_task_dir(value: Path) -> Path:
    task_dir = value.expanduser().resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task result directory does not exist: {task_dir}")
    if not (task_dir / "task.yaml").is_file():
        raise FileNotFoundError(
            f"input must be one task result directory containing task.yaml: {task_dir}"
        )
    result_path = task_dir / "task_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") not in {None, "completed"}:
            raise RuntimeError(f"task is not completed: {task_dir}")
    return task_dir


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


def _resolve_data_layout(
    task_dir: Path,
    task_payload: Mapping[str, Any],
    explicit: Path | None,
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    else:
        data_config = task_payload.get("data", {})
        raw_layout = data_config.get("layout") if isinstance(data_config, Mapping) else None
        if isinstance(raw_layout, str) and raw_layout:
            recorded = Path(raw_layout).expanduser()
            candidates.append(recorded)
            if not recorded.is_absolute():
                candidates.append(PROJECT_ROOT / recorded)
                candidates.append(task_dir / recorded)
            candidates.append(PROJECT_ROOT / "configs" / "data" / recorded.name)

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved
    if explicit is not None:
        raise FileNotFoundError(f"data layout does not exist: {explicit.expanduser().resolve()}")
    raise FileNotFoundError(
        "could not resolve the task's data layout on this machine; "
        f"checked {checked}. Pass the local file explicitly with --data-layout."
    )


def _default_output_dir(task_dir: Path, split: str, confidence: float) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    task_name = _safe_component(task_dir.name)
    confidence_name = f"{confidence:g}".replace(".", "_")
    return METHOD_ROOT / f"{timestamp}__{task_name}__{split}__conf-{confidence_name}"


def _review_filename(row: Mapping[str, Any]) -> str:
    image_path = Path(str(row["image_path"]))
    parent = _safe_component(image_path.parent.name)
    filename = _safe_component(image_path.name)
    return f"{parent}__{filename}"


def _deduplicate_review_names(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    owners: dict[str, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        candidate = _review_filename(row)
        key = candidate.casefold()
        owner = owners.get(key)
        if owner is not None and owner != sample_id:
            image_path = Path(str(row["image_path"]))
            digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8]
            candidate = f"{image_path.parent.name}__{image_path.stem}__{digest}{image_path.suffix}"
            candidate = _safe_component(candidate)
            key = candidate.casefold()
        owners[key] = sample_id
        names[sample_id] = candidate
    return names


def _create_image_link(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"source image does not exist: {source}")
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(
                f"cannot create hard link {destination} -> {source}. Hard links require "
                "source and output to be on the same volume; use --link-mode symlink if needed."
            ) from error
    else:
        try:
            destination.symlink_to(source)
        except OSError as error:
            hint = (
                " On Windows, enable Developer Mode or run the terminal with permission "
                "to create symbolic links."
                if os.name == "nt"
                else ""
            )
            raise OSError(f"cannot create symbolic link {destination} -> {source}.{hint}") from error


def _export_onnx(
    checkpoint: Path,
    yolov5_root: Path,
    imgsz: int,
    device: str,
) -> Path:
    export_script = yolov5_root / "export.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"YOLOv5 export.py does not exist: {export_script}")
    command = [
        sys.executable,
        str(export_script),
        "--weights",
        str(checkpoint),
        "--include",
        "onnx",
        "--imgsz",
        str(imgsz),
        "--batch-size",
        "1",
        "--device",
        device,
    ]
    subprocess.run(command, cwd=yolov5_root, check=True)
    onnx_path = checkpoint.with_suffix(".onnx")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"YOLOv5 reported success but did not create: {onnx_path}")
    return onnx_path


def _validate_onnx(onnx_path: Path, imgsz: int, class_count: int) -> dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "ONNX export validation requires both onnx and onnxruntime in the active environment"
        ) from error

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = [
        {"name": item.name, "shape": list(item.shape), "type": item.type}
        for item in session.get_inputs()
    ]
    outputs = [
        {"name": item.name, "shape": list(item.shape), "type": item.type}
        for item in session.get_outputs()
    ]
    if len(inputs) != 1 or list(inputs[0]["shape"]) != [1, 3, imgsz, imgsz]:
        raise ValueError(
            f"unexpected ONNX input; expected [1, 3, {imgsz}, {imgsz}], found {inputs}"
        )
    expected_columns = class_count + 5
    if not outputs or not outputs[0]["shape"] or outputs[0]["shape"][-1] != expected_columns:
        raise ValueError(
            f"unexpected YOLOv5 output; expected last dimension {expected_columns}, found {outputs}"
        )
    return {
        "checker": "passed",
        "runtime_session": "passed",
        "inputs": inputs,
        "outputs": outputs,
        "metadata": {item.key: item.value for item in model.metadata_props},
    }


def _write_anylabeling_config(
    path: Path,
    task_dir: Path,
    class_names: Sequence[str],
    confidence_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> None:
    task_digest = hashlib.sha1(str(task_dir).encode("utf-8")).hexdigest()[:8]
    payload = {
        "type": "yolov5",
        "name": f"yolov5-review-{task_digest}",
        "provider": "YOLO Retraining",
        "display_name": f"YOLOv5 Review - {task_dir.name}",
        "model_path": "best.onnx",
        "engine": "ort",
        "iou_threshold": iou_threshold,
        "conf_threshold": confidence_threshold,
        "max_det": max_det,
        "classes": list(class_names),
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _rows_by_sample_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    return grouped


def _clamped_box_points(
    row: Mapping[str, Any], image_width: int, image_height: int
) -> list[list[float]]:
    x1 = min(max(float(row["x1"]), 0.0), float(image_width))
    y1 = min(max(float(row["y1"]), 0.0), float(image_height))
    x2 = min(max(float(row["x2"]), 0.0), float(image_width))
    y2 = min(max(float(row["y2"]), 0.0), float(image_height))
    return [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]


def _comparison_shape(
    row: Mapping[str, Any],
    status: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    class_name = str(row["class_name"])
    is_prediction = "prediction_index" in row
    score = float(row["confidence"]) if is_prediction else None

    if status not in REVIEW_STATUS_COLORS:
        raise ValueError(f"unknown comparison status: {status}")

    return {
        "label": f"{status}/{class_name}",
        "score": score,
        "points": _clamped_box_points(row, image_width, image_height),
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
    }


def _build_anylabeling_comparison_payload(
    image_path: Path,
    review_filename: str,
    image_row: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    ground_truth_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    included_predictions = [
        row for row in prediction_rows if bool(row.get("included_at_conf", False))
    ]
    if len(included_predictions) != int(image_row["prediction_count_at_conf"]):
        raise ValueError(
            f"prediction geometry count mismatch for {image_row['sample_id']}: "
            f"expected {image_row['prediction_count_at_conf']}, found {len(included_predictions)}"
        )
    if len(ground_truth_rows) != int(image_row["ground_truth_count"]):
        raise ValueError(
            f"ground-truth geometry count mismatch for {image_row['sample_id']}: "
            f"expected {image_row['ground_truth_count']}, found {len(ground_truth_rows)}"
        )

    matched_predictions = [row for row in included_predictions if bool(row["is_tp_at_conf"])]
    false_positive_predictions = [
        row for row in included_predictions if bool(row["is_fp_at_conf"])
    ]
    shapes = [
        *(
            _comparison_shape(row, "GT", image_width, image_height)
            for row in ground_truth_rows
        ),
        *(
            _comparison_shape(row, "TP", image_width, image_height)
            for row in matched_predictions
        ),
        *(
            _comparison_shape(row, "FP", image_width, image_height)
            for row in false_positive_predictions
        ),
    ]
    return {
        "version": ANYLABELING_JSON_VERSION,
        "flags": {"yolo_retraining_model_review": True},
        "checked": False,
        "shapes": shapes,
        "description": None,
        "imagePath": review_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def _write_review_color_reference(
    output_dir: Path, class_names: Sequence[str]
) -> None:
    label_colors = {
        f"{status}/{class_name}": list(color)
        for status, color in REVIEW_STATUS_COLORS.items()
        for class_name in class_names
    }
    payload = {
        "shape_color": "manual",
        "label_colors": label_colors,
    }
    (output_dir / "anylabeling_color_snippet.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _build_review_directory(
    output_dir: Path,
    checkpoint: Path,
    image_rows: Sequence[Mapping[str, Any]],
    link_mode: str,
    config: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]] = (),
    ground_truth_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    fp_dir = output_dir / "false_positives"
    fn_dir = output_dir / "false_negatives"
    fp_dir.mkdir()
    fn_dir.mkdir()

    error_rows = [
        row for row in image_rows if int(row["fp_at_conf"]) > 0 or int(row["fn_at_conf"]) > 0
    ]
    predictions_by_sample = _rows_by_sample_id(prediction_rows)
    ground_truths_by_sample = _rows_by_sample_id(ground_truth_rows)
    review_names = _deduplicate_review_names(error_rows)
    jobs: list[tuple[str, Path, Path, Mapping[str, Any]]] = []
    for row in error_rows:
        source = Path(str(row["image_path"]))
        filename = review_names[str(row["sample_id"])]
        if int(row["fp_at_conf"]) > 0:
            jobs.append(("FP", source, fp_dir / filename, row))
        if int(row["fn_at_conf"]) > 0:
            jobs.append(("FN", source, fn_dir / filename, row))

    manifest_rows: list[dict[str, Any]] = []
    for error_type, source, destination, row in tqdm(
        jobs,
        desc="Creating image links and comparison JSON",
        unit="link",
        dynamic_ncols=True,
    ):
        _create_image_link(source, destination, link_mode)
        sample_id = str(row["sample_id"])
        comparison = _build_anylabeling_comparison_payload(
            source,
            destination.name,
            row,
            predictions_by_sample.get(sample_id, []),
            ground_truths_by_sample.get(sample_id, []),
        )
        _write_json(destination.with_suffix(".json"), comparison)
        manifest_rows.append(
            {
                **row,
                "error_type": error_type,
                "review_file": str(destination),
            }
        )

    print("Copying checkpoint: best.pt", flush=True)
    shutil.copy2(checkpoint, output_dir / "best.pt")
    _write_csv(output_dir / "review_manifest.csv", manifest_rows)
    _write_review_color_reference(output_dir, list(config.get("class_names", [])))

    fp_rows = [row for row in image_rows if int(row["fp_at_conf"]) > 0]
    fn_rows = [row for row in image_rows if int(row["fn_at_conf"]) > 0]
    summary = {
        **config,
        "output_dir": str(output_dir),
        "image_count": len(image_rows),
        "false_positive_image_count": len(fp_rows),
        "false_negative_image_count": len(fn_rows),
        "false_positive_box_count": sum(int(row["fp_at_conf"]) for row in image_rows),
        "false_negative_box_count": sum(int(row["fn_at_conf"]) for row in image_rows),
        "images_with_both_error_types": sum(
            int(row["fp_at_conf"]) > 0 and int(row["fn_at_conf"]) > 0
            for row in image_rows
        ),
        "comparison_json_count": len(jobs),
        "review_status_colors": REVIEW_STATUS_COLORS,
    }
    _write_json(output_dir / "review_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path, help="one task result directory containing task.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; omitted creates a timestamped directory under this method",
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation", "train"),
        default="test",
        help="dataset split to review (default: test)",
    )
    parser.add_argument("--checkpoint", default="best.pt", help="checkpoint under task_dir/checkpoints, or an explicit path")
    parser.add_argument("--data-layout", type=Path, default=None, help="local grouped data-layout YAML; normally inferred from task.yaml")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIDENCE, help=f"confidence threshold for FP/FN (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--match-iou-thres", type=float, default=0.5, help="class-aware matching IoU (default: 0.5)")
    parser.add_argument("--nms-iou-thres", type=float, default=0.6, help="YOLOv5 NMS IoU (default: 0.6)")
    parser.add_argument("--device", default="0", help="YOLOv5 device, e.g. 0 or cpu")
    parser.add_argument("--batch", type=int, default=16, help="inference batch size (default: 16)")
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4, help="dataloader workers (default: 0 on Windows, 4 elsewhere)")
    parser.add_argument("--imgsz", type=int, default=None, help="inference size; defaults to the value recorded in task.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 inference on CUDA")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--yolov5-root", type=Path, default=None, help="optional pinned YOLOv5 source root")
    parser.add_argument("--export-device", default="cpu", help="device used by YOLOv5 ONNX export (default: cpu)")
    parser.add_argument("--limit-images", type=int, default=0, help="analyze only the first N images for a smoke check; 0 means all")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink"),
        default="hardlink" if os.name == "nt" else "symlink",
        help="image-link type (default: hardlink on Windows, symlink elsewhere)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    conversion_started_at = datetime.now().astimezone()
    conversion_started = time.perf_counter()
    args = build_parser().parse_args(argv)
    if not 0 <= args.conf_thres <= 1:
        raise ValueError("conf-thres must be in [0, 1]")
    if not 0 < args.match_iou_thres <= 1 or not 0 < args.nms_iou_thres <= 1:
        raise ValueError("IoU thresholds must be in (0, 1]")
    if args.batch <= 0 or args.workers < 0 or args.max_det <= 0 or args.limit_images < 0:
        raise ValueError("batch/max-det must be positive; workers/limit-images must be non-negative")

    task_dir = _resolve_task_dir(args.task_dir)
    task_payload = _read_yaml(task_dir / "task.yaml")
    checkpoint = _resolve_checkpoint(task_dir, args.checkpoint)
    data_layout = _resolve_data_layout(task_dir, task_payload, args.data_layout)
    sample_ids = _read_sample_ids(task_dir, args.split)
    if args.limit_images:
        sample_ids = sample_ids[: args.limit_images]

    registry = _load_registry(data_layout)
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in registry["records"]]
    if missing_ids:
        raise KeyError(
            f"data layout cannot resolve {len(missing_ids)} task sample IDs; first values: "
            f"{missing_ids[:5]}"
        )

    backend_params = task_payload.get("backend", {}).get("params", {})
    recorded_imgsz = backend_params.get("imgsz", 640) if isinstance(backend_params, Mapping) else 640
    imgsz = args.imgsz if args.imgsz is not None else int(recorded_imgsz)
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(task_dir, args.split, args.conf_thres).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    setup_seconds = time.perf_counter() - conversion_started
    print(f"[1/6] Task: {task_dir}", flush=True)
    print(f"[2/6] Evaluating {len(sample_ids)} {args.split} images with {checkpoint.name}", flush=True)
    evaluation_started = time.perf_counter()
    _validate_torch_numpy_compatibility()
    runtime = YoloV5Runtime(
        args.yolov5_root,
        args.device,
        args.batch,
        args.workers,
        imgsz,
        args.half,
        args.nms_iou_thres,
        args.conf_thres,
        args.max_det,
    )
    with tempfile.TemporaryDirectory(prefix=".fp_fn_review_", dir=METHOD_ROOT) as temporary:
        temporary_dir = Path(temporary)
        manifest, data_yaml = _write_split_dataset_files(
            temporary_dir / "dataset", args.split, sample_ids, registry
        )
        model = runtime.load_model(checkpoint, data_yaml)
        _, image_rows, prediction_rows, ground_truth_rows, _ = runtime.evaluate_split(
            model,
            manifest,
            sample_ids,
            registry,
            {"split": args.split},
            temporary_dir / "evaluation",
            args.match_iou_thres,
            args.conf_thres,
        )
    evaluation_seconds = time.perf_counter() - evaluation_started

    print("[3/6] Constructing AnyLabeling review directory", flush=True)
    review_started = time.perf_counter()
    config = {
        "task_dir": str(task_dir),
        "checkpoint_source": str(checkpoint),
        "data_layout": str(data_layout),
        "split": args.split,
        "confidence_threshold": args.conf_thres,
        "match_iou_threshold": args.match_iou_thres,
        "nms_iou_threshold": args.nms_iou_thres,
        "imgsz": imgsz,
        "link_mode": args.link_mode,
        "class_names": registry["names"],
    }
    summary = _build_review_directory(
        output_dir,
        checkpoint,
        image_rows,
        args.link_mode,
        config,
        prediction_rows,
        ground_truth_rows,
    )
    review_seconds = time.perf_counter() - review_started

    print("[4/6] Exporting best.pt to fixed-shape best.onnx", flush=True)
    export_started = time.perf_counter()
    best_pt = output_dir / "best.pt"
    best_onnx = _export_onnx(best_pt, runtime.source_root, imgsz, args.export_device)
    export_seconds = time.perf_counter() - export_started

    print("[5/6] Validating ONNX and writing AnyLabeling YAML", flush=True)
    validation_started = time.perf_counter()
    onnx_validation = _validate_onnx(best_onnx, imgsz, len(registry["names"]))
    anylabeling_config = output_dir / "yolov5_anylabeling.yaml"
    _write_anylabeling_config(
        anylabeling_config,
        task_dir,
        registry["names"],
        args.conf_thres,
        args.nms_iou_thres,
        args.max_det,
    )
    summary.update(
        {
            "onnx_path": str(best_onnx),
            "anylabeling_config": str(anylabeling_config),
            "onnx_validation": onnx_validation,
        }
    )
    validation_seconds = time.perf_counter() - validation_started
    conversion_finished_at = datetime.now().astimezone()
    conversion_seconds = time.perf_counter() - conversion_started
    summary.update(
        {
            "conversion_started_at": conversion_started_at.isoformat(timespec="seconds"),
            "conversion_finished_at": conversion_finished_at.isoformat(timespec="seconds"),
            "conversion_duration_seconds": round(conversion_seconds, 3),
            "timings_seconds": {
                "setup": round(setup_seconds, 3),
                "evaluation": round(evaluation_seconds, 3),
                "review_directory_and_json": round(review_seconds, 3),
                "onnx_export": round(export_seconds, 3),
                "onnx_validation_and_config": round(validation_seconds, 3),
            },
        }
    )
    _write_json(output_dir / "review_summary.json", summary)
    print(
        "[6/6] Done: "
        f"{summary['false_positive_image_count']} FP images, "
        f"{summary['false_negative_image_count']} FN images, "
        f"{conversion_seconds:.1f}s ({conversion_seconds / 60:.2f} min) -> {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
