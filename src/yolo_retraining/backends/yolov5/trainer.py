from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOWED_PARAMS = {"batch", "imgsz", "device", "workers", "cache", "optimizer", "patience", "amp"}


def device_argument(device: Any) -> str:
    if isinstance(device, list):
        if not device:
            raise ValueError("backend.params.device list cannot be empty")
        return ",".join(str(item) for item in device)
    return str(device)


def validate_training_params(config: Mapping[str, Any]) -> None:
    params = config["backend"].get("params", {})
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        raise ValueError(f"unknown YOLOv5 backend params: {sorted(unknown)}")
    batch = int(params.get("batch", 0))
    imgsz = int(params.get("imgsz", 0))
    workers = int(params.get("workers", 0))
    if batch <= 0 or imgsz <= 0 or workers < 0:
        raise ValueError("backend.params batch/imgsz must be positive and workers must be non-negative")
    if params.get("amp", True) is not True:
        raise ValueError("original YOLOv5 v7.0 controls AMP automatically; backend.params.amp=false is not supported")
    optimizer = str(params.get("optimizer", "SGD"))
    if optimizer not in {"SGD", "Adam", "AdamW"}:
        raise ValueError("backend.params.optimizer must be SGD, Adam, or AdamW")
    device = params.get("device", "cpu")
    if isinstance(device, list) and len(device) > 1 and batch % len(device) != 0:
        raise ValueError("backend.params.batch must be divisible by the number of DDP devices")


def build_train_command(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    checkpoint: Path,
    data_yaml: Path,
    output_dir: Path,
) -> list[str]:
    validate_training_params(config)
    params = config["backend"]["params"]
    device = params.get("device", "cpu")
    devices = device if isinstance(device, list) else [device]
    command = [sys.executable]
    if len(devices) > 1:
        command.extend(["-m", "torch.distributed.run", "--nproc_per_node", str(len(devices))])
    command.extend(
        [
            str(Path(__file__).with_name("train_entry.py")),
            "--yolo-source-root",
            str(source_root),
            "--weights",
            str(checkpoint),
            "--data",
            str(data_yaml),
            "--epochs",
            str(int(config["budget"]["value"])),
            "--batch-size",
            str(int(params["batch"])),
            "--imgsz",
            str(int(params["imgsz"])),
            "--device",
            device_argument(device),
            "--workers",
            str(int(params.get("workers", 4))),
            "--optimizer",
            str(params.get("optimizer", "SGD")),
            "--patience",
            str(int(params.get("patience", 100))),
            "--seed",
            str(int(config["task"]["seed"])),
            "--project",
            str(output_dir),
            "--name",
            "train",
            "--exist-ok",
            "--noplots",
        ]
    )
    cache = params.get("cache", False)
    if cache:
        command.append("--cache")
        if isinstance(cache, str):
            command.append(cache)
    return command


def run_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"YOLOv5 subprocess failed with exit code {result.returncode}; see {log_path}")


def read_training_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key.strip(): _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _number(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def estimated_optimizer_steps(sample_count: int, batch: int, epochs: int) -> int:
    return math.ceil(sample_count / max(1, batch)) * epochs
