from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping


def training_arguments(config: Mapping[str, Any], *, data_yaml: Path, output_dir: Path) -> dict[str, Any]:
    params = dict(config["backend"].get("params", {}))
    allowed = {"batch", "imgsz", "device", "workers", "amp", "cache", "optimizer", "lr0", "lrf", "patience", "deterministic"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unknown Ultralytics backend params: {sorted(unknown)}")
    params.update(
        data=str(data_yaml),
        epochs=int(config["budget"]["value"]),
        project=str(output_dir),
        name="train",
        exist_ok=False,
        seed=int(config["task"]["seed"]),
        save=True,
        val=True,
        plots=False,
        verbose=False,
    )
    params.setdefault("deterministic", True)
    return params


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

