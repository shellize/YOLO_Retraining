from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from smoke_data import prepare_real_smoke_dataset
from yolo_retraining.backends.yolov5.source import validate_yolov5_source
from yolo_retraining.engine import TaskRunner


pytestmark = pytest.mark.smoke


def _real_data_root() -> Path:
    configured = os.environ.get("YOLO_RETRAINING_REAL_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    data_root = Path(__file__).parents[1] / "data"
    organized = data_root / "organized"
    return organized if organized.is_dir() else data_root / "self_improving"


def _config(tmp_path: Path, dataset_yaml: Path) -> dict:
    return {
        "task": {"name": "real-smoke", "label": "full-cold", "seed": 42, "output_root": str(tmp_path / "runs"), "parent_result": None},
        "data": {"catalog": {"smoke": str(dataset_yaml)}, "current": ["smoke"], "candidate": ["smoke"], "validation": ["smoke"], "test": ["smoke"]},
        "model": {"backend": "yolov5", "definition": "yolov5s.yaml"},
        "initialization": {"source": "pretrained", "checkpoint": "yolov5s.pt"},
        "select_policy": {"name": "full", "params": {}},
        "epoch_policy": {"name": "static", "params": {}},
        "budget": {"type": "epochs", "value": 1},
        "backend": {"params": {"batch": 4, "imgsz": 640, "device": 0, "workers": 2, "amp": True}},
        "evaluation": {"primary_metric": "map50_95", "test_scope": "seen", "evaluate_checkpoints": ["last", "best"]},
    }


def test_real_yolov5s_full_cold(tmp_path: Path) -> None:
    if os.environ.get("YOLO_RETRAINING_RUN_V5_SMOKE") != "1":
        pytest.skip("set YOLO_RETRAINING_RUN_V5_SMOKE=1 to run original YOLOv5s training")
    if not torch.cuda.is_available():
        pytest.skip("original YOLOv5s real smoke requires a CUDA device")
    prepared = prepare_real_smoke_dataset(_real_data_root(), tmp_path / "manifests")
    source_hashes = {
        sample.image: hashlib.sha256(sample.image.read_bytes()).hexdigest()
        for split in ("train", "val", "test")
        for sample in prepared[split]
    }
    output = TaskRunner(_config(tmp_path, Path(prepared["yaml"]))).run()
    result = json.loads((output / "task_result.json").read_text(encoding="utf-8"))
    status = json.loads((output / "task_status.json").read_text(encoding="utf-8"))
    assert status["status"] == result["status"] == "completed"
    assert (output / "checkpoints" / "last.pt").stat().st_size > 0
    assert (output / "checkpoints" / "best.pt").stat().st_size > 0
    assert len((output / "data" / "backend_train_images.txt").read_text(encoding="utf-8").splitlines()) == 64
    assert set(result["metrics"]) == {"last", "best"}
    for checkpoint in ("last", "best"):
        metrics = result["metrics"][checkpoint]["smoke"]
        assert {"map50_95", "map50", "precision", "recall"}.issubset(metrics)
        assert set(metrics["per_class_ap"]) == set(prepared["names"])
    assert result["backend"]["family"] == "original-yolov5"
    assert result["backend"]["source_commit"] == "915bbf294bb74c859f0b41f1c23bc395014ea679"
    assert result["backend"]["best_selection_metric"] == "0.1*map50+0.9*map50_95"
    assert result["backend"]["data_loader_adaptation"] == "read_only_incomplete_jpeg"
    assert source_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes}
    train_log = (output / "logs" / "yolov5_train.log").read_text(encoding="utf-8")
    assert "restored and saved" not in train_log
    for evaluation_log in (output / "backend" / "yolov5" / "evaluations").rglob("evaluation.log"):
        assert "restored and saved" not in evaluation_log.read_text(encoding="utf-8")
    assert not [path for path in output.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    source = validate_yolov5_source()
    inspect_path = tmp_path / "model_inspection.json"
    bridge = Path(__file__).parents[1] / "src" / "yolo_retraining" / "backends" / "yolov5" / "bridge.py"
    subprocess.run(
        [sys.executable, str(bridge), "inspect", "--root", source["root"], "--weights", str(output / "checkpoints" / "last.pt"), "--output", str(inspect_path)],
        check=True,
        cwd=source["root"],
    )
    inspection = json.loads(inspect_path.read_text(encoding="utf-8"))
    assert inspection["anchor_based"] is True
    assert inspection["has_objectness"] is True
    assert inspection["anchors_per_scale"] == 3
