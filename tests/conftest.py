from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml
from PIL import Image

from yolo_retraining.backends.base import DetectionBackend


def make_yolo_group(root: Path, group_id: str, *, names: list[str] | None = None) -> Path:
    group_root = root / group_id
    names = names or ["object"]
    for split_index, split in enumerate(("train", "val", "test")):
        image_dir = group_root / "images" / split
        label_dir = group_root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(2 if split == "train" else 1):
            image_path = image_dir / f"{group_id}_{split}_{index}.jpg"
            Image.new("RGB", (32, 24), color=(20 + split_index * 20, 30 + index * 10, 40)).save(image_path)
            (label_dir / f"{group_id}_{split}_{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    yaml_path = root / f"{group_id}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(group_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {index: name for index, name in enumerate(names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


@pytest.fixture
def catalog(tmp_path: Path) -> dict[str, str]:
    return {group: str(make_yolo_group(tmp_path, group)) for group in ("stage0", "stage1", "stage2")}


def task_config(tmp_path: Path, catalog: Mapping[str, str], *, current: list[str], candidate: list[str], label: str = "test", selection: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "task": {"name": "unit", "label": label, "seed": 42, "output_root": str(tmp_path / "runs"), "parent_result": None},
        "data": {"catalog": dict(catalog), "current": current, "candidate": candidate, "validation": candidate, "test": candidate},
        "model": {"backend": "yolov5", "definition": "yolov5s.yaml"},
        "initialization": {"source": "pretrained", "checkpoint": "yolov5s.pt"},
        "select_policy": selection or {"name": "full", "params": {}},
        "epoch_policy": {"name": "static", "params": {}},
        "budget": {"type": "epochs", "value": 1},
        "backend": {"params": {"batch": 2, "imgsz": 32, "device": "cpu", "workers": 0, "amp": True}},
        "evaluation": {"primary_metric": "map50_95", "test_scope": "seen", "evaluate_checkpoints": ["last", "best"]},
    }


class FakeBackend(DetectionBackend):
    capabilities = frozenset({"static_training", "evaluation"})

    def __init__(self, fail_train: bool = False) -> None:
        self.fail_train = fail_train
        self.train_requests: list[Mapping[str, Any]] = []

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def train(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.train_requests.append(request)
        if self.fail_train:
            raise RuntimeError("simulated training failure")
        root = Path(request["standard_dir"])
        checkpoints = root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        last, best = checkpoints / "last.pt", checkpoints / "best.pt"
        last.write_bytes(b"last")
        best.write_bytes(b"best")
        sample_count = len(request["selected_ids"])
        return {"last_checkpoint": str(last), "best_checkpoint": str(best), "history": [{"epoch": 0, "map50_95": 0.5}], "training_seconds": 1.0, "images_read": sample_count, "optimizer_steps": 1}

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        first = request["sample_ids"][0]
        group = first.split("::", 1)[0]
        score = 0.5 + (int(group.removeprefix("stage")) * 0.01 if group.startswith("stage") else 0.0)
        return {"map50_95": score, "map50": score + 0.1, "precision": 0.7, "recall": 0.6, "per_class_ap": {"object": score}, "sample_count": len(request["sample_ids"]), "evaluation_seconds": 0.1}
