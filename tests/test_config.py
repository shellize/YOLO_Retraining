from pathlib import Path

import pytest

from yolo_retraining.config import deep_merge, load_config


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    assert deep_merge({"a": {"x": 1, "items": [1]}}, {"a": {"y": 2, "items": [2]}}) == {"a": {"x": 1, "y": 2, "items": [2]}}


def test_checked_in_task_config_and_override() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "task" / "full_cold.yaml", ["task.seed=43"])
    assert config["backend"]["params"]["batch"] == 64
    assert config["task"]["seed"] == 43
    assert config["model"] == {"backend": "yolov5", "definition": "yolov5s.yaml"}
    assert config["initialization"] == {"source": "pretrained", "checkpoint": "yolov5s.pt"}
    assert Path(config["data"]["layout"]).is_absolute()
    assert config["data"]["validation"] == ["val"]
    assert config["data"]["test"] == ["test"]


def test_warm_config_requires_parent() -> None:
    root = Path(__file__).parents[1]
    with pytest.raises(ValueError, match="parent initialization"):
        load_config(root / "configs" / "task" / "full_warm.yaml")
