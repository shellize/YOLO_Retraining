from pathlib import Path

import pytest

from conftest import make_yolo_group
from yolo_retraining.data import build_registry


def test_registry_builds_stable_relative_ids(catalog: dict[str, str]) -> None:
    registry = build_registry({"stage0": catalog["stage0"]})
    ids = registry["groups"]["stage0"]["train"]
    assert ids == sorted(ids)
    assert ids[0].startswith("stage0::images/train/")
    assert str(Path(catalog["stage0"]).parent) not in ids[0]
    assert registry["names"] == ["object"]


def test_registry_rejects_label_schema_mismatch(tmp_path: Path) -> None:
    first = make_yolo_group(tmp_path, "first", names=["object"])
    second = make_yolo_group(tmp_path, "second", names=["different"])
    with pytest.raises(ValueError, match="label schema mismatch"):
        build_registry({"first": str(first), "second": str(second)})

