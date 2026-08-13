from pathlib import Path

import pytest
import yaml
from PIL import Image

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


def test_layout_maps_physical_batches_to_logical_groups(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for batch in ("0720_1", "0720_2", "0720_3"):
        image = root / "images" / batch / f"{batch}.jpg"
        label = root / "labels" / batch / f"{batch}.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(image)
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    layout = tmp_path / "layout.yaml"
    layout.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "names": {0: "object"},
                "groups": {
                    "test": {"split": "test", "images": ["images/0720_1"]},
                    "val": {"split": "val", "images": ["images/0720_2"]},
                    "stage0": {"split": "train", "images": ["images/0720_3"]},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    registry = build_registry({group: str(layout) for group in ("stage0", "val", "test")})
    assert registry["groups"]["stage0"]["train"] == ["stage0::images/0720_3/0720_3.jpg"]
    assert registry["groups"]["val"]["val"] == ["val::images/0720_2/0720_2.jpg"]
    assert registry["groups"]["test"]["test"] == ["test::images/0720_1/0720_1.jpg"]
