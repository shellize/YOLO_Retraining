from pathlib import Path

import pytest
import yaml
from PIL import Image

from yolo_retraining.data import build_registry, read_image_manifest, write_image_manifest


def _make_image(root: Path, name: str) -> Path:
    image = root / "images" / name
    label = root / "labels" / Path(name).with_suffix(".txt")
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16)).save(image)
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return image


def test_layout_accepts_explicit_manifest_without_copying_images(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    first = _make_image(root, "batch/first.jpg")
    second = _make_image(root, "batch/second.jpg")
    manifest = write_image_manifest([second, first], tmp_path / "manifests" / "train.txt")
    layout = tmp_path / "layout.yaml"
    layout.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "names": {0: "object"},
                "groups": {"stage0": {"split": "train", "manifest": str(manifest)}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    registry = build_registry({"stage0": str(layout)})
    assert registry["groups"]["stage0"]["train"] == [
        "stage0::images/batch/first.jpg",
        "stage0::images/batch/second.jpg",
    ]
    assert first.is_file() and second.is_file()


def test_manifest_can_preserve_input_order(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    first = _make_image(root, "batch/first.jpg")
    second = _make_image(root, "batch/second.jpg")
    manifest = write_image_manifest(
        [second, first, second],
        tmp_path / "manifests" / "ordered.txt",
        preserve_order=True,
    )

    assert read_image_manifest(manifest, dataset_root=root) == [second, first]


def test_manifest_rejects_duplicate_and_missing_images(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _make_image(root, "batch/image.jpg")
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text(f"{image}\n{image}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate image"):
        read_image_manifest(duplicate, dataset_root=root)

    missing = tmp_path / "missing.txt"
    missing.write_text("does-not-exist.jpg\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_image_manifest(missing, dataset_root=root)


def test_layout_rejects_images_and_manifest_together(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _make_image(root, "batch/image.jpg")
    manifest = write_image_manifest([image], tmp_path / "manifest.txt")
    layout = tmp_path / "layout.yaml"
    layout.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "names": ["object"],
                "groups": {
                    "stage0": {
                        "split": "train",
                        "images": "images/batch",
                        "manifest": str(manifest),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        build_registry({"stage0": str(layout)})


def test_layout_allows_explicit_nested_test_subsets(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    first = _make_image(root, "test/first.jpg")
    second = _make_image(root, "test/second.jpg")
    manifests = tmp_path / "manifests"
    full = write_image_manifest([first, second], manifests / "test.txt")
    filtered = write_image_manifest([first], manifests / "test_filtered.txt")
    difficult = write_image_manifest([second], manifests / "test_difficult.txt")
    layout = tmp_path / "layout.yaml"
    layout.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "names": {0: "object"},
                "groups": {
                    "test": {"split": "test", "manifest": str(full)},
                    "test_filtered": {
                        "split": "test",
                        "subset_of": "test",
                        "manifest": str(filtered),
                    },
                    "test_difficult": {
                        "split": "test",
                        "subset_of": "test",
                        "manifest": str(difficult),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = build_registry(
        {group: str(layout) for group in ("test", "test_filtered", "test_difficult")}
    )

    assert registry["subset_of"] == {"test_difficult": "test", "test_filtered": "test"}
    assert len(registry["groups"]["test"]["test"]) == 2
    assert len(registry["groups"]["test_filtered"]["test"]) == 1
    assert len(registry["groups"]["test_difficult"]["test"]) == 1
