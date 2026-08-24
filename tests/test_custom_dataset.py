from pathlib import Path

from PIL import Image

from data_analyse.custom_dataset.custom_dataset import allocate_split_counts, collect_entries, random_split, summarize_layout, write_layout
from yolo_retraining.data import write_image_manifest


def _images(root: Path, count: int) -> list[Path]:
    result = []
    for index in range(count):
        image = root / "images" / "batch" / f"{index:03d}.jpg"
        label = root / "labels" / "batch" / f"{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(image)
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        result.append(image)
    return result


def test_random_split_is_disjoint_deterministic_and_complete(tmp_path: Path) -> None:
    images = _images(tmp_path / "dataset", 10)
    fractions = {"train": 0.6, "val": 0.2, "test": 0.2}
    assert allocate_split_counts(10, fractions) == {"train": 6, "val": 2, "test": 2}
    first = random_split(images, seed=42, fractions=fractions)
    second = random_split(images, seed=42, fractions=fractions)
    assert first == second
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(images)


def test_custom_layout_round_trip_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images = _images(root, 3)
    train = write_image_manifest(images[:2], tmp_path / "manifests" / "train.txt")
    test = write_image_manifest(images[2:], tmp_path / "manifests" / "test.txt")
    layout = write_layout(
        output=tmp_path / "layout.yaml",
        dataset_root=root,
        names=["object"],
        groups={"stage0": ("train", train), "test": ("test", test)},
    )
    summary = summarize_layout(layout)
    assert summary["images"] == 3
    assert summary["groups"] == {"stage0": 2, "test": 1}
    assert summary["class_instances"] == {"object": 3}


def test_collect_entries_scans_directories_without_copying(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images = _images(root, 2)
    assert collect_entries(root, ["images/batch"]) == images
