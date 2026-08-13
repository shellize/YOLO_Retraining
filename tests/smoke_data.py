from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SmokeSample:
    image: Path
    label: Path
    classes: frozenset[int]


def _names(root: Path) -> list[str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"real smoke dataset is missing data.yaml: {data_yaml}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    value = payload.get("names")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        pairs = sorted((int(key), str(name)) for key, name in value.items())
        if [key for key, _ in pairs] != list(range(len(pairs))):
            raise ValueError("smoke dataset class ids must be contiguous from zero")
        return [name for _, name in pairs]
    raise ValueError(f"smoke dataset data.yaml has invalid names: {data_yaml}")


def _parse_label(path: Path, class_count: int) -> frozenset[int]:
    if not path.is_file():
        raise FileNotFoundError(f"image is missing its YOLO label: {path}")
    classes: set[int] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"invalid detection label at {path}:{line_number}: expected 5 fields")
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"non-numeric detection label at {path}:{line_number}") from error
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"invalid class id at {path}:{line_number}: {fields[0]}")
        x, y, width, height = coordinates
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"invalid normalized box at {path}:{line_number}: {coordinates}")
        classes.add(class_id)
    return frozenset(classes)


def scan_real_samples(root: Path) -> tuple[list[str], list[SmokeSample]]:
    root = root.resolve()
    names = _names(root)
    images_root = root / "images"
    labels_root = root / "labels"
    images = sorted(
        (path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix(),
    )
    if not images:
        raise ValueError(f"real smoke dataset contains no images: {images_root}")
    samples: list[SmokeSample] = []
    for image in images:
        relative = image.relative_to(images_root)
        label = (labels_root / relative).with_suffix(".txt")
        samples.append(SmokeSample(image.resolve(), label.resolve(), _parse_label(label, len(names))))
    return names, samples


def _take_coverage(
    pool: list[SmokeSample],
    *,
    class_count: int,
    minimum_per_class: int,
) -> list[SmokeSample]:
    selected: list[SmokeSample] = []
    counts = [0] * class_count
    while any(count < minimum_per_class for count in counts):
        deficits = {class_id for class_id, count in enumerate(counts) if count < minimum_per_class}
        best_index = -1
        best_score = 0
        for index, sample in enumerate(pool):
            score = len(sample.classes & deficits)
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0:
            missing = [class_id for class_id, count in enumerate(counts) if count < minimum_per_class]
            raise ValueError(f"cannot satisfy smoke class coverage for class ids {missing}")
        sample = pool.pop(best_index)
        selected.append(sample)
        for class_id in sample.classes:
            counts[class_id] += 1
    return selected


def _fill_partition(pool: list[SmokeSample], selected: list[SmokeSample], size: int) -> list[SmokeSample]:
    if len(selected) > size:
        raise ValueError(f"class coverage requires {len(selected)} samples but partition size is {size}")
    selected.extend(pool.pop() for _ in range(size - len(selected)))
    return selected


def _write_manifest(path: Path, samples: Iterable[SmokeSample]) -> Path:
    path.write_text("".join(f"{sample.image}\n" for sample in samples), encoding="utf-8")
    return path


def prepare_real_smoke_dataset(
    source_root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    train_size: int = 64,
    val_size: int = 16,
    test_size: int = 16,
) -> dict[str, object]:
    names, samples = scan_real_samples(source_root)
    required = train_size + val_size + test_size
    if len(samples) < required:
        raise ValueError(f"smoke dataset requires {required} distinct samples, found {len(samples)}")
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)
    # Reserve coverage for every split before random fill so train cannot consume
    # the only examples needed to make validation or test representative.
    train = _take_coverage(pool, class_count=len(names), minimum_per_class=4)
    val = _take_coverage(pool, class_count=len(names), minimum_per_class=2)
    test = _take_coverage(pool, class_count=len(names), minimum_per_class=2)
    train = _fill_partition(pool, train, train_size)
    val = _fill_partition(pool, val, val_size)
    test = _fill_partition(pool, test, test_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "train": _write_manifest(output_dir / "train.txt", train),
        "val": _write_manifest(output_dir / "val.txt", val),
        "test": _write_manifest(output_dir / "test.txt", test),
    }
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(source_root.resolve()),
                "train": str(manifests["train"].resolve()),
                "val": str(manifests["val"].resolve()),
                "test": str(manifests["test"].resolve()),
                "names": {index: name for index, name in enumerate(names)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {"yaml": data_yaml, "names": names, "train": train, "val": val, "test": test}
