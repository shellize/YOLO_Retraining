from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
CLASS_NAMES = ("large luggage", "stroller", "wheelchair", "flatbed truck")


@dataclass(frozen=True)
class Sample:
    image: Path
    relative_image: str
    batch: str
    frame: int
    classes: tuple[int, ...]
    instances: tuple[int, ...]
    time_bin: int

    @property
    def positive(self) -> bool:
        return bool(self.classes)

    @property
    def features(self) -> tuple[str, ...]:
        state = "positive" if self.positive else "background"
        return (state, *(f"class:{class_id}" for class_id in self.classes), f"time_bin:{self.time_bin}")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    study = script.parent.parent
    parser = argparse.ArgumentParser(description="Build the fixed image-random, multilabel-stratified 8:1:1 split.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=study
        / "experiment"
        / "variants"
        / "global_order_window20_tau0p990_label_aware_reviewed_final"
        / "manifest.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=study / "experiment" / "variants" / "random_stratified_s42_8_1_1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-bins", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate(total: int) -> dict[str, int]:
    raw = {split: total * FRACTIONS[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - counts[split]), split))
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def read_manifest(path: Path) -> list[Path]:
    manifest = path.resolve()
    images: list[Path] = []
    seen: set[Path] = set()
    for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        image = (manifest.parent / value).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"manifest image does not exist: {image}")
        if image in seen:
            raise ValueError(f"duplicate image in source manifest: {image}")
        seen.add(image)
        images.append(image)
    if not images:
        raise ValueError("source manifest contains no images")
    return images


def read_label(label: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts: Counter[int] = Counter()
    if label.is_file():
        for line_number, raw in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid YOLO row at {label}:{line_number}: {line!r}")
            class_id = int(fields[0])
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(f"class id outside schema at {label}:{line_number}: {class_id}")
            coordinates = [float(value) for value in fields[1:]]
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(f"coordinate outside [0, 1] at {label}:{line_number}")
            counts[class_id] += 1
    classes = tuple(sorted(counts))
    instances = tuple(counts.get(class_id, 0) for class_id in range(len(CLASS_NAMES)))
    return classes, instances


def load_samples(source_manifest: Path, *, time_bins: int) -> tuple[list[Sample], Path]:
    if time_bins <= 0:
        raise ValueError("time-bins must be positive")
    repo = Path(__file__).resolve().parents[4]
    dataset_root = (repo / "data" / "self_improving").resolve()
    base: list[tuple[Path, str, str, int, tuple[int, ...], tuple[int, ...]]] = []
    for image in read_manifest(source_manifest):
        relative = image.relative_to(dataset_root)
        if len(relative.parts) < 3 or relative.parts[0] != "images":
            raise ValueError(f"image is outside the expected images/<batch> layout: {image}")
        try:
            frame = int(image.stem.split("_", 1)[0])
        except ValueError as error:
            raise ValueError(f"image filename has no numeric global frame prefix: {image.name}") from error
        label = dataset_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
        classes, instances = read_label(label)
        base.append((image, relative.as_posix(), relative.parts[1], frame, classes, instances))
    base.sort(key=lambda item: (item[3], item[1].casefold()))
    if len({item[3] for item in base}) != len(base):
        raise ValueError("global frame numbers are not unique")
    samples = [
        Sample(
            image=image,
            relative_image=relative,
            batch=batch,
            frame=frame,
            classes=classes,
            instances=instances,
            time_bin=min(time_bins - 1, index * time_bins // len(base)),
        )
        for index, (image, relative, batch, frame, classes, instances) in enumerate(base)
    ]
    return samples, dataset_root


def stratified_split(samples: list[Sample], *, seed: int) -> dict[str, list[Sample]]:
    capacities = allocate(len(samples))
    feature_totals = Counter(feature for sample in samples for feature in sample.features)
    targets = {feature: allocate(total) for feature, total in feature_totals.items()}
    assigned_feature_counts = {split: Counter() for split in SPLITS}
    result = {split: [] for split in SPLITS}
    rng = random.Random(seed)
    random_tie = {sample.relative_image: rng.random() for sample in samples}
    ordered = sorted(
        samples,
        key=lambda sample: (
            min(feature_totals[feature] for feature in sample.features),
            sum(feature_totals[feature] for feature in sample.features),
            random_tie[sample.relative_image],
        ),
    )
    for sample in ordered:
        candidates = [split for split in SPLITS if len(result[split]) < capacities[split]]
        scored: list[tuple[float, float, float, str]] = []
        for split in candidates:
            feature_need = 0.0
            for feature in sample.features:
                target = targets[feature][split]
                deficit = target - assigned_feature_counts[split][feature]
                feature_need += deficit / max(1, target) / math.sqrt(feature_totals[feature])
            capacity_need = (capacities[split] - len(result[split])) / capacities[split]
            scored.append((feature_need, capacity_need, rng.random(), split))
        split = max(scored)[-1]
        result[split].append(sample)
        assigned_feature_counts[split].update(sample.features)
    for split in SPLITS:
        result[split].sort(key=lambda sample: (sample.frame, sample.relative_image.casefold()))
        if len(result[split]) != capacities[split]:
            raise RuntimeError(f"split size mismatch for {split}")
    return result


def write_manifest(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [Path(os.path.relpath(sample.image, path.parent)).as_posix() for sample in samples]
    path.write_text("\n".join(values) + "\n", encoding="utf-8", newline="\n")


def split_summary(samples: list[Sample], *, time_bins: int) -> dict:
    class_images = Counter(class_id for sample in samples for class_id in sample.classes)
    class_instances = [sum(sample.instances[class_id] for sample in samples) for class_id in range(len(CLASS_NAMES))]
    return {
        "images": len(samples),
        "positive_images": sum(sample.positive for sample in samples),
        "background_images": sum(not sample.positive for sample in samples),
        "class_positive_images": {CLASS_NAMES[index]: class_images[index] for index in range(len(CLASS_NAMES))},
        "class_instances": {CLASS_NAMES[index]: class_instances[index] for index in range(len(CLASS_NAMES))},
        "time_bins": {str(index): sum(sample.time_bin == index for sample in samples) for index in range(time_bins)},
        "frame_min": min(sample.frame for sample in samples),
        "frame_max": max(sample.frame for sample in samples),
        "per_batch": dict(sorted(Counter(sample.batch for sample in samples).items())),
    }


def main() -> int:
    args = parse_args()
    source_manifest = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    samples, dataset_root = load_samples(source_manifest, time_bins=args.time_bins)
    splits = stratified_split(samples, seed=args.seed)

    manifests = {}
    for split in SPLITS:
        manifest = output_dir / "manifests" / f"{split}.txt"
        write_manifest(splits[split], manifest)
        manifests[split] = manifest

    layout = {
        "path": Path(os.path.relpath(dataset_root, output_dir)).as_posix(),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "groups": {
            split: {
                "split": split,
                "manifest": Path(os.path.relpath(manifests[split], dataset_root)).as_posix(),
            }
            for split in SPLITS
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layout.yaml").write_text(
        yaml.safe_dump(layout, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )

    with (output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("split", "global_frame", "time_bin", "batch", "classes", "image"))
        for split in SPLITS:
            for sample in splits[split]:
                writer.writerow(
                    (split, sample.frame, sample.time_bin, sample.batch, ";".join(map(str, sample.classes)), sample.relative_image)
                )

    summary = {
        "status": "completed",
        "method": "image-level deterministic multilabel stratification with positive/background and global-time-bin coverage",
        "seed": args.seed,
        "fractions": FRACTIONS,
        "time_bins": args.time_bins,
        "source_manifest": os.path.relpath(source_manifest, output_dir).replace(os.sep, "/"),
        "source_manifest_sha256": sha256(source_manifest),
        "total_images": len(samples),
        "splits": {split: split_summary(splits[split], time_bins=args.time_bins) for split in SPLITS},
    }
    for split, manifest in manifests.items():
        summary["splits"][split]["manifest"] = f"manifests/{split}.txt"
        summary["splits"][split]["manifest_sha256"] = sha256(manifest)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    protocol = {
        "purpose": "Fixed train/val/test split of the reviewed deduplicated all-data pool",
        "source_variant": "global_order_window20_tau0p990_label_aware_reviewed_final",
        "split_variant": output_dir.name,
        "unit": "individual image",
        "grouping": None,
        "seed": args.seed,
        "fractions": FRACTIONS,
        "stratification_features": ["positive/background", "multilabel class presence", f"{args.time_bins} global-frame quantile bins"],
        "selection_boundary": "Validation is used for model selection; test remains fixed and is not used to revise the split or protocol.",
        "source_images_and_labels_mutated": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
