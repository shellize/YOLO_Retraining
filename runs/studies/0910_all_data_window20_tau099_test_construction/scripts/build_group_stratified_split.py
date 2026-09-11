from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from build_stratified_split import (
    CLASS_NAMES,
    FRACTIONS,
    SPLITS,
    Sample,
    allocate,
    load_samples,
    sha256,
    split_summary,
    write_manifest,
)


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = STUDY_ROOT / "experiment" / "variants" / "global_order_window20_tau0p990_label_aware_pair_reviewed_final" / "manifest.txt"
DEFAULT_GROUPS = STUDY_ROOT / "experiment" / "variants" / "target_track_groups_two_pass_final" / "image_split_groups.csv"
DEFAULT_OUTPUT = STUDY_ROOT / "experiment" / "variants" / "group_stratified_s42_8_1_1"


@dataclass(frozen=True)
class Unit:
    unit_id: str
    group_id: str
    samples: tuple[Sample, ...]
    features: Counter[str]

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def grouped(self) -> bool:
        return bool(self.group_id)


WEIGHTS = {
    "images": 20.0,
    "positive": 6.0,
    "background": 4.0,
    "class_image": 10.0,
    "class_instance": 4.0,
    "time_bin": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a group-aware stratified train/val/test split.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--search-trials", type=int, default=128)
    parser.add_argument("--large-group-threshold", type=int, default=10)
    parser.add_argument("--large-group-penalty", type=float, default=0.6)
    return parser.parse_args()


def sample_features(sample: Sample) -> Counter[str]:
    values: Counter[str] = Counter()
    values["images"] = 1
    values["positive" if sample.positive else "background"] = 1
    for class_id in sample.classes:
        values[f"class_image:{class_id}"] += 1
    for class_id, count in enumerate(sample.instances):
        values[f"class_instance:{class_id}"] += count
    values[f"time_bin:{sample.time_bin}"] += 1
    return values


def read_group_members(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image, group = row["image"], row["split_group_id"]
            if image in seen and seen[image] != group:
                raise ValueError(f"image occurs in multiple groups: {image}")
            seen[image] = group
            groups[group].append(image)
    return dict(groups)


def build_units(samples: list[Sample], group_members: dict[str, list[str]]) -> list[Unit]:
    by_image = {sample.relative_image: sample for sample in samples}
    grouped_images: set[str] = set()
    units: list[Unit] = []
    for group_id, images in sorted(group_members.items()):
        missing = sorted(set(images) - by_image.keys())
        if missing:
            raise ValueError(f"group contains images outside final manifest: {missing[:3]}")
        members = tuple(sorted((by_image[image] for image in images), key=lambda item: (item.frame, item.relative_image)))
        if any(sample.relative_image in grouped_images for sample in members):
            raise ValueError(f"group overlap detected at {group_id}")
        grouped_images.update(sample.relative_image for sample in members)
        features: Counter[str] = Counter()
        for sample in members:
            features.update(sample_features(sample))
        units.append(Unit(group_id, group_id, members, features))
    for sample in samples:
        if sample.relative_image not in grouped_images:
            units.append(Unit(f"image:{sample.relative_image}", "", (sample,), sample_features(sample)))
    return units


def feature_weight(feature: str) -> float:
    if feature.startswith("class_image:"):
        return WEIGHTS["class_image"]
    if feature.startswith("class_instance:"):
        return WEIGHTS["class_instance"]
    if feature.startswith("time_bin:"):
        return WEIGHTS["time_bin"]
    return WEIGHTS[feature]


def large_group_cost(unit: Unit, split: str, *, threshold: int, penalty: float) -> float:
    if split == "train" or not unit.grouped or unit.size < threshold:
        return 0.0
    return penalty * (unit.size / threshold) ** 2


def assignment_gain(
    unit: Unit,
    split: str,
    current: dict[str, Counter[str]],
    targets: dict[str, dict[str, float]],
    capacities: dict[str, int],
    *,
    threshold: int,
    penalty: float,
) -> float:
    if current[split]["images"] + unit.size > capacities[split]:
        return -math.inf
    score = 0.0
    for feature, value in unit.features.items():
        target = targets[split][feature]
        deficit = max(0.0, target - current[split][feature])
        scale = max(1.0, target)
        covered = min(value, deficit) / scale
        overflow = max(0.0, value - deficit) / scale
        score += feature_weight(feature) * (covered - 0.35 * overflow)
    capacity_need = (capacities[split] - current[split]["images"]) / capacities[split]
    return score + 0.05 * capacity_need - large_group_cost(unit, split, threshold=threshold, penalty=penalty)


def final_cost(
    current: dict[str, Counter[str]],
    targets: dict[str, dict[str, float]],
    assigned: dict[str, list[Unit]],
    *,
    threshold: int,
    penalty: float,
) -> float:
    value = 0.0
    for split in SPLITS:
        for feature, target in targets[split].items():
            scale = max(1.0, target)
            value += feature_weight(feature) * ((current[split][feature] - target) / scale) ** 2
        value += sum(large_group_cost(unit, split, threshold=threshold, penalty=penalty) for unit in assigned[split])
    return value


def split_units(
    units: list[Unit],
    *,
    seed: int,
    trials: int,
    large_group_threshold: int,
    large_group_penalty: float,
) -> tuple[dict[str, list[Unit]], float]:
    if trials < 1:
        raise ValueError("search-trials must be positive")
    totals: Counter[str] = Counter()
    for unit in units:
        totals.update(unit.features)
    capacities = allocate(totals["images"])
    targets = {split: {feature: total * FRACTIONS[split] for feature, total in totals.items()} for split in SPLITS}
    grouped = [unit for unit in units if unit.grouped]
    independent = [unit for unit in units if not unit.grouped]
    rarity = {
        unit.unit_id: sum(value / max(1, totals[feature]) for feature, value in unit.features.items() if feature.startswith("class_"))
        for unit in units
    }

    best: dict[str, list[Unit]] | None = None
    best_cost = math.inf
    for trial in range(trials):
        rng = random.Random(seed + 104729 * trial)
        group_order = sorted(grouped, key=lambda unit: (-unit.size, -rarity[unit.unit_id], rng.random()))
        independent_order = sorted(independent, key=lambda unit: (-rarity[unit.unit_id], rng.random()))
        current = {split: Counter() for split in SPLITS}
        assigned = {split: [] for split in SPLITS}
        for unit in (*group_order, *independent_order):
            choices = []
            for split in SPLITS:
                gain = assignment_gain(
                    unit,
                    split,
                    current,
                    targets,
                    capacities,
                    threshold=large_group_threshold,
                    penalty=large_group_penalty,
                )
                choices.append((gain, rng.random(), split))
            gain, _, chosen = max(choices)
            if not math.isfinite(gain):
                raise RuntimeError(f"no split has room for unit {unit.unit_id}")
            assigned[chosen].append(unit)
            current[chosen].update(unit.features)
        if any(current[split]["images"] != capacities[split] for split in SPLITS):
            raise RuntimeError("exact image capacities were not reached")
        cost = final_cost(
            current,
            targets,
            assigned,
            threshold=large_group_threshold,
            penalty=large_group_penalty,
        )
        if cost < best_cost:
            best, best_cost = assigned, cost
    if best is None:
        raise RuntimeError("split search produced no assignment")
    return best, best_cost


def group_summary(units: list[Unit], *, threshold: int) -> dict[str, int]:
    groups = [unit for unit in units if unit.grouped]
    return {
        "allocation_units": len(units),
        "group_count": len(groups),
        "grouped_images": sum(unit.size for unit in groups),
        "independent_images": sum(unit.size for unit in units if not unit.grouped),
        "large_group_threshold": threshold,
        "large_group_count": sum(unit.size >= threshold for unit in groups),
        "large_group_images": sum(unit.size for unit in groups if unit.size >= threshold),
        "largest_group": max((unit.size for unit in groups), default=1),
    }


def main() -> int:
    args = parse_args()
    source_manifest = args.source_manifest.resolve()
    groups_path = args.groups.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite group-aware split: {output}")
    samples, dataset_root = load_samples(source_manifest, time_bins=args.time_bins)
    units = build_units(samples, read_group_members(groups_path))
    assigned, objective = split_units(
        units,
        seed=args.seed,
        trials=args.search_trials,
        large_group_threshold=args.large_group_threshold,
        large_group_penalty=args.large_group_penalty,
    )

    split_samples: dict[str, list[Sample]] = {}
    sample_units: dict[str, Unit] = {}
    for split in SPLITS:
        split_samples[split] = sorted(
            (sample for unit in assigned[split] for sample in unit.samples),
            key=lambda sample: (sample.frame, sample.relative_image.casefold()),
        )
        for unit in assigned[split]:
            for sample in unit.samples:
                sample_units[sample.relative_image] = unit

    manifests: dict[str, Path] = {}
    for split in SPLITS:
        manifest = output / "manifests" / f"{split}.txt"
        write_manifest(split_samples[split], manifest)
        manifests[split] = manifest
    layout = {
        "path": Path(os.path.relpath(dataset_root, output)).as_posix(),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "groups": {
            split: {"split": split, "manifest": Path(os.path.relpath(manifests[split], dataset_root)).as_posix()}
            for split in SPLITS
        },
    }
    (output / "layout.yaml").write_text(yaml.safe_dump(layout, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")

    with (output / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("split", "allocation_unit", "split_group_id", "unit_size", "global_frame", "time_bin", "batch", "classes", "image"))
        for split in SPLITS:
            for sample in split_samples[split]:
                unit = sample_units[sample.relative_image]
                writer.writerow((split, unit.unit_id, unit.group_id, unit.size, sample.frame, sample.time_bin, sample.batch, ";".join(map(str, sample.classes)), sample.relative_image))

    split_group_stats = {split: group_summary(assigned[split], threshold=args.large_group_threshold) for split in SPLITS}
    summary = {
        "status": "completed",
        "method": "group-aware deterministic multi-objective stratification with large-group penalty on validation and test",
        "seed": args.seed,
        "search_trials": args.search_trials,
        "objective": objective,
        "fractions": FRACTIONS,
        "time_bins": args.time_bins,
        "source_manifest": os.path.relpath(source_manifest, output).replace(os.sep, "/"),
        "source_manifest_sha256": sha256(source_manifest),
        "group_constraints": os.path.relpath(groups_path, output).replace(os.sep, "/"),
        "group_constraints_sha256": sha256(groups_path),
        "total_images": len(samples),
        "allocation": split_group_stats,
        "splits": {split: split_summary(split_samples[split], time_bins=args.time_bins) for split in SPLITS},
    }
    for split, manifest in manifests.items():
        summary["splits"][split]["manifest"] = f"manifests/{split}.txt"
        summary["splits"][split]["manifest_sha256"] = sha256(manifest)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    protocol = {
        "purpose": "Fixed group-aware train/val/test split of the final reviewed data pool",
        "source_variant": source_manifest.parent.name,
        "group_variant": groups_path.parents[0].name,
        "split_variant": output.name,
        "unit": "reviewed split group or independent image",
        "hard_constraints": ["group integrity", "complete assignment", "no split overlap", "exact image-count allocation"],
        "soft_objectives": ["class-positive image balance", "class-instance balance", "positive/background balance", "global-time-bin balance", "large-group penalty on validation and test"],
        "seed": args.seed,
        "fractions": FRACTIONS,
        "time_bins": args.time_bins,
        "search_trials": args.search_trials,
        "large_group_threshold": args.large_group_threshold,
        "large_group_penalty": args.large_group_penalty,
        "selection_boundary": "Validation is used for model selection; test remains fixed and is not used to revise the split or protocol.",
        "source_images_and_labels_mutated": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
