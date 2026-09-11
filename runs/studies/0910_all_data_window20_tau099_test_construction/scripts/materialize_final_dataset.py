from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from build_stratified_split import CLASS_NAMES, Sample, load_samples, read_manifest, split_summary, write_manifest


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_VARIANT = STUDY_ROOT / "experiment" / "variants" / "group_stratified_s42_8_1_1"
DEFAULT_GROUPS = STUDY_ROOT / "experiment" / "variants" / "target_track_groups_two_pass_final" / "image_split_groups.csv"


@dataclass(frozen=True)
class StageUnit:
    unit_id: str
    group_id: str
    samples: tuple[Sample, ...]
    anchor_frame: int

    @property
    def size(self) -> int:
        return len(self.samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ordered train stages and materialize the final dataset as real files.")
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--stage-count", type=int, default=8)
    parser.add_argument("--time-bins", type=int, default=10)
    return parser.parse_args()


def read_group_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image, group = row["image"], row["split_group_id"]
            if image in result and result[image] != group:
                raise ValueError(f"image occurs in multiple groups: {image}")
            result[image] = group
    return result


def build_stage_units(samples: list[Sample], group_map: dict[str, str]) -> list[StageUnit]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    independent: list[Sample] = []
    for sample in samples:
        group_id = group_map.get(sample.relative_image, "")
        if group_id:
            grouped[group_id].append(sample)
        else:
            independent.append(sample)
    units: list[StageUnit] = []
    for group_id, members in grouped.items():
        ordered = tuple(sorted(members, key=lambda sample: (sample.frame, sample.relative_image)))
        anchor = ordered[len(ordered) // 2].frame
        units.append(StageUnit(group_id, group_id, ordered, anchor))
    units.extend(StageUnit(f"image:{sample.relative_image}", "", (sample,), sample.frame) for sample in independent)
    return sorted(units, key=lambda unit: (unit.anchor_frame, unit.samples[0].frame, unit.unit_id))


def split_ordered_units(units: list[StageUnit], stage_count: int) -> list[list[StageUnit]]:
    if stage_count < 1:
        raise ValueError("stage-count must be positive")
    if len(units) < stage_count:
        raise ValueError("fewer allocation units than stages")
    total = sum(unit.size for unit in units)
    cumulative = []
    running = 0
    for unit in units:
        running += unit.size
        cumulative.append(running)
    cuts = []
    previous = 0
    for number in range(1, stage_count):
        target = total * number / stage_count
        candidates = range(previous + 1, len(units) - (stage_count - number) + 1)
        cut = min(candidates, key=lambda index: (abs(cumulative[index - 1] - target), index))
        cuts.append(cut)
        previous = cut
    boundaries = [0, *cuts, len(units)]
    return [units[boundaries[index] : boundaries[index + 1]] for index in range(stage_count)]


def copy_real_file(source: Path, destination: Path) -> int:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"source must be a real file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)
    if not destination.is_file() or destination.is_symlink():
        raise RuntimeError(f"materialized destination is not a real file: {destination}")
    return destination.stat().st_size


def main() -> int:
    args = parse_args()
    variant = args.variant_dir.resolve()
    groups_path = args.groups.resolve()
    staging_root = args.staging_root.resolve()
    if staging_root.exists():
        raise FileExistsError(f"refusing to overwrite staging root: {staging_root}")
    source_dataset = (PROJECT_ROOT / "data" / "self_improving").resolve()

    split_paths = {name: variant / "manifests" / f"{name}.txt" for name in ("train", "val", "test")}
    split_images = {name: read_manifest(path) for name, path in split_paths.items()}
    all_paths = [path for name in ("train", "val", "test") for path in split_images[name]]
    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("train/val/test manifests overlap")
    samples, _ = load_samples(split_paths["train"], time_bins=args.time_bins)
    by_path = {sample.image: sample for sample in samples}
    train_samples = [by_path[path] for path in split_images["train"]]
    group_map = read_group_map(groups_path)
    stages = split_ordered_units(build_stage_units(train_samples, group_map), args.stage_count)

    stage_by_image: dict[str, str] = {}
    stage_samples: dict[str, list[Sample]] = {}
    for index, units in enumerate(stages):
        stage = f"stage{index}"
        members = sorted((sample for unit in units for sample in unit.samples), key=lambda sample: (sample.frame, sample.relative_image))
        stage_samples[stage] = members
        for sample in members:
            if sample.relative_image in stage_by_image:
                raise RuntimeError(f"train image occurs in multiple stages: {sample.relative_image}")
            stage_by_image[sample.relative_image] = stage
    if set(stage_by_image) != {sample.relative_image for sample in train_samples}:
        raise RuntimeError("ordered stages do not exactly cover train")
    group_stages: dict[str, set[str]] = defaultdict(set)
    for sample in train_samples:
        group_id = group_map.get(sample.relative_image, "")
        if group_id:
            group_stages[group_id].add(stage_by_image[sample.relative_image])
    violations = sorted(group for group, destinations in group_stages.items() if len(destinations) != 1)
    if violations:
        raise RuntimeError(f"train groups cross stage boundaries: {violations[:3]}")

    for stage, members in stage_samples.items():
        write_manifest(members, variant / "manifests" / f"{stage}.txt")
    layout = {
        "path": Path(os.path.relpath(source_dataset, variant)).as_posix(),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "groups": {
            "test": {"split": "test", "manifest": Path(os.path.relpath(split_paths["test"], source_dataset)).as_posix()},
            "val": {"split": "val", "manifest": Path(os.path.relpath(split_paths["val"], source_dataset)).as_posix()},
            **{
                stage: {
                    "split": "train",
                    "manifest": Path(os.path.relpath(variant / "manifests" / f"{stage}.txt", source_dataset)).as_posix(),
                }
                for stage in stage_samples
            },
        },
    }
    (variant / "layout.yaml").write_text(yaml.safe_dump(layout, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    with (variant / "stage_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("stage", "split_group_id", "global_frame", "time_bin", "classes", "image"))
        for stage, members in stage_samples.items():
            for sample in members:
                writer.writerow((stage, group_map.get(sample.relative_image, ""), sample.frame, sample.time_bin, ";".join(map(str, sample.classes)), sample.relative_image))

    destination_dataset = staging_root / "self_improving"
    image_bytes = 0
    label_bytes = 0
    for image in all_paths:
        relative = image.relative_to(source_dataset)
        image_bytes += copy_real_file(image, destination_dataset / relative)
        label = source_dataset / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
        label_relative = Path("labels") / Path(*relative.parts[1:]).with_suffix(".txt")
        label_bytes += copy_real_file(label, destination_dataset / label_relative)
    data_yaml = {
        "path": ".",
        "train": Path(os.path.relpath(split_paths["train"], source_dataset)).as_posix(),
        "val": Path(os.path.relpath(split_paths["val"], source_dataset)).as_posix(),
        "test": Path(os.path.relpath(split_paths["test"], source_dataset)).as_posix(),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (destination_dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")

    physical_images = list((destination_dataset / "images").rglob("*"))
    physical_labels = list((destination_dataset / "labels").rglob("*"))
    image_files = [path for path in physical_images if path.is_file()]
    label_files = [path for path in physical_labels if path.is_file()]
    symlinks = [path for path in staging_root.rglob("*") if path.is_symlink()]
    if len(image_files) != len(all_paths) or len(label_files) != len(all_paths) or symlinks:
        raise RuntimeError("materialized dataset validation failed")

    stage_report = {}
    for stage, members in stage_samples.items():
        frames = [sample.frame for sample in members]
        stats = split_summary(members, time_bins=args.time_bins)
        stats.update(
            {
                "group_count": len({group_map[sample.relative_image] for sample in members if sample.relative_image in group_map}),
                "frame_min": min(frames),
                "frame_max": max(frames),
                "manifest": f"manifests/{stage}.txt",
            }
        )
        stage_report[stage] = stats
    summary = {
        "status": "completed",
        "stage_policy": "train allocation units ordered by median global frame; contiguous cuts nearest equal image counts; split groups remain intact",
        "stage_count": args.stage_count,
        "train_images": len(train_samples),
        "stage_group_integrity_violations": 0,
        "stages": stage_report,
        "materialization": {
            "staging_root": str(staging_root),
            "images": len(image_files),
            "labels": len(label_files),
            "symlinks": 0,
            "image_bytes": image_bytes,
            "label_bytes": label_bytes,
        },
    }
    (variant / "materialization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
