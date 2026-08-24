from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import yaml

from yolo_retraining.data import build_registry, find_manifest_overlaps, read_image_manifest, write_image_manifest
from yolo_retraining.data.loaders import IMAGE_SUFFIXES, load_group


VALID_SPLITS = {"train", "val", "test"}


def _relative_path(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()


def _layout_payload(layout_path: Path) -> dict:
    payload = yaml.safe_load(layout_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), dict):
        raise ValueError(f"custom dataset source must be a groups-based layout: {layout_path}")
    return payload


def collect_layout_images(layout: Path | str, groups: Iterable[str] | None = None) -> tuple[list[Path], list[str], Path]:
    layout_path = Path(layout).expanduser().resolve()
    payload = _layout_payload(layout_path)
    group_ids = list(payload["groups"]) if groups is None else [str(group) for group in groups]
    images: list[Path] = []
    names: list[str] | None = None
    dataset_root: Path | None = None
    for group_id in group_ids:
        loaded = load_group(group_id, layout_path)
        if names is None:
            names = list(loaded["names"])
            dataset_root = Path(loaded["root"])
        for records in loaded["splits"].values():
            images.extend(Path(record["image_path"]) for record in records)
    unique = sorted(set(images), key=lambda item: item.as_posix().casefold())
    if not unique or names is None or dataset_root is None:
        raise ValueError(f"selected layout groups contain no images: {group_ids}")
    return unique, names, dataset_root


def collect_layout_groups(layout: Path | str, groups: Iterable[str] | None = None) -> tuple[dict[str, tuple[str, list[Path]]], list[str], Path]:
    layout_path = Path(layout).expanduser().resolve()
    payload = _layout_payload(layout_path)
    group_ids = list(payload["groups"]) if groups is None else [str(group) for group in groups]
    result: dict[str, tuple[str, list[Path]]] = {}
    names: list[str] | None = None
    dataset_root: Path | None = None
    for group_id in group_ids:
        loaded = load_group(group_id, layout_path)
        if names is None:
            names = list(loaded["names"])
            dataset_root = Path(loaded["root"])
        populated = [(split, records) for split, records in loaded["splits"].items() if records]
        if len(populated) != 1:
            raise ValueError(f"layout group must populate exactly one split: {group_id}")
        split, records = populated[0]
        result[group_id] = (split, [Path(record["image_path"]) for record in records])
    if not result or names is None or dataset_root is None:
        raise ValueError("selected layout groups contain no images")
    return result, names, dataset_root


def allocate_split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    if total < len(fractions):
        raise ValueError("the source dataset is too small for the requested non-empty splits")
    if set(fractions) != VALID_SPLITS or any(value <= 0 for value in fractions.values()):
        raise ValueError("train, val, and test fractions must all be positive")
    fraction_sum = sum(fractions.values())
    if abs(fraction_sum - 1.0) > 1e-8:
        raise ValueError(f"split fractions must sum to 1.0, got {fraction_sum}")
    raw = {name: total * value for name, value in fractions.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"split fractions produced an empty split: {counts}")
    return counts


def random_split(images: Iterable[Path], *, seed: int, fractions: dict[str, float]) -> dict[str, list[Path]]:
    shuffled = sorted({Path(image).resolve() for image in images}, key=lambda item: item.as_posix().casefold())
    random.Random(seed).shuffle(shuffled)
    counts = allocate_split_counts(len(shuffled), fractions)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}


def collect_entries(dataset_root: Path | str, entries: Iterable[str]) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    images: list[Path] = []
    for raw_entry in entries:
        entry = Path(raw_entry).expanduser()
        entry = (root / entry).resolve() if not entry.is_absolute() else entry.resolve()
        if entry.is_dir():
            images.extend(path.resolve() for path in entry.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        elif entry.is_file() and entry.suffix.lower() == ".txt":
            images.extend(read_image_manifest(entry, dataset_root=root))
        elif entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
            images.append(entry)
        else:
            raise FileNotFoundError(f"image entry does not exist or has no supported images: {entry}")
    normalized = [str(image).casefold() for image in images]
    if len(normalized) != len(set(normalized)):
        raise ValueError("image entries overlap or contain duplicate images")
    if not images:
        raise ValueError("image entries contain no images")
    return sorted(images, key=lambda item: item.as_posix().casefold())


def write_layout(
    *,
    output: Path | str,
    dataset_root: Path | str,
    names: list[str],
    groups: dict[str, tuple[str, Path]],
) -> Path:
    output_path = Path(output).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    manifest_images = {group_id: read_image_manifest(manifest, dataset_root=root) for group_id, (_, manifest) in groups.items()}
    overlaps = find_manifest_overlaps(manifest_images)
    if overlaps:
        image, owners = next(iter(overlaps.items()))
        raise ValueError(f"image appears in multiple custom groups: {image} ({owners})")
    payload = {
        "path": _relative_path(root, output_path.parent),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": split,
                "manifest": _relative_path(manifest.resolve(), root),
            }
            for group_id, (split, manifest) in groups.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path


def summarize_layout(layout: Path | str) -> dict:
    layout_path = Path(layout).expanduser().resolve()
    payload = _layout_payload(layout_path)
    registry = build_registry({group_id: str(layout_path) for group_id in payload["groups"]})
    class_counts = [0] * len(registry["names"])
    missing_labels = 0
    backgrounds = 0
    group_counts: dict[str, int] = {}
    for group_id, split_ids in registry["groups"].items():
        ids = [sample_id for values in split_ids.values() for sample_id in values]
        group_counts[group_id] = len(ids)
        for sample_id in ids:
            label = Path(registry["records"][sample_id]["label_path"])
            if not label.is_file():
                missing_labels += 1
                backgrounds += 1
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                backgrounds += 1
            for line in lines:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"invalid YOLO label row in {label}: {line!r}")
                class_id = int(fields[0])
                if class_id < 0 or class_id >= len(class_counts):
                    raise ValueError(f"class id {class_id} is outside the names schema in {label}")
                class_counts[class_id] += 1
    return {
        "layout": str(layout_path),
        "images": len(registry["records"]),
        "groups": group_counts,
        "background_images": backgrounds,
        "missing_label_files": missing_labels,
        "class_instances": dict(zip(registry["names"], class_counts)),
    }


def _parse_manifest_spec(value: str) -> tuple[str, str, Path]:
    fields = value.split(":", 2)
    if len(fields) != 3 or fields[1] not in VALID_SPLITS:
        raise argparse.ArgumentTypeError("manifest spec must be GROUP:SPLIT:PATH")
    return fields[0], fields[1], Path(fields[2]).expanduser().resolve()


def _parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("names must contain at least one comma-separated class name")
    return names


def _command_random_split(args: argparse.Namespace) -> dict:
    images, names, dataset_root = collect_layout_images(args.layout, args.groups)
    fractions = {"train": args.train, "val": args.val, "test": args.test}
    splits = random_split(images, seed=args.seed, fractions=fractions)
    output_dir = args.output_dir.resolve()
    manifests = {name: write_image_manifest(values, output_dir / "manifests" / f"{name}.txt") for name, values in splits.items()}
    groups = {name: (name, manifest) for name, manifest in manifests.items()}
    layout = write_layout(output=output_dir / "layout.yaml", dataset_root=dataset_root, names=names, groups=groups)
    summary = summarize_layout(layout)
    summary.update(seed=args.seed, fractions=fractions, source_layout=str(args.layout.resolve()), deliberate_frame_leakage=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _command_random_reassign(args: argparse.Namespace) -> dict:
    source_groups, names, dataset_root = collect_layout_groups(args.layout, args.groups)
    images = sorted(
        {image.resolve() for _, group_images in source_groups.values() for image in group_images},
        key=lambda item: item.as_posix().casefold(),
    )
    if len(images) != sum(len(group_images) for _, group_images in source_groups.values()):
        raise ValueError("source layout contains physical overlap between groups")
    random.Random(args.seed).shuffle(images)
    output_dir = args.output_dir.resolve()
    cursor = 0
    manifests: dict[str, Path] = {}
    group_specs: dict[str, tuple[str, Path]] = {}
    for group_id, (split, source_images) in source_groups.items():
        assigned = images[cursor : cursor + len(source_images)]
        cursor += len(source_images)
        manifest = write_image_manifest(assigned, output_dir / "manifests" / f"{group_id}.txt")
        manifests[group_id] = manifest
        group_specs[group_id] = (split, manifest)
    layout = write_layout(output=output_dir / "layout.yaml", dataset_root=dataset_root, names=names, groups=group_specs)
    summary = summarize_layout(layout)
    summary.update(
        seed=args.seed,
        source_layout=str(args.layout.resolve()),
        preserved_group_sizes={group: len(images) for group, (_, images) in source_groups.items()},
        deliberate_frame_leakage=True,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _command_build_layout(args: argparse.Namespace) -> dict:
    groups: dict[str, tuple[str, Path]] = {}
    for group_id, split, manifest in args.manifest:
        if group_id in groups:
            raise ValueError(f"duplicate group id: {group_id}")
        groups[group_id] = (split, manifest)
    layout = write_layout(output=args.output, dataset_root=args.dataset_root, names=args.names, groups=groups)
    return summarize_layout(layout)


def _command_write_manifest(args: argparse.Namespace) -> dict:
    images = collect_entries(args.dataset_root, args.images)
    manifest = write_image_manifest(images, args.output)
    return {"manifest": str(manifest), "images": len(images), "dataset_root": str(args.dataset_root.resolve())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate TXT-manifest dataset layouts without copying images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser("random-split", help="Create a deliberate frame-random split from a groups layout.")
    random_parser.add_argument("--layout", type=Path, required=True)
    random_parser.add_argument("--groups", nargs="*", help="Source groups; defaults to every group in the layout.")
    random_parser.add_argument("--output-dir", type=Path, required=True)
    random_parser.add_argument("--seed", type=int, default=42)
    random_parser.add_argument("--train", type=float, default=0.8)
    random_parser.add_argument("--val", type=float, default=0.1)
    random_parser.add_argument("--test", type=float, default=0.1)
    random_parser.set_defaults(handler=_command_random_split)

    reassign_parser = subparsers.add_parser(
        "random-reassign",
        help="Randomly reassign every image while preserving the original group ids, splits, and exact sizes.",
    )
    reassign_parser.add_argument("--layout", type=Path, required=True)
    reassign_parser.add_argument("--groups", nargs="*", help="Groups to reassign; defaults to every group in the layout.")
    reassign_parser.add_argument("--output-dir", type=Path, required=True)
    reassign_parser.add_argument("--seed", type=int, default=42)
    reassign_parser.set_defaults(handler=_command_random_reassign)

    layout_parser = subparsers.add_parser("build-layout", help="Build a layout from arbitrary existing manifests.")
    layout_parser.add_argument("--dataset-root", type=Path, required=True)
    layout_parser.add_argument("--names", type=_parse_names, required=True)
    layout_parser.add_argument("--manifest", type=_parse_manifest_spec, action="append", required=True, metavar="GROUP:SPLIT:PATH")
    layout_parser.add_argument("--output", type=Path, required=True)
    layout_parser.set_defaults(handler=_command_build_layout)

    manifest_parser = subparsers.add_parser("write-manifest", help="Create a manifest from image directories, images, or other manifests.")
    manifest_parser.add_argument("--dataset-root", type=Path, required=True)
    manifest_parser.add_argument("--images", nargs="+", required=True, help="Entries relative to dataset root unless absolute.")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.set_defaults(handler=_command_write_manifest)

    validate_parser = subparsers.add_parser("validate", help="Validate paths, overlap, labels, and class counts.")
    validate_parser.add_argument("--layout", type=Path, required=True)
    validate_parser.set_defaults(handler=lambda args: summarize_layout(args.layout))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
