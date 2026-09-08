"""Build two random and one source-ordered 8:1:1 layouts for strollerdifficult."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_retraining.data import read_image_manifest, write_image_manifest  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/stroller_protocol_0908/stroller_raw_source"
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "data_analyse/stroller_protocol_0908/results/difficult/variants/dedup_tau_0p990/manifests/stroller_raw.txt"
)
VARIANT_ROOT = STUDY_ROOT / "experiment/variants"
SPLIT_SEEDS = (41, 42)
TRAIN_FRACTION = 0.8
TEST_FRACTION = 0.1
VAL_FRACTION = 0.1
STAGE_COUNT = 8


def allocate_counts(total: int) -> dict[str, int]:
    train = round(total * TRAIN_FRACTION)
    test = round(total * TEST_FRACTION)
    val = total - train - test
    counts = {"train": train, "test": test, "val": val}
    if min(counts.values()) <= 0 or sum(counts.values()) != total:
        raise ValueError(f"invalid 8:1:1 allocation for {total}: {counts}")
    return counts


def stage_sizes(total: int) -> list[int]:
    base, remainder = divmod(total, STAGE_COUNT)
    return [base + int(index < remainder) for index in range(STAGE_COUNT)]


def read_names(dataset_root: Path) -> list[str]:
    classes_path = dataset_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"classes.txt does not exist: {classes_path}")
    names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"classes.txt is empty: {classes_path}")
    return names


def write_layout(
    variant_dir: Path,
    dataset_root: Path,
    names: list[str],
    groups: dict[str, tuple[str, list[Path]]],
    *,
    preserve_order: bool,
) -> Path:
    manifests = {
        group_id: write_image_manifest(
            images,
            variant_dir / "manifests" / f"{group_id}.txt",
            preserve_order=preserve_order,
        )
        for group_id, (_, images) in groups.items()
    }
    payload = {
        "path": Path(os.path.relpath(dataset_root, variant_dir)).as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": split,
                "manifest": Path(os.path.relpath(manifests[group_id], dataset_root)).as_posix(),
            }
            for group_id, (split, _) in groups.items()
        },
    }
    layout = variant_dir / "layout.yaml"
    layout.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return layout


def build(
    *,
    source_images: list[Path],
    dataset_root: Path,
    names: list[str],
    seed: int | None,
    split_mode: str,
    source_manifest: Path,
) -> dict[str, object]:
    if split_mode == "ordered":
        variant_dir = VARIANT_ROOT / "window10_tau0p990_ordered"
    else:
        if seed is None:
            raise ValueError("random split requires a seed")
        variant_dir = VARIANT_ROOT / f"window10_tau0p990_split_s{seed}"
    summary_path = variant_dir / "manifest_summary.json"
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if summary_path.is_file() and (variant_dir / "layout.yaml").is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"refusing to overwrite incomplete variant: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=True)

    counts = allocate_counts(len(source_images))
    if split_mode == "random":
        split_order = sorted(source_images, key=lambda image: image.as_posix().casefold())
        random.Random(seed).shuffle(split_order)
    else:
        split_order = list(source_images)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    train_images = split_order[:train_end]
    val_images = split_order[train_end:val_end]
    test_images = split_order[val_end:]
    sizes = stage_sizes(len(train_images))

    groups: dict[str, tuple[str, list[Path]]] = {
        "test": ("test", test_images),
        "val": ("val", val_images),
    }
    cursor = 0
    for index, size in enumerate(sizes):
        groups[f"stage{index}"] = ("train", train_images[cursor : cursor + size])
        cursor += size
    if cursor != len(train_images):
        raise AssertionError("stage sizes do not cover the training split")

    layout = write_layout(
        variant_dir,
        dataset_root,
        names,
        groups,
        preserve_order=split_mode == "ordered",
    )
    summary = {
        "study": STUDY_ROOT.name,
        "variant": variant_dir.name,
        "source_manifest": str(source_manifest),
        "dataset_root": str(dataset_root),
        "split_mode": split_mode,
        "seed": seed,
        "fractions": {"train": TRAIN_FRACTION, "test": TEST_FRACTION, "val": VAL_FRACTION},
        "split_counts": counts,
        "stage_sizes": {f"stage{index}": size for index, size in enumerate(sizes)},
        "image_count": len(source_images),
        "layout": str(layout),
        "ordering_source": (
            "dedup representative manifest order"
            if split_mode == "ordered"
            else "seeded shuffle of canonical paths"
        ),
        "protocol": (
            "strollerraw -> tau=0.99/window=10 representatives -> source-order 8:1:1 split"
            if split_mode == "ordered"
            else "strollerraw -> tau=0.99/window=10 representatives -> seeded random 8:1:1 split"
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (variant_dir / "protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    parser.add_argument("--split-mode", choices=("random", "ordered", "all"), default="all")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    source_images = read_image_manifest(source_manifest, dataset_root=dataset_root)
    names = read_names(dataset_root)
    modes = ["random", "ordered"] if args.split_mode == "all" else [args.split_mode]
    summaries: list[dict[str, object]] = []
    for mode in modes:
        if mode == "ordered":
            summaries.append(
                build(
                    source_images=source_images,
                    dataset_root=dataset_root,
                    names=names,
                    seed=None,
                    split_mode=mode,
                    source_manifest=source_manifest,
                )
            )
        else:
            for seed in args.seeds:
                summaries.append(
                    build(
                        source_images=source_images,
                        dataset_root=dataset_root,
                        names=names,
                        seed=seed,
                        split_mode=mode,
                        source_manifest=source_manifest,
                    )
                )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
