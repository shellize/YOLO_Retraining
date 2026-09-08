"""Build random or source-ordered 8:1:1 layouts from the retained manifest."""

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


SEEDS = (41, 42, 43)
SOURCE_MANIFEST = PROJECT_ROOT / "runs/studies/0907_stroller_window10_learning_curve/result/post_dedup_window10_tau_0p990/manifest.txt"
DATASET_ROOT = PROJECT_ROOT / "data/stroller"
VARIANT_ROOT = STUDY_ROOT / "experiment/variants"
TRAIN_FRACTION = 0.8
TEST_FRACTION = 0.1
VAL_FRACTION = 0.1
STAGE_COUNT = 8


def allocate_counts(total: int) -> dict[str, int]:
    train = round(total * TRAIN_FRACTION)
    test = round(total * TEST_FRACTION)
    val = total - train - test
    if min(train, test, val) <= 0 or train + test + val != total:
        raise ValueError(f"invalid 8:1:1 allocation for {total}: {(train, test, val)}")
    return {"train": train, "test": test, "val": val}


def stage_sizes(total: int) -> list[int]:
    base, remainder = divmod(total, STAGE_COUNT)
    return [base + int(index < remainder) for index in range(STAGE_COUNT)]


def write_layout(
    variant_dir: Path,
    groups: dict[str, tuple[str, Path]],
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
    names = [line.strip() for line in (DATASET_ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = {
        "path": Path(os.path.relpath(DATASET_ROOT, variant_dir)).as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": split,
                "manifest": Path(os.path.relpath(manifest, DATASET_ROOT)).as_posix(),
            }
            for group_id, (split, _) in groups.items()
            for manifest in [manifests[group_id]]
        },
    }
    layout = variant_dir / "layout.yaml"
    layout.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return layout


def build(seed: int | None, split_mode: str) -> dict:
    if split_mode == "ordered":
        variant_dir = VARIANT_ROOT / "window10_tau0p990_ordered"
    else:
        if seed is None:
            raise ValueError("random split mode requires a split seed")
        variant_dir = VARIANT_ROOT / f"window10_tau0p990_split_s{seed}"
    summary_path = variant_dir / "manifest_summary.json"
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if summary_path.is_file() and (variant_dir / "layout.yaml").is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"refusing to overwrite incomplete variant: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=True)

    images = read_image_manifest(SOURCE_MANIFEST, dataset_root=DATASET_ROOT)
    counts = allocate_counts(len(images))
    if split_mode == "random":
        split_order = sorted(images, key=lambda image: image.as_posix().casefold())
        random.Random(seed).shuffle(split_order)
    else:
        split_order = images
    train_end = counts["train"]
    test_end = train_end + counts["test"]
    train_images = split_order[:train_end]
    test_images = split_order[train_end:test_end]
    val_images = split_order[test_end:]
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

    layout = write_layout(variant_dir, groups, preserve_order=split_mode == "ordered")
    summary = {
        "study": STUDY_ROOT.name,
        "variant": variant_dir.name,
        "source_manifest": str(SOURCE_MANIFEST),
        "dataset_root": str(DATASET_ROOT),
        "split_mode": split_mode,
        "seed": seed,
        "fractions": {"train": TRAIN_FRACTION, "test": TEST_FRACTION, "val": VAL_FRACTION},
        "split_counts": counts,
        "stage_sizes": {f"stage{index}": size for index, size in enumerate(sizes)},
        "image_count": len(images),
        "layout": str(layout),
        "ordering_source": "retained manifest order (canonical path order from the deduplication output)",
        "protocol": (
            "window=10, tau=0.99 retained set -> source-order train/test/val contiguous slices "
            "-> source-order train stages"
            if split_mode == "ordered"
            else "window=10, tau=0.99 retained set -> random train/test/val split -> random train stage order"
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (variant_dir / "protocol.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--split-mode", choices=("random", "ordered"), default="random")
    args = parser.parse_args()
    if args.split_mode == "ordered":
        summaries = [build(None, args.split_mode)]
    else:
        summaries = [build(seed, args.split_mode) for seed in args.seeds]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
