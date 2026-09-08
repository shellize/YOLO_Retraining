"""Build nested difficult/easy splits for the m1/m2 comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Mapping

import yaml

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_retraining.data import read_image_manifest, write_image_manifest  # noqa: E402


DEFAULT_DIFFICULT_ROOT = PROJECT_ROOT / "data/stroller_raw"
DEFAULT_EASY_ROOT = PROJECT_ROOT / "data/stroller"
DEFAULT_DIFFICULT_MANIFEST = (
    PROJECT_ROOT
    / "data_analyse/stroller_protocol_0908/results/difficult/variants/dedup_tau_0p990/manifests/stroller_raw.txt"
)
DEFAULT_EASY_MANIFEST = (
    PROJECT_ROOT
    / "data_analyse/stroller_protocol_0908/results/easy/variants/dedup_tau_0p990/manifests/stroller.txt"
)
VARIANT_ROOT = STUDY_ROOT / "experiment/variants"
SPLIT_SEEDS = (41, 42)
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def read_names(dataset_root: Path) -> list[str]:
    classes_path = dataset_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"classes.txt does not exist: {classes_path}")
    names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"classes.txt is empty: {classes_path}")
    return names


def allocate_counts(total: int) -> dict[str, int]:
    train = round(total * TRAIN_FRACTION)
    val = round(total * VAL_FRACTION)
    test = total - train - val
    counts = {"train": train, "val": val, "test": test}
    if min(counts.values()) <= 0 or sum(counts.values()) != total:
        raise ValueError(f"invalid 8:1:1 allocation for {total}: {counts}")
    return counts


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix().casefold()


def keyed_paths(manifest: Path, root: Path) -> dict[str, Path]:
    paths = read_image_manifest(manifest, dataset_root=root)
    result: dict[str, Path] = {}
    for path in paths:
        key = relative_key(path, root)
        if key in result:
            raise ValueError(f"duplicate relative image identity: {key}")
        result[key] = path
    return result


def write_layout(
    layout_path: Path,
    dataset_root: Path,
    names: list[str],
    groups: Mapping[str, tuple[str, Path]],
) -> Path:
    payload = {
        "path": Path(os.path.relpath(dataset_root, layout_path.parent)).as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": split,
                "manifest": Path(os.path.relpath(manifest, dataset_root)).as_posix(),
            }
            for group_id, (split, manifest) in groups.items()
        },
    }
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return layout_path


def write_manifests(variant_dir: Path, split_groups: Mapping[str, list[Path]]) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for group_id, images in split_groups.items():
        if not images:
            raise ValueError(f"nested protocol produced an empty group: {group_id}")
        manifests[group_id] = write_image_manifest(
            images,
            variant_dir / "manifests" / f"{group_id}.txt",
        )
    return manifests


def build_split(
    *,
    seed: int,
    difficult_root: Path,
    difficult_manifest: Path,
    easy_root: Path,
    easy_manifest: Path,
    names: list[str],
) -> dict[str, object]:
    variant_dir = VARIANT_ROOT / f"split_s{seed}"
    summary_path = variant_dir / "summary.json"
    required = (
        summary_path,
        variant_dir / "m1_layout.yaml",
        variant_dir / "m2_layout.yaml",
        variant_dir / "eval_layouts/test1.yaml",
        variant_dir / "eval_layouts/test2.yaml",
        variant_dir / "eval_layouts/testhard.yaml",
    )
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if all(path.is_file() for path in required):
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"refusing to overwrite incomplete variant: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=True)

    difficult_by_key = keyed_paths(difficult_manifest, difficult_root)
    easy_by_key = keyed_paths(easy_manifest, easy_root)
    difficult_keys = set(difficult_by_key)
    easy_keys = set(easy_by_key)
    nested_keys = difficult_keys & easy_keys

    difficult_order = sorted(difficult_by_key.values(), key=lambda path: path.as_posix().casefold())
    random.Random(seed).shuffle(difficult_order)
    counts = allocate_counts(len(difficult_order))
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split1 = {
        "train1": difficult_order[:train_end],
        "val1": difficult_order[train_end:val_end],
        "test1": difficult_order[val_end:],
    }
    split2 = {
        "train2": [path for path in split1["train1"] if relative_key(path, difficult_root) in nested_keys],
        "val2": [path for path in split1["val1"] if relative_key(path, difficult_root) in nested_keys],
        "test2": [path for path in split1["test1"] if relative_key(path, difficult_root) in nested_keys],
    }
    testhard = [
        path
        for path in split1["test1"]
        if relative_key(path, difficult_root) not in nested_keys
    ]
    split_groups = {**split1, **split2, "testhard": testhard}
    manifests = write_manifests(variant_dir, split_groups)

    m1_layout = write_layout(
        variant_dir / "m1_layout.yaml",
        difficult_root,
        names,
        {
            "train1": ("train", manifests["train1"]),
            "val1": ("val", manifests["val1"]),
            "test1": ("test", manifests["test1"]),
        },
    )
    m2_layout = write_layout(
        variant_dir / "m2_layout.yaml",
        difficult_root,
        names,
        {
            "train2": ("train", manifests["train2"]),
            "val2": ("val", manifests["val2"]),
            "test1": ("test", manifests["test1"]),
        },
    )
    eval_layouts = {}
    for group_id in ("test1", "test2", "testhard"):
        eval_layouts[group_id] = write_layout(
            variant_dir / "eval_layouts" / f"{group_id}.yaml",
            difficult_root,
            names,
            {group_id: ("test", manifests[group_id])},
        )

    split2_counts = {group: len(images) for group, images in split2.items()}
    summary: dict[str, object] = {
        "study": STUDY_ROOT.name,
        "variant": variant_dir.name,
        "seed": seed,
        "difficult_root": str(difficult_root),
        "difficult_manifest": str(difficult_manifest),
        "easy_root": str(easy_root),
        "easy_manifest": str(easy_manifest),
        "identity_basis": "case-insensitive relative path below each source dataset root",
        "dedup_sets": {
            "difficult_retained": len(difficult_keys),
            "easy_retained": len(easy_keys),
            "nested_intersection": len(nested_keys),
            "easy_not_in_difficult": len(easy_keys - difficult_keys),
            "difficult_not_in_easy": len(difficult_keys - easy_keys),
        },
        "fractions": {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": TEST_FRACTION},
        "split1_counts": {group: len(images) for group, images in split1.items()},
        "split2_counts": split2_counts,
        "testhard_count": len(testhard),
        "layouts": {
            "m1": str(m1_layout),
            "m2": str(m2_layout),
            "test1": str(eval_layouts["test1"]),
            "test2": str(eval_layouts["test2"]),
            "testhard": str(eval_layouts["testhard"]),
        },
        "manifests": {group: str(path) for group, path in manifests.items()},
        "test_definitions": {
            "test1": "all difficult images assigned to the test split",
            "test2": "test1 intersected with the independently deduplicated easy identity set",
            "testhard": "test1 minus test2",
        },
        "protocol": (
            "difficult representatives -> seeded 8:1:1 split1 -> "
            "easy identity projection split2; m1 on train1/val1, m2 on train2/val2"
        ),
        "m3": "not run and not generated",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (variant_dir / "protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficult-root", type=Path, default=DEFAULT_DIFFICULT_ROOT)
    parser.add_argument("--difficult-manifest", type=Path, default=DEFAULT_DIFFICULT_MANIFEST)
    parser.add_argument("--easy-root", type=Path, default=DEFAULT_EASY_ROOT)
    parser.add_argument("--easy-manifest", type=Path, default=DEFAULT_EASY_MANIFEST)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    args = parser.parse_args()

    difficult_root = args.difficult_root.expanduser().resolve()
    difficult_manifest = args.difficult_manifest.expanduser().resolve()
    easy_root = args.easy_root.expanduser().resolve()
    easy_manifest = args.easy_manifest.expanduser().resolve()
    names = read_names(difficult_root)

    summaries = [
        build_split(
            seed=seed,
            difficult_root=difficult_root,
            difficult_manifest=difficult_manifest,
            easy_root=easy_root,
            easy_manifest=easy_manifest,
            names=names,
        )
        for seed in args.seeds
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
