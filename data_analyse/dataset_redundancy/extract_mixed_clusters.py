from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .redundancy_analysis import (
        _canonical_image_key,
        inspect_annotation,
        write_mixed_cluster_outputs,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from redundancy_analysis import (  # type: ignore[no-redef]
        _canonical_image_key,
        inspect_annotation,
        write_mixed_cluster_outputs,
    )


def _dataset_root_from_layout(layout_path: Path) -> Path:
    payload = yaml.safe_load(layout_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"layout must contain a YAML mapping: {layout_path}")
    declared_root = Path(payload.get("path", layout_path.parent)).expanduser()
    return (layout_path.parent / declared_root).resolve() if not declared_root.is_absolute() else declared_root.resolve()


def _read_cluster_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"cluster CSV contains no rows: {path}")
    required = {"group", "cluster_id", "image", "representative", "score_to_representative", "cluster_size"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"cluster CSV is missing columns {sorted(missing)}: {path}")
    return rows


def process_variant(variant_dir: Path, dataset_root: Path | None = None) -> dict[str, Any]:
    variant_dir = variant_dir.expanduser().resolve()
    cluster_path = variant_dir / "clusters.csv"
    if not cluster_path.is_file():
        raise FileNotFoundError(f"dedup variant has no clusters.csv: {variant_dir}")
    root = (dataset_root or _dataset_root_from_layout(variant_dir / "layout.yaml")).resolve()
    cluster_rows = _read_cluster_rows(cluster_path)
    annotations = {}
    for row in cluster_rows:
        image_value = row["image"]
        image_path = Path(image_value).expanduser()
        image_path = image_path if image_path.is_absolute() else root / image_path
        image_path = image_path.resolve()
        key = _canonical_image_key(image_value)
        info = inspect_annotation(image_path)
        previous = annotations.setdefault(key, info)
        if previous != info:
            raise ValueError(f"cluster CSV maps one image to inconsistent annotation paths: {image_value}")
    return {
        "variant": variant_dir.name,
        "dataset_root": str(root),
        **write_mixed_cluster_outputs(
            variant_dir,
            cluster_rows,
            annotations=annotations,
            dataset_root=root,
        ),
    }


def _variant_dirs(results_root: Path, names: list[str] | None) -> list[Path]:
    variants_root = results_root / "variants"
    if names:
        result = [variants_root / name for name in names]
    elif variants_root.is_dir():
        result = sorted(path for path in variants_root.iterdir() if path.is_dir() and path.name.startswith("dedup"))
    else:
        result = []
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract clusters containing both annotated and unannotated images from dedup variants."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="override the dataset root read from each variant layout")
    parser.add_argument("--variants", nargs="*", help="specific variant directory names; defaults to every dedup* directory")
    args = parser.parse_args(argv)

    results_root = args.results_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    variants = _variant_dirs(results_root, args.variants)
    if not variants:
        raise ValueError(f"no dedup* variants found below {results_root / 'variants'}")
    reports = [process_variant(variant, dataset_root) for variant in variants]
    aggregate = {
        "results_root": str(results_root),
        "variants": {report["variant"]: report for report in reports},
    }
    (results_root / "mixed_clusters_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
