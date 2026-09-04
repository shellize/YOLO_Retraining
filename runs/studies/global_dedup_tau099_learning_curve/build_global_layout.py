#!/usr/bin/env python3
"""Build the post-deduplication layout for the global tau=0.99 study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_analyse.dataset_redundancy.redundancy_analysis import (  # noqa: E402
    ImageRecord,
    inspect_annotation,
    load_embeddings,
    load_layout_records,
    representative_groups,
    temporal_similarity_edges,
)
from yolo_retraining.data import build_registry, write_image_manifest  # noqa: E402

THRESHOLD = 0.99
TEMPORAL_WINDOW = 1
DEFAULT_LAYOUT = PROJECT_ROOT / "configs/data/self_improving.yaml"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data_analyse/dataset_redundancy/results/20260824-182608/embeddings.npz"
DEFAULT_OUTPUT = STUDY_ROOT / "variants/dedup_tau_0p990"


def path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def inventory_fingerprint(records: Sequence[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        stat = record.path.stat()
        digest.update(f"{record.path.as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def remap_embeddings(records, cached_records, cached_embeddings):
    cached_by_path = {}
    for index, record in enumerate(cached_records):
        key = path_key(record.path)
        if key in cached_by_path:
            raise ValueError(f"duplicate cached image: {record.path}")
        cached_by_path[key] = index
    requested = {path_key(record.path) for record in records}
    archived = set(cached_by_path)
    if requested != archived:
        raise ValueError(
            f"embedding inventory mismatch: requested={len(requested)} cached={len(archived)} "
            f"missing={sorted(requested - archived)[:1]} extra={sorted(archived - requested)[:1]}"
        )
    import numpy as np
    return np.stack([cached_embeddings[cached_by_path[path_key(record.path)]] for record in records])


def split_stage_sizes(total: int, val_count: int, test_count: int, stage_count: int) -> list[int]:
    train_total = total - val_count - test_count
    if train_total < stage_count:
        raise ValueError(f"representative pool too small: total={total}, val={val_count}, test={test_count}")
    base, remainder = divmod(train_total, stage_count)
    return [base + int(index < remainder) for index in range(stage_count)]


def write_layout(output_dir: Path, dataset_root: Path, names: Sequence[str], groups: dict[str, tuple[str, list[Path]]]) -> Path:
    manifests = {}
    for group_id, (_, images) in groups.items():
        if not images:
            raise ValueError(f"empty generated group: {group_id}")
        manifests[group_id] = write_image_manifest(images, output_dir / "manifests" / f"{group_id}.txt")
    payload = {
        "path": Path(os.path.relpath(dataset_root, output_dir)).as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": split,
                "manifest": Path(os.path.relpath(manifests[group_id], dataset_root)).as_posix(),
            }
            for group_id, (split, _) in groups.items()
        },
    }
    layout_path = output_dir / "layout.yaml"
    import yaml
    layout_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    build_registry({group_id: str(layout_path) for group_id in groups})
    return layout_path


def annotation_summary(paths: list[Path]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for image in paths:
        info = inspect_annotation(image)
        if not info.file_exists:
            result["missing_label_file"] += 1
        elif not info.has_annotation:
            result["background_images"] += 1
        else:
            result["labeled_images"] += 1
            for line in info.label_path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    result[f"class_{line.split()[0]}_boxes"] += 1
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--stages", type=int, default=8)
    args = parser.parse_args()

    layout_path = args.layout.expanduser().resolve()
    embedding_path = args.embeddings.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if (output_dir / "layout.yaml").is_file() and (output_dir / "manifest_summary.json").is_file():
            print(f"reuse completed layout: {output_dir}")
            return 0
        raise FileExistsError(f"refusing to overwrite incomplete output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records, names, dataset_root, source_groups = load_layout_records(layout_path)
    cached_records, cached_embeddings, embedding_metadata = load_embeddings(embedding_path)
    embeddings = remap_embeddings(records, cached_records, cached_embeddings)
    if len(records) != 9573:
        raise ValueError(f"expected 9573 source images, got {len(records)}")
    fingerprint = inventory_fingerprint(records)
    cached_fingerprint = str(embedding_metadata.get("inventory_sha256", ""))
    if cached_fingerprint and fingerprint != cached_fingerprint:
        raise ValueError(f"inventory fingerprint mismatch: current={fingerprint} cached={cached_fingerprint}")

    edges = temporal_similarity_edges(records, embeddings, window=TEMPORAL_WINDOW, min_similarity=THRESHOLD)
    clusters = representative_groups(range(len(records)), edges, THRESHOLD)
    representatives = [records[representative].path for representative, _ in clusters]
    if len(representatives) != len({path_key(path) for path in representatives}):
        raise ValueError("duplicate representative image")

    shuffled = sorted(representatives, key=lambda path: path.as_posix().casefold())
    random.Random(args.seed).shuffle(shuffled)
    stage_sizes = split_stage_sizes(len(shuffled), args.val_count, args.test_count, args.stages)
    train_total = sum(stage_sizes)
    train_images = shuffled[:train_total]
    val_images = shuffled[train_total:train_total + args.val_count]
    test_images = shuffled[train_total + args.val_count:]
    if len(test_images) != args.test_count:
        raise ValueError("test count mismatch")

    groups: dict[str, tuple[str, list[Path]]] = {
        "test": ("test", test_images),
        "val": ("val", val_images),
    }
    cursor = 0
    for stage_index, stage_size in enumerate(stage_sizes):
        groups[f"stage{stage_index}"] = ("train", train_images[cursor:cursor + stage_size])
        cursor += stage_size

    layout_out = write_layout(output_dir, dataset_root, names, groups)
    audit_path = output_dir / "clusters.csv"
    fields = [
        "cluster_id", "cluster_size", "source_group", "source_split", "image",
        "has_annotation", "annotation_status", "representative", "selected",
    ]
    cluster_stats: Counter[str] = Counter()
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cluster_number, (representative, members) in enumerate(clusters):
            rep_path = records[representative].path
            infos = [inspect_annotation(records[index].path) for index in members]
            flags = [info.has_annotation for info in infos]
            kind = "pure_background" if not any(flags) else "pure_labeled" if all(flags) else "mixed"
            cluster_stats["clusters"] += 1
            cluster_stats["members"] += len(members)
            cluster_stats["non_singleton_clusters"] += int(len(members) > 1)
            cluster_stats[f"{kind}_clusters"] += 1
            for member, info in zip(members, infos):
                record = records[member]
                writer.writerow({
                    "cluster_id": f"cluster_{cluster_number:06d}",
                    "cluster_size": len(members),
                    "source_group": record.group,
                    "source_split": record.split,
                    "image": record.path.relative_to(dataset_root).as_posix(),
                    "has_annotation": int(info.has_annotation),
                    "annotation_status": info.status,
                    "representative": rep_path.relative_to(dataset_root).as_posix(),
                    "selected": int(member == representative),
                })

    split_summary = {
        group_id: {
            "split": split,
            "images": len(images),
            "annotation_summary": annotation_summary(images),
        }
        for group_id, (split, images) in groups.items()
    }
    summary = {
        "study": STUDY_ROOT.name,
        "protocol": {
            "threshold": THRESHOLD,
            "temporal_window": TEMPORAL_WINDOW,
            "policy": "keep one representative per cluster",
            "split_order": "global dedup -> fixed random test/val -> 8 random train stages",
            "seed": args.seed,
            "val_count": args.val_count,
            "test_count": args.test_count,
            "stage_count": args.stages,
        },
        "source_layout": str(layout_path),
        "source_groups": source_groups,
        "dataset_root": str(dataset_root),
        "source_images": len(records),
        "source_inventory_sha256": fingerprint,
        "source_embeddings": str(embedding_path),
        "embedding_metadata": embedding_metadata,
        "temporal_edges_at_threshold": len(edges),
        "clusters": len(clusters),
        "representative_images": len(representatives),
        "train_images_after_dedup": train_total,
        "validation_images_after_dedup": len(val_images),
        "test_images_after_dedup": len(test_images),
        "stage_sizes": {f"stage{index}": size for index, size in enumerate(stage_sizes)},
        "cluster_stats": dict(cluster_stats),
        "split_summary": split_summary,
        "layout": str(layout_out),
        "cluster_audit": str(audit_path),
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "protocol.json").write_text(json.dumps({
        "method": "legacy_adjacent_temporal_dedup",
        "threshold": THRESHOLD,
        "temporal_window": TEMPORAL_WINDOW,
        "embedding_cache": str(embedding_path),
        "embedding_inventory_sha256": cached_fingerprint,
        "current_inventory_sha256": fingerprint,
        "source_layout": str(layout_path),
        "output_layout": str(layout_out),
        "seed": args.seed,
        "val_count": args.val_count,
        "test_count": args.test_count,
        "stages": args.stages,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
