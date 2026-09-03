from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_analyse.dataset_redundancy.redundancy_analysis import (  # noqa: E402
    ImageRecord,
    SimilarityEdge,
    load_embeddings,
    load_layout_records,
    temporal_similarity_edges,
)
from yolo_retraining.data import build_registry, write_image_manifest  # noqa: E402
from yolo_retraining.data.loaders import yolo_label_path  # noqa: E402
from yolo_retraining.policies.selection.study import _label_count  # noqa: E402


DEFAULT_LAYOUT = PROJECT_ROOT / "configs" / "data" / "self_improving_balance_test.yaml"
DEFAULT_EMBEDDINGS = (
    PROJECT_ROOT / "data_analyse" / "dataset_redundancy" / "results" / "20260824-182608" / "embeddings.npz"
)
DEFAULT_BASELINE = (
    PROJECT_ROOT / "runs" / "tasks" /
    "Balance_Test__full-cold-stage0123__stage0+stage1+stage2+stage3__yolov5s__s42"
)


def threshold_label(threshold: float) -> str:
    return f"{threshold:.3f}".replace(".", "p")


def variant_names_for(threshold: float) -> tuple[str, str, str, str]:
    prefix = f"dedup_tau_{threshold_label(threshold)}"
    return (
        prefix,
        f"{prefix}_release_positive_clusters",
        f"{prefix}_release_positive_clusters_drop_pure_background",
        f"{prefix}_positive_representative",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balance label-contrast manifests for one threshold.")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=STUDY_ROOT)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--temporal-window", type=int, default=1)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args(argv)


def remap_embeddings(
    records: Sequence[ImageRecord],
    archived_records: Sequence[ImageRecord],
    archived_embeddings: np.ndarray,
) -> np.ndarray:
    by_path: dict[str, int] = {}
    for index, record in enumerate(archived_records):
        key = str(record.path.resolve()).casefold()
        if key in by_path:
            raise ValueError(f"embedding archive contains duplicate image: {record.path}")
        by_path[key] = index
    requested = {str(record.path.resolve()).casefold() for record in records}
    archived = set(by_path)
    if requested != archived:
        raise ValueError(
            "embedding archive inventory does not match layout: "
            f"missing={sorted(requested - archived)[:1]} extra={sorted(archived - requested)[:1]}"
        )
    return np.stack([archived_embeddings[by_path[str(record.path.resolve()).casefold()]] for record in records])


def greedy_groups(indices: Sequence[int], edges: Sequence[SimilarityEdge], threshold: float) -> list[tuple[int, list[int]]]:
    """The project's greedy grouping rule with sparse score summation."""
    allowed = set(indices)
    adjacency: dict[int, dict[int, float]] = {index: {} for index in indices}
    for edge in edges:
        if edge.score < threshold or edge.left not in allowed or edge.right not in allowed:
            continue
        adjacency[edge.left][edge.right] = edge.score
        adjacency[edge.right][edge.left] = edge.score
    unassigned = set(indices)
    result: list[tuple[int, list[int]]] = []
    while unassigned:
        representative = min(
            unassigned,
            key=lambda index: (
                -sum(neighbor in unassigned for neighbor in adjacency[index]),
                -sum(score for neighbor, score in adjacency[index].items() if neighbor in unassigned),
                index,
            ),
        )
        members = sorted({representative} | (set(adjacency[representative]) & unassigned))
        result.append((representative, members))
        unassigned.difference_update(members)
    return result


def write_layout(
    output_dir: Path,
    dataset_root: Path,
    names: Sequence[str],
    group_records: dict[str, list[ImageRecord]],
    selected_train: dict[str, list[Path]],
) -> Path:
    manifests: dict[str, Path] = {}
    for group_id, records in group_records.items():
        images = selected_train.get(group_id, [record.path for record in records])
        manifests[group_id] = write_image_manifest(images, output_dir / "manifests" / f"{group_id}.txt")
    layout_path = output_dir / "layout.yaml"
    payload = {
        "path": Path(os.path.relpath(dataset_root, layout_path.parent)).as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": records[0].split,
                "manifest": Path(os.path.relpath(manifests[group_id], dataset_root)).as_posix(),
            }
            for group_id, records in group_records.items()
        },
    }
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    build_registry({group_id: str(layout_path) for group_id in group_records})
    return layout_path


def ensure_baseline_reference(study_root: Path, baseline: Path) -> dict[str, str]:
    result_path = baseline / "task_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"completed full-data baseline is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"full-data baseline is not completed: {baseline}")
    tasks_dir = study_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    link = tasks_dir / "full_raw__existing_result"
    if link.is_symlink():
        if link.resolve() != baseline.resolve():
            raise ValueError(f"baseline link points elsewhere: {link} -> {link.resolve()}")
    elif link.exists():
        raise FileExistsError(f"baseline reference path is not a symlink: {link}")
    else:
        link.symlink_to(baseline.resolve(), target_is_directory=True)
    return {"result_dir": str(baseline.resolve()), "study_link": str(link.absolute())}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be strictly between zero and one")
    if args.temporal_window != 1:
        raise ValueError("this study fixes --temporal-window=1")
    layout = args.layout.expanduser().resolve()
    embeddings_path = args.embeddings.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    baseline = args.baseline_result.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, names, dataset_root, group_ids = load_layout_records(layout)
    archived_records, archived_embeddings, embedding_metadata = load_embeddings(embeddings_path)
    embeddings = remap_embeddings(records, archived_records, archived_embeddings)
    counts = [
        _label_count({"label_path": str(yolo_label_path(record.path))})
        for record in records
    ]
    edges = temporal_similarity_edges(records, embeddings, window=args.temporal_window, min_similarity=args.threshold)
    edge_scores = {(min(edge.left, edge.right), max(edge.left, edge.right)): edge.score for edge in edges}

    group_records: dict[str, list[ImageRecord]] = defaultdict(list)
    train_by_group: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        group_records[record.group].append(record)
        if record.split == "train":
            train_by_group[record.group].append(index)

    ordinary, release_positive, positive_only, positive_representative = variant_names_for(args.threshold)
    variant_names = (ordinary, release_positive, positive_only, positive_representative)
    selected: dict[str, dict[str, list[int]]] = {name: defaultdict(list) for name in variant_names}
    audit_rows: list[dict[str, object]] = []
    cluster_stats = {
        "clusters": 0,
        "non_singleton_clusters": 0,
        "positive_clusters": 0,
        "pure_background_clusters": 0,
        "mixed_clusters": 0,
        "background_representative_with_positive_member": 0,
    }
    cluster_number = 0
    for group_id in group_ids:
        for representative, members in greedy_groups(train_by_group.get(group_id, []), edges, args.threshold):
            cluster_id = f"{group_id}_{cluster_number:06d}"
            cluster_number += 1
            has_positive = any(counts[index] > 0 for index in members)
            has_background = any(counts[index] == 0 for index in members)
            positive_members = [index for index in members if counts[index] > 0]
            if positive_members and counts[representative] == 0:
                label_aware = min(
                    positive_members,
                    key=lambda index: (-edge_scores[(min(representative, index), max(representative, index))], index),
                )
            else:
                label_aware = representative

            selected[ordinary][group_id].append(representative)
            selected[release_positive][group_id].extend(members if has_positive else [representative])
            if has_positive:
                selected[positive_only][group_id].extend(members)
            selected[positive_representative][group_id].append(label_aware)

            cluster_stats["clusters"] += 1
            cluster_stats["non_singleton_clusters"] += len(members) > 1
            cluster_stats["positive_clusters"] += has_positive
            cluster_stats["pure_background_clusters"] += not has_positive
            cluster_stats["mixed_clusters"] += has_positive and has_background
            cluster_stats["background_representative_with_positive_member"] += has_positive and counts[representative] == 0
            for member in members:
                pair = (min(representative, member), max(representative, member))
                audit_rows.append(
                    {
                        "group": group_id,
                        "cluster_id": cluster_id,
                        "image": records[member].path.relative_to(dataset_root).as_posix(),
                        "label_count": counts[member],
                        "cluster_has_positive": int(has_positive),
                        "cluster_size": len(members),
                        "representative": records[representative].path.relative_to(dataset_root).as_posix(),
                        "positive_representative": records[label_aware].path.relative_to(dataset_root).as_posix(),
                        "score_to_representative": 1.0 if member == representative else edge_scores[pair],
                    }
                )

    variants_root = output_dir / "variants"
    variant_summaries: dict[str, dict[str, object]] = {}
    for variant_name in variant_names:
        variant_dir = variants_root / variant_name
        selected_paths = {
            group_id: [records[index].path for index in indices]
            for group_id, indices in selected[variant_name].items()
        }
        layout_path = write_layout(variant_dir, dataset_root, names, group_records, selected_paths)
        train_indices = [index for indices in selected[variant_name].values() for index in indices]
        train_manifest = write_image_manifest(
            [records[index].path for index in train_indices], variant_dir / "manifests" / "train.txt"
        )
        positive_images = sum(counts[index] > 0 for index in train_indices)
        variant_summaries[variant_name] = {
            "images": len(train_indices),
            "positive_images": positive_images,
            "background_images": len(train_indices) - positive_images,
            "layout": str(layout_path),
            "train_manifest": str(train_manifest),
            "per_group_images": {group_id: len(indices) for group_id, indices in selected[variant_name].items()},
        }

    audit_path = output_dir / f"clusters_tau_{threshold_label(args.threshold)}.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    original_train = [index for indices in train_by_group.values() for index in indices]
    baseline_reference = ensure_baseline_reference(output_dir, baseline)
    summary = {
        "study": output_dir.name,
        "source_layout": str(layout),
        "source_embeddings": str(embeddings_path),
        "embedding_metadata": embedding_metadata,
        "threshold": args.threshold,
        "temporal_window": args.temporal_window,
        "label_definition": "a YOLO label file containing at least one valid box",
        "full_raw": {
            "images": len(original_train),
            "positive_images": sum(counts[index] > 0 for index in original_train),
            "background_images": sum(counts[index] == 0 for index in original_train),
            **baseline_reference,
        },
        "cluster_stats": cluster_stats,
        "variants": variant_summaries,
        "cluster_audit": str(audit_path),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
