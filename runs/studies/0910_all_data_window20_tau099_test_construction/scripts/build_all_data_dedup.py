from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_analyse.dataset_redundancy.redundancy_analysis import (  # noqa: E402
    AnnotationInfo,
    ImageRecord,
    SimilarityEdge,
    inspect_annotation,
    load_embeddings,
    load_layout_records,
)
from yolo_retraining.data import write_image_manifest  # noqa: E402


DEFAULT_CONFIG = STUDY_ROOT / "config" / "protocol.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the full-data label-aware temporal-dedup candidate pool.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def image_key(path: Path) -> str:
    parts = list(path.parts)
    lowered = [part.casefold() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"image path has no images directory: {path}")
    index = len(lowered) - 1 - lowered[::-1].index("images")
    return Path(*parts[index:]).as_posix().casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_embeddings(
    records: Sequence[ImageRecord],
    archived_records: Sequence[ImageRecord],
    archived_embeddings: np.ndarray,
) -> np.ndarray:
    archived_by_key: dict[str, int] = {}
    for index, record in enumerate(archived_records):
        key = image_key(record.path)
        if key in archived_by_key:
            raise ValueError(f"embedding archive has duplicate image key: {key}")
        archived_by_key[key] = index
    requested = [image_key(record.path) for record in records]
    if len(set(requested)) != len(requested):
        raise ValueError("source layout contains duplicate images")
    missing = sorted(set(requested) - set(archived_by_key))
    extra = sorted(set(archived_by_key) - set(requested))
    if missing or extra:
        raise ValueError(
            "embedding inventory does not match source layout: "
            f"missing={missing[:1]} extra={extra[:1]}"
        )
    return np.stack([archived_embeddings[archived_by_key[key]] for key in requested]).astype(np.float32, copy=False)


def relative_image(record: ImageRecord, dataset_root: Path) -> str:
    return record.path.resolve().relative_to(dataset_root.resolve()).as_posix()


def global_frame_number(record: ImageRecord) -> int:
    token = record.path.name.split("_", 1)[0]
    if not token.isdecimal():
        raise ValueError(f"image filename does not begin with a numeric global frame id: {record.path}")
    return int(token)


def global_temporal_similarity_edges(
    embeddings: np.ndarray,
    *,
    window: int,
    min_similarity: float,
) -> list[SimilarityEdge]:
    edges: list[SimilarityEdge] = []
    for left in range(len(embeddings)):
        neighbors = list(range(left + 1, min(left + 1 + window, len(embeddings))))
        if not neighbors:
            continue
        scores = embeddings[neighbors] @ embeddings[left]
        for right, score in zip(neighbors, scores.tolist()):
            if score >= min_similarity:
                edges.append(SimilarityEdge(left=left, right=right, score=float(score)))
    return edges


def label_aware_representative_groups(
    indices: Sequence[int],
    edges: Sequence[SimilarityEdge],
    annotations: Sequence[AnnotationInfo],
    threshold: float,
) -> list[tuple[int, int, list[int]]]:
    allowed = set(indices)
    adjacency: dict[int, dict[int, float]] = {index: {} for index in indices}
    for edge in edges:
        if edge.score < threshold or edge.left not in allowed or edge.right not in allowed:
            continue
        adjacency[edge.left][edge.right] = edge.score
        adjacency[edge.right][edge.left] = edge.score

    def coverage_key(index: int, unassigned: set[int]) -> tuple[int, float, int]:
        return (
            -sum(neighbor in unassigned for neighbor in adjacency[index]),
            -sum(score for neighbor, score in adjacency[index].items() if neighbor in unassigned),
            index,
        )

    unassigned = set(indices)
    groups: list[tuple[int, int, list[int]]] = []
    while unassigned:
        ordinary_representative = min(unassigned, key=lambda index: coverage_key(index, unassigned))
        ordinary_members = {ordinary_representative} | (set(adjacency[ordinary_representative]) & unassigned)
        positive_candidates = [index for index in ordinary_members if annotations[index].has_annotation]
        if annotations[ordinary_representative].has_annotation or not positive_candidates:
            selected_representative = ordinary_representative
        else:
            selected_representative = min(
                positive_candidates,
                key=lambda index: coverage_key(index, unassigned),
            )
        members = sorted({selected_representative} | (set(adjacency[selected_representative]) & unassigned))
        groups.append((ordinary_representative, selected_representative, members))
        unassigned.difference_update(members)
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    layout_path = project_path(str(config["source_layout"]))
    embeddings_path = project_path(str(config["source_embeddings"]))
    variant_name = str(config["variant_name"])
    threshold = float(config["threshold"])
    temporal_window = int(config["temporal_window"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if temporal_window < 1:
        raise ValueError("temporal_window must be positive")

    variant_dir = STUDY_ROOT / "experiment" / "variants" / variant_name
    if variant_dir.exists() and any(variant_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing variant output: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=True)

    records, names, dataset_root, source_groups = load_layout_records(layout_path)
    records.sort(key=lambda record: (global_frame_number(record), record.path.as_posix().casefold()))
    frame_numbers = [global_frame_number(record) for record in records]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("source layout contains duplicate numeric global frame ids")
    archived_records, archived_embeddings, embedding_metadata = load_embeddings(embeddings_path)
    embeddings = remap_embeddings(records, archived_records, archived_embeddings)
    annotations = [inspect_annotation(record.path) for record in records]

    edges = global_temporal_similarity_edges(
        embeddings,
        window=temporal_window,
        min_similarity=threshold,
    )
    edge_scores = {
        (min(edge.left, edge.right), max(edge.left, edge.right)): float(edge.score)
        for edge in edges
    }
    by_batch: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_batch[record.path.parent.name].append(index)

    selected_indices: list[int] = []
    cluster_rows: list[dict[str, object]] = []
    per_batch: dict[str, dict[str, int]] = {
        batch: {
            "original_images": len(indices),
            "retained_images": 0,
            "removed_images": 0,
            "retained_positive_images": 0,
            "retained_background_images": 0,
        }
        for batch, indices in by_batch.items()
    }
    non_singleton_clusters = 0
    cross_batch_clusters = 0
    mixed_annotation_clusters = 0
    positive_override_clusters = 0

    groups = label_aware_representative_groups(range(len(records)), edges, annotations, threshold)
    for cluster_number, (ordinary_representative, selected_representative, members) in enumerate(groups):
        positive_members = [index for index in members if annotations[index].has_annotation]
        if selected_representative != ordinary_representative:
            positive_override_clusters += 1

        cluster_batches = list(dict.fromkeys(records[index].path.parent.name for index in members))
        cluster_id = f"global_w{temporal_window}_{cluster_number:06d}"
        selected_indices.append(selected_representative)
        if len(members) > 1:
            non_singleton_clusters += 1
        if len(cluster_batches) > 1:
            cross_batch_clusters += 1
        has_positive = bool(positive_members)
        has_background = any(not annotations[index].has_annotation for index in members)
        if has_positive and has_background:
            mixed_annotation_clusters += 1
        for member in members:
            pair = (min(selected_representative, member), max(selected_representative, member))
            cluster_rows.append(
                {
                    "batch": records[member].path.parent.name,
                    "cluster_batches": "|".join(cluster_batches),
                    "global_frame_number": global_frame_number(records[member]),
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "image": relative_image(records[member], dataset_root),
                    "has_annotation": int(annotations[member].has_annotation),
                    "annotation_status": annotations[member].status,
                    "ordinary_representative": relative_image(records[ordinary_representative], dataset_root),
                    "selected_representative": relative_image(records[selected_representative], dataset_root),
                    "score_to_selected_representative": 1.0
                    if member == selected_representative
                    else edge_scores[pair],
                }
            )

    for index in selected_indices:
        batch_stats = per_batch[records[index].path.parent.name]
        batch_stats["retained_images"] += 1
        if annotations[index].has_annotation:
            batch_stats["retained_positive_images"] += 1
        else:
            batch_stats["retained_background_images"] += 1
    for batch_stats in per_batch.values():
        batch_stats["removed_images"] = batch_stats["original_images"] - batch_stats["retained_images"]

    selected_set = set(selected_indices)
    residual_edges = [edge for edge in edges if edge.left in selected_set and edge.right in selected_set]
    if residual_edges:
        raise RuntimeError(
            "label-aware grouping left a temporal edge between retained representatives: "
            f"count={len(residual_edges)} first={residual_edges[0]}"
        )

    selected_indices.sort(key=lambda index: global_frame_number(records[index]))
    manifest_path = write_image_manifest(
        [records[index].path for index in selected_indices],
        variant_dir / "manifest.txt",
        preserve_order=True,
    )
    clusters_path = variant_dir / "clusters.csv"
    with clusters_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cluster_rows)

    protocol = {
        "study": STUDY_ROOT.name,
        "source_layout": str(layout_path.relative_to(PROJECT_ROOT).as_posix()),
        "source_embeddings": str(embeddings_path.relative_to(PROJECT_ROOT).as_posix()),
        "source_embedding_sha256": sha256(embeddings_path),
        "source_embedding_metadata": embedding_metadata,
        "source_groups": source_groups,
        "dataset_root": str(dataset_root.relative_to(PROJECT_ROOT).as_posix()),
        "image_count": len(records),
        "threshold": threshold,
        "temporal_window": temporal_window,
        "sequence_order": "numeric filename prefix across all physical batch directories",
        "first_global_frame_number": frame_numbers[0],
        "last_global_frame_number": frame_numbers[-1],
        "candidate_rule": "global numeric frame order; compare the next N available images, including batch boundaries",
        "grouping_rule": "label-aware greedy star grouping; group membership is recomputed from the selected representative",
        "representative_rule": "when the ordinary representative neighborhood contains a labeled image, choose the labeled candidate with best current coverage",
        "class_names": list(names),
    }
    (variant_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "completed",
        "original_images": len(records),
        "retained_images": len(selected_indices),
        "removed_images": len(records) - len(selected_indices),
        "retained_fraction": len(selected_indices) / len(records),
        "temporal_edges_at_threshold": len(edges),
        "clusters": len(selected_indices),
        "non_singleton_clusters": non_singleton_clusters,
        "cross_batch_clusters": cross_batch_clusters,
        "retained_temporal_conflicts": len(residual_edges),
        "mixed_annotation_clusters": mixed_annotation_clusters,
        "positive_representative_overrides": positive_override_clusters,
        "missing_label_files": sum(annotation.status == "missing" for annotation in annotations),
        "per_batch": per_batch,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "clusters_csv": clusters_path.name,
    }
    (variant_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
