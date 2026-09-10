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
    ImageRecord,
    inspect_annotation,
    load_embeddings,
    load_layout_records,
    representative_groups,
    temporal_similarity_edges,
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
    archived_records, archived_embeddings, embedding_metadata = load_embeddings(embeddings_path)
    embeddings = remap_embeddings(records, archived_records, archived_embeddings)
    annotations = [inspect_annotation(record.path) for record in records]

    edges = temporal_similarity_edges(
        records,
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
    per_batch: dict[str, dict[str, int]] = {}
    non_singleton_clusters = 0
    mixed_annotation_clusters = 0
    positive_override_clusters = 0

    for batch in sorted(by_batch, key=str.casefold):
        indices = sorted(by_batch[batch], key=lambda index: records[index].path.name.casefold())
        groups = representative_groups(indices, edges, threshold)
        batch_selected: list[int] = []
        for cluster_number, (ordinary_representative, members) in enumerate(groups):
            positive_members = [index for index in members if annotations[index].has_annotation]
            if annotations[ordinary_representative].has_annotation or not positive_members:
                selected_representative = ordinary_representative
            else:
                selected_representative = min(
                    positive_members,
                    key=lambda index: (
                        -edge_scores[(min(ordinary_representative, index), max(ordinary_representative, index))],
                        records[index].path.name.casefold(),
                    ),
                )
                positive_override_clusters += 1

            cluster_id = f"{batch}_w{temporal_window}_{cluster_number:06d}"
            batch_selected.append(selected_representative)
            if len(members) > 1:
                non_singleton_clusters += 1
            has_positive = bool(positive_members)
            has_background = any(not annotations[index].has_annotation for index in members)
            if has_positive and has_background:
                mixed_annotation_clusters += 1
            for member in sorted(members, key=lambda index: records[index].path.name.casefold()):
                pair = (min(ordinary_representative, member), max(ordinary_representative, member))
                cluster_rows.append(
                    {
                        "batch": batch,
                        "cluster_id": cluster_id,
                        "cluster_size": len(members),
                        "image": relative_image(records[member], dataset_root),
                        "has_annotation": int(annotations[member].has_annotation),
                        "annotation_status": annotations[member].status,
                        "ordinary_representative": relative_image(records[ordinary_representative], dataset_root),
                        "selected_representative": relative_image(records[selected_representative], dataset_root),
                        "score_to_ordinary_representative": 1.0
                        if member == ordinary_representative
                        else edge_scores[pair],
                    }
                )
        selected_indices.extend(batch_selected)
        per_batch[batch] = {
            "original_images": len(indices),
            "retained_images": len(batch_selected),
            "removed_images": len(indices) - len(batch_selected),
            "retained_positive_images": sum(annotations[index].has_annotation for index in batch_selected),
            "retained_background_images": sum(not annotations[index].has_annotation for index in batch_selected),
        }

    selected_indices.sort(key=lambda index: image_key(records[index].path))
    manifest_path = write_image_manifest(
        [records[index].path for index in selected_indices],
        variant_dir / "manifest.txt",
        preserve_order=True,
    )
    clusters_path = variant_dir / "clusters.csv"
    with clusters_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
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
        "candidate_rule": "same physical batch directory; filename ordered; forward temporal window",
        "grouping_rule": "project greedy representative grouping",
        "representative_rule": "prefer a non-empty YOLO-labeled member; otherwise keep ordinary representative",
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
