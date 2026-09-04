from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

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
    SimilarityEdge,
    inspect_annotation,
    load_embeddings,
    load_layout_records,
    representative_groups,
    temporal_similarity_edges,
)
from yolo_retraining.data import write_image_manifest  # noqa: E402


SOURCE_LAYOUT = (
    PROJECT_ROOT
    / "runs"
    / "studies"
    / "0826_balance_learning_curve_2batch"
    / "config"
    / "data"
    / "balance_2batch_ordered.yaml"
)
SOURCE_EMBEDDINGS = (
    PROJECT_ROOT
    / "data_analyse"
    / "dataset_redundancy"
    / "results"
    / "20260824-182608"
    / "embeddings.npz"
)
EXPECTED_GROUPS = ["test", "val", *(f"stage{index}" for index in range(8))]
TRAIN_GROUPS = [f"stage{index}" for index in range(8)]

TAU_099_VARIANT = "dedup_tau_0p990"
TAU_096_VARIANT = "dedup_tau_0p960_mixed_release_drop_pure_background"


def remap_embeddings(
    records: Sequence[ImageRecord],
    archived_records: Sequence[ImageRecord],
    archived_embeddings: np.ndarray,
    dataset_root: Path,
) -> np.ndarray:
    """Reuse cached features only when the physical image inventory matches exactly."""

    def requested_key(path: Path) -> str:
        return path.resolve().relative_to(dataset_root).as_posix().casefold()

    def archived_key(path: Path) -> str:
        parts = list(path.parts)
        lowered = [part.casefold() for part in parts]
        if "self_improving" not in lowered:
            raise ValueError(f"cached embedding path has no self_improving boundary: {path}")
        boundary = len(lowered) - 1 - lowered[::-1].index("self_improving")
        return Path(*parts[boundary + 1 :]).as_posix().casefold()

    archived_by_path: dict[str, int] = {}
    for index, record in enumerate(archived_records):
        key = archived_key(record.path)
        if key in archived_by_path:
            raise ValueError(f"embedding archive contains duplicate image: {record.path}")
        archived_by_path[key] = index

    requested = {requested_key(record.path) for record in records}
    archived = set(archived_by_path)
    if requested != archived:
        raise ValueError(
            "embedding inventory does not match the ordered Balance layout: "
            f"missing={sorted(requested - archived)[:1]} extra={sorted(archived - requested)[:1]}"
        )
    return np.stack(
        [archived_embeddings[archived_by_path[requested_key(record.path)]] for record in records]
    )


def write_layout(
    variant_dir: Path,
    dataset_root: Path,
    names: Sequence[str],
    group_records: dict[str, list[ImageRecord]],
    selected_train: dict[str, list[Path]],
) -> Path:
    manifests: dict[str, Path] = {}
    for group_id in EXPECTED_GROUPS:
        records = group_records[group_id]
        images = selected_train[group_id] if group_id in TRAIN_GROUPS else [record.path for record in records]
        manifests[group_id] = write_image_manifest(images, variant_dir / "manifests" / f"{group_id}.txt")

    merged_train = [image for group_id in TRAIN_GROUPS for image in selected_train[group_id]]
    write_image_manifest(merged_train, variant_dir / "manifests" / "train.txt")

    layout_path = variant_dir / "layout.yaml"
    payload = {
        "path": Path(os.path.relpath(dataset_root, layout_path.parent)).as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "groups": {
            group_id: {
                "split": group_records[group_id][0].split,
                "manifest": Path(os.path.relpath(manifests[group_id], dataset_root)).as_posix(),
            }
            for group_id in EXPECTED_GROUPS
        },
    }
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return layout_path


def cluster_type(has_annotations: list[bool]) -> str:
    if not any(has_annotations):
        return "pure_background"
    if all(has_annotations):
        return "pure_labeled"
    return "mixed"


def ordinary_selector(cluster_kind: str, representative: int, members: list[int]) -> list[int]:
    del cluster_kind, members
    return [representative]


def mixed_release_selector(cluster_kind: str, representative: int, members: list[int]) -> list[int]:
    if cluster_kind == "pure_background":
        return []
    if cluster_kind == "pure_labeled":
        return [representative]
    return list(members)


def build_variant(
    *,
    variant_name: str,
    threshold: float,
    policy: str,
    selector: Callable[[str, int, list[int]], list[int]],
    records: Sequence[ImageRecord],
    names: Sequence[str],
    dataset_root: Path,
    group_records: dict[str, list[ImageRecord]],
    train_indices: dict[str, list[int]],
    edges: Sequence[SimilarityEdge],
) -> dict[str, object]:
    selected_by_group: dict[str, list[Path]] = {}
    audit_rows: list[dict[str, object]] = []
    stats = {
        "clusters": 0,
        "non_singleton_clusters": 0,
        "pure_background_clusters": 0,
        "pure_labeled_clusters": 0,
        "mixed_clusters": 0,
        "selected_pure_background_images": 0,
        "unselected_pure_background_images": 0,
        "selected_pure_labeled_images": 0,
        "selected_mixed_images": 0,
    }

    for group_id in TRAIN_GROUPS:
        selected_indices: list[int] = []
        for cluster_number, (representative, members) in enumerate(
            representative_groups(train_indices[group_id], edges, threshold)
        ):
            annotations = [inspect_annotation(records[index].path) for index in members]
            kind = cluster_type([info.has_annotation for info in annotations])
            kept = selector(kind, representative, members)
            kept_set = set(kept)
            selected_indices.extend(kept)

            stats["clusters"] += 1
            stats["non_singleton_clusters"] += int(len(members) > 1)
            stats[f"{kind}_clusters"] += 1
            if kind == "pure_background":
                stats["selected_pure_background_images"] += len(kept)
                stats["unselected_pure_background_images"] += len(members) - len(kept)
            elif kind == "pure_labeled":
                stats["selected_pure_labeled_images"] += len(kept)
            else:
                stats["selected_mixed_images"] += len(kept)

            cluster_id = f"{group_id}_{cluster_number:06d}"
            for member, annotation in zip(members, annotations):
                audit_rows.append(
                    {
                        "group": group_id,
                        "cluster_id": cluster_id,
                        "cluster_type": kind,
                        "cluster_size": len(members),
                        "image": records[member].path.relative_to(dataset_root).as_posix(),
                        "has_annotation": int(annotation.has_annotation),
                        "annotation_status": annotation.status,
                        "representative": records[representative].path.relative_to(dataset_root).as_posix(),
                        "selected": int(member in kept_set),
                    }
                )

        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError(f"variant {variant_name} selected duplicate images in {group_id}")
        if not selected_indices:
            raise ValueError(f"variant {variant_name} produced an empty stage: {group_id}")
        selected_by_group[group_id] = [records[index].path for index in selected_indices]

    variant_dir = STUDY_ROOT / "experiment" / "variants" / variant_name
    layout_path = write_layout(
        variant_dir,
        dataset_root,
        names,
        group_records,
        selected_by_group,
    )
    audit_path = variant_dir / "clusters.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    per_stage = {group_id: len(selected_by_group[group_id]) for group_id in TRAIN_GROUPS}
    selected_images = sum(per_stage.values())
    return {
        "threshold": threshold,
        "policy": policy,
        "layout": str(layout_path),
        "cluster_audit": str(audit_path),
        "selected_images": selected_images,
        "retained_fraction": selected_images / sum(len(train_indices[group_id]) for group_id in TRAIN_GROUPS),
        "per_stage_images": per_stage,
        "cluster_stats": stats,
    }


def main() -> int:
    records, names, dataset_root, group_ids = load_layout_records(SOURCE_LAYOUT)
    if group_ids != EXPECTED_GROUPS:
        raise ValueError(f"ordered Balance layout groups changed: expected={EXPECTED_GROUPS} actual={group_ids}")
    source_payload = yaml.safe_load(SOURCE_LAYOUT.read_text(encoding="utf-8"))
    stage_order = {
        group_id: [Path(value).name for value in source_payload["groups"][group_id]["images"]]
        for group_id in TRAIN_GROUPS
    }

    archived_records, archived_embeddings, embedding_metadata = load_embeddings(SOURCE_EMBEDDINGS)
    embeddings = remap_embeddings(records, archived_records, archived_embeddings, dataset_root)
    edges = temporal_similarity_edges(records, embeddings, window=1, min_similarity=0.96)

    group_records: dict[str, list[ImageRecord]] = defaultdict(list)
    train_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        group_records[record.group].append(record)
        if record.group in TRAIN_GROUPS:
            if record.split != "train":
                raise ValueError(f"ordered train group is not marked train: {record.group} -> {record.split}")
            train_indices[record.group].append(index)

    variants = {
        TAU_099_VARIANT: build_variant(
            variant_name=TAU_099_VARIANT,
            threshold=0.99,
            policy="keep one representative per cluster",
            selector=ordinary_selector,
            records=records,
            names=names,
            dataset_root=dataset_root,
            group_records=group_records,
            train_indices=train_indices,
            edges=edges,
        ),
        TAU_096_VARIANT: build_variant(
            variant_name=TAU_096_VARIANT,
            threshold=0.96,
            policy=(
                "drop pure-background clusters; keep one representative for pure-labeled clusters; "
                "keep every member of mixed clusters"
            ),
            selector=mixed_release_selector,
            records=records,
            names=names,
            dataset_root=dataset_root,
            group_records=group_records,
            train_indices=train_indices,
            edges=edges,
        ),
    }
    summary = {
        "study": STUDY_ROOT.name,
        "source_layout": str(SOURCE_LAYOUT),
        "source_embeddings": str(SOURCE_EMBEDDINGS),
        "embedding_metadata": embedding_metadata,
        "temporal_window": 1,
        "label_definition": "a YOLO label file containing at least one non-empty row",
        "train_images": sum(len(train_indices[group_id]) for group_id in TRAIN_GROUPS),
        "stage_order": stage_order,
        "variants": variants,
    }
    summary_path = STUDY_ROOT / "experiment" / "variants" / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
