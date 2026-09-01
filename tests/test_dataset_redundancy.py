import csv
import json
from pathlib import Path

import numpy as np

from data_analyse.dataset_redundancy.redundancy_analysis import (
    ImageRecord,
    SimilarityEdge,
    annotation_index,
    build_parser,
    global_knn_edges,
    parse_thresholds,
    representative_groups,
    temporal_similarity_edges,
    write_mixed_cluster_outputs,
)


def test_temporal_edges_only_compare_within_directory_and_window(tmp_path: Path) -> None:
    records = [
        ImageRecord(tmp_path / "images" / "a" / "001.jpg", "stage0", "train"),
        ImageRecord(tmp_path / "images" / "a" / "002.jpg", "stage0", "train"),
        ImageRecord(tmp_path / "images" / "a" / "003.jpg", "stage0", "train"),
        ImageRecord(tmp_path / "images" / "b" / "001.jpg", "stage1", "train"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    edges = temporal_similarity_edges(records, embeddings, window=1, min_similarity=0.9)
    assert {(edge.left, edge.right) for edge in edges} == {(0, 1), (1, 2)}


def test_representative_groups_prevent_transitive_chain_merge() -> None:
    edges = [SimilarityEdge(0, 1, 0.99), SimilarityEdge(1, 2, 0.99)]
    groups = representative_groups([0, 1, 2], edges, 0.95)
    assert groups == [(1, [0, 1, 2])]

    edges = [SimilarityEdge(0, 1, 0.99), SimilarityEdge(1, 2, 0.99), SimilarityEdge(2, 3, 0.99)]
    groups = representative_groups([0, 1, 2, 3], edges, 0.95)
    assert all(representative in members for representative, members in groups)
    assert sorted(member for _, members in groups for member in members) == [0, 1, 2, 3]
    assert len(groups) == 2


def test_global_knn_returns_sparse_undirected_edges() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    edges = global_knn_edges(embeddings, top_k=1, block_size=2, device_name="cpu")
    assert any((edge.left, edge.right) == (0, 1) for edge in edges)
    assert all(edge.left < edge.right for edge in edges)


def test_redundancy_defaults_include_098_and_099() -> None:
    args = build_parser().parse_args(["--layout", "layout.yaml", "--output-dir", "out"])
    assert parse_thresholds("0.980,0.990") == [0.98, 0.99]
    assert 0.98 in args.thresholds
    assert 0.99 in args.thresholds


def test_mixed_cluster_outputs_select_labeled_and_unlabeled_members(tmp_path: Path) -> None:
    dataset_root = tmp_path
    labeled = dataset_root / "images" / "batch" / "001.jpg"
    unlabeled = dataset_root / "images" / "batch" / "002.jpg"
    singleton = dataset_root / "images" / "batch" / "003.jpg"
    for image in (labeled, unlabeled, singleton):
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
    (dataset_root / "labels" / "batch").mkdir(parents=True)
    (dataset_root / "labels" / "batch" / "001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset_root / "labels" / "batch" / "002.txt").write_text("\n", encoding="utf-8")
    (dataset_root / "labels" / "batch" / "003.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    records = [
        ImageRecord(labeled, "stage0", "train"),
        ImageRecord(unlabeled, "stage0", "train"),
        ImageRecord(singleton, "stage0", "train"),
    ]
    annotations = annotation_index(records, dataset_root)
    cluster_rows = [
        {
            "group": "stage0",
            "cluster_id": "stage0_000000",
            "image": "images/batch/001.jpg",
            "representative": "images/batch/001.jpg",
            "score_to_representative": "1.00000000",
            "cluster_size": "2",
        },
        {
            "group": "stage0",
            "cluster_id": "stage0_000000",
            "image": "images/batch/002.jpg",
            "representative": "images/batch/001.jpg",
            "score_to_representative": "0.99000000",
            "cluster_size": "2",
        },
        {
            "group": "stage0",
            "cluster_id": "stage0_000001",
            "image": "images/batch/003.jpg",
            "representative": "images/batch/003.jpg",
            "score_to_representative": "1.00000000",
            "cluster_size": "1",
        },
    ]

    variant_dir = tmp_path / "dedup_tau_0p980"
    summary = write_mixed_cluster_outputs(
        variant_dir,
        cluster_rows,
        annotations=annotations,
        dataset_root=dataset_root,
    )

    assert summary["total_clusters"] == 2
    assert summary["mixed_cluster_count"] == 1
    assert summary["mixed_member_count"] == 2
    with (variant_dir / "mixed_clusters.csv").open(newline="", encoding="utf-8") as handle:
        mixed_clusters = list(csv.DictReader(handle))
    assert mixed_clusters[0]["cluster_id"] == "stage0_000000"
    assert mixed_clusters[0]["annotated_count"] == "1"
    assert mixed_clusters[0]["unannotated_count"] == "1"
    with (variant_dir / "mixed_cluster_members.csv").open(newline="", encoding="utf-8") as handle:
        mixed_members = list(csv.DictReader(handle))
    assert {row["annotation_status"] for row in mixed_members} == {"labeled", "empty"}
    written_summary = json.loads((variant_dir / "mixed_cluster_summary.json").read_text(encoding="utf-8"))
    assert written_summary["mixed_cluster_count"] == 1
