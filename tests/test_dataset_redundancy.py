from pathlib import Path

import numpy as np

from data_analyse.dataset_redundancy.redundancy_analysis import (
    ImageRecord,
    SimilarityEdge,
    global_knn_edges,
    representative_groups,
    temporal_similarity_edges,
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
