from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageOps

from yolo_retraining.backends.yolov5.source import resolve_yolov5_root, validate_yolov5_source
from yolo_retraining.data import build_registry, write_image_manifest
from yolo_retraining.data.loaders import load_group, yolo_label_path


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    group: str
    split: str


@dataclass(frozen=True)
class SimilarityEdge:
    left: int
    right: int
    score: float


@dataclass(frozen=True)
class AnnotationInfo:
    label_path: Path
    file_exists: bool
    has_annotation: bool
    status: str


def inspect_annotation(image_path: Path | str) -> AnnotationInfo:
    """Return whether an image has at least one non-empty YOLO label row."""

    image = Path(image_path).expanduser().resolve()
    label_path = yolo_label_path(image)
    if not label_path.is_file():
        return AnnotationInfo(label_path=label_path, file_exists=False, has_annotation=False, status="missing")
    has_annotation = any(line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines())
    return AnnotationInfo(
        label_path=label_path,
        file_exists=True,
        has_annotation=has_annotation,
        status="labeled" if has_annotation else "empty",
    )


def _canonical_image_key(value: str) -> str:
    return Path(value).as_posix().casefold()


def annotation_index(records: Sequence[ImageRecord], dataset_root: Path) -> dict[str, AnnotationInfo]:
    return {
        _canonical_image_key(record.path.relative_to(dataset_root).as_posix()): inspect_annotation(record.path)
        for record in records
    }


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(item <= -1 or item >= 1 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values strictly between -1 and 1")
    return thresholds


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def load_layout_records(layout: Path | str, groups: Sequence[str] | None = None) -> tuple[list[ImageRecord], list[str], Path, list[str]]:
    layout_path = Path(layout).expanduser().resolve()
    payload = yaml.safe_load(layout_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), dict):
        raise ValueError(f"redundancy analysis requires a groups-based layout: {layout_path}")
    group_ids = list(payload["groups"]) if groups is None else [str(group) for group in groups]
    records: list[ImageRecord] = []
    names: list[str] | None = None
    dataset_root: Path | None = None
    owners: dict[str, str] = {}
    for group_id in group_ids:
        loaded = load_group(group_id, layout_path)
        if names is None:
            names = list(loaded["names"])
            dataset_root = Path(loaded["root"])
        for split, split_records in loaded["splits"].items():
            for record in split_records:
                path = Path(record["image_path"]).resolve()
                key = str(path).casefold()
                if key in owners:
                    raise ValueError(f"image occurs in multiple analysis groups: {path} ({owners[key]} and {group_id})")
                owners[key] = group_id
                records.append(ImageRecord(path=path, group=group_id, split=split))
    if not records or names is None or dataset_root is None:
        raise ValueError("selected layout groups contain no images")
    records.sort(key=lambda record: record.path.as_posix().casefold())
    return records, names, dataset_root, group_ids


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _inventory_fingerprint(records: Sequence[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        stat = record.path.stat()
        digest.update(f"{record.path.as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _load_yolov5_model(weights: Path, device: torch.device):
    source = validate_yolov5_source(resolve_yolov5_root())
    source_root = Path(source["root"])
    sys.path.insert(0, str(source_root))
    try:
        from models.experimental import attempt_load
        from utils.augmentations import letterbox
    finally:
        if sys.path[0] == str(source_root):
            sys.path.pop(0)
    model = attempt_load(str(weights), device=device, inplace=True, fuse=True)
    model.eval()
    if device.type == "cuda":
        model.half()
    return model, letterbox, source


def _prepare_image(path: Path, *, imgsz: int, stride: int, letterbox) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    image = letterbox(image, new_shape=(imgsz, imgsz), stride=stride, auto=False)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(image)


def extract_embeddings(
    records: Sequence[ImageRecord],
    *,
    weights: Path,
    device_name: str,
    imgsz: int,
    batch_size: int,
    layer_indices: Sequence[int],
    pool_grids: Sequence[int],
) -> tuple[np.ndarray, dict]:
    if len(layer_indices) != len(pool_grids) or any(grid < 1 for grid in pool_grids):
        raise ValueError("pool_grids must contain one positive size for each feature layer")
    device = resolve_device(device_name)
    model, letterbox, source = _load_yolov5_model(weights, device)
    module_list = getattr(model, "model", None)
    if module_list is None or max(layer_indices) >= len(module_list):
        raise ValueError(f"YOLOv5 model does not expose requested feature layers: {list(layer_indices)}")
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for index in layer_indices:
        hooks.append(module_list[index].register_forward_hook(lambda _module, _inputs, output, idx=index: captured.__setitem__(idx, output)))
    stride = int(model.stride.max())
    feature_batches: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                images = np.stack([_prepare_image(record.path, imgsz=imgsz, stride=stride, letterbox=letterbox) for record in batch_records])
                tensor = torch.from_numpy(images).to(device, non_blocking=True)
                tensor = tensor.half() if device.type == "cuda" else tensor.float()
                tensor /= 255.0
                captured.clear()
                model(tensor, augment=False, visualize=False)
                if set(captured) != set(layer_indices):
                    raise RuntimeError(f"failed to capture all YOLOv5 feature layers: {sorted(captured)}")
                pooled = [
                    F.adaptive_avg_pool2d(captured[index], grid).flatten(1).float()
                    for index, grid in zip(layer_indices, pool_grids)
                ]
                features = F.normalize(torch.cat(pooled, dim=1), p=2, dim=1)
                feature_batches.append(features.cpu().numpy().astype(np.float32, copy=False))
                print(f"embedded {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    embeddings = np.concatenate(feature_batches, axis=0)
    metadata = {
        "extractor": "yolov5s_multiscale_spatial_pyramid",
        "layers": list(layer_indices),
        "pool_grids": list(pool_grids),
        "imgsz": imgsz,
        "weights": str(weights.resolve()),
        "weights_sha256": _sha256(weights),
        "yolov5_commit": source["commit"],
        "yolov5_tag": source["tag"],
        "device": str(device),
        "embedding_shape": list(embeddings.shape),
        "inventory_sha256": _inventory_fingerprint(records),
    }
    return embeddings, metadata


def save_embeddings(path: Path, records: Sequence[ImageRecord], embeddings: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32, copy=False),
        paths=np.asarray([str(record.path) for record in records]),
        groups=np.asarray([record.group for record in records]),
        splits=np.asarray([record.split for record in records]),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_embeddings(path: Path | str) -> tuple[list[ImageRecord], np.ndarray, dict]:
    archive = np.load(Path(path).expanduser().resolve(), allow_pickle=False)
    records = [
        ImageRecord(Path(image_path), str(group), str(split))
        for image_path, group, split in zip(archive["paths"].tolist(), archive["groups"].tolist(), archive["splits"].tolist())
    ]
    embeddings = archive["embeddings"].astype(np.float32, copy=False)
    metadata = json.loads(str(archive["metadata"].item()))
    if len(records) != len(embeddings):
        raise ValueError("embedding archive path and feature counts disagree")
    return records, embeddings, metadata


def temporal_similarity_edges(
    records: Sequence[ImageRecord], embeddings: np.ndarray, *, window: int, min_similarity: float
) -> list[SimilarityEdge]:
    if window < 1:
        raise ValueError("temporal window must be positive")
    by_directory: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_directory[str(record.path.parent).casefold()].append(index)
    edges: list[SimilarityEdge] = []
    for indices in by_directory.values():
        indices.sort(key=lambda index: records[index].path.name.casefold())
        for offset, left in enumerate(indices):
            neighbors = indices[offset + 1 : offset + 1 + window]
            if not neighbors:
                continue
            scores = embeddings[neighbors] @ embeddings[left]
            for right, score in zip(neighbors, scores.tolist()):
                if score >= min_similarity:
                    edges.append(SimilarityEdge(left=left, right=right, score=float(score)))
    return edges


def global_knn_edges(
    embeddings: np.ndarray, *, top_k: int, block_size: int, device_name: str
) -> list[SimilarityEdge]:
    if top_k < 1 or top_k >= len(embeddings):
        raise ValueError("top_k must be between 1 and the number of images minus one")
    device = resolve_device(device_name)
    all_features = torch.from_numpy(embeddings).to(device)
    edge_scores: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for start in range(0, len(embeddings), block_size):
            end = min(start + block_size, len(embeddings))
            scores = all_features[start:end] @ all_features.T
            rows = torch.arange(end - start, device=device)
            scores[rows, torch.arange(start, end, device=device)] = -1
            values, neighbors = torch.topk(scores, k=top_k, dim=1)
            for row, (row_values, row_neighbors) in enumerate(zip(values.cpu().numpy(), neighbors.cpu().numpy())):
                left = start + row
                for right, score in zip(row_neighbors.tolist(), row_values.tolist()):
                    pair = (left, right) if left < right else (right, left)
                    edge_scores[pair] = max(edge_scores.get(pair, -1.0), float(score))
            print(f"global knn {end}/{len(embeddings)}", flush=True)
    return [SimilarityEdge(left, right, score) for (left, right), score in sorted(edge_scores.items())]


def cross_split_nearest(
    records: Sequence[ImageRecord], embeddings: np.ndarray, *, block_size: int, device_name: str
) -> list[tuple[int, int, float]]:
    train_indices = [index for index, record in enumerate(records) if record.split == "train"]
    eval_indices = [index for index, record in enumerate(records) if record.split in {"val", "test"}]
    if not train_indices or not eval_indices:
        return []
    device = resolve_device(device_name)
    train_features = torch.from_numpy(embeddings[train_indices]).to(device)
    result: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for start in range(0, len(eval_indices), block_size):
            block_indices = eval_indices[start : start + block_size]
            query = torch.from_numpy(embeddings[block_indices]).to(device)
            values, offsets = torch.max(query @ train_features.T, dim=1)
            for eval_index, train_offset, score in zip(block_indices, offsets.cpu().tolist(), values.cpu().tolist()):
                result.append((eval_index, train_indices[train_offset], float(score)))
    return result


def representative_groups(indices: Sequence[int], edges: Sequence[SimilarityEdge], threshold: float) -> list[tuple[int, list[int]]]:
    allowed = set(indices)
    adjacency: dict[int, dict[int, float]] = {index: {} for index in indices}
    for edge in edges:
        if edge.score < threshold or edge.left not in allowed or edge.right not in allowed:
            continue
        adjacency[edge.left][edge.right] = edge.score
        adjacency[edge.right][edge.left] = edge.score
    unassigned = set(indices)
    groups: list[tuple[int, list[int]]] = []
    while unassigned:
        representative = min(
            unassigned,
            key=lambda index: (
                -sum(other in unassigned for other in adjacency[index]),
                -sum(score for other, score in adjacency[index].items() if other in unassigned),
                index,
            ),
        )
        members = sorted({representative} | (set(adjacency[representative]) & unassigned))
        groups.append((representative, members))
        unassigned.difference_update(members)
    return groups


def _frame_rank(records: Sequence[ImageRecord]) -> dict[int, int]:
    by_directory: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_directory[str(record.path.parent).casefold()].append(index)
    ranks: dict[int, int] = {}
    for indices in by_directory.values():
        indices.sort(key=lambda index: records[index].path.name.casefold())
        for rank, index in enumerate(indices):
            ranks[index] = rank
    return ranks


def write_edges(path: Path, records: Sequence[ImageRecord], edges: Sequence[SimilarityEdge], dataset_root: Path) -> None:
    ranks = _frame_rank(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "right", "similarity", "left_group", "right_group", "same_directory", "frame_gap"])
        for edge in sorted(edges, key=lambda item: (-item.score, item.left, item.right)):
            left = records[edge.left]
            right = records[edge.right]
            same_directory = left.path.parent == right.path.parent
            writer.writerow(
                [
                    left.path.relative_to(dataset_root).as_posix(),
                    right.path.relative_to(dataset_root).as_posix(),
                    f"{edge.score:.8f}",
                    left.group,
                    right.group,
                    int(same_directory),
                    abs(ranks[edge.left] - ranks[edge.right]) if same_directory else "",
                ]
            )


def write_cross_split(path: Path, records: Sequence[ImageRecord], rows: Sequence[tuple[int, int, float]], dataset_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation_image", "evaluation_split", "evaluation_group", "nearest_train_image", "train_group", "similarity"])
        for eval_index, train_index, score in sorted(rows, key=lambda item: -item[2]):
            evaluation = records[eval_index]
            train = records[train_index]
            writer.writerow(
                [
                    evaluation.path.relative_to(dataset_root).as_posix(),
                    evaluation.split,
                    evaluation.group,
                    train.path.relative_to(dataset_root).as_posix(),
                    train.group,
                    f"{score:.8f}",
                ]
            )


def _variant_layout(
    *,
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


def _threshold_label(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def write_pair_gallery(
    path: Path,
    records: Sequence[ImageRecord],
    edges: Sequence[SimilarityEdge],
    *,
    limit: int = 30,
    target_score: float | None = None,
) -> None:
    if target_score is None:
        chosen = sorted(edges, key=lambda edge: (-edge.score, edge.left, edge.right))[:limit]
    else:
        eligible = [edge for edge in edges if edge.score >= target_score]
        chosen = sorted(eligible, key=lambda edge: (abs(edge.score - target_score), edge.left, edge.right))[:limit]
    if not chosen:
        return
    thumb_size = (320, 180)
    row_height = 220
    canvas = Image.new("RGB", (thumb_size[0] * 2, row_height * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, edge in enumerate(chosen):
        for column, index in enumerate((edge.left, edge.right)):
            with Image.open(records[index].path) as image:
                thumbnail = ImageOps.fit(image.convert("RGB"), thumb_size)
            canvas.paste(thumbnail, (column * thumb_size[0], row * row_height))
        draw.text((4, row * row_height + thumb_size[1] + 4), f"cos={edge.score:.5f}", fill="black")
        draw.text((110, row * row_height + thumb_size[1] + 4), records[edge.left].path.name, fill="black")
        draw.text((360, row * row_height + thumb_size[1] + 4), records[edge.right].path.name, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def build_mixed_cluster_report(
    cluster_rows: Sequence[Mapping[str, str]],
    *,
    annotations: Mapping[str, AnnotationInfo],
    dataset_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Select clusters containing both annotated and unannotated images."""

    clusters: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in cluster_rows:
        try:
            key = (row["group"], row["cluster_id"])
            image = row["image"]
        except KeyError as error:
            raise ValueError(f"cluster row is missing required field: {error.args[0]}") from error
        if _canonical_image_key(image) not in annotations:
            raise ValueError(f"no annotation status is available for cluster image: {image}")
        clusters.setdefault(key, []).append(row)

    mixed_clusters: list[dict[str, object]] = []
    mixed_members: list[dict[str, object]] = []
    by_group: dict[str, dict[str, int]] = {}
    for (group, cluster_id), members in clusters.items():
        member_annotations = [annotations[_canonical_image_key(row["image"])] for row in members]
        annotated_count = sum(info.has_annotation for info in member_annotations)
        unannotated_count = len(members) - annotated_count
        if not annotated_count or not unannotated_count:
            continue

        representative_row = next(
            (row for row in members if row["image"] == row["representative"]),
            members[0],
        )
        representative_info = annotations[_canonical_image_key(representative_row["image"])]
        mixed_clusters.append(
            {
                "group": group,
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "annotated_count": annotated_count,
                "unannotated_count": unannotated_count,
                "representative": representative_row["representative"],
                "representative_annotation_status": representative_info.status,
            }
        )
        group_summary = by_group.setdefault(
            group,
            {"mixed_cluster_count": 0, "mixed_member_count": 0, "annotated_count": 0, "unannotated_count": 0},
        )
        group_summary["mixed_cluster_count"] += 1
        group_summary["mixed_member_count"] += len(members)
        group_summary["annotated_count"] += annotated_count
        group_summary["unannotated_count"] += unannotated_count
        for row, info in zip(members, member_annotations):
            try:
                label_file = info.label_path.relative_to(dataset_root).as_posix()
            except ValueError:
                label_file = str(info.label_path)
            mixed_members.append(
                {
                    **dict(row),
                    "annotation_status": info.status,
                    "has_annotation": int(info.has_annotation),
                    "label_file_exists": int(info.file_exists),
                    "label_file": label_file,
                }
            )

    summary: dict[str, object] = {
        "total_clusters": len(clusters),
        "mixed_cluster_count": len(mixed_clusters),
        "mixed_member_count": len(mixed_members),
        "annotated_count": sum(int(row["annotated_count"]) for row in mixed_clusters),
        "unannotated_count": sum(int(row["unannotated_count"]) for row in mixed_clusters),
        "by_group": by_group,
    }
    return mixed_clusters, mixed_members, summary


def write_mixed_cluster_outputs(
    variant_dir: Path,
    cluster_rows: Sequence[Mapping[str, str]],
    *,
    annotations: Mapping[str, AnnotationInfo],
    dataset_root: Path,
) -> dict[str, object]:
    """Write machine-readable reports for mixed annotation clusters."""

    mixed_clusters, mixed_members, summary = build_mixed_cluster_report(
        cluster_rows,
        annotations=annotations,
        dataset_root=dataset_root,
    )
    variant_dir.mkdir(parents=True, exist_ok=True)
    with (variant_dir / "mixed_clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "cluster_id",
                "cluster_size",
                "annotated_count",
                "unannotated_count",
                "representative",
                "representative_annotation_status",
            ],
        )
        writer.writeheader()
        writer.writerows(mixed_clusters)
    with (variant_dir / "mixed_cluster_members.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "cluster_id",
                "image",
                "representative",
                "score_to_representative",
                "cluster_size",
                "annotation_status",
                "has_annotation",
                "label_file_exists",
                "label_file",
            ],
        )
        writer.writeheader()
        writer.writerows(mixed_members)
    report = {
        **summary,
        "files": {
            "clusters": "mixed_clusters.csv",
            "members": "mixed_cluster_members.csv",
        },
    }
    (variant_dir / "mixed_cluster_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


_THRESHOLD_WORKER_CONTEXT: dict[str, object] = {}


def _init_threshold_worker(
    records: Sequence[ImageRecord],
    names: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    temporal_edges: Sequence[SimilarityEdge],
    temporal_score: dict[tuple[int, int], float],
    group_records: dict[str, list[ImageRecord]],
    train_indices_by_group: dict[str, list[int]],
    random_seeds: Sequence[int],
    annotations: Mapping[str, AnnotationInfo],
) -> None:
    global _THRESHOLD_WORKER_CONTEXT
    _THRESHOLD_WORKER_CONTEXT = {
        "records": records,
        "names": names,
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "temporal_edges": temporal_edges,
        "temporal_score": temporal_score,
        "group_records": group_records,
        "train_indices_by_group": train_indices_by_group,
        "random_seeds": random_seeds,
        "annotations": annotations,
    }


def _analyze_threshold(
    *,
    threshold: float,
    records: Sequence[ImageRecord],
    names: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    temporal_edges: Sequence[SimilarityEdge],
    temporal_score: dict[tuple[int, int], float],
    group_records: dict[str, list[ImageRecord]],
    train_indices_by_group: dict[str, list[int]],
    random_seeds: Sequence[int],
    annotations: Mapping[str, AnnotationInfo],
) -> dict:
    label = _threshold_label(threshold)
    write_pair_gallery(
        output_dir / "audit" / f"boundary_tau_{label}.jpg",
        records,
        temporal_edges,
        target_score=threshold,
    )
    selected_by_group: dict[str, list[Path]] = {}
    cluster_rows: list[dict] = []
    cluster_sizes: list[int] = []
    for group_id, indices in train_indices_by_group.items():
        groups = representative_groups(indices, temporal_edges, threshold)
        selected_by_group[group_id] = [records[representative].path for representative, _ in groups]
        for cluster_number, (representative, members) in enumerate(groups):
            cluster_sizes.append(len(members))
            for member in members:
                score = 1.0 if member == representative else temporal_score[(min(representative, member), max(representative, member))]
                cluster_rows.append(
                    {
                        "group": group_id,
                        "cluster_id": f"{group_id}_{cluster_number:06d}",
                        "image": records[member].path.relative_to(dataset_root).as_posix(),
                        "representative": records[representative].path.relative_to(dataset_root).as_posix(),
                        "score_to_representative": f"{score:.8f}",
                        "cluster_size": len(members),
                    }
                )
    variant_dir = output_dir / "variants" / f"dedup_tau_{label}"
    layout = _variant_layout(
        output_dir=variant_dir,
        dataset_root=dataset_root,
        names=names,
        group_records=group_records,
        selected_train=selected_by_group,
    )
    with (variant_dir / "clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "cluster_id", "image", "representative", "score_to_representative", "cluster_size"])
        writer.writeheader()
        writer.writerows(cluster_rows)
    mixed_cluster_summary = write_mixed_cluster_outputs(
        variant_dir,
        cluster_rows,
        annotations=annotations,
        dataset_root=dataset_root,
    )

    original_train = sum(len(indices) for indices in train_indices_by_group.values())
    effective_train = sum(len(images) for images in selected_by_group.values())
    summary = {
        "layout": str(layout),
        "original_train_images": original_train,
        "effective_train_images": effective_train,
        "retained_fraction": effective_train / original_train,
        "redundant_images": original_train - effective_train,
        "non_singleton_clusters": sum(size > 1 for size in cluster_sizes),
        "max_cluster_size": max(cluster_sizes, default=1),
        "per_group_effective": {group: len(images) for group, images in selected_by_group.items()},
        "mixed_clusters": mixed_cluster_summary,
    }
    for seed in random_seeds:
        rng = random.Random(seed)
        random_selected = {
            group_id: rng.sample([records[index].path for index in indices], len(selected_by_group[group_id]))
            for group_id, indices in train_indices_by_group.items()
        }
        _variant_layout(
            output_dir=output_dir / "variants" / f"random_matched_tau_{label}_s{seed}",
            dataset_root=dataset_root,
            names=names,
            group_records=group_records,
            selected_train=random_selected,
        )
    return {"threshold": threshold, **summary}


def _run_threshold_worker(threshold: float) -> dict:
    if not _THRESHOLD_WORKER_CONTEXT:
        raise RuntimeError("threshold worker context was not initialized")
    return _analyze_threshold(threshold=threshold, **_THRESHOLD_WORKER_CONTEXT)


def analyze(
    *,
    records: Sequence[ImageRecord],
    embeddings: np.ndarray,
    names: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    thresholds: Sequence[float],
    temporal_window: int,
    top_k: int,
    block_size: int,
    device: str,
    random_seeds: Sequence[int],
    threshold_workers: int = 1,
) -> dict:
    if threshold_workers < 1:
        raise ValueError("threshold_workers must be at least 1")
    min_threshold = min(thresholds)
    temporal_edges = temporal_similarity_edges(records, embeddings, window=temporal_window, min_similarity=min_threshold)
    global_edges = global_knn_edges(embeddings, top_k=top_k, block_size=block_size, device_name=device)
    leakage = cross_split_nearest(records, embeddings, block_size=block_size, device_name=device)
    write_edges(output_dir / "temporal_similarity_pairs.csv", records, temporal_edges, dataset_root)
    write_edges(output_dir / "global_knn_pairs.csv", records, global_edges, dataset_root)
    write_cross_split(output_dir / "cross_split_nearest.csv", records, leakage, dataset_root)
    write_pair_gallery(output_dir / "audit" / "top_temporal_pairs.jpg", records, temporal_edges)

    group_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        group_records[record.group].append(record)
    train_indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.split == "train":
            train_indices_by_group[record.group].append(index)

    threshold_summaries: dict[str, dict] = {}
    temporal_score = {
        (min(edge.left, edge.right), max(edge.left, edge.right)): edge.score
        for edge in temporal_edges
    }
    annotations = annotation_index(records, dataset_root)
    worker_context = (
        records,
        names,
        dataset_root,
        output_dir,
        temporal_edges,
        temporal_score,
        group_records,
        train_indices_by_group,
        random_seeds,
        annotations,
    )
    if threshold_workers == 1 or len(thresholds) == 1:
        threshold_results = [_analyze_threshold(threshold=threshold, **dict(zip(
            ("records", "names", "dataset_root", "output_dir", "temporal_edges", "temporal_score", "group_records", "train_indices_by_group", "random_seeds", "annotations"),
            worker_context,
        ))) for threshold in thresholds]
    else:
        worker_count = min(threshold_workers, len(thresholds))
        # Spawn avoids inheriting a possibly initialized CUDA context when the
        # matrix stages ran on a GPU before threshold post-processing.
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_threshold_worker,
            initargs=worker_context,
        ) as executor:
            futures = {executor.submit(_run_threshold_worker, threshold): threshold for threshold in thresholds}
            threshold_results = []
            for future in as_completed(futures):
                result = future.result()
                threshold_results.append(result)
                print(
                    f"threshold {result['threshold']:.3f} completed; retained={result['retained_fraction']:.3%}",
                    flush=True,
                )
        threshold_results.sort(key=lambda result: result["threshold"])

    for result in threshold_results:
        threshold_summaries[str(result.pop("threshold"))] = result

    leakage_scores = [score for _, _, score in leakage]
    summary = {
        "images": len(records),
        "train_images": sum(record.split == "train" for record in records),
        "validation_images": sum(record.split == "val" for record in records),
        "test_images": sum(record.split == "test" for record in records),
        "temporal_window": temporal_window,
        "temporal_pairs_at_min_threshold": len(temporal_edges),
        "global_top_k": top_k,
        "global_edges": len(global_edges),
        "cross_split_nearest_mean": float(np.mean(leakage_scores)) if leakage_scores else None,
        "cross_split_nearest_median": float(np.median(leakage_scores)) if leakage_scores else None,
        "thresholds": threshold_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sequential-frame redundancy with fixed YOLOv5s pretrained features.")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", help="Layout groups to analyze; defaults to all groups.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--embeddings", type=Path, help="Reuse an existing embeddings.npz instead of extracting features.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=parse_int_list, default=[17, 20, 23])
    parser.add_argument("--pool-grids", type=parse_int_list, default=[4, 2, 1])
    parser.add_argument("--thresholds", type=parse_thresholds, default=[0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999])
    parser.add_argument("--temporal-window", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--random-seeds", type=parse_int_list, default=[41, 42, 43])
    parser.add_argument("--threshold-workers", type=int, default=1, help="parallel worker processes for independent threshold post-processing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threshold_workers < 1:
        raise ValueError("--threshold-workers must be at least 1")
    output_dir = args.output_dir.expanduser().resolve()
    layout = args.layout.expanduser().resolve()
    records, names, dataset_root, group_ids = load_layout_records(layout, args.groups)
    if args.embeddings:
        archived_records, embeddings, extraction_metadata = load_embeddings(args.embeddings)
        if [(record.path, record.group, record.split) for record in archived_records] != [
            (record.path, record.group, record.split) for record in records
        ]:
            raise ValueError("embedding archive does not match the selected layout records")
        embedding_path = args.embeddings.resolve()
    else:
        weights = args.weights.expanduser().resolve() if args.weights else resolve_yolov5_root() / "yolov5s.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"YOLOv5s pretrained weights do not exist: {weights}")
        embeddings, extraction_metadata = extract_embeddings(
            records,
            weights=weights,
            device_name=args.device,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            layer_indices=args.layers,
            pool_grids=args.pool_grids,
        )
        embedding_path = output_dir / "embeddings.npz"
        save_embeddings(embedding_path, records, embeddings, extraction_metadata)
    run_metadata = {
        **extraction_metadata,
        "source_layout": str(layout),
        "groups": group_ids,
        "dataset_root": str(dataset_root),
        "embeddings": str(embedding_path),
        "thresholds": args.thresholds,
        "temporal_window": args.temporal_window,
        "top_k": args.top_k,
        "random_seeds": args.random_seeds,
        "threshold_workers": args.threshold_workers,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = analyze(
        records=records,
        embeddings=embeddings,
        names=names,
        dataset_root=dataset_root,
        output_dir=output_dir,
        thresholds=args.thresholds,
        temporal_window=args.temporal_window,
        top_k=args.top_k,
        block_size=args.block_size,
        device=args.device,
        random_seeds=args.random_seeds,
        threshold_workers=args.threshold_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
