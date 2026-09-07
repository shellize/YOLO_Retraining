"""Compare adjacent-window deduplication choices at a fixed similarity threshold."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Directly executed Study scripts do not automatically expose the repository root.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_analyse.dataset_redundancy.redundancy_analysis import (
    ImageRecord,
    SimilarityEdge,
    representative_groups,
    temporal_similarity_edges,
)
from yolo_retraining.data import write_image_manifest


WINDOWS = (1, 3, 5, 10, 20, 50, 100, 200, 250)
THRESHOLD = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pair_rows(paths: list[Path], features: np.ndarray, top_k: int = 10) -> tuple[list[dict], float, int]:
    if len(paths) < 2:
        return [], 1.0, 0
    features = features.astype(np.float32, copy=False)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    scores = features @ features.T
    left, right = np.triu_indices(len(paths), k=1)
    values = scores[left, right]
    order = np.argsort(-values, kind="stable")[:top_k]
    rows = [
        {"rank": rank, "left": paths[int(left[position])].name, "right": paths[int(right[position])].name, "similarity": float(values[position])}
        for rank, position in enumerate(order, start=1)
    ]
    return rows, float(values.max()), int(np.count_nonzero(values >= THRESHOLD))


def write_html(path: Path, rows: list[dict]) -> None:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['window']}</td><td>{row['original_images']}</td><td>{row['retained_images']}</td>"
            f"<td>{row['retention']:.2%}</td><td>{row['temporal_edges']}</td><td>{row['clusters']}</td>"
            f"<td>{row['post_max_similarity']:.8f}</td><td>{row['post_pairs_at_or_above_tau']}</td>"
            f"<td><a href=\"{row['manifest_link']}\">manifest</a> · <a href=\"{row['pairs_link']}\">top pairs</a></td>"
            "</tr>"
        )
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>stroller_reviewed 邻近窗口扫描</title>
<style>
body{{margin:0;padding:24px;background:#f5f7fb;color:#172033;font:14px/1.5 "Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1280px;margin:auto;background:#fff;padding:22px;border:1px solid #d9e0eb;border-radius:12px}}
h1{{margin-top:0}}.note{{color:#647087;margin-bottom:16px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid #d9e0eb;text-align:left;white-space:nowrap}}th{{background:#eef2f7}}td:nth-child(7){{font-weight:700;color:#315efb}}
</style></head><body><main><h1>stroller_reviewed · τ=0.99 邻近窗口扫描</h1>
<div class="note">窗口表示按文件名排序后，每张图向后比较的邻居数量。去冗余后最高相似度是在对应保留集合内部重新计算的。</div>
<table><thead><tr><th>window</th><th>原图</th><th>保留</th><th>保留率</th><th>相似边</th><th>簇数</th><th>去冗余后最高相似度</th><th>保留集内 ≥τ 图片对</th><th>文件</th></tr></thead><tbody>{''.join(body)}</tbody></table>
</main></body></html>'''
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)
    (output_dir / "top_pairs").mkdir(exist_ok=True)

    archive = np.load(analysis_dir / "embeddings.npz", allow_pickle=True)
    records = [
        ImageRecord(Path(str(path)).expanduser().resolve(), str(group), str(split))
        for path, group, split in zip(archive["paths"].tolist(), archive["groups"].tolist(), archive["splits"].tolist())
    ]
    embeddings = archive["embeddings"].astype(np.float32, copy=False)
    if len(records) != len(embeddings):
        raise ValueError("embedding archive path and feature counts disagree")
    original_indices = [index for index, record in enumerate(records) if record.split == "train"]
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index in original_indices:
        group_indices[records[index].group].append(index)

    summary_rows: list[dict] = []
    for window in WINDOWS:
        edges: list[SimilarityEdge] = temporal_similarity_edges(records, embeddings, window=window, min_similarity=THRESHOLD)
        selected: list[int] = []
        cluster_sizes: list[int] = []
        for indices in group_indices.values():
            groups = representative_groups(indices, edges, THRESHOLD)
            selected.extend(representative for representative, _ in groups)
            cluster_sizes.extend(len(members) for _, members in groups)
        selected.sort(key=lambda index: records[index].path.as_posix().casefold())
        selected_paths = [records[index].path for index in selected]
        manifest_path = write_image_manifest(selected_paths, output_dir / "manifests" / f"stroller_reviewed_window_{window:02d}.txt")
        top_rows, post_max, post_above = pair_rows(selected_paths, embeddings[selected])
        pair_path = output_dir / "top_pairs" / f"window_{window:02d}.csv"
        with pair_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", "left", "right", "similarity"])
            writer.writeheader()
            writer.writerows(top_rows)
        summary_rows.append(
            {
                "window": window,
                "original_images": len(original_indices),
                "retained_images": len(selected),
                "retention": len(selected) / len(original_indices),
                "redundant_images": len(original_indices) - len(selected),
                "temporal_edges": len(edges),
                "clusters": len(selected),
                "non_singleton_clusters": sum(size > 1 for size in cluster_sizes),
                "max_cluster_size": max(cluster_sizes, default=1),
                "post_max_similarity": post_max,
                "post_pairs_at_or_above_tau": post_above,
                "post_top_left": top_rows[0]["left"] if top_rows else "",
                "post_top_right": top_rows[0]["right"] if top_rows else "",
                "manifest": str(manifest_path),
                "pairs": str(pair_path),
                "manifest_link": f"manifests/{manifest_path.name}",
                "pairs_link": f"top_pairs/{pair_path.name}",
            }
        )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "summary.json").write_text(json.dumps({"threshold": THRESHOLD, "windows": summary_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(output_dir / "window_sweep.html", summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "threshold": THRESHOLD, "windows": summary_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
