"""Write the tau=0.99 retained manifest and inspect its closest image pairs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np

from yolo_retraining.data import read_image_manifest, write_image_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Retained-image manifest; defaults to the original tau=0.99 variant manifest.")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def rel_image_path(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def html_page(payload: dict, *, dataset_root: Path, output_dir: Path) -> str:
    image_root = Path(os.path.relpath(dataset_root.resolve(), output_dir.resolve())).as_posix()
    serialized = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>stroller_reviewed τ=0.99 去冗余后最相似图片</title>
  <style>
    :root{{--bg:#f5f7fb;--surface:#fff;--ink:#172033;--muted:#647087;--line:#d9e0eb;--accent:#315efb;--shadow:0 10px 28px rgba(32,50,86,.08)}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}
    .app{{max-width:1500px;margin:0 auto;padding:24px}} h1{{margin:0 0 6px;font-size:25px}} .muted{{color:var(--muted)}}
    .summary{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}} .metric{{min-width:170px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 15px;box-shadow:var(--shadow)}} .metric small{{display:block;color:var(--muted)}} .metric strong{{display:block;font-size:22px;margin-top:2px}}
    .note{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-bottom:16px;color:var(--muted)}}
    #pairs{{display:grid;gap:12px}} .pair{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;box-shadow:var(--shadow)}} .pair-head{{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}} .score{{font-weight:700;color:var(--accent)}} .images{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}} .image-tile{{min-width:0;flex:1 1 320px}} .preview{{position:relative;display:block;width:100%;border:0;border-radius:7px;padding:0;overflow:hidden;background:#eef2f7;cursor:zoom-in}} .preview img{{display:block;width:100%;height:auto}} .image-tile.is-enlarged{{flex-basis:100%}} .image-tile.is-enlarged .preview{{width:min(100%,900px);cursor:zoom-out}} .caption{{padding:6px 2px 0;color:var(--muted);font-size:12px;overflow-wrap:anywhere}} @media(max-width:700px){{.app{{padding:14px}} .image-tile,.image-tile.is-enlarged{{flex-basis:100%}}}}
  </style>
</head>
<body><main class="app">
  <h1>stroller_reviewed · τ=0.99 去冗余后最近邻审计</h1>
  <div class="muted">仅在 0.99 去冗余后保留的图片集合内部计算余弦相似度；点击图片可在当前页面放大。</div>
  <section class="summary">
    <div class="metric"><small>保留图片数</small><strong>{payload["retained_images"]:,}</strong></div>
    <div class="metric"><small>检查的唯一图片对</small><strong>{payload["unique_pairs"]:,}</strong></div>
    <div class="metric"><small>最高相似度</small><strong>{payload["max_similarity"]:.8f}</strong></div>
    <div class="metric"><small>展示图片对</small><strong>{len(payload["pairs"]):,}</strong></div>
  </section>
  <div class="note">相似度使用分析阶段缓存的 YOLOv5 特征点积（特征已归一化，等价于余弦相似度）。这组图片是去冗余后的集合，不是原始集合中的相似度。</div>
  <section id="pairs"></section>
</main>
<script>
const DATA={serialized};
const ROOT={json.dumps(image_root, ensure_ascii=False)};
const esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const href=x=>ROOT+'/'+x;
document.getElementById('pairs').innerHTML=DATA.pairs.map((p,i)=>`<article class="pair"><div class="pair-head"><strong>#${{i+1}} · 去冗余后最相似图片对</strong><span class="score">similarity=${{p.similarity.toFixed(8)}}</span></div><div class="images"><div class="image-tile"><button type="button" class="preview" aria-pressed="false"><img src="${{href(p.left)}}" loading="lazy" alt="${{esc(p.left)}}"></button><div class="caption">${{esc(p.left)}}</div></div><div class="image-tile"><button type="button" class="preview" aria-pressed="false"><img src="${{href(p.right)}}" loading="lazy" alt="${{esc(p.right)}}"></button><div class="caption">${{esc(p.right)}}</div></div></div></article>`).join('');
document.getElementById('pairs').addEventListener('click',event=>{{const button=event.target.closest('button.preview');if(!button)return;const tile=button.closest('.image-tile');const enlarged=!tile.classList.contains('is-enlarged');tile.classList.toggle('is-enlarged',enlarged);button.setAttribute('aria-pressed',String(enlarged));}});
</script></body></html>
'''


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    source_manifest = args.manifest.expanduser().resolve() if args.manifest else analysis_dir / "variants" / "dedup_tau_0p990" / "manifests" / "stroller_reviewed.txt"
    retained_images = read_image_manifest(source_manifest, dataset_root=dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_image_manifest(retained_images, output_dir / "manifest.txt")

    archive = np.load(analysis_dir / "embeddings.npz", allow_pickle=True)
    archive_paths = []
    for value in archive["paths"].tolist():
        archive_path = Path(str(value))
        if archive_path.is_absolute():
            archive_paths.append(rel_image_path(archive_path, dataset_root))
        else:
            archive_paths.append(archive_path.as_posix())
    index_by_path = {path: index for index, path in enumerate(archive_paths)}
    retained_rel = [rel_image_path(image, dataset_root) for image in retained_images]
    missing = [path for path in retained_rel if path not in index_by_path]
    if missing:
        raise ValueError(f"retained images missing from embedding archive: {missing[0]}")
    indices = np.array([index_by_path[path] for path in retained_rel], dtype=np.int64)
    features = archive["embeddings"][indices].astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    scores = features @ features.T
    upper_i, upper_j = np.triu_indices(len(indices), k=1)
    upper_scores = scores[upper_i, upper_j]
    order = np.argsort(-upper_scores, kind="stable")[: args.top_k]

    pairs = [
        {
            "left": retained_rel[int(upper_i[position])],
            "right": retained_rel[int(upper_j[position])],
            "similarity": float(upper_scores[position]),
        }
        for position in order
    ]
    if not pairs:
        raise ValueError("at least two retained images are required")

    with (output_dir / "top_similar_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "left", "right", "similarity"])
        writer.writeheader()
        for rank, pair in enumerate(pairs, start=1):
            writer.writerow({"rank": rank, **pair, "similarity": f'{pair["similarity"]:.8f}'})

    payload = {
        "threshold": 0.99,
        "retained_images": len(retained_rel),
        "unique_pairs": int(len(upper_scores)),
        "max_similarity": float(upper_scores.max()),
        "pairs": pairs,
        "manifest": manifest_path.name,
        "embedding_archive": str((analysis_dir / "embeddings.npz").relative_to(output_dir.parent)),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "top_similar_pairs.html").write_text(html_page(payload, dataset_root=dataset_root, output_dir=output_dir), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "csv": str(output_dir / "top_similar_pairs.csv"), "html": str(output_dir / "top_similar_pairs.html"), "retained_images": len(retained_rel), "max_similarity": payload["max_similarity"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
