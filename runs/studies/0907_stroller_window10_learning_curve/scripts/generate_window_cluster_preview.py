"""Generate a preview for non-singleton clusters at one temporal window."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_analyse.dataset_redundancy.redundancy_analysis import (  # noqa: E402
    ImageRecord,
    representative_groups,
    temporal_similarity_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.99)
    return parser.parse_args()


def annotation_payload(dataset_root: Path, paths: set[str]) -> dict[str, dict]:
    result = {}
    for relative in sorted(paths):
        image = dataset_root / relative.removeprefix("images/")
        label_path = dataset_root / "labels" / f"{image.stem}.txt"
        boxes = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                c, x, y, w, h = fields
                boxes.append({"c": int(c), "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
        result[relative] = {"boxes": boxes}
    return result


def build_clusters(analysis_dir: Path, threshold: float, window: int) -> tuple[list[dict], int]:
    archive = np.load(analysis_dir / "embeddings.npz", allow_pickle=True)
    records = [
        ImageRecord(Path(str(path)).expanduser().resolve(), str(group), str(split))
        for path, group, split in zip(archive["paths"].tolist(), archive["groups"].tolist(), archive["splits"].tolist())
    ]
    embeddings = archive["embeddings"].astype(np.float32, copy=False)
    edges = temporal_similarity_edges(records, embeddings, window=window, min_similarity=threshold)
    edge_scores = {(min(edge.left, edge.right), max(edge.left, edge.right)): edge.score for edge in edges}
    train_by_group: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.split == "train":
            train_by_group[record.group].append(index)

    clusters = []
    annotation_paths = set()
    cluster_number = 0
    for group, indices in train_by_group.items():
        for representative, members in representative_groups(indices, edges, threshold):
            if len(members) <= 1:
                continue
            cluster_number += 1
            # All current records share the same dataset root; derive paths from the archive's images/ prefix.
            representative_rel = f"images/{records[representative].path.name}"
            member_rows = []
            for member in sorted(members, key=lambda index: records[index].path.name.casefold()):
                image_rel = f"images/{records[member].path.name}"
                pair = (min(representative, member), max(representative, member))
                score = 1.0 if member == representative else float(edge_scores[pair])
                member_rows.append({"image": image_rel, "score": score})
                annotation_paths.add(image_rel)
            clusters.append(
                {
                    "id": f"{group}_window{window}_{cluster_number:06d}",
                    "group": group,
                    "size": len(members),
                    "representative": representative_rel,
                    "members": member_rows,
                }
            )
    clusters.sort(key=lambda cluster: (-cluster["size"], cluster["id"]))
    return clusters, len(records)


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>stroller_reviewed window=10 非单例簇</title>
<style>
:root{--bg:#f5f7fb;--surface:#fff;--soft:#eef2f7;--ink:#172033;--muted:#647087;--line:#d9e0eb;--accent:#315efb;--green:#15803d;--shadow:0 10px 28px rgba(32,50,86,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}.app{max-width:1680px;margin:0 auto;padding:24px}h1{margin:0 0 5px;font-size:25px}.muted{color:var(--muted)}.summary{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.metric{min-width:160px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 15px;box-shadow:var(--shadow)}.metric small{color:var(--muted)}.metric strong{display:block;font-size:23px}.bar{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}.pagination{display:flex;flex-wrap:wrap;gap:6px}.page{border:1px solid var(--line);background:var(--surface);border-radius:8px;padding:6px 9px;cursor:pointer;font:inherit}.page[aria-current=page]{background:var(--accent);border-color:var(--accent);color:#fff}.page:disabled{opacity:.45;cursor:default}#grid{display:grid;gap:10px}.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}.head{display:flex;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line)}.head strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.meta{color:var(--muted);font-size:12px;white-space:nowrap}.members{display:flex;flex-wrap:wrap;gap:8px;padding:11px 12px}.tile{flex:0 0 190px;width:190px}.tile.tile-enlarged{flex-basis:570px;width:570px}.thumb{position:relative;display:block;width:100%;overflow:hidden;border:0;border-radius:6px;padding:0;background:var(--soft);color:inherit;font:inherit;text-align:left;cursor:zoom-in}.thumb img{width:100%;height:auto;display:block;object-fit:contain}.thumb.is-enlarged{cursor:zoom-out}.thumb.rep{outline:3px solid var(--green);outline-offset:-3px}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible}.box{fill:#2563eb;fill-opacity:.08;stroke:#2563eb;stroke-width:.006}.tag{position:absolute;left:5px;bottom:5px;padding:2px 5px;border-radius:4px;background:var(--green);color:#fff;font-size:10px}.caption{padding:5px 7px;color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}@media(max-width:820px){.app{padding:15px}.tile{flex-basis:160px;width:160px}.tile.tile-enlarged{flex-basis:480px;width:480px}}
</style></head><body><main class="app"><h1>stroller_reviewed · window=10 非单例簇</h1><div class="muted">τ=0.99；只展示簇内图片数量大于 1 的簇。绿色边框为代表图，点击图片可在当前页面放大。</div><section class="summary"><div class="metric"><small>非单例簇</small><strong id="clusters">—</strong></div><div class="metric"><small>涉及图片</small><strong id="members">—</strong></div><div class="metric"><small>最大簇</small><strong id="maxsize">—</strong></div><div class="metric"><small>页数</small><strong id="pages">—</strong></div></section><div class="bar"><div class="muted" id="count">加载中…</div><div class="pagination" id="pagination"></div></div><section id="grid"></section></main><script src="./cluster_preview_data.js"></script><script>
(()=>{const DATA=window.__CLUSTER_DATA__,ROOT="__IMAGE_ROOT__",PAGE_SIZE=12,state={page:1};const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),href=x=>ROOT+'/'+x;function overlay(image){const boxes=(DATA.annotations[image]||{}).boxes||[];return boxes.map(b=>{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h);return `<rect class="box" x="${x.toFixed(6)}" y="${y.toFixed(6)}" width="${Math.max(0,w).toFixed(6)}" height="${Math.max(0,h).toFixed(6)}"><title>stroller</title></rect>`}).join('')}function card(c){return `<article class="card"><div class="head"><strong>${esc(c.id)}</strong><span class="meta">${esc(c.group)} · ${c.size} 张</span></div><div class="members">${c.members.map(m=>{const rep=m.image===c.representative;return `<div class="tile"><button type="button" class="thumb${rep?' rep':''}" data-preview-image="${href(m.image)}" aria-pressed="false" aria-label="${esc(m.image)}"><img src="${href(m.image)}" loading="lazy" alt="${esc(m.image)}"><svg class="overlay" viewBox="0 0 1 1" preserveAspectRatio="none">${overlay(m.image)}</svg>${rep?'<span class="tag">代表图</span>':''}</button><div class="caption">${esc(m.image.split('/').pop())}<br>sim=${m.score.toFixed(6)}</div></div>`}).join('')}</div></article>`}function pages(total){const root=$("pagination"),p=state.page;root.innerHTML='';const add=(label,n,disabled=false,current=false)=>{const b=document.createElement('button');b.className='page';b.textContent=label;b.disabled=disabled;if(current)b.setAttribute('aria-current','page');b.onclick=()=>{state.page=n;render()};root.appendChild(b)};add('上一页',p-1,p===1);for(let n=1;n<=total;n++)add(String(n),n,false,n===p);add('下一页',p+1,p===total)}function render(){const total=Math.max(1,Math.ceil(DATA.clusters.length/PAGE_SIZE));state.page=Math.min(state.page,total);const rows=DATA.clusters.slice((state.page-1)*PAGE_SIZE,state.page*PAGE_SIZE);$("grid").innerHTML=rows.map(card).join('');$("count").textContent=`第 ${state.page}/${total} 页 · 当前 ${rows.length} 个簇`;pages(total)}$("grid").addEventListener('click',e=>{const b=e.target.closest('button[data-preview-image]');if(!b)return;const tile=b.closest('.tile'),on=!b.classList.contains('is-enlarged');b.classList.toggle('is-enlarged',on);tile.classList.toggle('tile-enlarged',on);b.setAttribute('aria-pressed',String(on))});$("clusters").textContent=DATA.clusters.length;$("members").textContent=DATA.clusters.reduce((n,c)=>n+c.size,0);$("maxsize").textContent=Math.max(...DATA.clusters.map(c=>c.size));$("pages").textContent=Math.ceil(DATA.clusters.length/PAGE_SIZE);render()})();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters, total_images = build_clusters(analysis_dir, args.threshold, args.window)
    annotation_paths = {member["image"] for cluster in clusters for member in cluster["members"]}
    annotations = annotation_payload(dataset_root, annotation_paths)
    payload = {"window": args.window, "threshold": args.threshold, "clusters": clusters, "annotations": annotations}
    (output_dir / "cluster_preview_data.js").write_text(
        "window.__CLUSTER_DATA__ = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8"
    )
    image_root = Path(os.path.relpath(dataset_root, output_dir)).as_posix()
    (output_dir / "cluster_preview.html").write_text(HTML.replace("__IMAGE_ROOT__", image_root), encoding="utf-8")
    with (output_dir / "clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_id", "group", "cluster_size", "image", "representative", "score_to_representative"])
        writer.writeheader()
        for cluster in clusters:
            for member in cluster["members"]:
                writer.writerow({"cluster_id": cluster["id"], "group": cluster["group"], "cluster_size": cluster["size"], "image": member["image"], "representative": cluster["representative"], "score_to_representative": f'{member["score"]:.8f}'})
    (output_dir / "summary.json").write_text(json.dumps({"window": args.window, "threshold": args.threshold, "original_images": total_images, "non_singleton_clusters": len(clusters), "member_images": sum(cluster["size"] for cluster in clusters), "max_cluster_size": max((cluster["size"] for cluster in clusters), default=0)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(output_dir / "cluster_preview.html"), "clusters": len(clusters), "member_images": sum(cluster["size"] for cluster in clusters)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
