from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from generate_positive_box_layout_pair_audit import CLASS_COLORS, CLASS_NAMES, frame, image_key, read_boxes, read_manifest


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_VARIANT = STUDY_ROOT / "experiment" / "variants" / "global_order_window20_tau0p990_label_aware_pair_reviewed_final"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data_analyse" / "dataset_redundancy" / "results" / "20260824-182608" / "embeddings.npz"
DEFAULT_OUTPUT = STUDY_ROOT / "result" / "final_positive_tau095_window10_cluster_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster final positive images for the last similarity audit.")
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=10)
    return parser.parse_args()


def page(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    root, colors = json.dumps(image_root, ensure_ascii=False), json.dumps(CLASS_COLORS)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最终正样本聚类审阅</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#667085;--line:#d6deea;--blue:#2563eb;--shadow:0 10px 28px rgba(32,50,86,.09)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1720px;margin:auto;padding:22px}}h1{{margin:0 0 6px}}.muted{{color:var(--muted)}}.summary,.toolbar,.members,.pages{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:17px 0}}.metric,.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:155px;padding:9px 13px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:21px}}.toolbar{{position:sticky;top:0;z-index:5;background:rgba(244,247,251,.96);padding:10px 0}}select,input,button{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px}}button{{cursor:pointer}}#grid{{display:grid;gap:12px}}.card{{padding:12px}}.head{{display:flex;justify-content:space-between;gap:10px;margin-bottom:9px}}.score{{font-weight:700;color:var(--blue)}}.tile{{width:240px;flex:0 0 240px}}.tile.big{{width:720px;flex-basis:720px}}.figure{{position:relative;background:#e9eef5;cursor:zoom-in}}img{{display:block;width:100%;height:auto}}svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.box{{fill-opacity:.06;stroke-width:.006}}.caption{{font-size:11px;color:var(--muted);padding-top:4px;overflow-wrap:anywhere}}.edges{{margin-top:8px;font-size:12px;color:var(--muted)}}.pages{{margin:14px 0}}.pages button[aria-current=page]{{background:var(--blue);color:#fff}}@media(max-width:760px){{main{{padding:12px}}.tile{{width:100%;flex-basis:100%}}}}
</style></head><body><main><h1>最终正样本 · tau=0.95, window=10 聚类审阅</h1><div class="muted">仅包含最终 manifest 中有有效标注的图片。先按全局帧号排序，再在正样本序列中比较后续 10 张；余弦相似度 ≥ 0.95 的边构成连通簇。默认按每个簇的最高内部相似度从高到低排列。</div><section class="summary"><div class="metric"><small>最终全部图片</small><strong>{payload['final_images']}</strong></div><div class="metric"><small>正样本图片</small><strong>{payload['positive_images']}</strong></div><div class="metric"><small>超阈值边</small><strong>{payload['edge_count']}</strong></div><div class="metric"><small>非单例簇</small><strong>{payload['cluster_count']}</strong></div><div class="metric"><small>簇内图片</small><strong>{payload['clustered_images']}</strong></div></section><div class="toolbar"><input id="search" placeholder="搜索文件名或簇 ID"><span class="muted" id="shown"></span></div><section id="grid"></section><div class="pages" id="pages"></div></main><script>
const DATA={data},ROOT={root},COLORS={colors},SIZE=10;let page=1;const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));function boxes(image){{return DATA.annotations[image].map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"></rect>`}}).join('')}}function tile(image){{return `<div class="tile"><div class="figure"><img loading="lazy" src="${{ROOT+'/'+image}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${{boxes(image)}}</svg></div><div class="caption">${{esc(image)}} · ${{DATA.annotations[image].length}} 个框</div></div>`}}function card(c){{return `<article class="card"><div class="head"><strong>${{esc(c.id)}} · ${{c.members.length}} 张 · frame ${{c.frame_min}}–${{c.frame_max}}</strong><span class="score">最高相似度=${{c.max_similarity.toFixed(8)}}</span></div><div class="members">${{c.members.map(tile).join('')}}</div><div class="edges">${{c.edges.map(e=>`${{e.left_frame}}–${{e.right_frame}}: ${{e.similarity.toFixed(8)}}`).join('　')}}</div></article>`}}function rows(){{const q=$('search').value.trim().toLowerCase();return q?DATA.clusters.filter(c=>c.id.toLowerCase().includes(q)||c.members.some(x=>x.toLowerCase().includes(q))):DATA.clusters}}function render(){{const all=rows(),total=Math.max(1,Math.ceil(all.length/SIZE));page=Math.min(page,total);$('grid').innerHTML=all.slice((page-1)*SIZE,page*SIZE).map(card).join('');$('shown').textContent=`${{all.length}} 个簇 · 第 ${{page}}/${{total}} 页`;$('pages').innerHTML='';for(let i=1;i<=total;i++){{if(total>14&&i>2&&i<total-1&&Math.abs(i-page)>1)continue;const b=document.createElement('button');b.textContent=i;b.setAttribute('aria-current',i===page?'page':'false');b.onclick=()=>{{page=i;render();scrollTo({{top:0,behavior:'smooth'}})}};$('pages').appendChild(b)}}}}$('search').oninput=()=>{{page=1;render()}};$('grid').onclick=e=>{{const t=e.target.closest('.tile');if(t)t.classList.toggle('big')}};render();
</script></body></html>'''


def main() -> int:
    args = parse_args(); variant, output = args.variant_dir.resolve(), args.output_dir.resolve()
    if not 0 <= args.threshold <= 1 or args.window < 1:
        raise ValueError("invalid threshold or window")
    dataset = (PROJECT_ROOT / "data" / "self_improving").resolve()
    images = read_manifest(variant / "manifest.txt")
    annotations = {}; positives = []
    for image in images:
        relative = image.relative_to(dataset).as_posix(); boxes = read_boxes(image, dataset)
        if boxes: positives.append(image); annotations[relative] = boxes
    archive = np.load(args.embeddings.resolve(), allow_pickle=True); features = archive["embeddings"].astype(np.float64); features /= np.linalg.norm(features, axis=1, keepdims=True)
    indices = {image_key(path): index for index, path in enumerate(archive["paths"].tolist())}
    parent = list(range(len(positives)))
    def find(x: int) -> int:
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    edges = []
    for left in range(len(positives)):
        left_feature = features[indices[image_key(positives[left])]]
        for right in range(left + 1, min(len(positives), left + args.window + 1)):
            similarity = float(left_feature @ features[indices[image_key(positives[right])]])
            if similarity >= args.threshold:
                union(left, right); edges.append((left, right, similarity))
    groups = {}
    for index in {item for edge in edges for item in edge[:2]}: groups.setdefault(find(index), []).append(index)
    edge_groups = {}
    for left, right, score in edges: edge_groups.setdefault(find(left), []).append((left, right, score))
    clusters = []
    for number, (root, members) in enumerate(groups.items(), start=1):
        members.sort(); cluster_edges = sorted(edge_groups[root], key=lambda x: (-x[2], x[0], x[1]))
        clusters.append({"id": f"positive_cluster_{number:04d}", "members": [positives[i].relative_to(dataset).as_posix() for i in members], "frame_min": min(frame(positives[i]) for i in members), "frame_max": max(frame(positives[i]) for i in members), "max_similarity": cluster_edges[0][2], "edges": [{"left_frame": frame(positives[a]), "right_frame": frame(positives[b]), "similarity": s} for a,b,s in cluster_edges]})
    clusters.sort(key=lambda c: (-c["max_similarity"], c["frame_min"]))
    for index, cluster in enumerate(clusters, start=1): cluster["id"] = f"positive_cluster_{index:04d}"
    payload = {"final_images": len(images), "positive_images": len(positives), "edge_count": len(edges), "cluster_count": len(clusters), "clustered_images": sum(len(c["members"]) for c in clusters), "threshold": args.threshold, "window": args.window, "class_names": CLASS_NAMES, "clusters": clusters, "annotations": annotations}
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.mkdir(parents=True, exist_ok=True); image_root = Path(os.path.relpath(dataset, output)).as_posix()
    (output / "cluster_review.html").write_text(page(payload, image_root), encoding="utf-8", newline="\n")
    with (output / "similarity_edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n"); writer.writerow(("left","right","similarity")); writer.writerows((positives[a].relative_to(dataset).as_posix(),positives[b].relative_to(dataset).as_posix(),f"{s:.9f}") for a,b,s in sorted(edges,key=lambda x:-x[2]))
    summary={key:payload[key] for key in ("final_images","positive_images","edge_count","cluster_count","clustered_images","threshold","window")}; summary.update({"status":"completed","html":"cluster_review.html","edge_csv":"similarity_edges.csv","order":"cluster maximum internal similarity descending"})
    (output / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
