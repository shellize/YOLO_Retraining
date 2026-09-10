from __future__ import annotations

import argparse
import csv
import json
import os
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

from data_analyse.dataset_redundancy.redundancy_analysis import load_embeddings  # noqa: E402
from yolo_retraining.data import read_image_manifest  # noqa: E402
from yolo_retraining.data.loaders import yolo_label_path  # noqa: E402


DEFAULT_CONFIG = STUDY_ROOT / "config" / "protocol.yaml"
CLASS_COLORS = ["#2563eb", "#16a34a", "#db2777", "#ea580c"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporal-cluster and post-dedup similarity audit pages.")
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


def rel_image(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def global_frame_number(path: Path | str) -> int:
    token = Path(path).name.split("_", 1)[0]
    if not token.isdecimal():
        raise ValueError(f"image filename does not begin with a numeric global frame id: {path}")
    return int(token)


def annotation_payload(dataset_root: Path, image_relatives: set[str]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for relative in sorted(image_relatives):
        image_path = dataset_root / relative
        label_path = yolo_label_path(image_path)
        boxes: list[dict[str, object]] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                class_id, x, y, width, height = fields
                try:
                    boxes.append(
                        {
                            "c": int(class_id),
                            "x": float(x),
                            "y": float(y),
                            "w": float(width),
                            "h": float(height),
                        }
                    )
                except ValueError:
                    continue
        payload[relative] = {"boxes": boxes, "label_count": len(boxes)}
    return payload


def load_clusters(path: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["cluster_id"]].append(row)
    clusters: list[dict[str, object]] = []
    for cluster_id, rows in grouped.items():
        if len(rows) <= 1:
            continue
        first = rows[0]
        clusters.append(
            {
                "id": cluster_id,
                "batches": first["cluster_batches"].split("|"),
                "size": int(first["cluster_size"]),
                "ordinary_representative": first["ordinary_representative"],
                "selected_representative": first["selected_representative"],
                "members": [
                    {
                        "image": row["image"],
                        "has_annotation": bool(int(row["has_annotation"])),
                        "similarity": float(row["score_to_selected_representative"]),
                    }
                    for row in rows
                ],
            }
        )
    clusters.sort(
        key=lambda cluster: (
            -int(cluster["size"]),
            -max(float(member["similarity"]) for member in cluster["members"] if member["image"] != cluster["selected_representative"]),
            str(cluster["id"]),
        )
    )
    return clusters


def top_pairs_blockwise(features: np.ndarray, count: int, block_size: int) -> list[tuple[int, int, float]]:
    if count < 1 or block_size < 1:
        raise ValueError("top pair count and block size must be positive")
    feature_count = len(features)
    if feature_count < 2:
        raise ValueError("at least two retained images are required")
    candidates: list[tuple[int, int, float]] = []
    for start in range(0, feature_count, block_size):
        end = min(start + block_size, feature_count)
        scores = features[start:end] @ features.T
        for local_row, global_row in enumerate(range(start, end)):
            scores[local_row, : global_row + 1] = -np.inf
        flat = scores.ravel()
        finite_count = int(np.isfinite(flat).sum())
        take = min(count, finite_count)
        if not take:
            continue
        positions = np.argpartition(flat, flat.size - take)[-take:]
        for position in positions.tolist():
            local_row, right = divmod(position, feature_count)
            score = float(scores[local_row, right])
            if np.isfinite(score):
                candidates.append((start + local_row, right, score))
        candidates = sorted(candidates, key=lambda row: (-row[2], row[0], row[1]))[:count]
    return candidates


def cluster_html(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全量数据全局顺序 window=20 时序去重簇审计</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#647087;--line:#d9e0eb;--blue:#2563eb;--green:#15803d;--orange:#ea580c;--shadow:0 10px 28px rgba(32,50,86,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1720px;margin:auto;padding:24px}}h1{{margin:0 0 5px;font-size:25px}}.muted{{color:var(--muted)}}.summary,.toolbar{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}.metric,.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:170px;padding:11px 14px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:22px}}select,input,button{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 9px}}button{{cursor:pointer}}#grid{{display:grid;gap:11px}}.card{{overflow:hidden}}.head{{display:flex;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line)}}.members{{display:flex;flex-wrap:wrap;gap:9px;padding:11px}}.tile{{width:200px;flex:0 0 200px}}.tile.big{{width:600px;flex-basis:600px}}.thumb{{position:relative;display:block;width:100%;padding:0;border:0;overflow:hidden;background:#eef2f7;cursor:zoom-in}}.thumb img{{display:block;width:100%;height:auto}}.thumb.big{{cursor:zoom-out}}.thumb.selected{{outline:4px solid var(--green);outline-offset:-4px}}.thumb.ordinary:not(.selected){{outline:4px solid var(--blue);outline-offset:-4px}}.overlay{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.box{{fill-opacity:.06;stroke-width:.006}}.tag{{position:absolute;left:5px;bottom:5px;background:rgba(23,32,51,.86);color:#fff;border-radius:4px;padding:2px 5px;font-size:10px}}.caption{{font-size:11px;color:var(--muted);padding:5px;overflow-wrap:anywhere}}.pages{{display:flex;gap:5px;flex-wrap:wrap;margin:14px 0}}.pages button[aria-current=page]{{background:var(--blue);color:#fff}}@media(max-width:800px){{main{{padding:14px}}.tile{{width:46%;flex-basis:46%}}.tile.big{{width:100%;flex-basis:100%}}}}
</style></head><body><main><h1>全量数据全局顺序 · window=20, tau=0.99 时序簇</h1><div class="muted">按文件名中的全局帧号跨 batch 比较。绿色：最终标注优先代表；蓝色：被替换的普通代表。显示真实 YOLO 框；点击图片放大。</div><section class="summary"><div class="metric"><small>原始图片</small><strong>{payload['original_images']:,}</strong></div><div class="metric"><small>保留代表</small><strong>{payload['retained_images']:,}</strong></div><div class="metric"><small>非单例簇</small><strong>{len(payload['clusters']):,}</strong></div><div class="metric"><small>跨 batch 簇</small><strong>{payload['cross_batch_clusters']:,}</strong></div><div class="metric"><small>标注代表替换</small><strong>{payload['positive_overrides']:,}</strong></div></section><div class="toolbar"><select id="batch"></select><select id="sort"><option value="size">簇大小降序</option><option value="similarity">最高相似度降序</option></select><input id="search" placeholder="文件名或簇 ID"><span class="muted" id="count"></span></div><div id="grid"></div><div class="pages" id="pages"></div></main><script>
const DATA={data},ROOT={json.dumps(image_root, ensure_ascii=False)},COLORS={json.dumps(CLASS_COLORS)};const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),state={{page:1}},SIZE=12;function boxes(path){{return (DATA.annotations[path]?.boxes||[]).map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"><title>${{esc(DATA.class_names[b.c]??b.c)}}</title></rect>`}}).join('')}}function rows(){{let r=DATA.clusters.slice(),q=$('search').value.toLowerCase(),b=$('batch').value;if(b!=='all')r=r.filter(x=>x.batches.includes(b));if(q)r=r.filter(x=>x.id.toLowerCase().includes(q)||x.members.some(m=>m.image.toLowerCase().includes(q)));if($('sort').value==='similarity')r.sort((a,b)=>Math.max(...b.members.filter(m=>m.image!==b.selected_representative).map(m=>m.similarity))-Math.max(...a.members.filter(m=>m.image!==a.selected_representative).map(m=>m.similarity)));return r}}function card(c){{return `<article class="card"><div class="head"><strong>${{esc(c.id)}}</strong><span class="muted">${{esc(c.batches.join(' / '))}} · ${{c.size}} 张</span></div><div class="members">${{c.members.map(m=>{{const selected=m.image===c.selected_representative,ordinary=m.image===c.ordinary_representative,tags=[selected?'标注优先代表':'',ordinary?'普通代表':''].filter(Boolean).join(' / ');return `<div class="tile"><button class="thumb${{selected?' selected':''}}${{ordinary?' ordinary':''}}"><img loading="lazy" src="${{ROOT+'/'+m.image}}"><svg class="overlay" viewBox="0 0 1 1" preserveAspectRatio="none">${{boxes(m.image)}}</svg>${{tags?`<span class="tag">${{tags}}</span>`:''}}</button><div class="caption">${{esc(m.image)}}<br>sim-to-selected=${{m.similarity.toFixed(8)}} · label=${{m.has_annotation?'yes':'no'}}</div></div>`}}).join('')}}</div></article>`}}function render(){{const all=rows(),total=Math.max(1,Math.ceil(all.length/SIZE));state.page=Math.min(state.page,total);$('grid').innerHTML=all.slice((state.page-1)*SIZE,state.page*SIZE).map(card).join('');$('count').textContent=`${{all.length}} 个簇 · 第 ${{state.page}}/${{total}} 页`;$('pages').innerHTML='';for(let i=1;i<=total;i++){{if(total>12&&i>2&&i<total-1&&Math.abs(i-state.page)>1)continue;const b=document.createElement('button');b.textContent=i;b.setAttribute('aria-current',i===state.page?'page':'false');b.onclick=()=>{{state.page=i;render();scrollTo({{top:0,behavior:'smooth'}})}};$('pages').appendChild(b)}}}}$('grid').onclick=e=>{{const b=e.target.closest('button.thumb');if(!b)return;b.classList.toggle('big');b.closest('.tile').classList.toggle('big')}};for(const id of ['batch','sort'])$(id).onchange=()=>{{state.page=1;render()}};$('search').oninput=()=>{{state.page=1;render()}};const batches=[...new Set(DATA.clusters.flatMap(c=>c.batches))].sort();$('batch').innerHTML='<option value="all">全部 batch</option>'+batches.map(b=>`<option>${{esc(b)}}</option>`).join('');render();
</script></body></html>'''


def pairs_html(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>去重后全局高相似图片人工审计</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#647087;--line:#d9e0eb;--blue:#2563eb;--green:#15803d;--red:#b91c1c;--orange:#c2410c;--shadow:0 10px 28px rgba(32,50,86,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{margin:0 0 5px;font-size:25px}}.muted{{color:var(--muted)}}.summary,.toolbar,.review{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:18px 0}}.metric,.pair{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:170px;padding:11px 14px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:22px}}button,select{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px;cursor:pointer}}.toolbar{{position:sticky;top:0;background:rgba(244,247,251,.94);backdrop-filter:blur(8px);padding:12px 0;z-index:5}}#pairs{{display:grid;gap:12px}}.pair{{padding:12px}}.head{{display:flex;justify-content:space-between;gap:10px;margin-bottom:10px}}.score{{font-weight:700;color:var(--blue)}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.tile button{{position:relative;display:block;width:100%;padding:0;border:0;overflow:hidden;background:#eef2f7}}.tile img{{display:block;width:100%;height:auto}}.tile.big{{grid-column:1/-1}}.tile.big button{{max-width:1000px}}.caption{{font-size:12px;color:var(--muted);padding:5px 2px;overflow-wrap:anywhere}}.choice.active[data-status=near_duplicate]{{background:var(--red);color:#fff}}.choice.active[data-status=same_view_distinct_time]{{background:var(--orange);color:#fff}}.choice.active[data-status=different_scene]{{background:var(--green);color:#fff}}.choice.active[data-status=unsure]{{background:var(--blue);color:#fff}}.review{{margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}}textarea{{width:100%;min-height:52px;border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}}.box{{fill-opacity:.06;stroke-width:.006}}@media(max-width:750px){{main{{padding:14px}}.images{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>去重后全局高相似图片 · 人工场景审计</h1><div class="muted">这些图片已通过全局顺序 window=20、tau=0.99 的标注感知时序去重。请区分连续近重复、同机位但不同时间、不同机位/场景；选择和备注保存在当前浏览器，可导出 CSV。</div><section class="summary"><div class="metric"><small>保留图片</small><strong>{payload['retained_images']:,}</strong></div><div class="metric"><small>展示图片对</small><strong>{len(payload['pairs']):,}</strong></div><div class="metric"><small>最高相似度</small><strong>{payload['max_similarity']:.8f}</strong></div><div class="metric"><small>window 内超阈值对</small><strong>{payload['above_threshold_within_window']:,}</strong></div><div class="metric"><small>已审阅</small><strong id="reviewed">0</strong></div></section><div class="toolbar"><select id="filter"><option value="all">全部</option><option value="unreviewed">未审阅</option><option value="near_duplicate">连续片段 / 近重复</option><option value="same_view_distinct_time">同机位但不同时间</option><option value="different_scene">不同机位 / 场景</option><option value="unsure">不确定</option></select><button id="export">导出审阅 CSV</button><button id="clear">清空本页审阅</button><span class="muted" id="shown"></span></div><section id="pairs"></section></main><script>
const DATA={data},ROOT={json.dumps(image_root, ensure_ascii=False)},COLORS={json.dumps(CLASS_COLORS)},KEY='0910_global_label_aware_post_dedup_review_v2';let review=JSON.parse(localStorage.getItem(KEY)||'{{}}');const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),save=()=>localStorage.setItem(KEY,JSON.stringify(review));function boxes(path){{return (DATA.annotations[path]?.boxes||[]).map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"><title>${{esc(DATA.class_names[b.c]??b.c)}}</title></rect>`}}).join('')}}function card(p){{const r=review[p.rank]||{{status:'',note:''}},tile=x=>`<div class="tile"><button data-zoom><img loading="lazy" src="${{ROOT+'/'+x}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">${{boxes(x)}}</svg></button><div class="caption">${{esc(x)}} · labels=${{DATA.annotations[x]?.label_count??0}}</div></div>`;return `<article class="pair" data-rank="${{p.rank}}"><div class="head"><strong>#${{p.rank}} · ${{p.same_batch?'同 batch':'跨 batch'}} · 序列距 ${{p.sequence_gap}}（帧号差 ${{p.global_frame_gap}}）</strong><span class="score">similarity=${{p.similarity.toFixed(8)}}</span></div><div class="images">${{tile(p.left)}}${{tile(p.right)}}</div><div class="review"><button class="choice ${{r.status==='near_duplicate'?'active':''}}" data-status="near_duplicate">连续片段 / 近重复</button><button class="choice ${{r.status==='same_view_distinct_time'?'active':''}}" data-status="same_view_distinct_time">同机位但不同时间</button><button class="choice ${{r.status==='different_scene'?'active':''}}" data-status="different_scene">不同机位 / 场景</button><button class="choice ${{r.status==='unsure'?'active':''}}" data-status="unsure">不确定</button><textarea placeholder="备注，例如同机位，时间不同、人员不同">${{esc(r.note||'')}}</textarea></div></article>`}}function visible(){{const f=$('filter').value;return DATA.pairs.filter(p=>f==='all'||(f==='unreviewed'?!review[p.rank]?.status:review[p.rank]?.status===f))}}function render(){{const rows=visible();$('pairs').innerHTML=rows.map(card).join('');$('shown').textContent=`显示 ${{rows.length}} / ${{DATA.pairs.length}} 对`;$('reviewed').textContent=Object.values(review).filter(x=>x.status).length}}$('filter').onchange=render;$('pairs').onclick=e=>{{const article=e.target.closest('.pair');if(!article)return;const rank=article.dataset.rank;if(e.target.closest('[data-zoom]')){{e.target.closest('.tile').classList.toggle('big');return}}const choice=e.target.closest('.choice');if(choice){{review[rank]=review[rank]||{{status:'',note:''}};review[rank].status=choice.dataset.status;save();render()}}}};$('pairs').oninput=e=>{{if(e.target.tagName!=='TEXTAREA')return;const rank=e.target.closest('.pair').dataset.rank;review[rank]=review[rank]||{{status:'',note:''}};review[rank].note=e.target.value;save()}};$('export').onclick=()=>{{const q=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=[['rank','left','right','similarity','same_batch','sequence_gap','global_frame_gap','status','note']];for(const p of DATA.pairs){{const r=review[p.rank]||{{}};rows.push([p.rank,p.left,p.right,p.similarity,p.same_batch?1:0,p.sequence_gap,p.global_frame_gap,r.status||'',r.note||''])}}const blob=new Blob([rows.map(r=>r.map(q).join(',')).join('\n')],{{type:'text/csv;charset=utf-8'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='global_label_aware_post_dedup_manual_review.csv';a.click();URL.revokeObjectURL(a.href)}};$('clear').onclick=()=>{{if(confirm('清空当前浏览器保存的全部人工审阅？')){{review={{}};save();render()}}}};render();
</script></body></html>'''


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8")) or {}
    embeddings_path = project_path(str(config["source_embeddings"]))
    variant_dir = STUDY_ROOT / "experiment" / "variants" / str(config["variant_name"])
    summary = json.loads((variant_dir / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((variant_dir / "protocol.json").read_text(encoding="utf-8"))
    dataset_root = project_path(str(protocol["dataset_root"]))
    clusters = load_clusters(variant_dir / "clusters.csv")
    cluster_images = {member["image"] for cluster in clusters for member in cluster["members"]}

    temporal_dir = STUDY_ROOT / "result" / str(config["temporal_result_name"])
    post_dir = STUDY_ROOT / "result" / str(config["post_dedup_result_name"])
    for output_dir in (temporal_dir, post_dir):
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing audit output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    cluster_annotations = annotation_payload(dataset_root, cluster_images)
    cluster_payload = {
        "original_images": summary["original_images"],
        "retained_images": summary["retained_images"],
        "positive_overrides": summary["positive_representative_overrides"],
        "cross_batch_clusters": summary["cross_batch_clusters"],
        "class_names": protocol["class_names"],
        "clusters": clusters,
        "annotations": cluster_annotations,
    }
    temporal_root = Path(os.path.relpath(dataset_root, temporal_dir)).as_posix()
    (temporal_dir / "cluster_preview.html").write_text(
        cluster_html(cluster_payload, temporal_root), encoding="utf-8"
    )
    (temporal_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "non_singleton_clusters": len(clusters),
                "member_images": sum(int(cluster["size"]) for cluster in clusters),
                "positive_representative_overrides": summary["positive_representative_overrides"],
                "html": "cluster_preview.html",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    retained_images = read_image_manifest(variant_dir / "manifest.txt", dataset_root=dataset_root)
    archived_records, archived_embeddings, embedding_metadata = load_embeddings(embeddings_path)
    archive_by_key = {image_key(record.path): index for index, record in enumerate(archived_records)}
    source_order = {
        image_key(record.path): rank
        for rank, record in enumerate(
            sorted(archived_records, key=lambda record: (global_frame_number(record.path), image_key(record.path)))
        )
    }
    retained_rel = [rel_image(path, dataset_root) for path in retained_images]
    missing = [path for path in retained_rel if path.casefold() not in archive_by_key]
    if missing:
        raise ValueError(f"retained image missing from embedding archive: {missing[0]}")
    indices = [archive_by_key[path.casefold()] for path in retained_rel]
    features = archived_embeddings[indices].astype(np.float32, copy=False)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    top_count = int(config["top_similar_pairs"])
    block_size = int(config["similarity_block_size"])
    top_rows = top_pairs_blockwise(features, top_count, block_size)
    pairs = [
        {
            "rank": rank,
            "left": retained_rel[left],
            "right": retained_rel[right],
            "similarity": score,
            "same_batch": Path(retained_rel[left]).parent == Path(retained_rel[right]).parent,
            "sequence_gap": abs(source_order[retained_rel[left].casefold()] - source_order[retained_rel[right].casefold()]),
            "global_frame_gap": abs(global_frame_number(retained_rel[left]) - global_frame_number(retained_rel[right])),
        }
        for rank, (left, right, score) in enumerate(top_rows, start=1)
    ]
    pair_images = {pair[side] for pair in pairs for side in ("left", "right")}
    pair_annotations = annotation_payload(dataset_root, pair_images)
    with (post_dir / "top_similar_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "left",
                "right",
                "similarity",
                "same_batch",
                "sequence_gap",
                "global_frame_gap",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({**pair, "similarity": f'{float(pair["similarity"]):.8f}', "same_batch": int(pair["same_batch"])})
    pair_payload = {
        "retained_images": len(retained_rel),
        "max_similarity": float(pairs[0]["similarity"]),
        "above_threshold_within_window": sum(
            float(pair["similarity"]) >= float(config["threshold"])
            and int(pair["sequence_gap"]) <= int(config["temporal_window"])
            for pair in pairs
        ),
        "class_names": protocol["class_names"],
        "pairs": pairs,
        "annotations": pair_annotations,
    }
    post_root = Path(os.path.relpath(dataset_root, post_dir)).as_posix()
    (post_dir / "top_similar_pairs.html").write_text(
        pairs_html(pair_payload, post_root), encoding="utf-8"
    )
    post_summary = {
        "status": "completed",
        "retained_images": len(retained_rel),
        "top_pairs": len(pairs),
        "max_similarity": float(pairs[0]["similarity"]),
        "same_batch_pairs": sum(bool(pair["same_batch"]) for pair in pairs),
        "cross_batch_pairs": sum(not bool(pair["same_batch"]) for pair in pairs),
        "pairs_at_or_above_threshold": sum(
            float(pair["similarity"]) >= float(config["threshold"]) for pair in pairs
        ),
        "pairs_at_or_above_threshold_within_window": pair_payload["above_threshold_within_window"],
        "embedding_metadata": embedding_metadata,
        "csv": "top_similar_pairs.csv",
        "html": "top_similar_pairs.html",
        "manual_review_export": "global_label_aware_post_dedup_manual_review.csv (downloaded by the browser)",
    }
    (post_dir / "summary.json").write_text(
        json.dumps(post_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "temporal": {
                    "non_singleton_clusters": len(clusters),
                    "member_images": sum(int(cluster["size"]) for cluster in clusters),
                    "positive_representative_overrides": summary["positive_representative_overrides"],
                    "html": str(temporal_dir / "cluster_preview.html"),
                },
                "post_dedup": post_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
