from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_VARIANT = (
    STUDY_ROOT
    / "experiment"
    / "variants"
    / "global_order_window20_tau0p990_label_aware_reviewed_final"
)
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data_analyse" / "dataset_redundancy" / "results" / "20260824-182608" / "embeddings.npz"
DEFAULT_OUTPUT = STUDY_ROOT / "result" / "retained_temporal_pair_audit"
CLASS_NAMES = ("large luggage", "stroller", "wheelchair", "flatbed truck")
CLASS_COLORS = ("#2563eb", "#16a34a", "#db2777", "#ea580c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit every adjacent pair retained after tau=0.99 temporal dedup.")
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frame-gap", type=int, default=20)
    return parser.parse_args()


def image_key(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    if normalized.casefold().startswith("images/"):
        return normalized.casefold()
    marker = "/images/"
    index = normalized.casefold().rfind(marker)
    if index < 0:
        raise ValueError(f"image path has no images directory: {path}")
    return normalized[index + 1 :].casefold()


def frame(path: Path) -> int:
    token = path.stem.split("_", 1)[0]
    if not token.isdecimal():
        raise ValueError(f"image has no numeric frame prefix: {path.name}")
    return int(token)


def source_timestamp(path: Path) -> str:
    _, separator, remainder = path.stem.partition("_")
    if not separator:
        return ""
    middle, tail_separator, _ = remainder.rpartition("_")
    return middle if tail_separator else remainder


def read_manifest(path: Path) -> list[Path]:
    manifest = path.resolve()
    images = []
    for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            image = (manifest.parent / value).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"manifest image does not exist: {image}")
            images.append(image)
    images.sort(key=lambda item: (frame(item), item.as_posix().casefold()))
    return images


def label_payload(image: Path, dataset_root: Path) -> dict[str, object]:
    relative = image.relative_to(dataset_root)
    label = dataset_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
    boxes = []
    if label.is_file():
        for line_number, raw in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid YOLO row at {label}:{line_number}: {line!r}")
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
            boxes.append({"c": class_id, "x": x, "y": y, "w": width, "h": height})
    return {"boxes": boxes, "classes": sorted({box["c"] for box in boxes}), "label_count": len(boxes)}


def page(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    root = json.dumps(image_root, ensure_ascii=False)
    colors = json.dumps(CLASS_COLORS)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>去重后连续帧链审计</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#647087;--line:#d9e0eb;--blue:#2563eb;--green:#15803d;--red:#b91c1c;--orange:#c2410c;--shadow:0 10px 28px rgba(32,50,86,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1720px;margin:auto;padding:24px}}h1{{margin:0 0 6px;font-size:26px}}.muted{{color:var(--muted)}}.summary,.toolbar,.review,.pages{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:18px 0}}.metric,.pair,.chain-card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:155px;padding:10px 13px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:21px}}button,select,input{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px}}button{{cursor:pointer}}.toolbar{{position:sticky;top:0;z-index:8;background:rgba(244,247,251,.95);backdrop-filter:blur(8px);padding:12px 0}}#pairs{{display:grid;gap:12px}}.pair,.chain-card{{padding:12px}}.pair.exact{{border:2px solid var(--orange)}}.head{{display:flex;justify-content:space-between;gap:12px;margin-bottom:9px}}.score{{font-weight:700;color:var(--blue)}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.chain-images{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}}.chain-images .tile{{width:250px;flex:0 0 250px}}.tile .figure{{position:relative;background:#e9eef5;cursor:zoom-in}}.tile img{{display:block;width:100%;height:auto}}.tile svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.box{{fill-opacity:.07;stroke-width:.006}}.caption{{font-size:12px;color:var(--muted);padding:5px 1px;overflow-wrap:anywhere}}.review{{margin-top:8px;padding-top:9px;border-top:1px solid var(--line)}}.choice.active[data-status=near_duplicate]{{background:var(--red);color:#fff}}.choice.active[data-status=same_event_distinct]{{background:var(--orange);color:#fff}}.choice.active[data-status=different]{{background:var(--green);color:#fff}}.choice.active[data-status=unsure]{{background:var(--blue);color:#fff}}textarea{{width:100%;min-height:45px;border:1px solid var(--line);border-radius:8px;padding:7px;font:inherit}}.tile.big{{width:100%;flex-basis:100%;grid-column:1/-1}}.tile.big .figure{{max-width:1200px}}.pages{{margin:16px 0}}.pages button[aria-current=page]{{background:var(--blue);color:#fff}}@media(max-width:760px){{main{{padding:14px}}.images{{grid-template-columns:1fr}}.chain-images .tile{{width:100%;flex-basis:100%}}}}
</style></head><body><main><h1>去重后连续帧链 · 时序漏洞审计</h1><div class="muted">这里不是相似度 top 200，而是最终 6,027 张数据按全局帧号排序后的全部相邻保留对。连续边的人工结论将在后续按连通分量合并，因此可表达任意长度的多图事件链。橙色边框表示文件名时间戳完全相同。</div>
<section class="summary"><div class="metric"><small>相邻候选对</small><strong>{payload['pair_count']}</strong></div><div class="metric"><small>帧距 1</small><strong>{payload['gap_one_pairs']}</strong></div><div class="metric"><small>多图连续链</small><strong>{payload['multi_image_chains']}</strong></div><div class="metric"><small>最长连续链</small><strong>{payload['max_chain_size']}</strong></div><div class="metric"><small>同时间戳</small><strong>{payload['same_timestamp_pairs']}</strong></div><div class="metric"><small>跨 split</small><strong>{payload['cross_split_pairs']}</strong></div><div class="metric"><small>已审阅</small><strong id="reviewed">0</strong></div></section>
<div class="toolbar"><select id="view"><option value="chains">完整连续链视图</option><option value="pairs">相邻两图 / 桥接对视图</option></select><select id="status"><option value="all">全部状态</option><option value="unreviewed">未审阅</option><option value="near_duplicate">连续帧 / 近重复</option><option value="same_event_distinct">同事件但应保留</option><option value="different">不同事件或链内有断点</option><option value="unsure">不确定</option></select><select id="scope"><option value="all">全部范围</option><option value="same_timestamp">含同时间戳</option><option value="gap1">仅帧距 1</option><option value="cross_split">含跨 split</option><option value="both_positive">双方正样本</option><option value="either_positive">至少一方正样本</option></select><select id="batch"><option value="all">全部 batch</option></select><select id="sort"><option value="size">链长度降序</option><option value="timestamp_first">同时间戳优先，其次帧距</option><option value="gap">帧距升序</option><option value="similarity">相似度降序</option></select><input id="search" placeholder="搜索文件名或链 ID"><button id="export">导出审阅 CSV</button><button id="clear">清空审阅</button><span class="muted" id="shown"></span></div><section id="pairs"></section><div class="pages" id="pages"></div>
</main><script>
const DATA={data},ROOT={root},COLORS={colors},KEY='0910_retained_temporal_pair_audit_v2';let review=JSON.parse(localStorage.getItem(KEY)||'{{}}'),page=1;const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),save=()=>localStorage.setItem(KEY,JSON.stringify(review));
function boxes(image){{return DATA.annotations[image].boxes.map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"><title>${{esc(DATA.class_names[b.c]??b.c)}}</title></rect>`}}).join('')}}
function pairRows(){{let out=DATA.pairs.slice(),s=$('status').value,scope=$('scope').value,b=$('batch').value,q=$('search').value.trim().toLowerCase();if(s==='unreviewed')out=out.filter(r=>!review['edge:'+r.rank]?.status);else if(s!=='all')out=out.filter(r=>review['edge:'+r.rank]?.status===s);if(scope==='same_timestamp')out=out.filter(r=>r.same_timestamp);if(scope==='gap1')out=out.filter(r=>r.frame_gap===1);if(scope==='cross_split')out=out.filter(r=>r.cross_split);if(scope==='both_positive')out=out.filter(r=>r.both_positive);if(scope==='either_positive')out=out.filter(r=>r.either_positive);if(b!=='all')out=out.filter(r=>r.batch===b);if(q)out=out.filter(r=>r.left.toLowerCase().includes(q)||r.right.toLowerCase().includes(q)||r.chain_id.toLowerCase().includes(q));const mode=$('sort').value;if(mode==='timestamp_first'||mode==='size')out.sort((a,b)=>Number(b.same_timestamp)-Number(a.same_timestamp)||a.frame_gap-b.frame_gap||b.similarity-a.similarity);if(mode==='gap')out.sort((a,b)=>a.frame_gap-b.frame_gap||b.similarity-a.similarity);if(mode==='similarity')out.sort((a,b)=>b.similarity-a.similarity||a.frame_gap-b.frame_gap);return out}}
function allChains(){{const grouped=new Map;for(const edge of DATA.pairs){{if(!edge.chain_id)continue;if(!grouped.has(edge.chain_id))grouped.set(edge.chain_id,[]);grouped.get(edge.chain_id).push(edge)}}return[...grouped].map(([id,edges])=>{{edges.sort((a,b)=>a.rank-b.rank);return{{id,edges,size:edges.length+1,members:[edges[0].left,...edges.map(e=>e.right)],batch:edges[0].batch,same_timestamp:edges.some(e=>e.same_timestamp),cross_split:edges.some(e=>e.cross_split),both_positive:edges.some(e=>e.both_positive),either_positive:edges.some(e=>e.either_positive),similarity:Math.max(...edges.map(e=>e.similarity)),frame_gap:1}}}})}}
function chainRows(){{let out=allChains(),s=$('status').value,scope=$('scope').value,b=$('batch').value,q=$('search').value.trim().toLowerCase();if(s==='unreviewed')out=out.filter(r=>!review['chain:'+r.id]?.status);else if(s!=='all')out=out.filter(r=>review['chain:'+r.id]?.status===s);if(scope==='same_timestamp')out=out.filter(r=>r.same_timestamp);if(scope==='cross_split')out=out.filter(r=>r.cross_split);if(scope==='both_positive')out=out.filter(r=>r.both_positive);if(scope==='either_positive')out=out.filter(r=>r.either_positive);if(b!=='all')out=out.filter(r=>r.batch===b);if(q)out=out.filter(r=>r.id.toLowerCase().includes(q)||r.members.some(x=>x.toLowerCase().includes(q)));const mode=$('sort').value;if(mode==='size')out.sort((a,b)=>b.size-a.size||a.id.localeCompare(b.id));if(mode==='timestamp_first')out.sort((a,b)=>Number(b.same_timestamp)-Number(a.same_timestamp)||b.size-a.size);if(mode==='similarity')out.sort((a,b)=>b.similarity-a.similarity||b.size-a.size);return out}}
function tile(image){{const a=DATA.annotations[image],names=a.classes.map(c=>DATA.class_names[c]).join(' / ')||'背景';return `<div class="tile"><div class="figure"><img loading="lazy" src="${{ROOT+'/'+image}}" alt="${{esc(image)}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${{boxes(image)}}</svg></div><div class="caption">${{esc(image)}}<br>${{esc(names)}} · ${{a.label_count}} 个框</div></div>`}}
function choices(r,chain){{return `<button class="choice ${{r.status==='near_duplicate'?'active':''}}" data-status="near_duplicate">${{chain?'整链近重复':'连续帧 / 近重复'}}</button><button class="choice ${{r.status==='same_event_distinct'?'active':''}}" data-status="same_event_distinct">${{chain?'整链同事件但应保留':'同事件但应保留'}}</button><button class="choice ${{r.status==='different'?'active':''}}" data-status="different">${{chain?'链内有断点':'不同事件'}}</button><button class="choice ${{r.status==='unsure'?'active':''}}" data-status="unsure">不确定</button>`}}
function pairCard(row){{const key='edge:'+row.rank,r=review[key]||{{status:'',note:''}},chain=row.chain_id?` · 连续链 ${{row.chain_id}}（${{row.chain_size}} 张）`:'';return `<article class="pair ${{row.same_timestamp?'exact':''}}" data-key="${{key}}"><div class="head"><strong>#${{row.rank}} · 帧距 ${{row.frame_gap}} · ${{esc(row.batch)}} · ${{esc(row.left_split)}}/${{esc(row.right_split)}}${{row.same_timestamp?' · 同时间戳':''}}${{chain}}</strong><span class="score">similarity=${{row.similarity.toFixed(8)}}</span></div><div class="images">${{tile(row.left)}}${{tile(row.right)}}</div><div class="review">${{choices(r,false)}}<textarea placeholder="备注">${{esc(r.note||'')}}</textarea></div></article>`}}
function chainCard(row){{const key='chain:'+row.id,r=review[key]||{{status:'',note:''}},splits=[...new Set(row.edges.flatMap(e=>[e.left_split,e.right_split]))].sort().join('/');return `<article class="chain-card" data-key="${{key}}"><div class="head"><strong>${{row.id}} · 共 ${{row.size}} 张 · ${{esc(row.batch)}} · frame ${{row.members[0].split('/').pop().split('_')[0]}}–${{row.members.at(-1).split('/').pop().split('_')[0]}} · split=${{esc(splits)}}</strong><span class="score">最高相似度=${{row.similarity.toFixed(8)}}</span></div><div class="chain-images">${{row.members.map(tile).join('')}}</div><div class="review">${{choices(r,true)}}<textarea placeholder="如果链内有断点，请备注大致帧号，然后切换到两图视图标具体边界">${{esc(r.note||'')}}</textarea></div></article>`}}
function render(){{const chainMode=$('view').value==='chains',all=chainMode?chainRows():pairRows(),size=chainMode?4:20,total=Math.max(1,Math.ceil(all.length/size));page=Math.min(page,total);$('pairs').innerHTML=all.slice((page-1)*size,page*size).map(chainMode?chainCard:pairCard).join('');$('shown').textContent=`显示 ${{all.length}} ${{chainMode?'条链':'对'}} · 第 ${{page}}/${{total}} 页`;$('reviewed').textContent=Object.values(review).filter(x=>x.status).length;$('pages').innerHTML='';for(let i=1;i<=total;i++){{if(total>14&&i>2&&i<total-1&&Math.abs(i-page)>1)continue;const b=document.createElement('button');b.textContent=i;b.setAttribute('aria-current',i===page?'page':'false');b.onclick=()=>{{page=i;render();scrollTo({{top:0,behavior:'smooth'}})}};$('pages').appendChild(b)}}}}
for(const id of ['view','status','scope','batch','sort'])$(id).onchange=()=>{{page=1;render()}};$('search').oninput=()=>{{page=1;render()}};$('pairs').onclick=e=>{{const owner=e.target.closest('[data-key]');if(!owner)return;const choice=e.target.closest('.choice');if(choice){{const key=owner.dataset.key;review[key]=review[key]||{{status:'',note:''}};review[key].status=choice.dataset.status;save();render();return}}const tile=e.target.closest('.tile');if(tile)tile.classList.toggle('big')}};$('pairs').oninput=e=>{{if(e.target.tagName!=='TEXTAREA')return;const key=e.target.closest('[data-key]').dataset.key;review[key]=review[key]||{{status:'',note:''}};review[key].note=e.target.value;save()}};
$('export').onclick=()=>{{const q=v=>'"'+String(v??'').replaceAll('"','""')+'"',out=[['record_type','record_id','chain_id','chain_size','left','right','frame_gap','similarity','same_timestamp','left_split','right_split','status','note']];for(const chain of allChains()){{const r=review['chain:'+chain.id]||{{}};out.push(['chain',chain.id,chain.id,chain.size,chain.members[0],chain.members.at(-1),'','','','','',r.status||'',r.note||''])}}for(const row of DATA.pairs){{const r=review['edge:'+row.rank]||{{}};out.push(['edge',row.rank,row.chain_id,row.chain_size,row.left,row.right,row.frame_gap,row.similarity,row.same_timestamp?1:0,row.left_split,row.right_split,r.status||'',r.note||''])}}const blob=new Blob(['\ufeff'+out.map(r=>r.map(q).join(',')).join('\\n')],{{type:'text/csv;charset=utf-8'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='retained_temporal_chain_manual_review.csv';a.click();URL.revokeObjectURL(a.href)}};$('clear').onclick=()=>{{if(confirm('清空当前浏览器保存的全部审阅？')){{review={{}};save();render()}}}};$('batch').innerHTML='<option value="all">全部 batch</option>'+[...new Set(DATA.pairs.map(r=>r.batch))].sort().map(b=>`<option>${{esc(b)}}</option>`).join('');render();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    variant_dir = args.variant_dir.expanduser().resolve()
    embeddings_path = args.embeddings.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.max_frame_gap < 1:
        raise ValueError("max-frame-gap must be positive")
    dataset_root = (PROJECT_ROOT / "data" / "self_improving").resolve()
    images = read_manifest(variant_dir / "manifest.txt")
    archive = np.load(embeddings_path, allow_pickle=True)
    embeddings = archive["embeddings"].astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    archive_indices = {image_key(path): index for index, path in enumerate(archive["paths"].tolist())}

    assignments_path = STUDY_ROOT / "experiment" / "variants" / "random_stratified_s42_8_1_1" / "assignments.csv"
    with assignments_path.open(encoding="utf-8", newline="") as handle:
        assignments = {row["image"]: row["split"] for row in csv.DictReader(handle)}

    annotations = {}
    for image in images:
        relative = image.relative_to(dataset_root).as_posix()
        annotations[relative] = label_payload(image, dataset_root)

    chains: list[list[Path]] = []
    current_chain = [images[0]]
    for image in images[1:]:
        previous = current_chain[-1]
        if image.parent == previous.parent and frame(image) - frame(previous) == 1:
            current_chain.append(image)
        else:
            chains.append(current_chain)
            current_chain = [image]
    chains.append(current_chain)
    chain_info: dict[Path, tuple[str, int]] = {}
    for index, chain in enumerate(chains, start=1):
        chain_id = f"gap1_{index:04d}"
        for image in chain:
            chain_info[image] = (chain_id, len(chain))

    pairs = []
    for left_path, right_path in zip(images, images[1:]):
        gap = frame(right_path) - frame(left_path)
        if gap > args.max_frame_gap:
            continue
        left = left_path.relative_to(dataset_root).as_posix()
        right = right_path.relative_to(dataset_root).as_posix()
        left_index = archive_indices[image_key(left)]
        right_index = archive_indices[image_key(right)]
        left_positive = bool(annotations[left]["label_count"])
        right_positive = bool(annotations[right]["label_count"])
        left_chain_id, left_chain_size = chain_info[left_path]
        right_chain_id, _ = chain_info[right_path]
        pair_chain_id = left_chain_id if left_chain_id == right_chain_id else ""
        pairs.append(
            {
                "rank": len(pairs) + 1,
                "left": left,
                "right": right,
                "frame_gap": gap,
                "similarity": float(embeddings[left_index] @ embeddings[right_index]),
                "same_timestamp": source_timestamp(left_path) == source_timestamp(right_path),
                "chain_id": pair_chain_id,
                "chain_size": left_chain_size if pair_chain_id else 0,
                "batch": left_path.parent.name if left_path.parent == right_path.parent else f"{left_path.parent.name}/{right_path.parent.name}",
                "left_split": assignments[left],
                "right_split": assignments[right],
                "cross_split": assignments[left] != assignments[right],
                "both_positive": left_positive and right_positive,
                "either_positive": left_positive or right_positive,
            }
        )
    payload = {
        "retained_images": len(images),
        "pair_count": len(pairs),
        "gap_one_pairs": sum(row["frame_gap"] == 1 for row in pairs),
        "same_timestamp_pairs": sum(row["same_timestamp"] for row in pairs),
        "cross_split_pairs": sum(row["cross_split"] for row in pairs),
        "multi_image_chains": sum(len(chain) > 1 for chain in chains),
        "images_in_multi_image_chains": sum(len(chain) for chain in chains if len(chain) > 1),
        "max_chain_size": max(map(len, chains)),
        "class_names": CLASS_NAMES,
        "pairs": pairs,
        "annotations": annotations,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(os.path.relpath(dataset_root, output_dir)).as_posix()
    (output_dir / "temporal_pair_audit.html").write_text(page(payload, image_root), encoding="utf-8", newline="\n")
    similarity_bands = Counter(
        "<0.95" if row["similarity"] < 0.95 else "0.95-0.98" if row["similarity"] < 0.98 else "0.98-0.99"
        for row in pairs
        if row["frame_gap"] == 1
    )
    summary = {
        "status": "completed",
        "scope": f"consecutive retained neighbors with global frame gap <= {args.max_frame_gap}",
        "retained_images": len(images),
        "pair_count": len(pairs),
        "gap_one_pairs": payload["gap_one_pairs"],
        "same_timestamp_pairs": payload["same_timestamp_pairs"],
        "cross_split_pairs": payload["cross_split_pairs"],
        "multi_image_chains": payload["multi_image_chains"],
        "images_in_multi_image_chains": payload["images_in_multi_image_chains"],
        "max_chain_size": payload["max_chain_size"],
        "gap_one_similarity_bands": dict(sorted(similarity_bands.items())),
        "html": "temporal_pair_audit.html",
        "browser_export": "retained_temporal_chain_manual_review.csv",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
