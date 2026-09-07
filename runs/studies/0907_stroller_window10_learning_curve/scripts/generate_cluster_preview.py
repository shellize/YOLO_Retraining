"""Generate a self-contained browser preview for adjacent redundancy clusters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


THRESHOLDS = (0.980, 0.990)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def threshold_key(value: float) -> str:
    return f"{value:.3f}"


def variant_label(value: float) -> str:
    return threshold_key(value).replace(".", "p")


def load_annotations(layout_path: Path) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
    # The layout is deliberately resolved without importing the training registry.
    layout_payload = yaml.safe_load(layout_path.read_text(encoding="utf-8")) or {}
    root_value = Path(layout_payload.get("path", layout_path.parent))
    dataset_root = (layout_path.parent / root_value).resolve() if not root_value.is_absolute() else root_value.resolve()
    classes_path = dataset_root / "classes.txt"
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    images_dir = dataset_root / "images"
    annotations: dict[str, dict[str, Any]] = {}
    groups = [str(group) for group in layout_payload.get("groups", {})]
    for image in sorted(images_dir.iterdir(), key=lambda item: item.name.casefold()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        relative = image.relative_to(dataset_root).as_posix()
        label_path = dataset_root / "labels" / f"{image.stem}.txt"
        boxes: list[dict[str, float | int]] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                boxes.append({"c": int(fields[0]), "x": float(fields[1]), "y": float(fields[2]), "w": float(fields[3]), "h": float(fields[4])})
        annotations[relative] = {"status": "labeled" if boxes else "empty", "boxes": boxes}
    return classes, annotations, groups


def build_payload(analysis_dir: Path, layout_path: Path) -> dict[str, Any]:
    classes, annotations, groups = load_annotations(layout_path)
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    datasets: dict[str, Any] = {}
    for value in THRESHOLDS:
        key = threshold_key(value)
        variant = analysis_dir / "variants" / f"dedup_tau_{variant_label(value)}"
        variant_summary = summary.get("thresholds", {}).get(str(value), summary.get("thresholds", {}).get(key, {}))
        datasets[key] = {
            "summary": variant_summary,
            "clusters": read_csv(variant / "clusters.csv"),
            "mixedClusters": read_csv(variant / "mixed_clusters.csv"),
            "mixedMembers": read_csv(variant / "mixed_cluster_members.csv"),
        }
    return {"classes": classes, "groups": groups, "annotations": annotations, "thresholds": datasets}


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>stroller_reviewed 邻近冗余度预览</title>
  <style>
    :root{--bg:#f5f7fb;--surface:#fff;--soft:#eef2f7;--ink:#172033;--muted:#647087;--line:#d9e0eb;--accent:#315efb;--green:#15803d;--orange:#b45309;--shadow:0 10px 28px rgba(32,50,86,.08)}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}a{color:inherit}.app{max-width:1680px;margin:0 auto;padding:24px}header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:20px}h1{margin:0 0 5px;font-size:25px}.subtitle{color:var(--muted)}.links{display:flex;flex-wrap:wrap;gap:8px}.links a,.page{border:1px solid var(--line);background:var(--surface);border-radius:8px;padding:7px 10px;text-decoration:none}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:12px;margin-bottom:18px}.metric{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:13px 15px;box-shadow:var(--shadow)}.metric small{color:var(--muted)}.metric strong{display:block;margin-top:2px;font-size:23px}.workspace{display:grid;grid-template-columns:270px minmax(0,1fr);gap:18px;align-items:start}aside{position:sticky;top:14px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:15px;box-shadow:var(--shadow)}.control{margin-bottom:14px}label{display:block;margin-bottom:5px;color:var(--muted);font-size:12px}input,select{width:100%;border:1px solid var(--line);border-radius:8px;padding:9px 10px;background:var(--surface);color:var(--ink);font:inherit}.note{color:var(--muted);font-size:12px;margin:-3px 0 14px}.stage{display:grid;gap:7px;border-top:1px solid var(--line);padding-top:12px}.stage div{display:flex;justify-content:space-between;background:var(--soft);padding:8px;border-radius:8px}.legend{border-top:1px solid var(--line);margin-top:15px;padding-top:12px;color:var(--muted);font-size:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--green)}main{min-width:0}.bar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}.count{color:var(--muted)}.pagination{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.page{cursor:pointer;font:inherit;padding:6px 9px}.page[aria-current=page]{background:var(--accent);border-color:var(--accent);color:#fff}.page:disabled{opacity:.45;cursor:default}#grid{display:grid;gap:10px}.card{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}.head{display:flex;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line)}.head strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.meta{color:var(--muted);font-size:12px;white-space:nowrap}.members{display:flex;flex-wrap:wrap;gap:8px;padding:11px 12px}.tile{flex:0 0 190px;width:190px}.tile.tile-enlarged{flex-basis:570px;width:570px}.thumb{position:relative;display:block;width:100%;overflow:hidden;border:0;border-radius:6px;padding:0;background:var(--soft);color:inherit;font:inherit;text-align:left;cursor:zoom-in}.thumb img{width:100%;height:auto;display:block;object-fit:contain}.thumb.is-enlarged{cursor:zoom-out}.thumb.rep{outline:3px solid var(--green);outline-offset:-3px}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible}.box{fill:#2563eb;fill-opacity:.08;stroke:#2563eb;stroke-width:.006}.tag{position:absolute;left:5px;bottom:5px;padding:2px 5px;border-radius:4px;background:var(--green);color:#fff;font-size:10px}.caption{padding:5px 7px;color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.empty{background:var(--surface);border:1px dashed var(--line);border-radius:12px;padding:35px;text-align:center;color:var(--muted)}@media(max-width:820px){.app{padding:15px}header{display:block}.links{margin-top:12px}.workspace{grid-template-columns:1fr}aside{position:static}.tile{flex-basis:160px;width:160px}.tile.tile-enlarged{flex-basis:480px;width:480px}}
  </style>
</head>
<body><div class="app">
  <header><div><h1 id="title">stroller_reviewed 邻近冗余度预览</h1><div class="subtitle">同一 images 目录内按文件名排序，temporal-window=1；绿色边框为每簇代表图，点击缩略图可放大。</div></div><nav class="links"><a href="./summary.json" target="_blank">summary.json</a><a href="./temporal_similarity_pairs.csv" target="_blank">相似度边</a><a href="./run_metadata.json" target="_blank">run metadata</a></nav></header>
  <section class="summary"><div class="metric"><small>阈值 τ</small><strong id="tau">—</strong></div><div class="metric"><small>图片总数</small><strong id="images">—</strong></div><div class="metric"><small>簇数量</small><strong id="clusters">—</strong></div><div class="metric"><small>代表保留率</small><strong id="retention">—</strong></div><div class="metric"><small>最大簇大小</small><strong id="maxsize">—</strong></div></section>
  <div class="workspace"><aside><div class="control"><label for="threshold">聚类阈值 τ</label><select id="threshold"></select></div><div class="note">τ 越高，聚类条件越严格。两个阈值只用于并列审计，不代表已选定训练阈值。</div><div class="control"><label for="group">数据组</label><select id="group"><option value="all">全部</option></select></div><div class="control"><label for="size">聚类大小</label><select id="size"><option value="all">全部</option><option value="1">单图</option><option value="2">2 张</option><option value="3">3 张</option><option value="4">4 张以上</option></select></div><div class="control"><label for="sort">排序</label><select id="sort"><option value="score-asc">最低相似度：低到高</option><option value="size-desc">聚类大小：大到小</option><option value="score-desc">最低相似度：高到低</option><option value="id-asc">聚类编号</option></select></div><div class="control"><label for="search">搜索文件名或簇编号</label><input id="search" type="search" placeholder="例如 000123 或 reviewed_000001"></div><div class="stage" id="stage"></div><div class="legend"><span class="dot"></span>代表图 · 所有成员均来自同一 reviewed 图片池 · 当前框线来自 labels/ 下的 YOLO TXT</div></aside><main><div class="bar"><div class="count" id="count">正在加载…</div><div class="pagination" id="pagination"></div></div><section id="grid"></section><div id="empty" class="empty" hidden>没有符合筛选条件的聚类。</div></main></div>
</div><script src="./cluster_preview_data.js"></script><script>
(() => { const DATA=window.__CLUSTER_DATA__, RAW_ROOT="../../../../../data/stroller/", PAGE_SIZE=12, state={threshold:"",rows:[],filtered:[],page:1}; const $=id=>document.getElementById(id), fmt=new Intl.NumberFormat("zh-CN"); const esc=x=>String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); const imgHref=x=>RAW_ROOT+x; const variant=t=>`./variants/dedup_tau_${t.replace(".","p")}`;
function groupRows(rows){const map=new Map(); for(const r of rows){let c=map.get(r.cluster_id);if(!c){c={id:r.cluster_id,group:r.group,size:Number(r.cluster_size),members:[]};map.set(r.cluster_id,c)} c.members.push({image:r.image,rep:r.representative,score:Number(r.score_to_representative)})} return [...map.values()].map(c=>{c.members.sort((a,b)=>a.image.localeCompare(b.image));c.representative=c.members.find(x=>x.image===x.rep)||c.members[0];c.minScore=Math.min(...c.members.map(x=>x.score));return c})}
function overlay(image){const info=DATA.annotations[image]||{}, boxes=info.boxes||[];return boxes.map(b=>{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h);return `<rect class="box" x="${x.toFixed(6)}" y="${y.toFixed(6)}" width="${Math.max(0,w).toFixed(6)}" height="${Math.max(0,h).toFixed(6)}"><title>stroller</title></rect>`}).join("")}
function update(){const d=DATA.thresholds[state.threshold],s=d.summary||{},rows=state.rows;$("tau").textContent=state.threshold;$("images").textContent=fmt.format(Number(s.original_train_images||rows.reduce((n,c)=>n+c.size,0)));$("clusters").textContent=fmt.format(rows.length);$("retention").textContent=s.retained_fraction==null?((rows.length/(Number(s.original_train_images)||1))*100).toFixed(1)+"%":(Number(s.retained_fraction)*100).toFixed(1)+"%";$("maxsize").textContent=fmt.format(Number(s.max_cluster_size||Math.max(...rows.map(c=>c.size),0))); const groups=[...new Set(rows.map(c=>c.group))].sort();$("stage").innerHTML=groups.map(g=>{const a=rows.filter(c=>c.group===g);return `<div><span>${esc(g)}</span><strong>${fmt.format(a.length)} / ${fmt.format(a.reduce((n,c)=>n+c.size,0))}</strong></div>`}).join(""); apply()}
function apply(){const g=$("group").value,z=$("size").value,q=$("search").value.trim().toLowerCase(),sort=$("sort").value;state.filtered=state.rows.filter(c=>(g==="all"||c.group===g)&&(z==="all"||(z==="4"?c.size>=4:c.size===Number(z)))&&(!q||c.id.toLowerCase().includes(q)||c.members.some(m=>m.image.toLowerCase().includes(q))));state.filtered.sort((a,b)=>sort==="score-asc"?a.minScore-b.minScore||a.id.localeCompare(b.id):sort==="score-desc"?b.minScore-a.minScore||a.id.localeCompare(b.id):sort==="size-desc"?b.size-a.size||a.id.localeCompare(b.id):a.id.localeCompare(b.id));state.page=1;render()}
function card(c){const thumbs=c.members.map(m=>{const rep=m.image===c.representative.image;return `<div class="tile"><button type="button" class="thumb ${rep?"rep":""}" data-preview-image="${imgHref(m.image)}" aria-pressed="false" aria-label="${esc(m.image)} · 相似度 ${m.score.toFixed(6)}"><img src="${imgHref(m.image)}" loading="lazy" alt="${esc(m.image)}"><svg class="overlay" viewBox="0 0 1 1" preserveAspectRatio="none">${overlay(m.image)}</svg>${rep?'<span class="tag">代表图</span>':''}</button><div class="caption">${esc(m.image.split("/").pop())}<br>sim=${m.score.toFixed(6)}</div></div>`}).join("");return `<article class="card"><div class="head"><strong>${esc(c.id)}</strong><span class="meta">${esc(c.group)} · ${c.size} 张 · min sim ${c.minScore.toFixed(6)}</span></div><div class="members">${thumbs}</div></article>`}
function toggleThumbnail(button){const tile=button.closest(".tile"),enlarged=!button.classList.contains("is-enlarged");button.classList.toggle("is-enlarged",enlarged);tile.classList.toggle("tile-enlarged",enlarged);button.setAttribute("aria-pressed",String(enlarged))}
$("grid").addEventListener("click",event=>{const button=event.target.closest("button[data-preview-image]");if(!button)return;event.preventDefault();toggleThumbnail(button)})
function pages(){const total=Math.max(1,Math.ceil(state.filtered.length/PAGE_SIZE)),p=state.page,root=$("pagination");root.innerHTML="";const add=(label,n,disabled=false,current=false)=>{const b=document.createElement("button");b.className="page";b.textContent=label;b.disabled=disabled;if(current)b.setAttribute("aria-current","page");b.onclick=()=>{state.page=n;render();scrollTo({top:0,behavior:"smooth"})};root.appendChild(b)};add("上一页",p-1,p===1);for(let n=1;n<=total;n++){if(total>9&&n>2&&n<total-1&&Math.abs(n-p)>1){if(n===3||n===total-2){const s=document.createElement("span");s.textContent="…";root.appendChild(s)}continue}add(String(n),n,false,n===p)}add("下一页",p+1,p===total)}
function render(){const totalPages=Math.max(1,Math.ceil(state.filtered.length/PAGE_SIZE));state.page=Math.min(state.page,totalPages);const start=(state.page-1)*PAGE_SIZE,pageRows=state.filtered.slice(start,start+PAGE_SIZE);$("count").textContent=state.filtered.length?`τ=${state.threshold} · 第 ${state.page}/${totalPages} 页 · 当前 ${pageRows.length} 个簇`:"没有符合筛选条件的聚类";$("grid").innerHTML=pageRows.map(card).join("");$("empty").hidden=state.filtered.length!==0;pages()}
for(const id of ["group","size","sort"]){$(id).onchange=apply}$("search").oninput=apply;$("threshold").onchange=()=>{state.threshold=$("threshold").value;state.rows=groupRows(DATA.thresholds[state.threshold].clusters);update()}; const keys=Object.keys(DATA.thresholds).sort();$("threshold").innerHTML=keys.map(k=>`<option value="${k}">τ=${k}</option>`).join("");const groups=DATA.groups||[];$("group").innerHTML='<option value="all">全部</option>'+groups.map(g=>`<option value="${esc(g)}">${esc(g)}</option>`).join("");state.threshold=keys[0];state.rows=groupRows(DATA.thresholds[state.threshold].clusters);update();
})();
</script></body></html>
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    layout_path = args.layout.expanduser().resolve()
    payload = build_payload(analysis_dir, layout_path)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (analysis_dir / "cluster_preview_data.js").write_text(f"window.__CLUSTER_DATA__ = {serialized};\n", encoding="utf-8")
    (analysis_dir / "cluster_preview.html").write_text(HTML, encoding="utf-8")
    print(json.dumps({"html": str(analysis_dir / "cluster_preview.html"), "data": str(analysis_dir / "cluster_preview_data.js"), "thresholds": list(payload["thresholds"]), "groups": payload["groups"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
