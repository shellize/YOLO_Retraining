from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from generate_positive_box_layout_pair_audit import CLASS_NAMES, frame, read_boxes, read_manifest


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_VARIANT = STUDY_ROOT / "experiment" / "variants" / "global_order_window20_tau0p990_label_aware_pair_reviewed_final"
DEFAULT_OUTPUT = STUDY_ROOT / "result" / "target_track_candidate_audit"


@dataclass
class Node:
    image: str
    sequence_index: int
    frame: int
    box_index: int
    class_id: int
    x: float
    y: float
    w: float
    h: float
    tight: np.ndarray
    context: np.ndarray


@dataclass
class Track:
    nodes: list[Node] = field(default_factory=list)
    link_scores: list[float] = field(default_factory=list)

    @property
    def class_id(self) -> int:
        return self.nodes[0].class_id

    @property
    def last_index(self) -> int:
        return self.nodes[-1].sequence_index

    def prototype(self, attribute: str) -> np.ndarray:
        values = np.stack([getattr(node, attribute) for node in self.nodes])
        result = values.mean(axis=0)
        norm = np.linalg.norm(result)
        return result / norm if norm else result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-track candidates in the post-dedup sequence.")
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-grouped-images", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--min-tight-similarity", type=float, default=0.55)
    parser.add_argument("--max-center-distance", type=float, default=0.20)
    parser.add_argument("--max-size-distance", type=float, default=0.22)
    parser.add_argument("--tight-weight", type=float, default=0.60)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--geometry-weight", type=float, default=0.15)
    return parser.parse_args()


def crop(image: np.ndarray, box: dict[str, float | int], padding: float) -> np.ndarray:
    height, width = image.shape[:2]
    box_width = float(box["w"]) * (1 + 2 * padding)
    box_height = float(box["h"]) * (1 + 2 * padding)
    x1 = max(0, int(round((float(box["x"]) - box_width / 2) * width)))
    y1 = max(0, int(round((float(box["y"]) - box_height / 2) * height)))
    x2 = min(width, int(round((float(box["x"]) + box_width / 2) * width)))
    y2 = min(height, int(round((float(box["y"]) + box_height / 2) * height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty target crop")
    return image[y1:y2, x1:x2]


def descriptor(value: np.ndarray) -> np.ndarray:
    resized = cv2.resize(value, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(gray).reshape(-1)
    histogram /= max(float(np.linalg.norm(histogram)), 1e-12)
    hog /= max(float(np.linalg.norm(hog)), 1e-12)
    combined = np.concatenate((histogram * math.sqrt(0.45), hog * math.sqrt(0.55))).astype(np.float32)
    combined /= max(float(np.linalg.norm(combined)), 1e-12)
    return combined


def appearance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(left @ right, 0.0, 1.0))


def match_score(track: Track, node: Node, *, tight_weight: float, context_weight: float, geometry_weight: float) -> tuple[float, float, float, float]:
    last = track.nodes[-1]
    center_distance = math.hypot(node.x - last.x, node.y - last.y)
    size_distance = math.hypot(node.w - last.w, node.h - last.h)
    tight_prototype = appearance(track.prototype("tight"), node.tight)
    tight_last = appearance(last.tight, node.tight)
    tight_similarity = 0.6 * tight_prototype + 0.4 * tight_last
    context_similarity = 0.6 * appearance(track.prototype("context"), node.context) + 0.4 * appearance(last.context, node.context)
    geometry = math.exp(-(center_distance / 0.10 + size_distance / 0.12))
    score = tight_weight * tight_similarity + context_weight * context_similarity + geometry_weight * geometry
    return score, tight_similarity, center_distance, size_distance


def build_tracks(nodes_by_index: dict[int, list[Node]], *, window: int, min_score: float, min_tight: float, max_center: float, max_size: float, tight_weight: float, context_weight: float, geometry_weight: float) -> list[Track]:
    tracks: list[Track] = []
    for sequence_index in sorted(nodes_by_index):
        image_nodes = nodes_by_index[sequence_index]
        for class_id in sorted({node.class_id for node in image_nodes}):
            nodes = [node for node in image_nodes if node.class_id == class_id]
            active = [track for track in tracks if track.class_id == class_id and sequence_index - track.last_index <= window]
            if not active:
                tracks.extend(Track(nodes=[node]) for node in nodes)
                continue
            scores = np.full((len(active), len(nodes)), -1.0, dtype=np.float64)
            details: dict[tuple[int, int], tuple[float, float, float, float]] = {}
            for row, track in enumerate(active):
                for column, node in enumerate(nodes):
                    detail = match_score(track, node, tight_weight=tight_weight, context_weight=context_weight, geometry_weight=geometry_weight)
                    details[row, column] = detail
                    score, tight, center, size = detail
                    if score >= min_score and tight >= min_tight and center <= max_center and size <= max_size:
                        scores[row, column] = score
            rows, columns = linear_sum_assignment(-scores)
            matched_nodes: set[int] = set()
            for row, column in zip(rows.tolist(), columns.tolist()):
                if scores[row, column] < 0:
                    continue
                active[row].nodes.append(nodes[column])
                active[row].link_scores.append(float(scores[row, column]))
                matched_nodes.add(column)
            tracks.extend(Track(nodes=[node]) for column, node in enumerate(nodes) if column not in matched_nodes)
    return [track for track in tracks if len(track.nodes) >= 2]


def html(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    root = json.dumps(image_root, ensure_ascii=False)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>目标轨迹确认</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#667085;--line:#d6deea;--blue:#2563eb;--green:#15803d;--red:#b42318;--orange:#c2410c;--shadow:0 8px 24px rgba(32,50,86,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1720px;margin:auto;padding:22px}}h1{{margin:0 0 5px}}.muted{{color:var(--muted)}}.summary,.toolbar,.crops,.review,.pages{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:16px 0}}.metric,.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:150px;padding:9px 13px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:21px}}button,select{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px;cursor:pointer}}.toolbar{{position:sticky;top:0;z-index:5;background:rgba(244,247,251,.96);padding:10px 0}}#grid{{display:grid;gap:12px}}.card{{padding:12px}}.head{{display:flex;justify-content:space-between;gap:10px;margin-bottom:9px}}.crop{{width:220px;flex:0 0 220px}}canvas{{display:block;width:220px;height:180px;background:#e9eef5;border-radius:7px}}.caption{{font-size:11px;color:var(--muted);padding-top:4px;overflow-wrap:anywhere}}.review{{margin-top:10px;padding-top:9px;border-top:1px solid var(--line)}}.choice.active[data-status=same_track]{{background:var(--green);color:#fff}}.choice.active[data-status=not_same_track]{{background:var(--red);color:#fff}}.choice.active[data-status=unsure]{{background:var(--orange);color:#fff}}.pages{{margin:14px 0}}.pages button[aria-current=page]{{background:var(--blue);color:#fff}}@media(max-width:700px){{main{{padding:12px}}.crop{{width:100%;flex-basis:100%}}canvas{{width:100%}}}}
</style></head><body><main><h1>目标轨迹确认</h1><div class="muted">每张卡片是一条候选目标轨迹，图片按去重后序列位置排列。只需判断这些目标是否属于同一轨迹。</div><section class="summary"><div class="metric"><small>候选轨迹</small><strong>{payload['track_count']}</strong></div><div class="metric"><small>轨迹目标裁剪</small><strong>{payload['node_count']}</strong></div><div class="metric"><small>最长轨迹</small><strong>{payload['max_track_size']}</strong></div><div class="metric"><small>已审阅</small><strong id="reviewed">0</strong></div></section><div class="toolbar"><select id="status"><option value="all">全部</option><option value="unreviewed">未审阅</option><option value="same_track">同一轨迹</option><option value="not_same_track">不是同一轨迹</option><option value="unsure">不确定</option></select><button id="export">导出审阅 CSV</button><button id="clear">清空审阅</button><span class="muted" id="shown"></span></div><section id="grid"></section><div id="pages" class="pages"></div></main><script>
const DATA={data},ROOT={root},KEY='0910_target_track_candidate_review_v1',SIZE=10;let review=JSON.parse(localStorage.getItem(KEY)||'{{}}'),page=1;const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),save=()=>localStorage.setItem(KEY,JSON.stringify(review));function rows(){{const s=$('status').value;return DATA.tracks.filter(t=>s==='all'||(s==='unreviewed'?!review[t.id]:review[t.id]===s))}}function draw(canvas,node){{const image=new Image;image.onload=()=>{{const pad=.18,w=node.w*(1+2*pad),h=node.h*(1+2*pad),sx=Math.max(0,(node.x-w/2)*image.naturalWidth),sy=Math.max(0,(node.y-h/2)*image.naturalHeight),sw=Math.min(w*image.naturalWidth,image.naturalWidth-sx),sh=Math.min(h*image.naturalHeight,image.naturalHeight-sy),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,sx,sy,sw,sh,0,0,canvas.width,canvas.height)}};image.src=ROOT+'/'+node.image}}function card(track){{const value=review[track.id]||'';return `<article class="card" data-id="${{track.id}}"><div class="head"><strong>${{esc(track.id)}} · ${{esc(DATA.class_names[track.class_id]??track.class_id)}} · ${{track.nodes.length}} 张</strong></div><div class="crops">${{track.nodes.map((node,index)=>`<div class="crop"><canvas width="440" height="360" data-node="${{index}}"></canvas><div class="caption">${{esc(node.image)}}<br>位置 ${{node.sequence_index}} · 框 ${{node.box_index}}</div></div>`).join('')}}</div><div class="review"><button class="choice ${{value==='same_track'?'active':''}}" data-status="same_track">同一轨迹</button><button class="choice ${{value==='not_same_track'?'active':''}}" data-status="not_same_track">不是同一轨迹</button><button class="choice ${{value==='unsure'?'active':''}}" data-status="unsure">不确定</button></div></article>`}}function render(){{const all=rows(),total=Math.max(1,Math.ceil(all.length/SIZE));page=Math.min(page,total);const visible=all.slice((page-1)*SIZE,page*SIZE);$('grid').innerHTML=visible.map(card).join('');document.querySelectorAll('.card').forEach((card,index)=>card.querySelectorAll('canvas').forEach(canvas=>draw(canvas,visible[index].nodes[Number(canvas.dataset.node)])));$('reviewed').textContent=Object.keys(review).length;$('shown').textContent=`${{all.length}} 条 · 第 ${{page}}/${{total}} 页`;$('pages').innerHTML='';for(let i=1;i<=total;i++){{if(total>14&&i>2&&i<total-1&&Math.abs(i-page)>1)continue;const b=document.createElement('button');b.textContent=i;b.setAttribute('aria-current',i===page?'page':'false');b.onclick=()=>{{page=i;render();scrollTo({{top:0,behavior:'smooth'}})}};$('pages').appendChild(b)}}}}$('status').onchange=()=>{{page=1;render()}};$('grid').onclick=e=>{{const choice=e.target.closest('.choice');if(!choice)return;review[e.target.closest('.card').dataset.id]=choice.dataset.status;save();render()}};$('export').onclick=()=>{{const q=v=>'"'+String(v??'').replaceAll('"','""')+'"',out=[['track_id','class_id','class_name','node_count','images','sequence_indices','status']];for(const t of DATA.tracks)out.push([t.id,t.class_id,DATA.class_names[t.class_id]??t.class_id,t.nodes.length,t.nodes.map(n=>n.image+'#box'+n.box_index).join('|'),t.nodes.map(n=>n.sequence_index).join('|'),review[t.id]||'']);const blob=new Blob(['\ufeff'+out.map(r=>r.map(q).join(',')).join('\\n')],{{type:'text/csv;charset=utf-8'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='target_track_manual_review.csv';a.click();URL.revokeObjectURL(a.href)}};$('clear').onclick=()=>{{if(confirm('清空全部轨迹判断？')){{review={{}};save();render()}}}};render();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    if args.window < 1:
        raise ValueError("window must be positive")
    if not math.isclose(args.tight_weight + args.context_weight + args.geometry_weight, 1.0, abs_tol=1e-9):
        raise ValueError("tight/context/geometry weights must sum to 1")
    variant, output = args.variant_dir.resolve(), args.output_dir.resolve()
    dataset = (PROJECT_ROOT / "data" / "self_improving").resolve()
    images = read_manifest(variant / "manifest.txt")
    excluded_images: set[str] = set()
    if args.exclude_grouped_images:
        with args.exclude_grouped_images.resolve().open(encoding="utf-8-sig", newline="") as handle:
            excluded_images = {row["image"].replace("\\", "/").casefold() for row in csv.DictReader(handle)}
    nodes_by_index: dict[int, list[Node]] = defaultdict(list)
    object_count = 0
    for sequence_index, image_path in enumerate(images):
        boxes = read_boxes(image_path, dataset)
        if not boxes:
            continue
        relative = image_path.relative_to(dataset).as_posix()
        if relative.casefold() in excluded_images:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")
        for box_index, box in enumerate(boxes):
            nodes_by_index[sequence_index].append(Node(relative, sequence_index, frame(image_path), box_index, int(box["c"]), float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"]), descriptor(crop(image, box, 0.03)), descriptor(crop(image, box, 0.22))))
            object_count += 1
    tracks = build_tracks(nodes_by_index, window=args.window, min_score=args.min_score, min_tight=args.min_tight_similarity, max_center=args.max_center_distance, max_size=args.max_size_distance, tight_weight=args.tight_weight, context_weight=args.context_weight, geometry_weight=args.geometry_weight)
    tracks.sort(key=lambda track: (-min(track.link_scores), -len(track.nodes), track.nodes[0].sequence_index))
    rows = []
    for index, track in enumerate(tracks, start=1):
        rows.append({"id": f"track_{index:04d}", "class_id": track.class_id, "track_score": min(track.link_scores), "nodes": [{"image": node.image, "sequence_index": node.sequence_index, "frame": node.frame, "box_index": node.box_index, "x": node.x, "y": node.y, "w": node.w, "h": node.h} for node in track.nodes]})
    payload = {"final_images": len(images), "labeled_objects": object_count, "track_count": len(rows), "node_count": sum(len(row["nodes"]) for row in rows), "max_track_size": max(len(row["nodes"]) for row in rows), "class_names": CLASS_NAMES, "tracks": rows}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_root = Path(os.path.relpath(dataset, output)).as_posix()
    browser_export = f"{output.name}_manual_review.csv"
    page = html(payload, image_root).replace("0910_target_track_candidate_review_v1", f"0910_{output.name}_review_v1").replace("target_track_manual_review.csv", browser_export)
    (output / "track_review.html").write_text(page, encoding="utf-8", newline="\n")
    summary = {"status": "completed", "source_variant": variant.name, "sequence_definition": "position in the complete post-dedup manifest", "window": args.window, "excluded_grouped_images": len(excluded_images), "feature": "tight/context HSV histogram plus HOG", "feature_weights": {"tight": args.tight_weight, "context": args.context_weight, "geometry": args.geometry_weight}, "track_score": "minimum accepted link score in the trajectory; tracks sorted descending", "matching": {"min_score": args.min_score, "min_tight_similarity": args.min_tight_similarity, "max_center_distance": args.max_center_distance, "max_size_distance": args.max_size_distance}, **{key: payload[key] for key in ("final_images", "labeled_objects", "track_count", "node_count", "max_track_size")}, "html": "track_review.html", "browser_export": browser_export}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
