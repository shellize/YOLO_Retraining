from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_VARIANT = (
    STUDY_ROOT
    / "experiment"
    / "variants"
    / "global_order_window20_tau0p990_label_aware_reviewed_final"
)
DEFAULT_ASSIGNMENTS = (
    STUDY_ROOT
    / "experiment"
    / "variants"
    / "random_stratified_s42_8_1_1"
    / "assignments.csv"
)
DEFAULT_EMBEDDINGS = (
    PROJECT_ROOT
    / "data_analyse"
    / "dataset_redundancy"
    / "results"
    / "20260824-182608"
    / "embeddings.npz"
)
DEFAULT_OUTPUT = STUDY_ROOT / "result" / "positive_box_layout_pair_audit"
CLASS_NAMES = ("large luggage", "stroller", "wheelchair", "flatbed truck")
CLASS_COLORS = ("#2563eb", "#16a34a", "#db2777", "#ea580c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank adjacent positive image pairs with equal box counts by box-layout distance."
    )
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-frame-gap", type=int, default=1)
    parser.add_argument("--max-frame-gap", type=int, default=1)
    parser.add_argument("--min-cosine", type=float)
    return parser.parse_args()


def frame(path: Path) -> int:
    value = path.stem.split("_", 1)[0]
    if not value.isdecimal():
        raise ValueError(f"image has no numeric frame prefix: {path.name}")
    return int(value)


def image_key(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    if normalized.casefold().startswith("images/"):
        return normalized.casefold()
    marker = "/images/"
    index = normalized.casefold().rfind(marker)
    if index < 0:
        raise ValueError(f"image path has no images directory: {path}")
    return normalized[index + 1 :].casefold()


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
    return sorted(images, key=lambda item: (frame(item), item.as_posix().casefold()))


def read_boxes(image: Path, dataset_root: Path) -> list[dict[str, float | int]]:
    relative = image.relative_to(dataset_root)
    label = dataset_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
    boxes: list[dict[str, float | int]] = []
    if not label.is_file():
        return boxes
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
    return boxes


def box_iou(left: dict[str, float | int], right: dict[str, float | int]) -> float:
    left_x1, left_y1 = float(left["x"]) - float(left["w"]) / 2, float(left["y"]) - float(left["h"]) / 2
    left_x2, left_y2 = float(left["x"]) + float(left["w"]) / 2, float(left["y"]) + float(left["h"]) / 2
    right_x1, right_y1 = float(right["x"]) - float(right["w"]) / 2, float(right["y"]) - float(right["h"]) / 2
    right_x2, right_y2 = float(right["x"]) + float(right["w"]) / 2, float(right["y"]) + float(right["h"]) / 2
    intersection = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1)) * max(
        0.0, min(left_y2, right_y2) - max(left_y1, right_y1)
    )
    union = float(left["w"]) * float(left["h"]) + float(right["w"]) * float(right["h"]) - intersection
    return intersection / union if union > 0 else 0.0


def layout_metrics(
    left_boxes: list[dict[str, float | int]], right_boxes: list[dict[str, float | int]]
) -> tuple[float, float, int]:
    """Match geometry as unordered sets; class labels do not influence the assignment."""
    left = np.asarray([[box["x"], box["y"], box["w"], box["h"]] for box in left_boxes], dtype=np.float64)
    right = np.asarray([[box["x"], box["y"], box["w"], box["h"]] for box in right_boxes], dtype=np.float64)
    center_distance = np.linalg.norm(left[:, None, :2] - right[None, :, :2], axis=2)
    size_distance = np.linalg.norm(left[:, None, 2:] - right[None, :, 2:], axis=2)
    cost = center_distance + 0.5 * size_distance
    left_indices, right_indices = linear_sum_assignment(cost)
    mean_cost = float(cost[left_indices, right_indices].mean())
    mean_iou = float(
        np.mean([box_iou(left_boxes[i], right_boxes[j]) for i, j in zip(left_indices, right_indices)])
    )
    class_mismatches = sum(left_boxes[i]["c"] != right_boxes[j]["c"] for i, j in zip(left_indices, right_indices))
    return mean_cost, mean_iou, class_mismatches


def class_signature(boxes: list[dict[str, float | int]]) -> str:
    counts = Counter(int(box["c"]) for box in boxes)
    return " + ".join(f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else class_id}×{count}" for class_id, count in sorted(counts.items()))


def html(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    root = json.dumps(image_root, ensure_ascii=False)
    colors = json.dumps(CLASS_COLORS)
    review_key = json.dumps(payload["review_key"], ensure_ascii=False)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>正样本框布局双图审阅</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#667085;--line:#d6deea;--blue:#2563eb;--red:#b42318;--green:#15803d;--orange:#c2410c;--shadow:0 10px 28px rgba(32,50,86,.09)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.48 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}h1{{margin:0 0 6px;font-size:25px}}.muted{{color:var(--muted)}}.summary,.toolbar,.review,.nav{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:16px 0}}.metric,.pair{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:160px;padding:9px 13px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:20px}}button,select,input,textarea{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px}}button{{cursor:pointer}}.toolbar{{position:sticky;top:0;z-index:5;background:rgba(244,247,251,.96);padding:11px 0;backdrop-filter:blur(8px)}}.pair{{padding:13px}}.head{{display:flex;justify-content:space-between;gap:10px;margin-bottom:9px}}.score{{color:var(--blue);font-weight:700}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:13px}}.figure{{position:relative;background:#e9eef5}}.figure img{{display:block;width:100%;height:auto}}.figure svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.box{{fill-opacity:.06;stroke-width:.005}}.caption{{font-size:12px;color:var(--muted);padding-top:5px;overflow-wrap:anywhere}}.review{{margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}}.choice.active[data-status=duplicate]{{background:var(--red);color:white}}.choice.active[data-status=not_duplicate]{{background:var(--green);color:white}}.choice.active[data-status=unsure]{{background:var(--orange);color:white}}textarea{{width:100%;min-height:46px}}.nav{{justify-content:center;margin:14px 0}}.nav strong{{min-width:120px;text-align:center}}@media(max-width:760px){{main{{padding:12px}}.images{{grid-template-columns:1fr}}.head{{display:block}}}}
</style></head><body><main><h1>正样本保留序列相邻图片 · 框布局双图审阅</h1><div class="muted">候选条件：{payload['candidate_rule_zh']}。框布局距离 = 最优一一匹配后的平均〔中心距离 + 0.5 × 宽高距离〕，坐标均已归一化；越小只表示框布局越接近，不等同于必然重复。类别不参与匹配，以免漏掉标注类别不一致的近重复帧。</div>
<section class="summary"><div class="metric"><small>候选图片对</small><strong>{payload['candidate_count']}</strong></div><div class="metric"><small>当前跨 split</small><strong>{payload['cross_split_count']}</strong></div><div class="metric"><small>布局距离范围</small><strong>{payload['distance_min']:.4f}–{payload['distance_max']:.4f}</strong></div><div class="metric"><small>已审阅</small><strong id="reviewed">0</strong></div></section>
<div class="toolbar"><select id="scope"><option value="all">全部候选</option><option value="cross">仅跨 split</option><option value="same">仅同 split</option></select><select id="status"><option value="all">全部状态</option><option value="unreviewed">未审阅</option><option value="duplicate">重复</option><option value="not_duplicate">不重复</option><option value="unsure">不确定</option></select><input id="maxDistance" type="number" min="0" step="0.001" placeholder="最大布局距离"><button id="export">导出审阅 CSV</button><button id="clear">清空审阅</button><span class="muted" id="shown"></span></div>
<section id="content"></section><div class="nav"><button id="prev">上一对</button><strong id="position"></strong><button id="next">下一对</button></div>
</main><script>
const DATA={data},ROOT={root},COLORS={colors},KEY={review_key};let review=JSON.parse(localStorage.getItem(KEY)||'{{}}'),index=0;const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),save=()=>localStorage.setItem(KEY,JSON.stringify(review));
function filtered(){{let rows=DATA.pairs.slice(),scope=$('scope').value,status=$('status').value,max=parseFloat($('maxDistance').value);if(scope==='cross')rows=rows.filter(r=>r.cross_split);if(scope==='same')rows=rows.filter(r=>!r.cross_split);if(status==='unreviewed')rows=rows.filter(r=>!review[r.rank]?.status);else if(status!=='all')rows=rows.filter(r=>review[r.rank]?.status===status);if(Number.isFinite(max))rows=rows.filter(r=>r.layout_distance<=max);return rows}}
function boxes(item){{return item.boxes.map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"><title>${{esc(DATA.class_names[b.c]??b.c)}}</title></rect>`}}).join('')}}
function tile(item,signature){{return `<div><div class="figure"><img src="${{ROOT+'/'+item.image}}" alt="${{esc(item.image)}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${{boxes(item)}}</svg></div><div class="caption">${{esc(item.image)}}<br>${{esc(signature)}} · split=${{esc(item.split)}}</div></div>`}}
function render(){{const rows=filtered();index=Math.max(0,Math.min(index,Math.max(0,rows.length-1)));$('reviewed').textContent=Object.values(review).filter(x=>x.status).length;$('shown').textContent=`筛选后 ${{rows.length}} 对`;if(!rows.length){{$('content').innerHTML='<div class="pair">没有符合当前筛选条件的候选。</div>';$('position').textContent='0 / 0';return}}const row=rows[index],r=review[row.rank]||{{status:'',note:''}};$('position').textContent=`${{index+1}} / ${{rows.length}}`;$('content').innerHTML=`<article class="pair" data-rank="${{row.rank}}"><div class="head"><strong>#${{row.rank}} · frame ${{row.left_frame}} / ${{row.right_frame}}（帧号差 ${{row.frame_gap}}） · ${{esc(row.batch)}} · ${{esc(row.left.split)}}/${{esc(row.right.split)}}${{row.cross_split?' · 跨 split':''}}</strong><span class="score">布局距离=${{row.layout_distance.toFixed(6)}} · 匹配 IoU=${{row.mean_matched_iou.toFixed(4)}} · 图像相似度=${{row.cosine_similarity.toFixed(6)}}</span></div><div class="images">${{tile(row.left,row.left_signature)}}${{tile(row.right,row.right_signature)}}</div><div class="review"><button class="choice ${{r.status==='duplicate'?'active':''}}" data-status="duplicate">重复</button><button class="choice ${{r.status==='not_duplicate'?'active':''}}" data-status="not_duplicate">不重复</button><button class="choice ${{r.status==='unsure'?'active':''}}" data-status="unsure">不确定</button><span class="muted">匹配类别不一致 ${{row.class_mismatches}} 个</span><textarea placeholder="备注；若准备以此作为停止附近，建议再向后审阅一小段">${{esc(r.note||'')}}</textarea></div></article>`}}
for(const id of ['scope','status'])$(id).onchange=()=>{{index=0;render()}};$('maxDistance').oninput=()=>{{index=0;render()}};$('prev').onclick=()=>{{index=Math.max(0,index-1);render()}};$('next').onclick=()=>{{index=Math.min(filtered().length-1,index+1);render()}};$('content').onclick=e=>{{const choice=e.target.closest('.choice');if(!choice)return;const rank=e.target.closest('[data-rank]').dataset.rank;review[rank]=review[rank]||{{status:'',note:''}};review[rank].status=choice.dataset.status;save();render()}};$('content').oninput=e=>{{if(e.target.tagName!=='TEXTAREA')return;const rank=e.target.closest('[data-rank]').dataset.rank;review[rank]=review[rank]||{{status:'',note:''}};review[rank].note=e.target.value;save()}};
$('export').onclick=()=>{{const q=v=>'"'+String(v??'').replaceAll('"','""')+'"',out=[['rank','left','right','left_frame','right_frame','frame_gap','batch','box_count','left_signature','right_signature','layout_distance','mean_matched_iou','class_mismatches','cosine_similarity','left_split','right_split','cross_split','status','note']];for(const row of DATA.pairs){{const r=review[row.rank]||{{}};out.push([row.rank,row.left.image,row.right.image,row.left_frame,row.right_frame,row.frame_gap,row.batch,row.box_count,row.left_signature,row.right_signature,row.layout_distance,row.mean_matched_iou,row.class_mismatches,row.cosine_similarity,row.left.split,row.right.split,row.cross_split?1:0,r.status||'',r.note||''])}}const blob=new Blob(['\ufeff'+out.map(row=>row.map(q).join(',')).join('\\n')],{{type:'text/csv;charset=utf-8'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=DATA.browser_export;a.click();URL.revokeObjectURL(a.href)}};$('clear').onclick=()=>{{if(confirm('清空当前浏览器保存的全部审阅？')){{review={{}};save();render()}}}};document.addEventListener('keydown',e=>{{if(e.target.matches('input,textarea,select'))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()}});render();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    variant_dir = args.variant_dir.expanduser().resolve()
    assignments_path = args.assignments.expanduser().resolve()
    embeddings_path = args.embeddings.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.min_frame_gap < 1 or args.max_frame_gap < args.min_frame_gap:
        raise ValueError("frame-gap bounds must satisfy 1 <= min <= max")
    if args.min_cosine is not None and not -1.0 <= args.min_cosine <= 1.0:
        raise ValueError("min-cosine must be within [-1, 1]")
    dataset_root = (PROJECT_ROOT / "data" / "self_improving").resolve()
    images = read_manifest(variant_dir / "manifest.txt")

    with assignments_path.open(encoding="utf-8-sig", newline="") as handle:
        assignments = {row["image"]: row["split"] for row in csv.DictReader(handle)}
    archive = np.load(embeddings_path, allow_pickle=True)
    embeddings = archive["embeddings"].astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    archive_indices = {image_key(path): index for index, path in enumerate(archive["paths"].tolist())}

    annotations: dict[str, list[dict[str, float | int]]] = {}
    for image in images:
        relative = image.relative_to(dataset_root).as_posix()
        annotations[relative] = read_boxes(image, dataset_root)

    pairs = []
    for left_path, right_path in zip(images, images[1:]):
        frame_gap = frame(right_path) - frame(left_path)
        if frame_gap < args.min_frame_gap or frame_gap > args.max_frame_gap:
            continue
        left_relative = left_path.relative_to(dataset_root).as_posix()
        right_relative = right_path.relative_to(dataset_root).as_posix()
        left_boxes, right_boxes = annotations[left_relative], annotations[right_relative]
        if not left_boxes or len(left_boxes) != len(right_boxes):
            continue
        left_index, right_index = archive_indices[image_key(left_relative)], archive_indices[image_key(right_relative)]
        cosine_similarity = float(embeddings[left_index] @ embeddings[right_index])
        if args.min_cosine is not None and cosine_similarity <= args.min_cosine:
            continue
        layout_distance, mean_iou, class_mismatches = layout_metrics(left_boxes, right_boxes)
        left_split, right_split = assignments[left_relative], assignments[right_relative]
        pairs.append(
            {
                "left": {"image": left_relative, "boxes": left_boxes, "split": left_split},
                "right": {"image": right_relative, "boxes": right_boxes, "split": right_split},
                "left_frame": frame(left_path),
                "right_frame": frame(right_path),
                "frame_gap": frame_gap,
                "batch": left_path.parent.name if left_path.parent == right_path.parent else f"{left_path.parent.name}/{right_path.parent.name}",
                "box_count": len(left_boxes),
                "left_signature": class_signature(left_boxes),
                "right_signature": class_signature(right_boxes),
                "layout_distance": layout_distance,
                "mean_matched_iou": mean_iou,
                "class_mismatches": class_mismatches,
                "cosine_similarity": cosine_similarity,
                "cross_split": left_split != right_split,
            }
        )
    pairs.sort(key=lambda row: (row["layout_distance"], -row["cosine_similarity"], row["left_frame"]))
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank

    if not pairs:
        raise RuntimeError("no positive adjacent pairs with equal box counts")
    cosine_rule = f"、图像特征余弦相似度严格大于 {args.min_cosine:g}" if args.min_cosine is not None else ""
    candidate_rule_zh = (
        f"去重后保留序列中相邻、全局帧号差 {args.min_frame_gap}～{args.max_frame_gap}、"
        f"两张图都有有效标注、总框数相同{cosine_rule}"
    )
    browser_export = f"{output_dir.name}_manual_review.csv"
    payload = {
        "candidate_count": len(pairs),
        "cross_split_count": sum(row["cross_split"] for row in pairs),
        "distance_min": pairs[0]["layout_distance"],
        "distance_max": pairs[-1]["layout_distance"],
        "class_names": CLASS_NAMES,
        "candidate_rule_zh": candidate_rule_zh,
        "review_key": f"0910_{output_dir.name}_v1",
        "browser_export": browser_export,
        "pairs": pairs,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(os.path.relpath(dataset_root, output_dir)).as_posix()
    (output_dir / "pair_review.html").write_text(html(payload, image_root), encoding="utf-8", newline="\n")
    with (output_dir / "candidate_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "rank", "left", "right", "left_frame", "right_frame", "frame_gap", "batch", "box_count",
            "left_signature", "right_signature", "layout_distance", "mean_matched_iou",
            "class_mismatches", "cosine_similarity", "left_split", "right_split", "cross_split",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "left": row["left"]["image"],
                    "right": row["right"]["image"],
                    "left_frame": row["left_frame"],
                    "right_frame": row["right_frame"],
                    "frame_gap": row["frame_gap"],
                    "batch": row["batch"],
                    "box_count": row["box_count"],
                    "left_signature": row["left_signature"],
                    "right_signature": row["right_signature"],
                    "layout_distance": f'{row["layout_distance"]:.9f}',
                    "mean_matched_iou": f'{row["mean_matched_iou"]:.9f}',
                    "class_mismatches": row["class_mismatches"],
                    "cosine_similarity": f'{row["cosine_similarity"]:.9f}',
                    "left_split": row["left"]["split"],
                    "right_split": row["right"]["split"],
                    "cross_split": int(row["cross_split"]),
                }
            )
    summary = {
        "status": "completed",
        "candidate_rule": (
            f"consecutive retained neighbors; global frame gap {args.min_frame_gap}..{args.max_frame_gap}; "
            f"both positive; equal total box counts"
            + (f"; cosine similarity > {args.min_cosine:g}" if args.min_cosine is not None else "")
        ),
        "layout_distance": "Hungarian mean(center L2 + 0.5 * size L2), normalized YOLO coordinates, class-agnostic",
        "candidate_count": len(pairs),
        "cross_split_count": payload["cross_split_count"],
        "distance_min": payload["distance_min"],
        "distance_max": payload["distance_max"],
        "html": "pair_review.html",
        "candidate_csv": "candidate_pairs.csv",
        "browser_export": browser_export,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
