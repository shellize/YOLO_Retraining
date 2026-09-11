from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from generate_positive_box_layout_pair_audit import CLASS_COLORS, CLASS_NAMES, frame, read_manifest


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_VARIANT = STUDY_ROOT / "experiment" / "variants" / "global_order_window20_tau0p990_label_aware_pair_reviewed_final"
DEFAULT_GROUPS = STUDY_ROOT / "experiment" / "variants" / "target_track_groups_two_pass_final" / "image_split_groups.csv"
DEFAULT_OUTPUT = STUDY_ROOT / "result" / "final_positive_group_overview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display every final positive image in sequence and split-group order.")
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_groups(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image = row["image"]
            if image in result and result[image] != row["split_group_id"]:
                raise ValueError(f"image occurs in multiple split groups: {image}")
            result[image] = row["split_group_id"]
    return result


def read_boxes(image: Path, dataset_root: Path) -> list[dict[str, float | int]]:
    relative = image.relative_to(dataset_root)
    label = dataset_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
    boxes: list[dict[str, float | int]] = []
    if not label.is_file():
        return boxes
    for raw in label.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.strip().split()
        if len(fields) != 5:
            continue
        class_id = int(fields[0])
        x, y, width, height = (float(value) for value in fields[1:])
        boxes.append({"c": class_id, "x": x, "y": y, "w": width, "h": height})
    return boxes


def page(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    root = json.dumps(image_root, ensure_ascii=False)
    colors = json.dumps(CLASS_COLORS)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最终正样本与轨迹 group 总览</title><style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#667085;--line:#d6deea;--blue:#2563eb;--shadow:0 8px 22px rgba(32,50,86,.08)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1720px;margin:auto;padding:20px}}h1{{margin:0 0 5px;font-size:25px}}.muted{{color:var(--muted)}}.summary,.toolbar,.pages{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:15px 0}}.metric,.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}}.metric{{min-width:150px;padding:8px 12px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:20px}}.toolbar{{position:sticky;top:0;z-index:5;background:rgba(244,247,251,.96);padding:10px 0}}select,input,button{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:7px;padding:7px 9px}}button{{cursor:pointer}}#grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}}.card{{overflow:hidden}}.figure{{position:relative;background:#e9eef5;cursor:zoom-in}}.figure img{{display:block;width:100%;height:auto}}.figure svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.box{{fill-opacity:.06;stroke-width:.006}}.caption{{padding:7px 8px;font-size:11px;overflow-wrap:anywhere}}.group{{display:inline-block;border-radius:999px;padding:2px 7px;margin-bottom:3px;color:#fff;font-weight:650}}.independent{{background:#7b8495!important}}.card.big{{grid-column:span 3}}.pages{{margin:14px 0;justify-content:center}}.pages button[aria-current=page]{{background:var(--blue);color:#fff}}@media(max-width:1100px){{#grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}@media(max-width:700px){{main{{padding:12px}}#grid{{grid-template-columns:1fr 1fr}}.card.big{{grid-column:1/-1}}}}
</style></head><body><main><h1>最终正样本与轨迹 group 总览</h1><div class="muted">仅展示最终 5,763 张 manifest 中有有效标注的图片。默认按完整 manifest 顺序排列；可切换为按 group 聚合。相同 group 必须在后续划分中进入同一个 split，“独立”表示尚无人工确认的跨图片轨迹约束。</div><section class="summary"><div class="metric"><small>最终全部图片</small><strong>{payload['final_images']}</strong></div><div class="metric"><small>有标注图片</small><strong>{payload['positive_images']}</strong></div><div class="metric"><small>已绑定图片</small><strong>{payload['grouped_positive_images']}</strong></div><div class="metric"><small>独立图片</small><strong>{payload['independent_positive_images']}</strong></div><div class="metric"><small>图片级 group</small><strong>{payload['group_count']}</strong></div></section><div class="toolbar"><select id="order"><option value="sequence">按图片顺序</option><option value="group">按 group</option></select><select id="scope"><option value="all">全部</option><option value="grouped">仅已分组</option><option value="independent">仅独立</option></select><input id="search" placeholder="搜索图片、batch 或 group"><span class="muted" id="shown"></span></div><section id="grid"></section><div class="pages" id="pages"></div></main><script>
const DATA={data},ROOT={root},COLORS={colors},SIZE=100;let current=1;const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));function groupColor(group){{if(!group)return '#7b8495';let h=0;for(const c of group)h=(h*31+c.charCodeAt(0))%360;return `hsl(${{h}} 58% 43%)`}}function boxes(row){{return row.boxes.map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"><title>${{esc(DATA.class_names[b.c]??b.c)}}</title></rect>`}}).join('')}}function rows(){{const scope=$('scope').value,q=$('search').value.trim().toLowerCase();let result=DATA.images.filter(r=>(scope==='all'||(scope==='grouped'&&r.group)||(scope==='independent'&&!r.group))&&(!q||r.image.toLowerCase().includes(q)||r.batch.toLowerCase().includes(q)||(r.group||'独立').toLowerCase().includes(q)));if($('order').value==='group')result.sort((a,b)=>{{if(a.group&&b.group)return a.group.localeCompare(b.group)||a.sequence_index-b.sequence_index;if(a.group)return-1;if(b.group)return 1;return a.sequence_index-b.sequence_index}});return result}}function card(r){{const label=r.group||'独立',cls=r.group?'group':'group independent';return `<article class="card"><div class="figure"><img loading="lazy" src="${{ROOT+'/'+r.image}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${{boxes(r)}}</svg></div><div class="caption"><span class="${{cls}}" style="background:${{groupColor(r.group)}}">${{esc(label)}}</span><br><strong>#${{r.sequence_index+1}} · frame ${{r.frame}} · ${{esc(r.batch)}}</strong><br>${{esc(r.image)}} · ${{r.boxes.length}} 个框</div></article>`}}function render(){{const all=rows(),pages=Math.max(1,Math.ceil(all.length/SIZE));current=Math.min(current,pages);$('grid').innerHTML=all.slice((current-1)*SIZE,current*SIZE).map(card).join('');$('shown').textContent=`显示 ${{all.length}} 张 · 第 ${{current}}/${{pages}} 页`;$('pages').innerHTML='';for(let i=1;i<=pages;i++){{if(pages>16&&i>2&&i<pages-1&&Math.abs(i-current)>1)continue;const b=document.createElement('button');b.textContent=i;b.setAttribute('aria-current',i===current?'page':'false');b.onclick=()=>{{current=i;render();scrollTo({{top:0,behavior:'smooth'}})}};$('pages').appendChild(b)}}}}for(const id of ['order','scope'])$(id).onchange=()=>{{current=1;render()}};$('search').oninput=()=>{{current=1;render()}};$('grid').onclick=e=>{{const card=e.target.closest('.card');if(card)card.classList.toggle('big')}};render();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    variant = args.variant_dir.resolve()
    groups_path = args.groups.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite positive group overview: {output}")

    dataset = (PROJECT_ROOT / "data" / "self_improving").resolve()
    images = read_manifest(variant / "manifest.txt")
    groups = read_groups(groups_path)
    rows: list[dict[str, object]] = []
    for sequence_index, image in enumerate(images):
        boxes = read_boxes(image, dataset)
        if not boxes:
            continue
        relative = image.relative_to(dataset).as_posix()
        rows.append(
            {
                "image": relative,
                "sequence_index": sequence_index,
                "frame": frame(image),
                "batch": image.parent.name,
                "group": groups.get(relative),
                "boxes": boxes,
            }
        )

    grouped = sum(row["group"] is not None for row in rows)
    payload = {
        "final_images": len(images),
        "positive_images": len(rows),
        "grouped_positive_images": grouped,
        "independent_positive_images": len(rows) - grouped,
        "group_count": len(set(groups.values())),
        "class_names": CLASS_NAMES,
        "images": rows,
    }
    output.mkdir(parents=True)
    image_root = Path(os.path.relpath(dataset, output)).as_posix()
    (output / "positive_group_overview.html").write_text(page(payload, image_root), encoding="utf-8", newline="\n")
    summary = {key: payload[key] for key in ("final_images", "positive_images", "grouped_positive_images", "independent_positive_images", "group_count")}
    summary.update({"status": "completed", "order": "complete final manifest sequence", "html": "positive_group_overview.html"})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
