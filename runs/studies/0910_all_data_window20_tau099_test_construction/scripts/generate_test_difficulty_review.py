from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_SPLIT_DIR = STUDY_ROOT / "experiment" / "variants" / "random_stratified_s42_8_1_1"
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "result" / "test_positive_difficulty_review"
CLASS_NAMES = ("large luggage", "stroller", "wheelchair", "flatbed truck")
CLASS_COLORS = ("#2563eb", "#16a34a", "#db2777", "#ea580c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a browser review page for positive images in the fixed test split.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_manifest(path: Path) -> list[Path]:
    manifest = path.resolve()
    images: list[Path] = []
    for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            image = (manifest.parent / value).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"manifest image does not exist: {image}")
            images.append(image)
    return images


def label_path(image: Path, dataset_root: Path) -> Path:
    relative = image.relative_to(dataset_root)
    if len(relative.parts) < 3 or relative.parts[0] != "images":
        raise ValueError(f"unexpected image layout: {image}")
    return dataset_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")


def read_boxes(path: Path) -> list[dict[str, float | int]]:
    boxes: list[dict[str, float | int]] = []
    if not path.is_file():
        return boxes
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO row at {path}:{line_number}: {line!r}")
        class_id = int(fields[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"class id outside schema at {path}:{line_number}: {class_id}")
        x, y, width, height = (float(value) for value in fields[1:])
        boxes.append({"c": class_id, "x": x, "y": y, "w": width, "h": height})
    return boxes


def html(payload: dict[str, object], image_root: str) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    root = json.dumps(image_root, ensure_ascii=False)
    colors = json.dumps(CLASS_COLORS, ensure_ascii=False)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定 test 正样本难度审阅</title>
<style>
:root{{--bg:#f4f7fb;--surface:#fff;--ink:#172033;--muted:#647087;--line:#d9e0eb;--blue:#2563eb;--green:#15803d;--red:#b91c1c;--amber:#b45309;--shadow:0 10px 28px rgba(32,50,86,.08)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1700px;margin:auto;padding:24px}}h1{{margin:0 0 6px;font-size:26px}}.muted{{color:var(--muted)}}
.summary,.toolbar,.review,.pages{{display:flex;flex-wrap:wrap;gap:9px;align-items:center}}.summary{{margin:18px 0}}.metric,.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}.metric{{min-width:155px;padding:10px 13px}}.metric small{{display:block;color:var(--muted)}}.metric strong{{font-size:21px}}
button,select,input{{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px}}button{{cursor:pointer}}.toolbar{{position:sticky;top:0;z-index:8;background:rgba(244,247,251,.95);backdrop-filter:blur(8px);padding:12px 0}}
#grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.card{{overflow:hidden}}.figure{{position:relative;background:#e9eef5;cursor:zoom-in}}.figure img{{display:block;width:100%;height:auto}}.figure svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.box{{fill-opacity:.07;stroke-width:.006}}.caption{{padding:8px 10px;border-top:1px solid var(--line);font-size:12px;overflow-wrap:anywhere}}.review{{padding:0 10px 10px}}.choice.active[data-status=difficult]{{background:var(--red);border-color:var(--red);color:#fff}}.choice.active[data-status=normal]{{background:var(--green);border-color:var(--green);color:#fff}}.choice.active[data-status=unsure]{{background:var(--amber);border-color:var(--amber);color:#fff}}textarea{{width:100%;min-height:48px;border:1px solid var(--line);border-radius:8px;padding:7px;font:inherit}}
.card.big{{grid-column:1/-1}}.card.big .figure{{max-width:1200px;cursor:zoom-out}}.pages{{margin:16px 0}}.pages button[aria-current=page]{{background:var(--blue);color:#fff}}.kbd{{font-family:Consolas,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:1px 5px;background:#fff}}
@media(max-width:1050px){{#grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:680px){{main{{padding:14px}}#grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>固定 test 正样本 · 难度审阅</h1>
<div class="muted">仅展示带有效标注的 test 图片。红色“困难”表示后续从 filtered test 排除；未审阅、正常和不确定默认保留。选择与备注保存在当前浏览器。</div>
<section class="summary"><div class="metric"><small>完整 test</small><strong>{payload['test_images']}</strong></div><div class="metric"><small>待审正样本</small><strong>{payload['positive_images']}</strong></div><div class="metric"><small>背景（不展示）</small><strong>{payload['background_images']}</strong></div><div class="metric"><small>已审阅</small><strong id="reviewed">0</strong></div><div class="metric"><small>困难</small><strong id="difficult">0</strong></div></section>
<div class="toolbar"><select id="status"><option value="all">全部状态</option><option value="unreviewed">未审阅</option><option value="difficult">困难</option><option value="normal">正常</option><option value="unsure">不确定</option></select><select id="class"><option value="all">全部类别</option></select><select id="batch"><option value="all">全部 batch</option></select><select id="sort"><option value="frame">全局帧号升序</option><option value="labels_desc">标注框数量降序</option></select><input id="search" placeholder="搜索文件名"><button id="export">导出审阅 CSV</button><button id="clear">清空本页审阅</button><span class="muted" id="shown"></span></div>
<p class="muted">快捷键：鼠标停在卡片上，按 <span class="kbd">D</span> 困难、<span class="kbd">N</span> 正常、<span class="kbd">U</span> 不确定。</p><section id="grid"></section><div class="pages" id="pages"></div>
</main><script>
const DATA={data},ROOT={root},COLORS={colors},KEY='0910_fixed_test_positive_difficulty_review_s42_v1',SIZE=30;
let review=JSON.parse(localStorage.getItem(KEY)||'{{}}'),page=1,hovered=null;const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),save=()=>localStorage.setItem(KEY,JSON.stringify(review));
function boxSvg(row){{return row.boxes.map(b=>{{const x=Math.max(0,b.x-b.w/2),y=Math.max(0,b.y-b.h/2),w=Math.min(1-x,b.w),h=Math.min(1-y,b.h),color=COLORS[b.c%COLORS.length];return `<rect class="box" x="${{x}}" y="${{y}}" width="${{w}}" height="${{h}}" style="stroke:${{color}};fill:${{color}}"><title>${{esc(DATA.class_names[b.c]??b.c)}}</title></rect>`}}).join('')}}
function setStatus(rank,status){{review[rank]=review[rank]||{{status:'',note:''}};review[rank].status=status;save();render()}}
function rows(){{let out=DATA.rows.slice(),s=$('status').value,c=$('class').value,b=$('batch').value,q=$('search').value.trim().toLowerCase();if(s==='unreviewed')out=out.filter(r=>!review[r.rank]?.status);else if(s!=='all')out=out.filter(r=>review[r.rank]?.status===s);if(c!=='all')out=out.filter(r=>r.classes.includes(Number(c)));if(b!=='all')out=out.filter(r=>r.batch===b);if(q)out=out.filter(r=>r.image.toLowerCase().includes(q));if($('sort').value==='labels_desc')out.sort((a,b)=>b.boxes.length-a.boxes.length||a.frame-b.frame);return out}}
function card(row){{const r=review[row.rank]||{{status:'',note:''}},names=row.classes.map(c=>DATA.class_names[c]).join(' / ');return `<article class="card" data-rank="${{row.rank}}"><div class="figure"><img loading="lazy" src="${{ROOT+'/'+row.image}}" alt="${{esc(row.image)}}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${{boxSvg(row)}}</svg></div><div class="caption"><strong>#${{row.rank}} · frame ${{row.frame}} · ${{esc(row.batch)}}</strong><br>${{esc(names)}} · ${{row.boxes.length}} 个框<br>${{esc(row.image)}}</div><div class="review"><button class="choice ${{r.status==='difficult'?'active':''}}" data-status="difficult">困难</button><button class="choice ${{r.status==='normal'?'active':''}}" data-status="normal">正常</button><button class="choice ${{r.status==='unsure'?'active':''}}" data-status="unsure">不确定</button><textarea placeholder="困难原因或备注">${{esc(r.note||'')}}</textarea></div></article>`}}
function render(){{const all=rows(),total=Math.max(1,Math.ceil(all.length/SIZE));page=Math.min(page,total);$('grid').innerHTML=all.slice((page-1)*SIZE,page*SIZE).map(card).join('');$('shown').textContent=`显示 ${{all.length}} / ${{DATA.rows.length}} 张 · 第 ${{page}}/${{total}} 页`;$('reviewed').textContent=Object.values(review).filter(x=>x.status).length;$('difficult').textContent=Object.values(review).filter(x=>x.status==='difficult').length;$('pages').innerHTML='';for(let i=1;i<=total;i++){{if(total>14&&i>2&&i<total-1&&Math.abs(i-page)>1)continue;const button=document.createElement('button');button.textContent=i;button.setAttribute('aria-current',i===page?'page':'false');button.onclick=()=>{{page=i;render();scrollTo({{top:0,behavior:'smooth'}})}};$('pages').appendChild(button)}}}}
for(const id of ['status','class','batch','sort'])$(id).onchange=()=>{{page=1;render()}};$('search').oninput=()=>{{page=1;render()}};
$('grid').onclick=e=>{{const card=e.target.closest('.card');if(!card)return;const choice=e.target.closest('.choice');if(choice){{setStatus(card.dataset.rank,choice.dataset.status);return}}if(e.target.closest('.figure'))card.classList.toggle('big')}};$('grid').onmouseover=e=>{{const card=e.target.closest('.card');if(card)hovered=card.dataset.rank}};$('grid').onmouseout=e=>{{if(e.target.closest('.card'))hovered=null}};$('grid').oninput=e=>{{if(e.target.tagName!=='TEXTAREA')return;const rank=e.target.closest('.card').dataset.rank;review[rank]=review[rank]||{{status:'',note:''}};review[rank].note=e.target.value;save()}};
document.onkeydown=e=>{{if(!hovered||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;const status={{d:'difficult',n:'normal',u:'unsure'}}[e.key.toLowerCase()];if(status)setStatus(hovered,status)}};
$('export').onclick=()=>{{const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"',out=[['rank','image','global_frame','batch','classes','label_count','status','note']];for(const row of DATA.rows){{const r=review[row.rank]||{{}};out.push([row.rank,row.image,row.frame,row.batch,row.classes.join(';'),row.boxes.length,r.status||'',r.note||''])}}const blob=new Blob(['\ufeff'+out.map(row=>row.map(quote).join(',')).join('\n')],{{type:'text/csv;charset=utf-8'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='test_positive_difficulty_manual_review.csv';a.click();URL.revokeObjectURL(a.href)}};
$('clear').onclick=()=>{{if(confirm('清空当前浏览器保存的全部难度审阅？')){{review={{}};save();render()}}}};
$('class').innerHTML='<option value="all">全部类别</option>'+DATA.class_names.map((name,index)=>`<option value="${{index}}">${{esc(name)}}</option>`).join('');$('batch').innerHTML='<option value="all">全部 batch</option>'+[...new Set(DATA.rows.map(r=>r.batch))].sort().map(b=>`<option>${{esc(b)}}</option>`).join('');render();
</script></body></html>'''


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    dataset_root = (PROJECT_ROOT / "data" / "self_improving").resolve()
    test_images = read_manifest(split_dir / "manifests" / "test.txt")
    rows = []
    for image in test_images:
        boxes = read_boxes(label_path(image, dataset_root))
        if not boxes:
            continue
        relative = image.relative_to(dataset_root).as_posix()
        frame = int(image.stem.split("_", 1)[0])
        rows.append(
            {
                "rank": len(rows) + 1,
                "image": relative,
                "frame": frame,
                "batch": image.relative_to(dataset_root).parts[1],
                "classes": sorted({int(box["c"]) for box in boxes}),
                "boxes": boxes,
            }
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing review output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_images": len(test_images),
        "positive_images": len(rows),
        "background_images": len(test_images) - len(rows),
        "class_names": CLASS_NAMES,
        "rows": rows,
    }
    image_root = Path(os.path.relpath(dataset_root, output_dir)).as_posix()
    (output_dir / "review.html").write_text(html(payload, image_root), encoding="utf-8", newline="\n")
    summary = {
        "status": "completed",
        "source_split": "random_stratified_s42_8_1_1/test",
        "test_images": len(test_images),
        "positive_images_for_review": len(rows),
        "background_images_not_shown": len(test_images) - len(rows),
        "review_semantics": {
            "difficult": "exclude from the later filtered test",
            "normal": "keep",
            "unsure": "keep unless explicitly changed to difficult",
            "unreviewed": "keep",
        },
        "html": "review.html",
        "browser_export": "test_positive_difficulty_manual_review.csv",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
