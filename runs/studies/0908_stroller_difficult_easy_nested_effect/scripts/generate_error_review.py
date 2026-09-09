"""Generate an offline, per-image visual review for the nested stroller study.

The page deliberately keeps the three views separate for every image:

* m1: a checkpoint trained on train1 (difficult representatives),
* m2: a checkpoint trained on train2 (the easy projection), and
* ground truth: the label file belonging to the raw stroller image.

The script runs inference only.  It never trains, copies images, or changes a
checkpoint.  The generated page uses relative links back to data/stroller_raw
so the page remains usable after the repository is moved to another checkout.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_retraining.backends import create_backend  # noqa: E402
from yolo_retraining.config import load_config  # noqa: E402
from yolo_retraining.data import build_registry  # noqa: E402
from yolo_retraining.data.loaders import yolo_label_path  # noqa: E402


SEEDS = (41, 42)
MODELS = ("m1", "m2")
GROUPS = ("test1", "test2", "testhard")
RESULT_ROOT = STUDY_ROOT / "result/evaluation"
OUTPUT_ROOT = STUDY_ROOT / "result/error_review"
CONFIG_ROOT = STUDY_ROOT / "config"
IOU_THRESHOLD = 0.5
INFERENCE_CONFIDENCE = 0.25


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def task_dir(model: str, seed: int) -> Path:
    labels = {"m1": "m1-difficult", "m2": "m2-easy"}
    groups = {"m1": "train1", "m2": "train2"}
    return STUDY_ROOT / "experiment/task" / (
        f"{model}_s{seed}__{labels[model]}__{groups[model]}__yolov5s__s42"
    )


def checkpoint_for(model: str, seed: int) -> Path:
    directory = task_dir(model, seed)
    result = read_json(directory / "task_result.json")
    if result.get("status") != "completed":
        raise ValueError(f"task is not complete: {directory}")
    checkpoint = directory / str(result["artifacts"]["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_variant_summary(seed: int) -> dict[str, Any]:
    return read_json(STUDY_ROOT / "experiment/variants" / f"split_s{seed}" / "summary.json")


def canonical(path: Path | str) -> str:
    return str(Path(path).resolve()).casefold()


def records_for_layout(layout_path: Path, group: str) -> tuple[dict[str, Any], list[str]]:
    registry = build_registry({group: str(layout_path)})
    sample_ids = list(registry["groups"][group]["test"])
    if not sample_ids:
        raise ValueError(f"evaluation group is empty: {group}")
    return registry, sample_ids


def load_yolo_boxes(label_path: Path) -> list[dict[str, Any]]:
    if not label_path.is_file():
        raise FileNotFoundError(f"ground-truth label does not exist: {label_path}")
    boxes: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        class_id = int(float(fields[0]))
        x, y, width, height = (float(value) for value in fields[1:5])
        x1 = x - width / 2.0
        y1 = y - height / 2.0
        x2 = x + width / 2.0
        y2 = y + height / 2.0
        boxes.append(
            {
                "class_id": class_id,
                "box": [
                    max(0.0, min(1.0, x1)),
                    max(0.0, min(1.0, y1)),
                    max(0.0, min(1.0, x2)),
                    max(0.0, min(1.0, y2)),
                ],
            }
        )
    return boxes


def prediction_box(raw: Iterable[float], width: int, height: int) -> dict[str, Any]:
    values = [float(value) for value in raw]
    if len(values) < 6:
        raise ValueError(f"YOLOv5 prediction row has fewer than six values: {values}")
    x1, y1, x2, y2, confidence, class_id = values[:6]
    return {
        "class_id": int(class_id),
        "confidence": confidence,
        "box": [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ],
    }


def box_iou(left: list[float], right: list[float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def classify_predictions(
    predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True)
    matched: set[int] = set()
    tp = 0
    fp = 0
    for prediction in ordered:
        candidates = [
            (box_iou(prediction["box"], target["box"]), index)
            for index, target in enumerate(ground_truth)
            if index not in matched and int(target["class_id"]) == int(prediction["class_id"])
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= IOU_THRESHOLD:
            matched.add(best_index)
            prediction["match"] = "tp"
            prediction["matched_iou"] = best_iou
            tp += 1
        else:
            prediction["match"] = "fp"
            prediction["matched_iou"] = best_iou
            fp += 1
    fn = len(ground_truth) - len(matched)
    return {
        "boxes": ordered,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "wrong": bool(fp or fn),
        "correct": not bool(fp or fn),
    }


def infer_model(
    *,
    model: str,
    seed: int,
    registry: dict[str, Any],
    sample_ids: list[str],
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, list[list[float]]]:
    config = load_config(CONFIG_ROOT / f"{model}_s{seed}.yaml")
    inference_config = copy.deepcopy(config)
    # The visual review is intentionally serialized on CPU so it does not
    # compete with a separately running training process on the server.
    inference_config["backend"]["params"]["device"] = "cpu"
    backend = create_backend(inference_config)
    backend.validate_config(inference_config)
    return backend.predict(
        {
            "config": inference_config,
            "registry": registry,
            "sample_ids": sample_ids,
            "checkpoint": str(checkpoint),
            "output_dir": output_dir,
        }
    )


def image_record(
    *,
    record: dict[str, Any],
    groups: list[str],
    names: list[str],
    predictions: dict[str, dict[str, list[list[float]]]],
    seed: int,
    model_checkpoints: dict[str, Path],
) -> dict[str, Any]:
    image_path = Path(record["image_path"]).resolve()
    label_path = Path(record["label_path"]).resolve()
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - training env supplies Pillow
        raise RuntimeError("Pillow is required to generate the visual review") from error
    with Image.open(image_path) as image:
        width, height = image.size
    ground_truth = load_yolo_boxes(label_path)
    sample_id = str(record["sample_id"])
    models: dict[str, Any] = {}
    for model in MODELS:
        raw_predictions = predictions[model].get(sample_id, [])
        boxes = [prediction_box(row, width, height) for row in raw_predictions]
        models[model] = classify_predictions(boxes, ground_truth)
        models[model]["checkpoint"] = str(model_checkpoints[model])
    parts = list(image_path.parts)
    lowered = [part.casefold() for part in parts]
    image_index = len(lowered) - 1 - lowered[::-1].index("images")
    return {
        "id": sample_id,
        "filename": image_path.name,
        "image": Path(*parts[image_index:]).as_posix(),
        "groups": groups,
        "width": width,
        "height": height,
        "ground_truth": ground_truth,
        "models": models,
        "seed": seed,
        "class_names": names,
    }


def build_seed_payload(seed: int) -> dict[str, Any]:
    variant = load_variant_summary(seed)
    test1_layout = Path(variant["layouts"]["test1"])
    test1_registry, sample_ids = records_for_layout(test1_layout, "test1")
    records = test1_registry["records"]

    paths_by_group: dict[str, set[str]] = {}
    for group in GROUPS:
        group_registry, group_ids = records_for_layout(Path(variant["layouts"][group]), group)
        paths_by_group[group] = {
            canonical(group_registry["records"][sample_id]["image_path"])
            for sample_id in group_ids
        }
    test1_paths = {canonical(records[sample_id]["image_path"]) for sample_id in sample_ids}
    if paths_by_group["test1"] != test1_paths:
        paths_by_group["test1"] = test1_paths
    if paths_by_group["test2"] & paths_by_group["testhard"]:
        raise ValueError(f"test2 and testhard overlap for seed {seed}")
    if paths_by_group["test2"] | paths_by_group["testhard"] != test1_paths:
        raise ValueError(f"test2 and testhard do not partition test1 for seed {seed}")

    model_checkpoints = {model: checkpoint_for(model, seed) for model in MODELS}
    predictions: dict[str, dict[str, list[list[float]]]] = {}
    for model in MODELS:
        cache_dir = OUTPUT_ROOT / "prediction_cache" / f"split_s{seed}" / model
        predictions[model] = infer_model(
            model=model,
            seed=seed,
            registry=test1_registry,
            sample_ids=sample_ids,
            checkpoint=model_checkpoints[model],
            output_dir=cache_dir,
        )

    names = list(test1_registry["names"])
    payload_records = []
    for sample_id in sample_ids:
        source_record = records[sample_id]
        path_key = canonical(source_record["image_path"])
        groups = [group for group in GROUPS if path_key in paths_by_group[group]]
        if not groups:
            raise ValueError(f"test1 sample has no display group: {source_record['image_path']}")
        payload_records.append(
            image_record(
                record=source_record,
                groups=groups,
                names=names,
                predictions=predictions,
                seed=seed,
                model_checkpoints=model_checkpoints,
            )
        )
    return {
        "seed": seed,
        "records": payload_records,
        "counts": {group: sum(group in row["groups"] for row in payload_records) for group in GROUPS},
        "models": {model: str(model_checkpoints[model]) for model in MODELS},
    }


def page_html() -> str:
    return r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>0908 stroller nested effect · visual review</title>
  <style>
    :root { color-scheme: light; --ink:#17202b; --muted:#657180; --line:#dce3e8; --paper:#f5f7f8; --card:#fff; --blue:#277d98; --orange:#df704d; --green:#2f9b67; --red:#c84a4a; --shadow:0 12px 32px rgba(28,48,61,.08); }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--paper); font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
    .app { max-width:1880px; margin:0 auto; padding:26px 30px 48px; }
    header { margin-bottom:22px; }
    .eyebrow { color:var(--blue); font-size:12px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
    h1 { margin:7px 0 8px; font-size:clamp(25px,3vw,42px); line-height:1.08; letter-spacing:-.03em; }
    .intro { max-width:960px; margin:0; color:var(--muted); font-size:15px; line-height:1.7; }
    .protocol { margin-top:12px; color:#51606d; font-size:12px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:12px; align-items:end; padding:15px; background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
    .control { display:flex; min-width:145px; flex-direction:column; gap:5px; }
    .control.search { min-width:240px; flex:1 1 250px; }
    label, .control-label { color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.04em; text-transform:uppercase; }
    select, input { width:100%; height:38px; padding:0 11px; color:var(--ink); background:#fbfcfc; border:1px solid #cdd7dd; border-radius:9px; font:inherit; outline:none; }
    select:focus, input:focus { border-color:var(--blue); box-shadow:0 0 0 3px rgba(39,125,152,.13); }
    .body-grid { display:grid; grid-template-columns:250px minmax(0,1fr); gap:22px; align-items:start; margin-top:22px; }
    aside { position:sticky; top:15px; padding:18px; background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
    aside h2 { margin:0 0 13px; font-size:14px; }
    aside p { margin:10px 0; color:var(--muted); font-size:12px; line-height:1.65; }
    .legend { display:grid; gap:9px; margin:14px 0 19px; }
    .legend-row { display:flex; align-items:center; gap:8px; color:#4d5b67; font-size:12px; }
    .swatch { width:12px; height:12px; border:2px solid currentColor; border-radius:3px; }
    .swatch.gt { color:var(--green); } .swatch.m1 { color:var(--blue); } .swatch.m2 { color:var(--orange); } .swatch.fp { color:var(--red); }
    .summary { display:grid; grid-template-columns:repeat(5,minmax(110px,1fr)); gap:10px; margin-bottom:17px; }
    .stat { min-height:80px; padding:13px 14px; background:var(--card); border:1px solid var(--line); border-radius:13px; }
    .stat .value { display:block; margin-top:4px; font-size:25px; font-weight:800; letter-spacing:-.04em; }
    .stat .name { color:var(--muted); font-size:11px; font-weight:700; }
    .results-head { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; margin:0 0 13px; color:var(--muted); font-size:13px; }
    .results-head strong { color:var(--ink); }
    .grid { display:grid; grid-template-columns:minmax(0,1fr); gap:17px; }
    .card { overflow:hidden; background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
    .card-head { display:flex; justify-content:space-between; gap:12px; padding:15px 16px 12px; border-bottom:1px solid #edf0f2; }
    .card-head h3 { overflow:hidden; margin:2px 0 0; max-width:390px; font-size:15px; text-overflow:ellipsis; white-space:nowrap; }
    .serial { color:var(--blue); font-size:11px; font-weight:800; }
    .tags { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:5px; }
    .tag, .badge { display:inline-flex; align-items:center; width:max-content; padding:3px 7px; border-radius:999px; font-size:10px; font-weight:800; white-space:nowrap; }
    .tag { color:#52606b; background:#eef2f4; } .tag.hard { color:#8a4334; background:#fff0eb; }
    .card-meta { display:flex; flex-wrap:wrap; gap:6px 14px; padding:9px 16px 0; color:var(--muted); font-size:11px; }
    .triptych { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; padding:12px; }
    .tile { min-width:0; cursor:zoom-in; border:1px solid #dfe5e9; border-radius:10px; background:#f8fafb; transition:all .16s ease; }
    .tile:hover { border-color:#9dbbc4; box-shadow:0 5px 16px rgba(28,48,61,.08); }
    .tile.zoomed { grid-column:1/-1; cursor:zoom-out; }
    .tile-title { display:flex; justify-content:space-between; gap:5px; padding:8px 9px 6px; font-size:11px; font-weight:800; }
    .tile-title .sub { color:var(--muted); font-size:10px; font-weight:600; }
    .tile.good .badge { color:#216b49; background:#e9f7ef; } .tile.bad .badge { color:#943838; background:#ffeceb; }
    .figure { position:relative; overflow:hidden; aspect-ratio:16/9; background:#17202b; }
    .figure img { display:block; width:100%; height:100%; object-fit:contain; }
    .figure svg { position:absolute; inset:0; z-index:2; width:100%; height:100%; pointer-events:none; overflow:visible; }
    .box { fill-opacity:.10; stroke-width:.006; }
    .box.gt { fill:var(--green); stroke:var(--green); } .box.m1 { fill:var(--blue); stroke:var(--blue); } .box.m2 { fill:var(--orange); stroke:var(--orange); } .box.fp { fill:var(--red); stroke:var(--red); stroke-dasharray:.012 .006; }
    .box-label { font-size:.032px; font-weight:800; }
    .tile-foot { min-height:29px; padding:6px 9px 8px; color:var(--muted); font-size:10px; line-height:1.35; }
    .empty { display:none; padding:50px 20px; text-align:center; color:var(--muted); background:var(--card); border:1px dashed #c5d0d6; border-radius:16px; }
    .pagination { display:flex; justify-content:center; align-items:center; gap:12px; margin-top:18px; }
    .pagination button { min-width:76px; height:34px; color:var(--ink); background:var(--card); border:1px solid #cdd7dd; border-radius:8px; cursor:pointer; }
    .pagination button:disabled { cursor:not-allowed; opacity:.45; }
    .page-number { color:var(--muted); font-size:12px; }
    code { padding:1px 4px; background:#eef2f4; border-radius:4px; font-size:11px; }
    @media (max-width:980px) { .body-grid { grid-template-columns:1fr; } aside { position:static; } .summary { grid-template-columns:repeat(3,1fr); } }
    @media (max-width:650px) { .app { padding:18px 12px 35px; } .summary { grid-template-columns:repeat(2,1fr); } .grid { grid-template-columns:1fr; } .triptych { grid-template-columns:1fr; } .tile.zoomed { grid-column:auto; } }
  </style>
</head>
<body>
<div class="app">
  <header>
    <div class="eyebrow">0908 · nested effect · offline review</div>
    <h1>m1 / m2 对同一张图的三视图对照</h1>
    <p class="intro">每张样本分别显示 m1 预测、m2 预测和真实标注。点击任意一个图块可以放大；预测框与真实框不会叠加在同一张图中，因此可以直接观察两个模型各自漏检、误检或定位偏差。</p>
    <div class="protocol">视觉状态定义：推理显示置信度为 0.25；按类别进行一对一匹配，IoU ≥ 0.50 且没有额外误检才记为“正确”。该状态用于筛图，不替代 canonical AP 评估。</div>
  </header>
  <section class="toolbar" aria-label="筛选工具">
    <div class="control"><span class="control-label">split seed</span><select id="seed"></select></div>
    <div class="control"><span class="control-label">数据集合</span><select id="group"><option value="test1">test1 · 全部 test</option><option value="test2">test2 · easy 子集</option><option value="testhard">testhard · test1 − test2</option></select></div>
    <div class="control"><span class="control-label">模型状态</span><select id="failure"><option value="all">全部图片</option><option value="m1_wrong">m1 没做对</option><option value="m2_wrong">m2 没做对</option><option value="either_wrong">任一模型没做对</option><option value="both_wrong">m1 与 m2 都没做对</option><option value="m1_correct">仅看 m1 正确</option><option value="m2_correct">仅看 m2 正确</option></select></div>
    <div class="control"><span class="control-label">每页</span><select id="page-size"><option value="6">6</option><option value="12" selected>12</option><option value="24">24</option></select></div>
    <div class="control search"><label for="search">文件名搜索</label><input id="search" type="search" placeholder="例如 00143_...jpg"></div>
  </section>
  <div class="body-grid">
    <aside>
      <h2>读图说明</h2>
      <div class="legend">
        <div class="legend-row"><span class="swatch m1"></span> m1 预测框</div>
        <div class="legend-row"><span class="swatch m2"></span> m2 预测框</div>
        <div class="legend-row"><span class="swatch gt"></span> 真实标注框</div>
        <div class="legend-row"><span class="swatch fp"></span> 未匹配预测（误检）</div>
      </div>
      <p><strong>test1</strong> 是 difficult 的完整测试划分；<strong>test2</strong> 是其中属于 easy 的图片；<strong>testhard</strong> 是 <code>test1 - test2</code>。</p>
      <p>m1 在 train1/difficult 上训练，m2 在 train2/easy 投影上训练。seed 只切换对应 split 和对应 checkpoint。</p>
      <p>图像链接指向仓库的 <code>data/stroller_raw</code>，页面没有复制图片。</p>
    </aside>
    <main>
      <div class="summary" id="summary"></div>
      <div class="results-head"><span id="shown"></span><span>点击图块切换语义放大</span></div>
      <div id="grid" class="grid"></div>
      <div id="empty" class="empty">当前筛选没有图片。</div>
      <div class="pagination"><button id="prev">上一页</button><span id="page-number" class="page-number"></span><button id="next">下一页</button></div>
    </main>
  </div>
</div>
<script src="review_data.js"></script>
<script>
(() => {
  const DATA = window.__ERROR_REVIEW_DATA__;
  const state = { seed: String(DATA.seeds[0]), group: 'test1', failure: 'all', search: '', page: 1, pageSize: 12 };
  const $ = (id) => document.getElementById(id);
  const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const seedData = () => DATA.by_seed[state.seed];
  const selectedRows = () => {
    const query = state.search.trim().toLowerCase();
    return seedData().records.filter((row) => {
      const inGroup = row.groups.includes(state.group);
      const inSearch = !query || row.filename.toLowerCase().includes(query);
      const m1Wrong = row.models.m1.wrong;
      const m2Wrong = row.models.m2.wrong;
      const states = { all:true, m1_wrong:m1Wrong, m2_wrong:m2Wrong, either_wrong:m1Wrong || m2Wrong, both_wrong:m1Wrong && m2Wrong, m1_correct:!m1Wrong, m2_correct:!m2Wrong };
      return inGroup && inSearch && states[state.failure];
    });
  };
  const modelText = (model) => {
    const item = model === 'm1' ? 'm1' : 'm2';
    const value = seedData().records.length ? seedData().records[0].models[item] : null;
    return value ? item : item;
  };
  const boxSvg = (boxes, kind, names) => {
    const rows = boxes.map((box, index) => {
      const [x1,y1,x2,y2] = box.box;
      const width = Math.max(0, x2-x1), height = Math.max(0, y2-y1);
      const className = kind === 'gt' ? 'gt' : (box.match === 'fp' ? 'fp' : kind);
      const label = kind === 'gt' ? `GT ${names[box.class_id] || box.class_id}` : `${names[box.class_id] || box.class_id} ${(box.confidence * 100).toFixed(1)}%`;
      return `<rect class="box ${className}" x="${x1}" y="${y1}" width="${width}" height="${height}"><title>${escapeHtml(label)}</title></rect>`;
    }).join('');
    return `<svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">${rows}</svg>`;
  };
  const tile = (row, model, names) => {
    const result = row.models[model];
    const status = result.correct ? '正确' : '错误';
    const color = model === 'm1' ? 'm1' : 'm2';
    const note = `${result.tp} TP · ${result.fp} FP · ${result.fn} FN`;
    const element = document.createElement('section');
    element.className = `tile ${result.correct ? 'good' : 'bad'}`;
    element.setAttribute('aria-label', `${model} ${status}`);
    element.innerHTML = `<div class="tile-title"><span>${model} <span class="sub">${model === 'm1' ? 'difficult' : 'easy'}</span></span><span class="badge">${status}</span></div><div class="figure" style="aspect-ratio:${row.width}/${row.height}"><img loading="lazy" src="${escapeHtml(DATA.image_root + row.image)}" alt="${escapeHtml(row.filename)} · ${model}">${boxSvg(result.boxes, color, names)}</div><div class="tile-foot">${note} · 点击放大</div>`;
    element.addEventListener('click', () => {
      const enlarged = element.classList.toggle('zoomed');
      element.setAttribute('aria-expanded', String(enlarged));
    });
    return element;
  };
  const makeCard = (row, index) => {
    const names = DATA.class_names;
    const card = document.createElement('article');
    const groupTags = row.groups.map((group) => `<span class="tag ${group === 'testhard' ? 'hard' : ''}">${group}</span>`).join('');
    card.className = 'card';
    card.innerHTML = `<div class="card-head"><div><div class="serial">#${String(index + 1).padStart(3, '0')} · seed ${row.seed}</div><h3 title="${escapeHtml(row.filename)}">${escapeHtml(row.filename)}</h3></div><div class="tags">${groupTags}</div></div><div class="card-meta"><span>GT ${row.ground_truth.length}</span><span>尺寸 ${row.width}×${row.height}</span><span>m1 ${row.models.m1.correct ? '✓' : '×'} · m2 ${row.models.m2.correct ? '✓' : '×'}</span></div><div class="triptych"></div>`;
    const triptych = card.querySelector('.triptych');
    triptych.appendChild(tile(row, 'm1', names));
    triptych.appendChild(tile(row, 'm2', names));
    const truth = document.createElement('section');
    truth.className = 'tile good';
    truth.setAttribute('aria-label', '真实标注');
    truth.innerHTML = `<div class="tile-title"><span>真实标注 <span class="sub">ground truth</span></span><span class="badge">${row.ground_truth.length} 个框</span></div><div class="figure" style="aspect-ratio:${row.width}/${row.height}"><img loading="lazy" src="${escapeHtml(DATA.image_root + row.image)}" alt="${escapeHtml(row.filename)} · ground truth">${boxSvg(row.ground_truth, 'gt', names)}</div><div class="tile-foot">绿色框为标签 · 点击放大</div>`;
    truth.addEventListener('click', () => { const enlarged = truth.classList.toggle('zoomed'); truth.setAttribute('aria-expanded', String(enlarged)); });
    triptych.appendChild(truth);
    return card;
  };
  const renderSummary = (rows) => {
    const all = seedData().records.filter((row) => row.groups.includes(state.group));
    const m1Wrong = all.filter((row) => row.models.m1.wrong).length;
    const m2Wrong = all.filter((row) => row.models.m2.wrong).length;
    const bothWrong = all.filter((row) => row.models.m1.wrong && row.models.m2.wrong).length;
    const items = [['样本数', all.length], ['当前显示', rows.length], ['m1 错误', m1Wrong], ['m2 错误', m2Wrong], ['都错误', bothWrong]];
    $('summary').innerHTML = items.map(([name,value]) => `<div class="stat"><span class="name">${name}</span><span class="value">${value}</span></div>`).join('');
  };
  const render = () => {
    const rows = selectedRows();
    renderSummary(rows);
    const pages = Math.max(1, Math.ceil(rows.length / state.pageSize));
    state.page = Math.min(state.page, pages);
    const start = (state.page - 1) * state.pageSize;
    const current = rows.slice(start, start + state.pageSize);
    $('grid').replaceChildren(...current.map((row, index) => makeCard(row, start + index)));
    $('empty').style.display = current.length ? 'none' : 'block';
    $('shown').innerHTML = `<strong>${current.length}</strong> / ${rows.length} 张 · seed ${state.seed} · ${escapeHtml(state.group)}`;
    $('page-number').textContent = `${state.page} / ${pages}`;
    $('prev').disabled = state.page <= 1;
    $('next').disabled = state.page >= pages;
  };
  DATA.seeds.forEach((seed) => { const option = document.createElement('option'); option.value = seed; option.textContent = `seed ${seed}`; $('seed').appendChild(option); });
  $('seed').value = state.seed;
  $('seed').addEventListener('change', (event) => { state.seed = event.target.value; state.page = 1; render(); });
  $('group').addEventListener('change', (event) => { state.group = event.target.value; state.page = 1; render(); });
  $('failure').addEventListener('change', (event) => { state.failure = event.target.value; state.page = 1; render(); });
  $('page-size').addEventListener('change', (event) => { state.pageSize = Number(event.target.value); state.page = 1; render(); });
  $('search').addEventListener('input', (event) => { state.search = event.target.value; state.page = 1; render(); });
  $('prev').addEventListener('click', () => { state.page -= 1; render(); window.scrollTo({top:0, behavior:'smooth'}); });
  $('next').addEventListener('click', () => { state.page += 1; render(); window.scrollTo({top:0, behavior:'smooth'}); });
  render();
})();
</script>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    by_seed: dict[str, Any] = {}
    for seed in args.seeds:
        if seed not in SEEDS:
            raise ValueError(f"unsupported nested-study seed: {seed}; choices={SEEDS}")
        by_seed[str(seed)] = build_seed_payload(seed)

    data = {
        "study": STUDY_ROOT.name,
        "seeds": args.seeds,
        "class_names": list(by_seed[str(args.seeds[0])]["records"][0]["class_names"]),
        "image_root": "../../../../../data/stroller_raw/",
        "inference": {
            "checkpoint": "best.pt from each completed m1/m2 task",
            "device": "cpu",
            "confidence": INFERENCE_CONFIDENCE,
            "matching_iou": IOU_THRESHOLD,
            "nms": "YOLOv5 AutoShape defaults (conf=0.25, iou=0.45)",
        },
        "by_seed": by_seed,
    }
    data_path = OUTPUT_ROOT / "review_data.js"
    data_path.write_text(
        "window.__ERROR_REVIEW_DATA__ = "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    html_path = OUTPUT_ROOT / "error_review.html"
    html_path.write_text(page_html(), encoding="utf-8")
    summary = {
        "status": "completed",
        "study": STUDY_ROOT.name,
        "page": str(html_path.relative_to(STUDY_ROOT)),
        "data": str(data_path.relative_to(STUDY_ROOT)),
        "seeds": args.seeds,
        "records_by_seed": {seed: len(payload["records"]) for seed, payload in by_seed.items()},
        "group_counts_by_seed": {seed: payload["counts"] for seed, payload in by_seed.items()},
        "visual_correctness": "strict image-level status: no unmatched prediction and no missed ground-truth box at IoU 0.50",
        "prediction_source": "fresh inference from completed best.pt checkpoints; no training was started",
        "image_source": "relative links to data/stroller_raw; no image copies were created",
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
