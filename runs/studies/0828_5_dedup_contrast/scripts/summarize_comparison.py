from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
RESULT_ROOT = STUDY_ROOT / "result" / "strategy_comparison"
TASK_ROOT = STUDY_ROOT / "experiment" / "task"

EXPERIMENTS = [
    {
        "name": "full_raw_640_300ep",
        "label": "全量原始 baseline（640，配置 300 epoch）",
        "result_dir": PROJECT_ROOT / "runs" / "tasks" / "Balance_Test__full-cold-stage0123__stage0+stage1+stage2+stage3__yolov5s__s42",
        "log": PROJECT_ROOT / "runs" / "tasks" / "Balance_Test__full-cold-stage0123__stage0+stage1+stage2+stage3__yolov5s__s42" / "logs" / "yolov5_train.log",
    },
    {
        "name": "full_raw_sequence_100ep",
        "label": "全量原始 baseline（Sequence 最终 stage，100 epoch）",
        "protocol": "Sequence final stage",
        "result_dir": PROJECT_ROOT / "runs" / "sequences" / "Balance_Test__seq-full-cold__stage0-stage3__yolov5s__s42" / "tasks" / "003__stage3__full-cold",
        "log": PROJECT_ROOT / "runs" / "sequences" / "Balance_Test__seq-full-cold__stage0-stage3__yolov5s__s42" / "tasks" / "003__stage3__full-cold" / "logs" / "yolov5_train.log",
    },
    {
        "name": "dedup_tau_0p900",
        "label": "普通去冗余",
    },
    {
        "name": "dedup_tau_0p900_release_positive_clusters",
        "label": "释放含标注簇、纯背景簇留代表",
    },
    {
        "name": "dedup_tau_0p900_release_positive_clusters_drop_pure_background",
        "label": "释放含标注簇、删除纯背景簇",
    },
    {
        "name": "dedup_tau_0p900_positive_representative",
        "label": "有标注优先单代表",
    },
]

TASK_SUFFIX = "__full-cold-stage0123__stage0+stage1+stage2+stage3__yolov5s__s42"


def load_experiment(spec: dict[str, str | Path]) -> dict[str, object]:
    result_dir = Path(spec.get("result_dir", TASK_ROOT / str(spec["name"]))).resolve()
    task_path = result_dir / "task.yaml"
    result_path = result_dir / "task_result.json"
    if not task_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"completed Task artifacts are missing under {result_dir}")
    task = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"Task is not completed: {result_dir}")
    backend_params = task.get("backend", {}).get("params", {})
    best_test = result.get("metrics", {}).get("best", {}).get("test", {})
    last_test = result.get("metrics", {}).get("last", {}).get("test", {})
    log_path = Path(spec.get("log", STUDY_ROOT / "logs" / f"{spec['name']}.log"))
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    epochs = [int(value) for value in re.findall(r"(\d+) epochs completed", log_text)]
    cost = result.get("cost", {})
    selected_count = result.get("selection", {}).get("group_counts", {}).get("candidate")
    return {
        "name": spec["name"],
        "label": spec["label"],
        "protocol": spec.get("protocol", "single Task"),
        "result_dir": str(result_dir),
        "train_images": selected_count,
        "imgsz": backend_params.get("imgsz"),
        "budget_epochs": task.get("budget", {}).get("value"),
        # The sequence task omits patience; the YOLOv5 backend supplies its
        # effective default of 100.  Keep that effective value in the report.
        "patience": backend_params.get("patience", 100),
        "actual_epochs": epochs[-1] if epochs else None,
        "best_map50_95": best_test.get("map50_95"),
        "best_map50": best_test.get("map50"),
        "last_map50_95": last_test.get("map50_95"),
        "last_map50": last_test.get("map50"),
        "training_seconds": cost.get("training_seconds"),
    }


def write_outputs(rows: list[dict[str, object]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_ROOT / "comparison.csv"
    fields = [
        "name",
        "label",
        "protocol",
        "train_images",
        "imgsz",
        "budget_epochs",
        "patience",
        "actual_epochs",
        "best_map50_95",
        "best_map50",
        "last_map50_95",
        "last_map50",
        "training_seconds",
        "result_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Balance tau=0.90 label-contrast comparison",
        "",
        "所有行均使用 balance test/val、YOLOv5s、640 分辨率和 full-cold 初始化。",
        "除 Sequence 参考行外，其余为 seed 42 的单 Task；Sequence 行取现有四阶段 full-cold 实验的最终 stage（该 stage 的 seed 为 45）。",
        "`best` 指验证集选择出的 best checkpoint 在 test 上的指标。",
        "",
        "| 实验 | 协议 | 训练图数 | imgsz | 配置 epoch | patience | 实际 epoch | best mAP50-95 | best mAP50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['label']} | {row['protocol']} | {row['train_images']} | {row['imgsz']} | "
            f"{row['budget_epochs']} | {row['patience']} | {row['actual_epochs']} | "
            f"{float(row['best_map50_95']):.4f} | {float(row['best_map50']):.4f} |"
        )
    markdown.extend(
        [
            "",
            "注意：Sequence 参考行来自 `runs/sequences/Balance_Test__seq-full-cold__stage0-stage3__yolov5s__s42` 的最终 stage。它每个 stage 冷启动训练 100 epoch，最终 stage 使用累计 7573 张训练图；四个 tau=0.90 实验则是单 Task、配置 300 epoch/patience=100。因此该行是同分辨率的 100-epoch 全量参考，但不是与四个 tau=0.90 单 Task 完全相同的训练协议。",
            "",
            "结果路径和原始 Task 配置见 `comparison.csv`。",
        ]
    )
    (RESULT_ROOT / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    for spec in EXPERIMENTS:
        if "result_dir" not in spec:
            spec = {
                **spec,
                "result_dir": TASK_ROOT / f"{spec['name']}{TASK_SUFFIX}",
                "log": STUDY_ROOT / "logs" / f"{spec['name']}.log",
            }
        rows.append(load_experiment(spec))
    write_outputs(rows)
    print((RESULT_ROOT / "comparison.md").resolve())
    print((RESULT_ROOT / "comparison.csv").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
