from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "sequences"
OUTPUT_DIR = Path(__file__).resolve().parent

SEQUENCE_DIRS = {
    "full-cold": "Balance_Test__seq-full-cold__stage0-stage3__yolov5s__s42",
    "full-warm": "Balance_Test__seq-full-warm__stage0-stage3__yolov5s__s42",
    "current-only": "Balance_Test__seq-current-only__stage0-stage3__yolov5s__s42",
    "random-replay": "Balance_Test__seq-random-replay__stage0-stage3__yolov5s__s42",
}


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for sequence, directory_name in SEQUENCE_DIRS.items():
        sequence_dir = RUNS_DIR / directory_name
        task_dirs = sorted((sequence_dir / "tasks").glob("*__stage*__*"))
        if len(task_dirs) != 4:
            raise RuntimeError(f"Expected 4 stages for {sequence}, found {len(task_dirs)}")

        for task_dir in task_dirs:
            match = re.match(r"^(\d{3})__stage(\d)__", task_dir.name)
            if not match:
                raise RuntimeError(f"Cannot parse stage from {task_dir}")
            task_result_path = task_dir / "task_result.json"
            payload = json.loads(task_result_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]["best"]["test"]
            rows.append(
                {
                    "sequence": sequence,
                    "stage": int(match.group(2)),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "mAP50": float(metrics["map50"]),
                    "mAP50_95": float(metrics["map50_95"]),
                    "source": str(task_result_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )

    rows.sort(key=lambda row: (list(SEQUENCE_DIRS).index(row["sequence"]), row["stage"]))
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 rows, found {len(rows)}")
    return rows


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_csv(rows: list[dict]) -> None:
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence", "stage", "precision", "recall", "mAP50", "mAP50_95", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "precision": fmt(row["precision"]), "recall": fmt(row["recall"]), "mAP50": fmt(row["mAP50"]), "mAP50_95": fmt(row["mAP50_95"])})


def draw_chart(rows: list[dict]) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "current-only": "#2563eb",
        "full-cold": "#dc2626",
        "full-warm": "#059669",
        "random-replay": "#d97706",
    }
    labels = {
        "current-only": "Current Only",
        "full-cold": "Full Cold",
        "full-warm": "Full Warm",
        "random-replay": "Random Replay",
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)
    for sequence in SEQUENCE_DIRS:
        sequence_rows = [row for row in rows if row["sequence"] == sequence]
        ax.plot(
            [row["stage"] for row in sequence_rows],
            [row["mAP50"] for row in sequence_rows],
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=colors[sequence],
            label=labels[sequence],
        )

    ax.set_title("Balance_Test 四种 Sequence 的 mAP50 变化", fontsize=15, pad=14)
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("mAP50", fontsize=12)
    ax.set_xticks([0, 1, 2, 3], ["Stage 0", "Stage 1", "Stage 2", "Stage 3"])
    ax.set_ylim(0.40, 0.55)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.01, 0.01, "指标取各 stage 的 best checkpoint / test 结果；纵轴为聚焦区间 0.40–0.55。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_DIR / "map50_by_sequence.png", bbox_inches="tight")
    plt.close(fig)


def draw_precision_recall_chart(rows: list[dict]) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "current-only": "#2563eb",
        "full-cold": "#dc2626",
        "full-warm": "#059669",
        "random-replay": "#d97706",
    }
    labels = {
        "current-only": "Current Only",
        "full-cold": "Full Cold",
        "full-warm": "Full Warm",
        "random-replay": "Random Replay",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), dpi=160, sharex=True)
    for ax, metric, title, y_limits in zip(
        axes,
        ["precision", "recall"],
        ["Precision", "Recall"],
        [(0.45, 0.85), (0.35, 0.50)],
    ):
        for sequence in SEQUENCE_DIRS:
            sequence_rows = [row for row in rows if row["sequence"] == sequence]
            ax.plot(
                [row["stage"] for row in sequence_rows],
                [row[metric] for row in sequence_rows],
                marker="o",
                linewidth=2.0,
                markersize=6,
                color=colors[sequence],
                label=labels[sequence],
            )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Stage", fontsize=11)
        ax.set_xticks([0, 1, 2, 3], ["Stage 0", "Stage 1", "Stage 2", "Stage 3"])
        ax.set_ylim(*y_limits)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric value", fontsize=11)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Balance_Test 四种 Sequence 的 Precision / Recall 变化", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall_by_sequence.png", bbox_inches="tight")
    plt.close(fig)


def write_markdown(rows: list[dict]) -> None:
    by_sequence = {sequence: [row for row in rows if row["sequence"] == sequence] for sequence in SEQUENCE_DIRS}
    stage3 = {sequence: by_sequence[sequence][-1] for sequence in SEQUENCE_DIRS}
    best_final = max(stage3, key=lambda sequence: stage3[sequence]["mAP50"])
    best_stage2 = max(rows, key=lambda row: row["mAP50"])
    highest_precision = max(stage3, key=lambda sequence: stage3[sequence]["precision"])
    highest_recall = max(stage3, key=lambda sequence: stage3[sequence]["recall"])

    lines = [
        "# Balance_Test 四种 Sequence 结果分析",
        "",
        "## 数据口径",
        "",
        "- 实验范围：`project_v2/runs/sequences` 下的 4 个 `Balance_Test` sequence，每个包含 `stage0`–`stage3`。",
        "- 评价集合：每个 stage 的 `best checkpoint / test`。",
        "- 选择原因：这些运行的 `best_selection_metric` 为 `map50`，因此 precision、recall 和 mAP50 均来自同一 best checkpoint 评价结果。",
        "- 说明：4 个 sequence 的训练策略、初始化或 replay 组成不同，以下是结果的描述性比较，不把 sequence 间差异直接解释为单一因素的因果效果。",
        "",
        "## Precision 与 Recall",
        "",
        "| Sequence | Stage 0 | Stage 1 | Stage 2 | Stage 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for sequence in SEQUENCE_DIRS:
        cells = [f"P={fmt(row['precision'])}; R={fmt(row['recall'])}" for row in by_sequence[sequence]]
        lines.append(f"| `{sequence}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## mAP50",
        "",
        "| Sequence | Stage 0 | Stage 1 | Stage 2 | Stage 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for sequence in SEQUENCE_DIRS:
        cells = [fmt(row["mAP50"]) for row in by_sequence[sequence]]
        lines.append(f"| `{sequence}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## mAP50 折线图",
        "",
        "![Balance_Test 四种 sequence 的 mAP50 折线图](map50_by_sequence.png)",
        "",
        "## Precision / Recall 折线图",
        "",
        "![Balance_Test 四种 sequence 的 Precision 和 Recall 折线图](precision_recall_by_sequence.png)",
        "",
        "## 结果观察",
        "",
        f"1. Stage 0 四条曲线重合，mAP50 均为 `{fmt(by_sequence['current-only'][0]['mAP50'])}`，说明四个 sequence 使用了相同的初始 stage0 结果。",
        f"2. 最终 Stage 3 的 mAP50 以 `{best_final}` 最高，为 `{fmt(stage3[best_final]['mAP50'])}`；其次是 `random-replay` `{fmt(stage3['random-replay']['mAP50'])}`、`full-warm` `{fmt(stage3['full-warm']['mAP50'])}` 和 `current-only` `{fmt(stage3['current-only']['mAP50'])}`。",
        f"3. 全部 16 个 stage 中的最高 mAP50 出现在 `{best_stage2['sequence']}` 的 Stage {best_stage2['stage']}，为 `{fmt(best_stage2['mAP50'])}`；该 sequence 到 Stage 3 回落到 `{fmt(stage3[best_stage2['sequence']]['mAP50'])}`，但 recall 从 `{fmt(by_sequence[best_stage2['sequence']][2]['recall'])}` 回升到 `{fmt(by_sequence[best_stage2['sequence']][3]['recall'])}`。",
        f"4. Stage 3 的 precision 最高是 `{highest_precision}`（`{fmt(stage3[highest_precision]['precision'])}`），recall 最高是 `{highest_recall}`（`{fmt(stage3[highest_recall]['recall'])}`）；因此最终阶段存在明显的 precision–recall 取舍，不能只看单一指标。",
        "",
        "## 原始结果来源",
        "",
        "详细行级数据已保存为 [`metrics.csv`](metrics.csv)。每一行的 `source` 字段指向对应的 `task_result.json`。",
    ]
    (OUTPUT_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    draw_chart(rows)
    draw_precision_recall_chart(rows)
    write_markdown(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
