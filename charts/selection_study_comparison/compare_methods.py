from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


WORKSPACE = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE.parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "sequences"
STAGES = ("Stage 0", "Stage 1", "Stage 2", "Stage 3")
FIXED_STAGES = ("Stage 1", "Stage 2", "Stage 3")
FIXED_CONFIDENCES = (0.2, 0.3, 0.4, 0.5, 0.6)
METRICS = {
    "map50": ("mAP50", "map50.png"),
    "recall": ("Recall", "recall.png"),
    "precision": ("Precision", "precision.png"),
}


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    label_zh: str
    sequence_dir: str
    family: str
    baseline: bool = False
    shared_stage0: bool = False


METHODS = (
    Method(
        "full_cold",
        "Full Cold (baseline)",
        "全量冷启动（Baseline）",
        "Balance_Test__seq-full-cold__stage0-stage3__yolov5s__s42",
        "baseline",
        baseline=True,
    ),
    Method(
        "current_only",
        "Current Only",
        "仅当前阶段",
        "Balance_Test__seq-current-only__stage0-stage3__yolov5s__s42",
        "existing method",
    ),
    Method(
        "positive_cum",
        "Positive Only + Cum Cold",
        "仅正样本 + 累积冷启动",
        "B640__seq-positive-only__cum-cold__stage0-stage3__yolov5s__s42",
        "positive-only",
        shared_stage0=True,
    ),
    Method(
        "positive_current",
        "Positive Only + Current Warm",
        "仅正样本 + 当前阶段热启动",
        "B640__seq-positive-only__current-warm__stage0-stage3__yolov5s__s42",
        "positive-only",
        shared_stage0=True,
    ),
    Method(
        "pred_cum",
        "Pred Positive + Cum Cold",
        "预测触发样本 + 累积冷启动",
        "B640__seq-pred-positive__cum-cold__stage0-stage3__yolov5s__s42",
        "pred-positive",
        shared_stage0=True,
    ),
    Method(
        "pred_current",
        "Pred Positive + Current Warm",
        "预测触发样本 + 当前阶段热启动",
        "B640__seq-pred-positive__current-warm__stage0-stage3__yolov5s__s42",
        "pred-positive",
        shared_stage0=True,
    ),
    Method(
        "error_cum",
        "Error Hard + Cum Cold",
        "错误困难样本 + 累积冷启动",
        "B640__seq-error-hard__cum-cold__stage0-stage3__yolov5s__s42",
        "error-hard",
        shared_stage0=True,
    ),
)

SHARED_STAGE0 = (
    RUNS_ROOT
    / "Balance_Test__seq-full-warm__stage0-stage3__yolov5s__s42"
    / "tasks"
    / "000__stage0__full-warm"
    / "task_result.json"
)


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_sequence(method: Method) -> Path:
    sequence = RUNS_ROOT / method.sequence_dir
    result_path = sequence / "sequence_result.json"
    status_path = sequence / "sequence_status.json"
    if not result_path.is_file():
        status = read_json(status_path).get("status") if status_path.is_file() else "missing"
        raise FileNotFoundError(f"method {method.label} is not complete (status={status}): {sequence}")
    result = read_json(result_path)
    if result.get("status") != "completed":
        raise ValueError(f"method {method.label} sequence_result is not completed: {result_path}")
    return sequence


def task_result_for_stage(method: Method, sequence: Path, stage: int) -> Path:
    if stage == 0 and method.shared_stage0:
        return SHARED_STAGE0
    matches = sorted((sequence / "tasks").glob(f"{stage:03d}__stage{stage}__*/task_result.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one Stage {stage} TaskResult for {method.label}, found {len(matches)} under {sequence}"
        )
    return matches[0]


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        sequence = validate_sequence(method)
        for stage, stage_label in enumerate(STAGES):
            result_path = task_result_for_stage(method, sequence, stage)
            result = read_json(result_path)
            try:
                metrics = result["metrics"]["best"]["test"]
            except KeyError as error:
                raise KeyError(f"missing metrics.best.test in {result_path}") from error
            rows.append(
                {
                    "method_key": method.key,
                    "method": method.label,
                    "family": method.family,
                    "is_baseline": method.baseline,
                    "stage": stage,
                    "stage_label": stage_label,
                    "map50": float(metrics["map50"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "task_result": str(result_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )
    return rows


def write_csv_table(rows: list[dict[str, object]]) -> None:
    fields = [
        "method_key",
        "method",
        "family",
        "is_baseline",
        "stage",
        "stage_label",
        "map50",
        "recall",
        "precision",
        "task_result",
    ]
    with (WORKSPACE / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "map50": f"{row['map50']:.8f}", "recall": f"{row['recall']:.8f}", "precision": f"{row['precision']:.8f}"})


def rows_by_method(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_key"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["stage"]))
    return grouped


def padded_domain(values: list[float]) -> tuple[float, float]:
    minimum, maximum = min(values), max(values)
    span = max(maximum - minimum, 0.08)
    padding = span * 0.12
    return max(0.0, minimum - padding), min(1.0, maximum + padding)


def plot_metric(rows: list[dict[str, object]], metric: str, title: str, filename: str) -> None:
    grouped = rows_by_method(rows)
    values = [float(row[metric]) for row in rows]
    ymin, ymax = padded_domain(values)
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P", "X")
    line_styles = ("--", ":", "-", "-.", "-", "-.", ":")

    fig, ax = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    for index, method in enumerate(METHODS):
        method_rows = grouped[method.key]
        ax.plot(
            range(len(STAGES)),
            [float(row[metric]) for row in method_rows],
            label=method.label,
            color=colors[index % len(colors)],
            marker=markers[index],
            linestyle=line_styles[index],
            linewidth=3.0 if method.baseline else 2.0,
            markersize=8 if method.baseline else 6,
            zorder=5 if method.baseline else 3,
        )

    ax.set_title(f"Selection Study: {title} by Stage", fontsize=18, pad=16)
    ax.set_xlabel("Training stage", fontsize=13)
    ax.set_ylabel(f"Best checkpoint test {title} (0–1)", fontsize=13)
    ax.set_xticks(range(len(STAGES)), STAGES)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.7, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    fig.savefig(WORKSPACE / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_metric_table(grouped: dict[str, list[dict[str, object]]], metric: str, title: str) -> list[str]:
    lines = [f"## {title}", "", "| 方法 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | 平均值 |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        values = [float(row[metric]) for row in grouped[method.key]]
        formatted = " | ".join(f"{value:.4f}" for value in values)
        lines.append(f"| {method.label_zh} | {formatted} | {sum(values) / len(values):.4f} |")
    lines.append("")
    return lines


def write_metrics_markdown(rows: list[dict[str, object]]) -> None:
    grouped = rows_by_method(rows)
    lines = [
        "# Sequence 样本筛选实验指标",
        "",
        "所有数值均来自 `metrics.best.test`。全量冷启动是 baseline；六种对比方法为“仅当前阶段”和五条已经完成的 B640 样本筛选序列。",
        "",
    ]
    for metric, (title, _filename) in METRICS.items():
        lines.extend(markdown_metric_table(grouped, metric, title))
    (WORKSPACE / "metrics_tables.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(rows: list[dict[str, object]]) -> None:
    grouped = rows_by_method(rows)
    baseline = grouped["full_cold"]
    non_baseline = METHODS[1:]
    lines = [
        "# Sequence 样本筛选实验比较总结",
        "",
        "## 比较范围",
        "",
        "- Checkpoint 和数据划分：`best/test`。",
        "- Baseline：Balance Test 全量冷启动。",
        "- 对比方法：仅当前阶段，以及五条已经完成的 B640 样本筛选序列。",
        "- 所有 B640 序列复用同一个已完成的 Full Warm Stage 0 结果。",
        "- “错误困难样本 + 当前阶段热启动”未纳入，因为其 Sequence 状态仍为 `running`，且 Stage 1 没有 `task_result.json`。",
        "- GradNorm Top-K 和 Random Top-K 未纳入，因为当前没有对应结果目录。",
        "",
        "## 各阶段最佳方法",
        "",
        "| 指标 | Stage 0 | Stage 1 | Stage 2 | Stage 3 |",
        "|---|---|---|---|---|",
    ]
    for metric, (title, _filename) in METRICS.items():
        winners = []
        for stage in range(len(STAGES)):
            stage_rows = [row for row in rows if int(row["stage"]) == stage]
            best = max(float(row[metric]) for row in stage_rows)
            labels = [method.label_zh for method in METHODS if any(str(row["method_key"]) == method.key and abs(float(row[metric]) - best) < 1e-12 for row in stage_rows)]
            winners.append(f"{', '.join(labels)} ({best:.4f})")
        lines.append(f"| {title} | " + " | ".join(winners) + " |")

    lines.extend(
        [
            "",
            "## Stage 3 相对全量冷启动 Baseline 的差值",
            "",
            "| 方法 | ΔmAP50 | ΔRecall | ΔPrecision |",
            "|---|---:|---:|---:|",
        ]
    )
    baseline_stage3 = baseline[3]
    for method in METHODS[1:]:
        row = grouped[method.key][3]
        lines.append(
            f"| {method.label_zh} | {float(row['map50']) - float(baseline_stage3['map50']):+.4f} | "
            f"{float(row['recall']) - float(baseline_stage3['recall']):+.4f} | "
            f"{float(row['precision']) - float(baseline_stage3['precision']):+.4f} |"
        )

    best_new_stage3_map = max(non_baseline, key=lambda method: float(grouped[method.key][3]["map50"]))
    best_mean_precision = max(METHODS, key=lambda method: sum(float(row["precision"]) for row in grouped[method.key]) / 4)
    pred_current = grouped["pred_current"]
    positive_cum = grouped["positive_cum"]
    current_only = grouped["current_only"]
    lines.extend(
        [
            "",
            "## 主要比较结论",
            "",
            f"- 全量冷启动在 Stage 1～3 的 mAP50 均为最高，四阶段平均 mAP50 也是最高的（{sum(float(row['map50']) for row in baseline) / 4:.4f}）。",
            f"- Stage 3 中最接近 baseline 的非 baseline 方法是“{best_new_stage3_map.label_zh}”，mAP50 为 {float(grouped[best_new_stage3_map.key][3]['map50']):.4f}，仍比全量冷启动低 {abs(float(grouped[best_new_stage3_map.key][3]['map50']) - float(baseline_stage3['map50'])):.4f}。",
            f"- “预测触发样本 + 当前阶段热启动”在 Stage 1（{float(pred_current[1]['recall']):.4f}）和 Stage 2（{float(pred_current[2]['recall']):.4f}）取得最高 Recall，但 Stage 3 降至 {float(pred_current[3]['recall']):.4f}，优势没有持续。",
            f"- “{best_mean_precision.label_zh}”的平均 Precision 最高（{sum(float(row['precision']) for row in grouped[best_mean_precision.key]) / 4:.4f}）；“仅正样本 + 累积冷启动”在 Stage 3 的 Precision 也与全量冷启动几乎持平（{float(positive_cum[3]['precision']):.4f} 对 {float(baseline_stage3['precision']):.4f}）。",
            f"- “仅当前阶段”在 Stage 3 获得最高 Precision（{float(current_only[3]['precision']):.4f}），但 Recall 只有 {float(current_only[3]['recall']):.4f}，说明 Precision 的表面提升并没有转化为更加均衡的检测性能。",
        ]
    )
    lines.extend(
        [
            "",
            "以上图表是对已完成实验的描述性比较。由于当前只有一个 seed，不能据此声称差异具有统计显著性。",
            "",
        ]
    )
    (WORKSPACE / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def fixed_sweep_path(method: Method, stage: int) -> Path:
    sequence = validate_sequence(method)
    task_result = task_result_for_stage(method, sequence, stage)
    return task_result.parent / "backend" / "yolov5" / "evaluations" / "best" / "test" / "confidence_sweep.json"


def collect_fixed_confidence_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHODS[2:]:
        for stage in range(1, 4):
            sweep_path = fixed_sweep_path(method, stage)
            if not sweep_path.is_file():
                raise FileNotFoundError(f"fixed-confidence artifact is missing for {method.label}, Stage {stage}: {sweep_path}")
            sweep = read_json(sweep_path)
            thresholds = [float(value) for value in sweep["thresholds"]]
            for confidence in FIXED_CONFIDENCES:
                try:
                    index = next(index for index, value in enumerate(thresholds) if abs(value - confidence) < 1e-9)
                except StopIteration as error:
                    raise ValueError(f"confidence {confidence} is missing from {sweep_path}") from error
                rows.append(
                    {
                        "method_key": method.key,
                        "method": method.label,
                        "method_zh": method.label_zh,
                        "stage": stage,
                        "stage_label": FIXED_STAGES[stage - 1],
                        "confidence": confidence,
                        "precision": float(sweep["overall"]["precision"][index]),
                        "recall": float(sweep["overall"]["recall"][index]),
                        "source": str(sweep_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    }
                )
    return rows


def write_fixed_confidence_csv(rows: list[dict[str, object]]) -> None:
    fields = ["method_key", "method", "method_zh", "stage", "stage_label", "confidence", "precision", "recall", "source"]
    with (WORKSPACE / "fixed_confidence_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "confidence": f"{row['confidence']:.1f}",
                    "precision": f"{row['precision']:.8f}",
                    "recall": f"{row['recall']:.8f}",
                }
            )


def plot_fixed_confidence(rows: list[dict[str, object]], confidence: float, metric: str, metric_zh: str) -> None:
    selected = [row for row in rows if abs(float(row["confidence"]) - confidence) < 1e-9]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault(str(row["method_key"]), []).append(row)
    values = [float(row[metric]) for row in selected]
    ymin, ymax = padded_domain(values)
    colors = plt.get_cmap("tab10").colors
    markers = ("^", "D", "v", "P", "X")
    lines = ("-", "-.", "-", "-.", ":")
    fig, ax = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    for index, method in enumerate(METHODS[2:]):
        method_rows = sorted(grouped[method.key], key=lambda row: int(row["stage"]))
        ax.plot(
            range(1, 4),
            [float(row[metric]) for row in method_rows],
            label=method.label,
            color=colors[(index + 2) % len(colors)],
            marker=markers[index],
            linestyle=lines[index],
            linewidth=2.0,
            markersize=7,
        )
    ax.set_title(f"Fixed confidence {confidence:.1f}: {metric_zh} by Stage", fontsize=18, pad=16)
    ax.set_xlabel("Training stage", fontsize=13)
    ax.set_ylabel(f"Fixed-confidence test {metric_zh} (0–1)", fontsize=13)
    ax.set_xticks(range(1, 4), FIXED_STAGES)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.7, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    filename = f"{metric}_conf{confidence:.1f}".replace(".", "p") + ".png"
    fig.savefig(WORKSPACE / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_fixed_confidence_tables(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 固定置信度 Precision / Recall 指标",
        "",
        "本组结果只使用已经保存多置信度产物的五条 B640 筛选序列，比较 Stage 1～3；Full Cold、Current Only 和共享 Stage 0 没有纳入。",
        "",
    ]
    for confidence in FIXED_CONFIDENCES:
        lines.extend([f"## 置信度 {confidence:.1f}", "", "| 方法 | Stage | Precision | Recall |", "|---|---:|---:|---:|"])
        selected = [row for row in rows if abs(float(row["confidence"]) - confidence) < 1e-9]
        for row in sorted(selected, key=lambda row: (str(row["method_zh"]), int(row["stage"]))):
            lines.append(f"| {row['method_zh']} | {row['stage_label']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} |")
        lines.append("")
    (WORKSPACE / "fixed_confidence_tables.md").write_text("\n".join(lines), encoding="utf-8")


def without_first_heading(text: str, *, demote: bool = False) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    if demote:
        lines = [f"#{line}" if line.startswith("## ") else line for line in lines]
    return "\n".join(lines)


def write_complete_report() -> None:
    summary = without_first_heading((WORKSPACE / "comparison_summary.md").read_text(encoding="utf-8"), demote=True)
    tables = without_first_heading((WORKSPACE / "metrics_tables.md").read_text(encoding="utf-8"), demote=True)
    fixed_tables = without_first_heading((WORKSPACE / "fixed_confidence_tables.md").read_text(encoding="utf-8"), demote=True)
    fixed_figures: list[str] = []
    for confidence in FIXED_CONFIDENCES:
        token = f"{confidence:.1f}".replace(".", "p")
        fixed_figures.extend(
            [
                f"### 置信度 {confidence:.1f} 的 Precision",
                "",
                f"![固定置信度 {confidence:.1f} Precision](precision_conf{token}.png)",
                "",
                f"### 置信度 {confidence:.1f} 的 Recall",
                "",
                f"![固定置信度 {confidence:.1f} Recall](recall_conf{token}.png)",
                "",
            ]
        )
    report = "\n".join(
        [
            "# Sequence 样本筛选实验完整对比报告",
            "",
            "本报告比较六种已完成方法与 Balance Test 全量冷启动 baseline。所有指标均使用 `metrics.best.test`。",
            "",
            "## 各阶段 mAP50",
            "",
            "![mAP50 comparison](map50.png)",
            "",
            "## 各阶段 Recall",
            "",
            "![Recall comparison](recall.png)",
            "",
            "## 各阶段 Precision",
            "",
            "![Precision comparison](precision.png)",
            "",
            "## 比较与结论",
            "",
            summary,
            "",
            "## 完整指标表",
            "",
            tables,
            "",
            "## 固定置信度 Precision / Recall",
            "",
            "Full Cold、Current Only 和共享 Stage 0 没有保存多置信度产物，因此本节只比较五条已有固定置信度结果的 B640 序列，阶段从 Stage 1 开始。",
            "",
            *fixed_figures,
            fixed_tables,
            "",
            "## 复现信息",
            "",
            "- 长表格式数值数据：[`metrics.csv`](metrics.csv)",
            "- 指标提取和绘图脚本：[`compare_methods.py`](compare_methods.py)",
            "- 固定置信度指标：[`fixed_confidence_metrics.csv`](fixed_confidence_metrics.csv)",
            "- 固定置信度表格：[`fixed_confidence_tables.md`](fixed_confidence_tables.md)",
            "",
        ]
    )
    (WORKSPACE / "complete_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    write_csv_table(rows)
    write_metrics_markdown(rows)
    write_summary(rows)
    for metric, (title, filename) in METRICS.items():
        plot_metric(rows, metric, title, filename)
    fixed_rows = collect_fixed_confidence_rows()
    write_fixed_confidence_csv(fixed_rows)
    write_fixed_confidence_tables(fixed_rows)
    for confidence in FIXED_CONFIDENCES:
        plot_fixed_confidence(fixed_rows, confidence, "precision", "Precision")
        plot_fixed_confidence(fixed_rows, confidence, "recall", "Recall")
    write_complete_report()
    print(f"wrote {len(rows)} regular metric rows, {len(fixed_rows)} fixed-confidence rows, 13 plots, and complete_report.md to {WORKSPACE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
