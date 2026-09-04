"""Generate the completed Global Dedup tau=0.99 learning-curve summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


STUDY_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = STUDY_DIR / "result" / "learning_curve"
SEQUENCES_DIR = STUDY_DIR / "experiment" / "sequence"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def find_completed_sequence() -> Path:
    sequence_dirs = sorted(path for path in SEQUENCES_DIR.iterdir() if path.is_dir())
    if len(sequence_dirs) != 1:
        raise ValueError(f"Expected exactly one sequence, found {len(sequence_dirs)}")
    sequence_dir = sequence_dirs[0]
    sequence_status = read_json(sequence_dir / "sequence_status.json")
    if sequence_status.get("status") != "completed":
        raise ValueError(f"Sequence is not completed: {sequence_dir}")
    if sequence_status.get("completed_tasks") != 8:
        raise ValueError(f"Expected 8 completed tasks: {sequence_status}")
    return sequence_dir


def collect_rows(sequence_dir: Path) -> list[dict[str, Any]]:
    task_dirs = sorted(path for path in (sequence_dir / "tasks").iterdir() if path.is_dir())
    if len(task_dirs) != 8:
        raise ValueError(f"Expected 8 task directories, found {len(task_dirs)}")

    rows: list[dict[str, Any]] = []
    for stage, task_dir in enumerate(task_dirs):
        status = read_json(task_dir / "task_status.json")
        result = read_json(task_dir / "task_result.json")
        if status.get("status") != "completed" or result.get("status") != "completed":
            raise ValueError(f"Task is not completed: {task_dir}")

        metrics = result.get("metrics", {}).get("best", {}).get("test", {})
        cost = result.get("cost", {})
        required_metrics = ("map50_95", "map50", "precision", "recall")
        missing = [name for name in required_metrics if metrics.get(name) is None]
        if missing:
            raise ValueError(f"Missing best/test metrics {missing}: {task_dir}")
        if cost.get("selected_count") is None:
            raise ValueError(f"Missing selected_count: {task_dir}")

        rows.append(
            {
                "stage": stage,
                "cumulative_train_images": int(cost["selected_count"]),
                "test_map50_95": float(metrics["map50_95"]),
                "test_map50": float(metrics["map50"]),
                "test_precision": float(metrics["precision"]),
                "test_recall": float(metrics["recall"]),
                "checkpoint": "best",
                "test_images": 1000,
                "task_dir": str(task_dir.relative_to(STUDY_DIR)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    output_path = RESULTS_DIR / "learning_curve_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    x = [row["cumulative_train_images"] for row in rows]
    colors = {
        "map50_95": "#176B87",
        "map50": "#64CCC5",
        "precision": "#E07A5F",
        "recall": "#F2CC8F",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#BAC4CF",
            "axes.labelcolor": "#263746",
            "xtick.color": "#52616B",
            "ytick.color": "#52616B",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True)
    fig.patch.set_facecolor("#F7F9FB")

    panels = [
        (
            axes[0],
            [("test_map50_95", "mAP@0.5:0.95", colors["map50_95"]), ("test_map50", "mAP@0.5", colors["map50"])],
            "Mean Average Precision",
        ),
        (
            axes[1],
            [("test_precision", "Precision", colors["precision"]), ("test_recall", "Recall", colors["recall"])],
            "Precision and Recall",
        ),
    ]

    for axis, series, title in panels:
        axis.set_facecolor("white")
        for key, label, color in series:
            values = [row[key] for row in rows]
            axis.plot(
                x,
                values,
                color=color,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.2,
                linewidth=2.4,
                label=label,
            )
            axis.annotate(
                f"{values[-1]:.3f}",
                (x[-1], values[-1]),
                xytext=(-7, 9),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
        axis.set_title(title, pad=12)
        axis.set_xlabel("Cumulative training images")
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:,}" for value in x], rotation=35, ha="right")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8, alpha=0.9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")

    axes[0].set_ylabel("Test metric (best checkpoint)")
    axes[0].set_ylim(0.20, 0.68)
    axes[1].set_ylim(0.35, 0.92)
    fig.suptitle("Global Dedup τ=0.99 — Full Cold Learning Curve", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.015,
        "Seed 42  |  fixed test set: 1,000 images  |  each point uses the task's best checkpoint",
        ha="center",
        color="#52616B",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93), w_pad=3.0)

    png_path = RESULTS_DIR / "learning_curve.png"
    svg_path = RESULTS_DIR / "learning_curve.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, svg_path


def main() -> int:
    sequence_dir = find_completed_sequence()
    rows = collect_rows(sequence_dir)
    csv_path = write_csv(rows)
    png_path, svg_path = plot(rows)
    summary = {
        "study": STUDY_DIR.name,
        "sequence_status": "completed",
        "completed_stages": len(rows),
        "metric_source": "task_result.json -> metrics.best.test",
        "x_axis": "task_result.json -> cost.selected_count",
        "test_images": 1000,
        "final_stage": rows[-1],
        "outputs": {
            "csv": csv_path.name,
            "png": png_path.name,
            "svg": svg_path.name,
        },
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
