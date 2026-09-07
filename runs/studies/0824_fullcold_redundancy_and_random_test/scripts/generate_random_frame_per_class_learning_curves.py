"""Generate per-class AP@0.5:0.95 curves for the random-frame sequence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDY_DIR = Path(__file__).resolve().parents[1]
SEQUENCE_DIR = (
    STUDY_DIR
    / "experiment"
    / "sequence"
    / "random_frame_s42__seq-full-cold__stage0-stage3__yolov5s__s42"
)
RESULTS_DIR = STUDY_DIR / "result" / "random_frame_per_class_learning_curve"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def collect_rows() -> list[dict[str, Any]]:
    status = read_json(SEQUENCE_DIR / "sequence_status.json")
    if status.get("status") != "completed" or status.get("completed_tasks") != 4:
        raise ValueError(f"Sequence is not complete: {SEQUENCE_DIR}")

    task_dirs = sorted(path for path in (SEQUENCE_DIR / "tasks").iterdir() if path.is_dir())
    if len(task_dirs) != 4:
        raise ValueError(f"Expected 4 tasks, found {len(task_dirs)}: {SEQUENCE_DIR}")

    rows: list[dict[str, Any]] = []
    classes: set[str] | None = None
    for stage, task_dir in enumerate(task_dirs):
        task_status = read_json(task_dir / "task_status.json")
        task_result = read_json(task_dir / "task_result.json")
        if task_status.get("status") != "completed" or task_result.get("status") != "completed":
            raise ValueError(f"Task is not complete: {task_dir}")

        metrics = task_result.get("metrics", {}).get("best", {}).get("test", {})
        cost = task_result.get("cost", {})
        per_class_ap = metrics.get("per_class_ap")
        selected_count = cost.get("selected_count")
        if not isinstance(per_class_ap, dict):
            raise ValueError(f"Missing per-class AP metric: {task_dir}")
        if selected_count is None:
            raise ValueError(f"Missing selected_count: {task_dir}")
        current_classes = set(per_class_ap)
        if classes is None:
            classes = current_classes
        elif classes != current_classes:
            raise ValueError(f"Class set changes across stages: {task_dir}")

        for class_name in sorted(current_classes):
            rows.append(
                {
                    "stage": stage,
                    "cumulative_train_images": int(selected_count),
                    "class_name": class_name,
                    "ap50_95": float(per_class_ap[class_name]),
                    "test_images": int(metrics.get("sample_count", 0)),
                    "task_dir": str(task_dir.relative_to(STUDY_DIR)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "per_class_ap_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict[str, Any]]) -> Path:
    classes = sorted({str(row["class_name"]) for row in rows})
    x_values = sorted({int(row["cumulative_train_images"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True)
    axes_flat = list(axes.flat)
    for axis, class_name in zip(axes_flat, classes):
        selected = sorted(
            (row for row in rows if row["class_name"] == class_name),
            key=lambda row: row["stage"],
        )
        axis.plot(
            [row["cumulative_train_images"] for row in selected],
            [row["ap50_95"] for row in selected],
            marker="o",
            markersize=5,
            linewidth=2.2,
            color="#176B87",
            label="random frame, split seed 42",
        )
        axis.set_title(class_name, fontweight="bold")
        axis.set_ylabel("Test AP@0.5:0.95")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("Cumulative training images")
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"{value:,}" for value in x_values], rotation=35, ha="right")
    for axis in axes_flat[len(classes) :]:
        axis.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(
        "Random-frame experiment: per-class AP learning curves",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.89), w_pad=2.8, h_pad=2.4)
    path = RESULTS_DIR / "per_class_ap50_95_learning_curve.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="#F7F9FB")
    plt.close(fig)
    return path


def main() -> int:
    rows = collect_rows()
    if not rows:
        raise ValueError("No per-class AP rows were found")
    csv_path = write_csv(rows)
    plot_path = plot(rows)
    summary = {
        "study": STUDY_DIR.name,
        "sequence": SEQUENCE_DIR.name,
        "metric_source": "task_result.json -> metrics.best.test.per_class_ap",
        "metric_definition": "per-class AP@0.5:0.95; per-class AP@0.5 is not available in this historical result",
        "x_axis": "task_result.json -> cost.selected_count",
        "test_images_per_stage": sorted({int(row["test_images"]) for row in rows}),
        "outputs": {"csv": csv_path.name, "plot": plot_path.name},
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
