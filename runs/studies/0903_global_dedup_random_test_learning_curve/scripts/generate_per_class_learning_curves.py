"""Generate per-class mAP learning curves from canonical task results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDY_DIR = Path(__file__).resolve().parents[1]
SEQUENCES_DIR = STUDY_DIR / "experiment" / "sequence"
RESULTS_DIR = STUDY_DIR / "result" / "per_class_learning_curve"

SEQUENCE_CANDIDATES = {
    "split_s42": (
        (
            "GlobalDedupTau099_PostSplit__seq-full-cold__stage0-stage7__yolov5s__s42",
        ),
        "split seed 42",
    ),
    "split_s41": (
        (
            "GlobalDedupTau099_PostSplit_RandomS41_Rerun__seq-full-cold__stage0-stage7__yolov5s__s42",
            "GlobalDedupTau099_PostSplit_RandomS41__seq-full-cold__stage0-stage7__yolov5s__s42",
        ),
        "split seed 41",
    ),
    "split_s43": (
        (
            "GlobalDedupTau099_PostSplit_RandomS43_Rerun__seq-full-cold__stage0-stage7__yolov5s__s42",
            "GlobalDedupTau099_PostSplit_RandomS43__seq-full-cold__stage0-stage7__yolov5s__s42",
        ),
        "split seed 43",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def completed_sequence(sequence_dir: Path) -> bool:
    status_path = sequence_dir / "sequence_status.json"
    if not status_path.is_file():
        return False
    status = read_json(status_path)
    return status.get("status") == "completed" and status.get("completed_tasks") == 8


def collect_sequence(experiment: str, label: str, sequence_dir: Path) -> list[dict[str, Any]]:
    task_dirs = sorted(path for path in (sequence_dir / "tasks").iterdir() if path.is_dir())
    if len(task_dirs) != 8:
        raise ValueError(f"Expected 8 tasks, found {len(task_dirs)}: {sequence_dir}")

    rows: list[dict[str, Any]] = []
    classes: set[str] | None = None
    for stage, task_dir in enumerate(task_dirs):
        task_status = read_json(task_dir / "task_status.json")
        task_result = read_json(task_dir / "task_result.json")
        if task_status.get("status") != "completed" or task_result.get("status") != "completed":
            raise ValueError(f"Task is not complete: {task_dir}")
        metrics = task_result.get("metrics", {}).get("best", {}).get("test", {})
        cost = task_result.get("cost", {})
        ap50 = metrics.get("per_class_ap50")
        ap50_95 = metrics.get("per_class_ap")
        if not isinstance(ap50, dict) or not isinstance(ap50_95, dict):
            raise ValueError(f"Missing per-class AP metrics: {task_dir}")
        if set(ap50) != set(ap50_95):
            raise ValueError(f"AP50/AP50-95 class sets differ: {task_dir}")
        if classes is None:
            classes = set(ap50)
        elif classes != set(ap50):
            raise ValueError(f"Class set changes across stages: {task_dir}")
        if cost.get("selected_count") is None:
            raise ValueError(f"Missing selected_count: {task_dir}")
        for class_name in sorted(classes):
            rows.append(
                {
                    "experiment": experiment,
                    "label": label,
                    "stage": stage,
                    "cumulative_train_images": int(cost["selected_count"]),
                    "class_name": class_name,
                    "map50": float(ap50[class_name]),
                    "map50_95": float(ap50_95[class_name]),
                    "test_images": int(metrics.get("sample_count", 0)),
                    "task_dir": str(task_dir.relative_to(STUDY_DIR)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "per_class_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict[str, Any]], metric: str, filename: str, title: str) -> Path:
    classes = sorted({str(row["class_name"]) for row in rows})
    experiments = list(dict.fromkeys((str(row["experiment"]), str(row["label"])) for row in rows))
    colors = {"split_s42": "#176B87", "split_s41": "#E07A5F", "split_s43": "#7B61A8"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True)
    axes_flat = list(axes.flat)
    for axis, class_name in zip(axes_flat, classes):
        axis.set_facecolor("white")
        for experiment, label in experiments:
            selected = sorted(
                (row for row in rows if row["experiment"] == experiment and row["class_name"] == class_name),
                key=lambda row: row["stage"],
            )
            axis.plot(
                [row["cumulative_train_images"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                markersize=5,
                linewidth=2.0,
                color=colors.get(experiment, "#52616B"),
                label=label,
            )
        axis.set_title(class_name, fontweight="bold")
        axis.set_ylabel("Test mAP")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Cumulative training images")
        axis.set_xticks(sorted({int(row["cumulative_train_images"]) for row in rows}))
        axis.set_xticklabels(
            [f"{value:,}" for value in sorted({int(row["cumulative_train_images"]) for row in rows})],
            rotation=35,
            ha="right",
        )
    for axis in axes_flat[len(classes) :]:
        axis.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.89), w_pad=2.8, h_pad=2.4)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="#F7F9FB")
    plt.close(fig)
    return path


def main() -> int:
    rows: list[dict[str, Any]] = []
    included: list[str] = []
    omitted: list[str] = []
    resolved_sequences: dict[str, str] = {}
    for experiment, (candidates, label) in SEQUENCE_CANDIDATES.items():
        sequence_name = next(
            (candidate for candidate in candidates if completed_sequence(SEQUENCES_DIR / candidate)),
            None,
        )
        if sequence_name is None:
            omitted.append(experiment)
            continue
        included.append(experiment)
        resolved_sequences[experiment] = sequence_name
        rows.extend(collect_sequence(experiment, label, SEQUENCES_DIR / sequence_name))
    if not rows:
        raise ValueError("No completed sequence is available")

    csv_path = write_csv(rows)
    ap50_path = plot(rows, "map50", "per_class_map50_learning_curve.png", "Per-class mAP@0.5 learning curves")
    ap50_95_path = plot(
        rows,
        "map50_95",
        "per_class_map50_95_learning_curve.png",
        "Per-class mAP@0.5:0.95 learning curves",
    )
    summary = {
        "study": STUDY_DIR.name,
        "included_experiments": included,
        "omitted_incomplete_experiments": omitted,
        "resolved_sequences": resolved_sequences,
        "metric_source": "task_result.json -> metrics.best.test.per_class_ap50/per_class_ap",
        "x_axis": "task_result.json -> cost.selected_count",
        "outputs": {
            "csv": csv_path.name,
            "map50_plot": ap50_path.name,
            "map50_95_plot": ap50_95_path.name,
        },
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
