"""Generate learning-curve plots for the random train/val/test assignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SEQUENCE_DIR = (
    PROJECT_ROOT
    / "runs"
    / "studies"
    / "0824_fullcold_redundancy_and_random_test"
    / "experiment"
    / "sequence"
    / "random_frame_s42__seq-full-cold__stage0-stage3__yolov5s__s42"
)
EXPERIMENT = "random_frame_s42"
CLASS_NAMES = ["large luggage", "stroller", "wheelchair", "flatbed truck"]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_per_class_ap50(task_path: Path) -> dict[str, float]:
    """Read YOLOv5's per-class mAP50 column from its evaluation log.

    The older bridge used for this run returned only ``per_class_ap``
    (AP50-95) in JSON, although ``val.py`` printed the class AP50 values.
    """

    log_path = task_path / "backend" / "yolov5" / "evaluations" / "best" / "test" / "evaluation.log"
    if not log_path.is_file():
        raise ValueError(f"missing evaluation log for per-class AP50: {log_path}")

    values: dict[str, float] = {}
    class_names = sorted(CLASS_NAMES, key=len, reverse=True)
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        for class_name in class_names:
            if not line.startswith(class_name):
                continue
            fields = line[len(class_name) :].split()
            # images, instances, precision, recall, AP50, AP50-95
            if len(fields) < 6:
                continue
            try:
                values[class_name] = float(fields[-2])
            except ValueError:
                continue
            break

    missing = [name for name in CLASS_NAMES if name not in values]
    if missing:
        raise ValueError(f"evaluation log has no per-class AP50 for {missing}: {log_path}")
    return values


def load_rows() -> list[dict[str, Any]]:
    sequence_result = _read_json(SEQUENCE_DIR / "sequence_result.json")
    if sequence_result.get("status") != "completed":
        raise ValueError(f"sequence is not completed: {SEQUENCE_DIR}")

    task_paths = sequence_result.get("tasks", [])
    if len(task_paths) != 4:
        raise ValueError(f"expected four completed stages, got {len(task_paths)}")

    rows: list[dict[str, Any]] = []
    for stage, raw_task_path in enumerate(task_paths):
        # Historical JSON keeps the server path from before the Study was
        # normalized. Resolve by task basename against the current checkout.
        task_path = SEQUENCE_DIR / "tasks" / Path(raw_task_path).name
        task = _read_json(task_path / "task_result.json")
        if task.get("status") != "completed":
            raise ValueError(f"task is not completed: {task_path}")
        metrics = task.get("metrics", {}).get("best", {}).get("test", {})
        per_class_ap = metrics.get("per_class_ap", {})
        if not isinstance(per_class_ap, dict):
            raise ValueError(f"missing per-class AP values: {task_path}")
        missing = [name for name in CLASS_NAMES if name not in per_class_ap]
        if missing:
            raise ValueError(f"missing classes {missing} in {task_path}")
        per_class_ap50 = _read_per_class_ap50(task_path)

        cost = task.get("cost", {})
        source = task_path / "task_result.json"
        try:
            source_name = str(source.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            source_name = str(source.resolve())
        row: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "stage": stage,
            "selected_count": cost.get("selected_count"),
            "candidate_count": cost.get("candidate_count"),
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "source": source_name,
        }
        for class_name in CLASS_NAMES:
            row[f"ap50.{class_name}"] = per_class_ap50[class_name]
            row[f"ap50_95.{class_name}"] = per_class_ap[class_name]
        rows.append(row)

    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "experiment",
        "stage",
        "selected_count",
        "candidate_count",
        "map50",
        "map50_95",
        "precision",
        "recall",
        *[f"ap50.{name}" for name in CLASS_NAMES],
        *[f"ap50_95.{name}" for name in CLASS_NAMES],
        "source",
    ]
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [row["selected_count"] for row in rows]

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=180)
    axis.plot(x_values, [row["map50"] for row in rows], marker="o", linewidth=2.2, label="Test mAP50")
    axis.plot(
        x_values,
        [row["map50_95"] for row in rows],
        marker="o",
        linewidth=2.2,
        label="Test mAP50-95",
    )
    axis.set_xlabel("Cumulative training images")
    axis.set_ylabel("Metric (best checkpoint / random test)")
    axis.set_title("Random-test Full Cold learning curve")
    axis.set_xticks(x_values)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "map_learning_curve.png", bbox_inches="tight")
    plt.close(fig)

    def plot_per_class(prefix: str, ylabel: str, title: str, filename: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for axis, class_name in zip(axes.flat, CLASS_NAMES):
            axis.plot(
                x_values,
                [row[f"{prefix}.{class_name}"] for row in rows],
                marker="o",
                linewidth=1.9,
                color="#2563eb",
            )
            axis.set_title(class_name)
            axis.set_ylabel(ylabel)
            axis.set_xticks(x_values)
            axis.grid(alpha=0.25)
        for axis in axes[-1]:
            axis.set_xlabel("Cumulative training images")
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close(fig)

    plot_per_class(
        "ap50",
        "Test AP50",
        "Random-test per-class AP50 learning curves",
        "per_class_ap50_learning_curve.png",
    )
    plot_per_class(
        "ap50_95",
        "Test AP50-95",
        "Random-test per-class AP50-95 learning curves",
        "per_class_ap_learning_curve.png",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_csv(rows)
    plot_curves(rows)
    summary = {
        "experiment": EXPERIMENT,
        "sequence": str(SEQUENCE_DIR.resolve()),
        "test_split": "randomly reassigned test group",
        "rows": len(rows),
        "metrics_csv": str((OUTPUT_DIR / "metrics.csv").resolve()),
        "map_plot": str((OUTPUT_DIR / "map_learning_curve.png").resolve()),
        "per_class_ap50_plot": str((OUTPUT_DIR / "per_class_ap50_learning_curve.png").resolve()),
        "per_class_plot": str((OUTPUT_DIR / "per_class_ap_learning_curve.png").resolve()),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
