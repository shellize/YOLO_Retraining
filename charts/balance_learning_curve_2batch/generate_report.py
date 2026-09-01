"""Generate the metrics and plots stored alongside this script."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _parse_sequence(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sequence must use EXPERIMENT=OUTPUT_DIR")
    experiment, raw_path = value.split("=", 1)
    if not experiment:
        raise argparse.ArgumentTypeError("sequence experiment name cannot be empty")
    return experiment, Path(raw_path).expanduser().resolve()


def _stage_batches(sequence_dir: Path) -> dict[str, list[str]]:
    sequence = yaml.safe_load((sequence_dir / "sequence.yaml").read_text(encoding="utf-8")) or {}
    layout_path = Path(sequence["data"]["layout"])
    layout = yaml.safe_load(layout_path.read_text(encoding="utf-8")) or {}
    result: dict[str, list[str]] = {}
    for arrival in sequence["arrivals"]:
        stage = str(arrival["id"])
        images = layout["groups"][stage]["images"]
        result[stage] = [Path(value).name.removeprefix("0720_") for value in images]
    return result


def collect_sequence(experiment: str, sequence_dir: Path) -> list[dict[str, Any]]:
    sequence_result = _read_json(sequence_dir / "sequence_result.json")
    if sequence_result.get("status") != "completed":
        raise ValueError(f"sequence is not completed: {sequence_dir}")
    stage_batches = _stage_batches(sequence_dir)
    cumulative_batches: list[str] = []
    rows: list[dict[str, Any]] = []
    for stage_index, task_path in enumerate(sequence_result.get("tasks", [])):
        task = _read_json(Path(task_path) / "task_result.json")
        metrics = task.get("metrics", {}).get("best", {}).get("test", {})
        per_class_ap50 = metrics.get("per_class_ap50")
        per_class_ap50_95 = metrics.get("per_class_ap", {})
        if not isinstance(per_class_ap50, dict):
            raise ValueError(f"task result does not contain per-class AP50: {task_path}")
        cost = task.get("cost", {})
        stage_id = f"stage{stage_index}"
        new_batches = stage_batches[stage_id]
        cumulative_batches.extend(new_batches)
        row: dict[str, Any] = {
            "experiment": experiment,
            "stage": stage_index,
            "new_batches": "+".join(new_batches),
            "cumulative_batches": "+".join(cumulative_batches),
            "batch_count": len(cumulative_batches),
            "selected_count": cost.get("selected_count"),
            "images_read": cost.get("images_read"),
            "optimizer_steps": cost.get("optimizer_steps"),
            "training_seconds": cost.get("training_seconds"),
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "task_output": task.get("output_dir"),
        }
        for class_name, value in per_class_ap50.items():
            row[f"ap50.{class_name}"] = value
        for class_name, value in per_class_ap50_95.items():
            row[f"ap50_95.{class_name}"] = value
        rows.append(row)
    if len(rows) != 8:
        raise ValueError(f"expected eight completed learning-curve stages, got {len(rows)}: {sequence_dir}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "ordered": "Ordered batches",
        "random_s41": "Random batches (seed 41)",
        "random_s43": "Random batches (seed 43)",
    }
    experiments = list(dict.fromkeys(str(row["experiment"]) for row in rows))

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for experiment in experiments:
        selected = [row for row in rows if row["experiment"] == experiment]
        axis.plot(
            [row["selected_count"] for row in selected],
            [row["map50"] for row in selected],
            marker="o",
            linewidth=2,
            label=labels.get(experiment, experiment),
        )
    axis.set_xlabel("Cumulative training images")
    axis.set_ylabel("Test mAP50 (best checkpoint)")
    axis.set_title("Balance Test Full Cold learning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "map50_learning_curve.png", dpi=180)
    plt.close(fig)

    class_names = ["large luggage", "stroller", "wheelchair", "flatbed truck"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, class_name in zip(axes.flat, class_names):
        for experiment in experiments:
            selected = [row for row in rows if row["experiment"] == experiment]
            axis.plot(
                [row["selected_count"] for row in selected],
                [row.get(f"ap50.{class_name}") for row in selected],
                marker="o",
                linewidth=1.8,
                label=labels.get(experiment, experiment),
            )
        axis.set_title(class_name)
        axis.set_ylabel("Test AP50")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Cumulative training images")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Per-class AP50 learning curves", y=0.995)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "per_class_ap_learning_curve.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Balance Test two-batch Full Cold learning curves.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", type=_parse_sequence, action="append", required=True)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for experiment, sequence_dir in args.sequence:
        if experiment in seen:
            raise ValueError(f"duplicate experiment name: {experiment}")
        seen.add(experiment)
        rows.extend(collect_sequence(experiment, sequence_dir))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    _plot(rows, output_dir)
    payload = {
        "experiments": list(seen),
        "rows": len(rows),
        "metrics_csv": str((output_dir / "metrics.csv").resolve()),
        "map50_plot": str((output_dir / "map50_learning_curve.png").resolve()),
        "per_class_plot": str((output_dir / "per_class_ap_learning_curve.png").resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
