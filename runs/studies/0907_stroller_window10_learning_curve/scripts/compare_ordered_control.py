"""Compare the completed random split curves with the source-ordered control."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_learning_curve import CURVES, collect_curve, validate


STUDY_DIR = Path(__file__).resolve().parents[1]
SEQUENCES_DIR = STUDY_DIR / "experiment" / "sequence"
RESULTS_DIR = STUDY_DIR / "result" / "learning_curve"
ORDERED_SEQUENCE = "StrollerWindow10Tau099_Ordered__seq-full-cold__stage0-stage7__yolov5s__s42"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def collect_ordered_curve() -> list[dict[str, Any]]:
    sequence_dir = SEQUENCES_DIR / ORDERED_SEQUENCE
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    rows = collect_curve(-1, ORDERED_SEQUENCE)
    for row in rows:
        row["split_mode"] = "ordered"
    return rows


def validate_ordered(random_rows: list[dict[str, Any]], ordered_rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_protocol = validate(random_rows)
    ordered_x = [row["cumulative_train_images"] for row in sorted(ordered_rows, key=lambda item: item["stage"])]
    if ordered_x != random_protocol["x_points"]:
        raise ValueError(f"ordered and random cumulative training points differ: {ordered_x} != {random_protocol['x_points']}")
    if any(row["test_images"] != 80 for row in ordered_rows):
        raise ValueError("the ordered control requires 80 test images")
    return {"random_protocol": random_protocol, "ordered_x_points": ordered_x}


def write_csv(rows: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "metrics_with_ordered_control.csv"
    fields = [
        "curve",
        "split_seed",
        "split_mode",
        "stage",
        "cumulative_train_images",
        "test_map50_95",
        "test_map50",
        "test_precision",
        "test_recall",
        "test_images",
        "task_dir",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def plot(random_rows: list[dict[str, Any]], ordered_rows: list[dict[str, Any]]) -> Path:
    colors = {41: "#E07A5F", 42: "#176B87", 43: "#7B61A8"}
    labels = {41: "random split seed 41", 42: "random split seed 42", 43: "random split seed 43"}
    ordered = sorted(ordered_rows, key=lambda row: row["stage"])
    x = [row["cumulative_train_images"] for row in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), sharex=True)
    fig.patch.set_facecolor("#F7F9FB")
    for axis, metric, title in zip(
        axes,
        ("test_map50_95", "test_map50"),
        ("mAP@0.5:0.95", "mAP@0.5"),
    ):
        axis.set_facecolor("white")
        for split_seed in (41, 42, 43):
            curve = sorted(
                (row for row in random_rows if row["split_seed"] == split_seed),
                key=lambda row: row["stage"],
            )
            axis.plot(
                [row["cumulative_train_images"] for row in curve],
                [row[metric] for row in curve],
                marker="o",
                linewidth=2.1,
                markersize=5.2,
                color=colors[split_seed],
                label=labels[split_seed],
            )
        axis.plot(
            x,
            [row[metric] for row in ordered],
            marker="D",
            linestyle="--",
            linewidth=2.6,
            markersize=5.4,
            color="#172033",
            label="ordered source order",
        )
        axis.set_title(title, fontweight="bold", pad=12)
        axis.set_xlabel("Cumulative training images")
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:,}" for value in x], rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Test metric (best checkpoint)")
    fig.suptitle("stroller_reviewed window=10 τ=0.99 — Ordered Split Control", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.015,
        "Full Cold | random curves use split seeds 41/42/43 | ordered curve preserves the retained manifest order",
        ha="center",
        color="#52616B",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93), w_pad=3.0)
    path = RESULTS_DIR / "learning_curve_with_ordered_control.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(RESULTS_DIR / "learning_curve_with_ordered_control.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    random_rows = [row for seed, sequence in CURVES.items() for row in collect_curve(seed, sequence)]
    ordered_rows = collect_ordered_curve()
    protocol = validate_ordered(random_rows, ordered_rows)
    for row in random_rows:
        row["curve"] = f"random_split_s{row['split_seed']}"
        row["split_mode"] = "random"
    for row in ordered_rows:
        row["curve"] = "ordered"
    csv_path = write_csv(random_rows + ordered_rows)
    image_path = plot(random_rows, ordered_rows)
    summary = {
        "study": STUDY_DIR.name,
        "comparison": "random split seeds versus source-ordered split control",
        "random_split_seeds": [41, 42, 43],
        "ordered_sequence": ORDERED_SEQUENCE,
        "training_seed_schedule": "42-49 for every sequence",
        "metric_source": "task_result.json -> metrics.best.test",
        "x_axis": "task_result.json -> cost.selected_count",
        "test_images_per_curve": 80,
        "protocol_validation": protocol,
        "outputs": {
            "csv": csv_path.name,
            "png": image_path.name,
            "svg": "learning_curve_with_ordered_control.svg",
        },
    }
    (RESULTS_DIR / "ordered_control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
