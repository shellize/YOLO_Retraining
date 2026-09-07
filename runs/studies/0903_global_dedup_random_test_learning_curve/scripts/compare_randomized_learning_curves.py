"""Compare the completed learning curves across post-deduplication split seeds."""

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
RESULTS_DIR = STUDY_DIR / "result" / "randomization_comparison"

CURVE_CANDIDATES = {
    42: (
        "GlobalDedupTau099_PostSplit__seq-full-cold__stage0-stage7__yolov5s__s42",
    ),
    41: (
        "GlobalDedupTau099_PostSplit_RandomS41__seq-full-cold__stage0-stage7__yolov5s__s42",
        "GlobalDedupTau099_PostSplit_RandomS41_Rerun__seq-full-cold__stage0-stage7__yolov5s__s42",
    ),
    43: (
        "GlobalDedupTau099_PostSplit_RandomS43__seq-full-cold__stage0-stage7__yolov5s__s42",
        "GlobalDedupTau099_PostSplit_RandomS43_Rerun__seq-full-cold__stage0-stage7__yolov5s__s42",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def collect_curve(split_seed: int, sequence_name: str) -> list[dict[str, Any]]:
    sequence_dir = SEQUENCES_DIR / sequence_name
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    status = read_json(sequence_dir / "sequence_status.json")
    if status.get("status") != "completed" or status.get("completed_tasks") != 8:
        raise ValueError(f"Sequence is not complete: {sequence_dir} ({status})")

    task_dirs = sorted(path for path in (sequence_dir / "tasks").iterdir() if path.is_dir())
    if len(task_dirs) != 8:
        raise ValueError(f"Expected 8 task directories, found {len(task_dirs)}: {sequence_dir}")

    rows: list[dict[str, Any]] = []
    for stage, task_dir in enumerate(task_dirs):
        task_status = read_json(task_dir / "task_status.json")
        task_result = read_json(task_dir / "task_result.json")
        if task_status.get("status") != "completed" or task_result.get("status") != "completed":
            raise ValueError(f"Task is not complete: {task_dir}")
        metrics = task_result.get("metrics", {}).get("best", {}).get("test", {})
        cost = task_result.get("cost", {})
        required = ("map50_95", "map50", "precision", "recall")
        missing = [name for name in required if metrics.get(name) is None]
        if missing or cost.get("selected_count") is None:
            raise ValueError(f"Missing canonical metrics/cost {missing}: {task_dir}")
        rows.append(
            {
                "split_seed": split_seed,
                "stage": stage,
                "cumulative_train_images": int(cost["selected_count"]),
                "test_map50_95": float(metrics["map50_95"]),
                "test_map50": float(metrics["map50"]),
                "test_precision": float(metrics["precision"]),
                "test_recall": float(metrics["recall"]),
                "test_images": 1000,
                "task_dir": str(task_dir.relative_to(STUDY_DIR)),
            }
        )
    return rows


def resolve_curve_names() -> dict[int, str]:
    resolved: dict[int, str] = {}
    for split_seed, candidates in CURVE_CANDIDATES.items():
        for sequence_name in candidates:
            sequence_dir = SEQUENCES_DIR / sequence_name
            status_path = sequence_dir / "sequence_status.json"
            if not status_path.is_file():
                continue
            status = read_json(status_path)
            if status.get("status") == "completed" and status.get("completed_tasks") == 8:
                resolved[split_seed] = sequence_name
                break
        if split_seed not in resolved:
            choices = ", ".join(candidates)
            raise FileNotFoundError(f"No completed sequence for split seed {split_seed}; checked: {choices}")
    return resolved


def validate_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["split_seed"]), []).append(row)
    x_by_seed = {
        seed: [row["cumulative_train_images"] for row in sorted(seed_rows, key=lambda item: item["stage"])]
        for seed, seed_rows in by_seed.items()
    }
    reference_x = x_by_seed[42]
    if any(values != reference_x for values in x_by_seed.values()):
        raise ValueError(f"Cumulative training-image points differ across seeds: {x_by_seed}")
    if any(row["test_images"] != 1000 for row in rows):
        raise ValueError("The comparison requires 1,000 test images per curve")

    per_stage: list[dict[str, Any]] = []
    for stage in range(8):
        stage_rows = [row for row in rows if row["stage"] == stage]
        map50_95 = [row["test_map50_95"] for row in stage_rows]
        map50 = [row["test_map50"] for row in stage_rows]
        per_stage.append(
            {
                "stage": stage,
                "cumulative_train_images": reference_x[stage],
                "map50_95_min": min(map50_95),
                "map50_95_max": max(map50_95),
                "map50_95_range": max(map50_95) - min(map50_95),
                "map50_min": min(map50),
                "map50_max": max(map50),
                "map50_range": max(map50) - min(map50),
            }
        )
    return {"x_points": reference_x, "per_stage_range": per_stage}


def write_csv(rows: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "metrics_by_split_seed.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict[str, Any]]) -> Path:
    colors = {42: "#176B87", 41: "#E07A5F", 43: "#7B61A8"}
    labels = {42: "split seed 42", 41: "split seed 41", 43: "split seed 43"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True)
    fig.patch.set_facecolor("#F7F9FB")
    for axis, metric, title in zip(
        axes,
        ("test_map50_95", "test_map50"),
        ("mAP@0.5:0.95", "mAP@0.5"),
    ):
        axis.set_facecolor("white")
        for split_seed in (42, 41, 43):
            curve = sorted(
                (row for row in rows if row["split_seed"] == split_seed),
                key=lambda row: row["stage"],
            )
            x = [row["cumulative_train_images"] for row in curve]
            y = [row[metric] for row in curve]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2.3,
                markersize=5.5,
                color=colors[split_seed],
                label=labels[split_seed],
            )
        axis.set_title(title, fontweight="bold", pad=12)
        axis.set_xlabel("Cumulative training images")
        axis.set_xticks(sorted({row["cumulative_train_images"] for row in rows}))
        axis.set_xticklabels(
            [f"{value:,}" for value in sorted({row["cumulative_train_images"] for row in rows})],
            rotation=35,
            ha="right",
        )
        axis.set_ylim(0.20, 0.68)
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Test metric (best checkpoint)")
    fig.suptitle("Global Dedup τ=0.99 — Split-seed robustness", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.015,
        "Full Cold | training seed schedule fixed at 42–49 | each split has 1,000 test images",
        ha="center",
        color="#52616B",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93), w_pad=3.0)
    path = RESULTS_DIR / "learning_curve_by_split_seed.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    resolved = resolve_curve_names()
    rows = [
        row
        for split_seed, sequence_name in resolved.items()
        for row in collect_curve(split_seed, sequence_name)
    ]
    protocol = validate_protocol(rows)
    csv_path = write_csv(rows)
    png_path = plot(rows)
    summary = {
        "study": STUDY_DIR.name,
        "comparison": "post-deduplication random split seed",
        "split_seeds": [42, 41, 43],
        "resolved_sequences": {str(seed): name for seed, name in resolved.items()},
        "training_seed_schedule": "42-49 for every sequence",
        "metric_source": "task_result.json -> metrics.best.test",
        "x_axis": "task_result.json -> cost.selected_count",
        "test_images_per_curve": 1000,
        "protocol_validation": protocol,
        "outputs": {"csv": csv_path.name, "png": png_path.name},
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
