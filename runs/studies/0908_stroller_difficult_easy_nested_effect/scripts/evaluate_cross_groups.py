"""Evaluate m1 and m2 on test1, test2, and testhard using the same backend."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_retraining.backends import create_backend  # noqa: E402
from yolo_retraining.config import load_config  # noqa: E402
from yolo_retraining.data import build_registry  # noqa: E402


SEEDS = (41, 42)
GROUPS = ("test1", "test2", "testhard")
MODELS = ("m1", "m2")
RESULT_ROOT = STUDY_ROOT / "result/evaluation"
CONFIG_ROOT = STUDY_ROOT / "config"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def task_dir(model: str, seed: int) -> Path:
    labels = {"m1": "m1-difficult", "m2": "m2-easy"}
    groups = {"m1": "train1", "m2": "train2"}
    return (
        STUDY_ROOT
        / "experiment/task"
        / f"{model}_s{seed}__{labels[model]}__{groups[model]}__yolov5s__s42"
    )


def config_path(model: str, seed: int) -> Path:
    return CONFIG_ROOT / f"{model}_s{seed}.yaml"


def checkpoint_for(model: str, seed: int) -> Path:
    directory = task_dir(model, seed)
    result = read_json(directory / "task_result.json")
    if result.get("status") != "completed":
        raise ValueError(f"task is not complete: {directory}")
    checkpoint = directory / str(result["artifacts"]["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_variant_summary(seed: int) -> dict[str, Any]:
    return read_json(STUDY_ROOT / "experiment/variants" / f"split_s{seed}" / "summary.json")


def evaluate_one(
    *,
    model: str,
    seed: int,
    group: str,
    checkpoint: Path,
    config: dict[str, Any],
    layout_path: Path,
) -> dict[str, Any]:
    output_dir = RESULT_ROOT / f"split_s{seed}" / model / "best" / group
    result_path = output_dir / "result.json"
    if result_path.is_file():
        payload = read_json(result_path)
        if payload.get("status") == "completed":
            return payload
        raise ValueError(f"existing evaluation result is not completed: {result_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite incomplete evaluation output: {output_dir}")

    registry = build_registry({group: str(layout_path)})
    sample_ids = list(registry["groups"][group]["test"])
    if not sample_ids:
        raise ValueError(f"evaluation group is empty: {group}")
    backend = create_backend(config)
    backend.validate_config(config)
    evaluation = backend.evaluate(
        {
            "config": config,
            "registry": registry,
            "sample_ids": sample_ids,
            "checkpoint": str(checkpoint),
            "checkpoint_name": "best",
            "evaluation_group": group,
            "output_dir": output_dir,
        }
    )
    payload = {
        "status": "completed",
        "model": model,
        "split_seed": seed,
        "group": group,
        "checkpoint": str(checkpoint),
        "layout": str(layout_path),
        "sample_count": len(sample_ids),
        "metrics": {
            key: float(evaluation[key])
            for key in ("map50_95", "map50", "precision", "recall")
            if evaluation.get(key) is not None
        },
        "backend_evaluation": evaluation,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def collect_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        variant = load_variant_summary(seed)
        eval_layouts = variant["layouts"]
        for model in MODELS:
            config = load_config(config_path(model, seed))
            checkpoint = checkpoint_for(model, seed)
            for group in GROUPS:
                payload = evaluate_one(
                    model=model,
                    seed=seed,
                    group=group,
                    checkpoint=checkpoint,
                    config=config,
                    layout_path=Path(eval_layouts[group]),
                )
                metrics = payload["metrics"]
                rows.append(
                    {
                        "split_seed": seed,
                        "model": model,
                        "test_group": group,
                        "sample_count": payload["sample_count"],
                        "map50_95": metrics.get("map50_95"),
                        "map50": metrics.get("map50"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "evaluation_dir": str(
                            (RESULT_ROOT / f"split_s{seed}" / model / "best" / group).relative_to(STUDY_ROOT)
                        ),
                    }
                )
    return rows


def write_comparison(rows: list[dict[str, Any]]) -> Path:
    by_key = {(int(row["split_seed"]), row["model"], row["test_group"]): row for row in rows}
    comparison: list[dict[str, Any]] = []
    for seed in SEEDS:
        for group in GROUPS:
            m1 = by_key[(seed, "m1", group)]
            m2 = by_key[(seed, "m2", group)]
            row: dict[str, Any] = {
                "split_seed": seed,
                "test_group": group,
                "sample_count": m1["sample_count"],
            }
            for metric in ("map50_95", "map50", "precision", "recall"):
                left = float(m1[metric])
                right = float(m2[metric])
                row[f"m1_{metric}"] = left
                row[f"m2_{metric}"] = right
                row[f"m2_minus_m1_{metric}"] = right - left
            comparison.append(row)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_ROOT / "comparison_by_split.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(comparison[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)

    grouped: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        group_rows = [row for row in comparison if row["test_group"] == group]
        grouped[group] = {}
        for metric in ("map50_95", "map50", "precision", "recall"):
            grouped[group][f"m1_{metric}_mean"] = sum(row[f"m1_{metric}"] for row in group_rows) / len(group_rows)
            grouped[group][f"m2_{metric}_mean"] = sum(row[f"m2_{metric}"] for row in group_rows) / len(group_rows)
            grouped[group][f"m2_minus_m1_{metric}_mean"] = sum(
                row[f"m2_minus_m1_{metric}"] for row in group_rows
            ) / len(group_rows)

    summary = {
        "study": STUDY_ROOT.name,
        "status": "completed",
        "models": {
            "m1": "trained on train1 from strollerdifficult",
            "m2": "trained on train2, the easy projection of train1",
        },
        "seeds": list(SEEDS),
        "groups": list(GROUPS),
        "checkpoint": "best selected using the model-specific validation split",
        "primary_readouts": {
            "m2_test1_vs_m1_test1": "easy-only training generalization to the full difficult test split",
            "m2_test2_vs_m1_test2": "effect of adding difficult samples on the easy test subset",
            "m2_testhard_vs_m1_testhard": "direct difficult-subset comparison",
        },
        "important_boundary": (
            "m3 is not included; m1 and m2 differ in both sample difficulty and training-set size/distribution, "
            "so deltas are practical training-protocol effects rather than a pure matched-size causal estimate"
        ),
        "mean_by_test_group": grouped,
        "comparison_csv": csv_path.name,
    }
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path


def main() -> int:
    rows = collect_results()
    path = write_comparison(rows)
    print(json.dumps({"rows": len(rows), "comparison_csv": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
