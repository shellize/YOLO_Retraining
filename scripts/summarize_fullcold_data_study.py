from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def collect_sequence(experiment: str, sequence_dir: Path) -> list[dict[str, Any]]:
    sequence_result = _read_json(sequence_dir / "sequence_result.json")
    if sequence_result.get("status") != "completed":
        raise ValueError(f"sequence is not completed: {sequence_dir}")
    rows: list[dict[str, Any]] = []
    for stage, task_path in enumerate(sequence_result.get("tasks", [])):
        task = _read_json(Path(task_path) / "task_result.json")
        metrics = task.get("metrics", {}).get("best", {}).get("test", {})
        cost = task.get("cost", {})
        per_class = metrics.get("per_class_ap", {})
        row: dict[str, Any] = {
            "experiment": experiment,
            "stage": stage,
            "task_output": task.get("output_dir"),
            "selected_count": cost.get("selected_count"),
            "candidate_count": cost.get("candidate_count"),
            "images_read": cost.get("images_read"),
            "optimizer_steps": cost.get("optimizer_steps"),
            "training_seconds": cost.get("training_seconds"),
            "map50_95": metrics.get("map50_95"),
            "map50": metrics.get("map50"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "large_luggage_ap": per_class.get("large luggage"),
        }
        for class_name, value in per_class.items():
            row[f"ap.{class_name}"] = value
        rows.append(row)
    if not rows:
        raise ValueError(f"sequence contains no task results: {sequence_dir}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize serial full-cold data experiments.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", type=_parse_sequence, action="append", required=True)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    sequence_paths: dict[str, str] = {}
    for experiment, sequence_dir in args.sequence:
        if experiment in sequence_paths:
            raise ValueError(f"duplicate experiment name: {experiment}")
        sequence_paths[experiment] = str(sequence_dir)
        rows.extend(collect_sequence(experiment, sequence_dir))

    baseline_name = args.sequence[0][0]
    baseline_last = max((row for row in rows if row["experiment"] == baseline_name), key=lambda row: row["stage"])
    last_rows: list[dict[str, Any]] = []
    for experiment, _ in args.sequence:
        last = max((row for row in rows if row["experiment"] == experiment), key=lambda row: row["stage"])
        enriched = dict(last)
        for metric in ("map50_95", "map50", "large_luggage_ap"):
            baseline_value = baseline_last.get(metric)
            current_value = last.get(metric)
            enriched[f"delta_vs_{baseline_name}.{metric}"] = (
                None if baseline_value is None or current_value is None else current_value - baseline_value
            )
        last_rows.append(enriched)

    output_dir = args.output_dir.expanduser().resolve()
    _write_csv(output_dir / "comparison_all_stages.csv", rows)
    _write_csv(output_dir / "comparison_final_stage.csv", last_rows)
    payload = {
        "baseline": baseline_name,
        "sequences": sequence_paths,
        "final_stage": last_rows,
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
