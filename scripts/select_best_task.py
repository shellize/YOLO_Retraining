#!/usr/bin/env python3
"""Select the best completed task by a nested metric in task_result.json."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def best_history_value(task_dir: Path, metric: str) -> float:
    history_path = task_dir / "metrics" / "train_history.csv"
    if not history_path.is_file():
        raise SystemExit(f"training history not found: {history_path}")
    with history_path.open(newline="", encoding="utf-8") as handle:
        values = [float(row[metric]) for row in csv.DictReader(handle) if row.get(metric)]
    if not values:
        raise SystemExit(f"metric has no values: {metric} in {history_path}")
    return max(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-metric", required=True, help="validation-history metric, e.g. metrics/mAP_0.5:0.95")
    parser.add_argument("tasks", nargs="+", help="LABEL=TASK_DIRECTORY entries")
    args = parser.parse_args()

    candidates: list[tuple[str, float, Path]] = []
    for item in args.tasks:
        if "=" not in item:
            raise SystemExit(f"task entry must be LABEL=TASK_DIRECTORY: {item}")
        label, raw_path = item.split("=", 1)
        task_dir = Path(raw_path)
        result_path = task_dir / "task_result.json"
        if not result_path.is_file():
            raise SystemExit(f"task result not found: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise SystemExit(f"task is not completed: {task_dir}")
        candidates.append((label, best_history_value(task_dir, args.history_metric), task_dir))

    winner = max(candidates, key=lambda item: item[1])
    print(winner[0])
    for label, value, task_dir in candidates:
        print(f"[selection] {label}: {value:.6f} ({task_dir})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
