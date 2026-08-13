from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Mapping

from yolo_retraining.config import BUILTIN_DEFAULTS, deep_merge, dump_yaml, validate_sequence_config, validate_task_config
from yolo_retraining.evaluation import summarize_matrix

from .output import create_output, read_json, sanitize, sequence_output_path, status_payload, write_csv, write_json
from .task_runner import TaskRunner


class SequenceRunner:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        validate_sequence_config(self.config)
        self.output_dir = sequence_output_path(self.config)

    def run(self) -> Path:
        create_output(self.output_dir)
        dump_yaml(self.config, self.output_dir / "sequence.yaml")
        status_path = self.output_dir / "sequence_status.json"
        write_json(status_path, status_payload("running", completed_tasks=0))
        tasks_dir = self.output_dir / "tasks"
        tasks_dir.mkdir()
        results: list[dict[str, Any]] = []
        catalog: dict[str, str] = {}
        seen_groups: list[str] = []
        previous_result: Path | None = None
        try:
            for index, arrival in enumerate(self.config["arrivals"]):
                group_id = str(arrival["id"])
                seen_groups.append(group_id)
                if "data" in arrival:
                    catalog[group_id] = str(arrival["data"])
                task_config = self._task_config(index, group_id, seen_groups, catalog, previous_result)
                task_dir = tasks_dir / f"{index:03d}__{sanitize(group_id)}__{sanitize(task_config['task']['label'])}"
                result_dir = TaskRunner(task_config, output_dir=task_dir).run()
                result = read_json(result_dir / "task_result.json")
                results.append(result)
                previous_result = result_dir
                write_json(status_path, status_payload("running", completed_tasks=len(results), current_task=index + 1))
            summary = summarize_matrix([{"last_metrics": result["metrics"].get("last", {})} for result in results])
            self._write_summaries(results, summary)
            write_json(self.output_dir / "sequence_result.json", {"schema_version": 1, "status": "completed", "output_dir": str(self.output_dir.resolve()), "tasks": [result["output_dir"] for result in results], "summary": summary})
            write_json(status_path, status_payload("completed", completed_tasks=len(results)))
            return self.output_dir
        except Exception as error:
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(status_path, status_payload("failed", completed_tasks=len(results), error_type=type(error).__name__, error=str(error)))
            raise

    def _task_config(self, index: int, group_id: str, seen_groups: list[str], catalog: Mapping[str, str], previous_result: Path | None) -> dict[str, Any]:
        task_config = deep_merge(BUILTIN_DEFAULTS, self.config["task_template"])
        override = self.config.get("task_overrides", {}).get(group_id, {})
        task_config = deep_merge(task_config, override)
        sequence = self.config["sequence"]
        task_config["task"] = deep_merge(
            task_config.get("task", {}),
            {
                "name": f"{index:03d}",
                "label": sequence.get("label", task_config.get("task", {}).get("label", "sequence")),
                "seed": int(sequence.get("seed", 42)) + index,
                "output_root": str((self.output_dir / "tasks").resolve()),
                "parent_result": str(previous_result.resolve()) if previous_result else None,
            },
        )
        candidate_groups = list(seen_groups) if self.config["scope_rule"]["candidate"] == "all_seen" else [group_id]
        layout_data = self.config.get("data")
        if isinstance(layout_data, Mapping) and layout_data.get("layout"):
            task_config["data"] = {
                "layout": str(layout_data["layout"]),
                "current": [group_id],
                "candidate": candidate_groups,
                "validation": list(layout_data["validation"]),
                "test": list(layout_data["test"]),
            }
        else:
            evaluation_groups = list(seen_groups) if self.config["scope_rule"].get("evaluation", "seen") == "seen" else [group_id]
            task_config["data"] = {"catalog": dict(catalog), "current": [group_id], "candidate": candidate_groups, "validation": evaluation_groups, "test": evaluation_groups}
        task_config["initialization"] = dict(self.config["initialization"]["first" if index == 0 else "subsequent"])
        validate_task_config(task_config)
        return task_config

    def _write_summaries(self, results: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
        metric_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            row: dict[str, Any] = {"task": index, "task_label": result["task_label"]}
            for group, metrics in result["metrics"].get("last", {}).items():
                row[f"{group}.map50_95"] = metrics["map50_95"]
            row["seen_mean"] = summary["seen_mean"][index]
            metric_rows.append(row)
            cost_rows.append({"task": index, **result["cost"]})
            selection_rows.append({"task": index, **result["selection"]["group_counts"], **result["selection"]["metadata"]})
        write_csv(self.output_dir / "summary" / "metrics_by_task.csv", metric_rows)
        write_csv(self.output_dir / "summary" / "cost_by_task.csv", cost_rows)
        write_csv(self.output_dir / "summary" / "selection_by_task.csv", selection_rows)
