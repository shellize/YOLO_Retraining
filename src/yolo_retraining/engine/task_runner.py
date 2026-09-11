from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from yolo_retraining.backends import DetectionBackend, create_backend
from yolo_retraining.config import dump_yaml, validate_task_config
from yolo_retraining.data import build_registry, catalog_from_data_config, resolve_scope
from yolo_retraining.evaluation import build_cost
from yolo_retraining.policies import create_epoch_policy, create_selection_policy
from yolo_retraining.visualization import write_task_tensorboard

from .output import create_output, read_json, relative_to, status_payload, task_output_path, write_csv, write_json, write_lines


class TaskRunner:
    def __init__(self, config: Mapping[str, Any], *, backend: DetectionBackend | None = None, output_dir: Path | None = None) -> None:
        self.config = dict(config)
        validate_task_config(self.config)
        self.backend = backend or create_backend(self.config)
        self.output_dir = Path(output_dir) if output_dir is not None else task_output_path(self.config)

    def run(self) -> Path:
        create_output(self.output_dir)
        print(f"[Task] output: {self.output_dir.resolve()}", flush=True)
        dump_yaml(self.config, self.output_dir / "task.yaml")
        status_path = self.output_dir / "task_status.json"
        write_json(status_path, status_payload("created"))
        try:
            registry = build_registry(catalog_from_data_config(self.config["data"]))
            scope = resolve_scope(registry, self.config["data"])
            self._write_scope(scope)
            test_count = sum(len(ids) for ids in scope["test_by_group"].values())
            print(
                f"[Task] data: current={len(scope['current_ids'])}, candidate={len(scope['candidate_ids'])}, "
                f"validation={len(scope['validation_ids'])}, test={test_count}",
                flush=True,
            )
            write_json(status_path, status_payload("selecting"))
            selection_started = time.perf_counter()
            selection_policy = create_selection_policy(self.config["select_policy"])
            epoch_policy = create_epoch_policy(self.config["epoch_policy"])
            selection = selection_policy.select(
                {
                    **scope,
                    "seed": int(self.config["task"]["seed"]),
                    "registry": registry,
                    "config": self.config,
                    "backend": self.backend,
                    "output_dir": self.output_dir,
                }
            )
            selected_ids = epoch_policy.build_plan(selection["selected_ids"], epoch=0, seed=int(self.config["task"]["seed"]))
            if not selected_ids:
                raise ValueError("selection policy returned no training samples")
            selection_seconds = time.perf_counter() - selection_started
            self._write_selection(selection, selected_ids)
            group_counts = {key: len(value) for key, value in selection.get("groups", {}).items()}
            print(
                f"[Task] selection: policy={self.config['select_policy']['name']}, selected={len(selected_ids)}, "
                f"groups={group_counts}, time={selection_seconds:.3f}s",
                flush=True,
            )
            initial_checkpoint = self._resolve_initialization(registry["names"])
            self.backend.validate_config(self.config)
            backend_dir = self.output_dir / "backend" / self.backend.output_namespace
            write_json(status_path, status_payload("training", selected_count=len(selected_ids)))
            print(
                f"[Task] training: backend={self.config['model']['backend']}, epochs={self.config['budget']['value']}, "
                f"batch={self.config['backend']['params']['batch']}, device={self.config['backend']['params']['device']}",
                flush=True,
            )
            training = self.backend.train(
                {
                    "config": self.config,
                    "registry": registry,
                    "selected_ids": selected_ids,
                    "validation_ids": scope["validation_ids"],
                    "initial_checkpoint": initial_checkpoint,
                    "standard_dir": self.output_dir,
                    "raw_dir": backend_dir,
                }
            )
            print(f"[Task] training completed: {float(training.get('training_seconds', 0.0)):.1f}s", flush=True)
            write_csv(self.output_dir / "metrics" / "train_history.csv", training.get("history", []))
            write_json(status_path, status_payload("evaluating"))
            metrics, evaluation_seconds = self._evaluate(registry, scope, training)
            write_json(self.output_dir / "metrics" / "evaluation.json", metrics)
            device_count = self._device_count()
            cost = build_cost(selection_seconds, training, evaluation_seconds, device_count)
            cost.update(candidate_count=len(scope["candidate_ids"]), selected_count=len(selected_ids), replay_count=len(selection.get("groups", {}).get("replay", [])))
            write_json(self.output_dir / "metrics" / "cost.json", cost)
            tensorboard_dir = write_task_tensorboard(
                self.output_dir,
                self.config,
                training.get("history", []),
                metrics,
                cost,
            )
            result = self._result(registry, selection, training, metrics, cost, tensorboard_dir=tensorboard_dir)
            write_json(self.output_dir / "task_result.json", result)
            write_json(status_path, status_payload("completed"))
            print(
                f"[Task] completed: evaluation={evaluation_seconds:.1f}s, output={self.output_dir.resolve()}",
                flush=True,
            )
            return self.output_dir
        except Exception as error:
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(status_path, status_payload("failed", error_type=type(error).__name__, error=str(error)))
            print(f"[Task] failed: {error}; details={self.output_dir / 'logs' / 'error.txt'}", flush=True)
            raise

    def _write_scope(self, scope: Mapping[str, Any]) -> None:
        data_dir = self.output_dir / "data"
        write_lines(data_dir / "current_ids.txt", scope["current_ids"])
        write_lines(data_dir / "candidate_ids.txt", scope["candidate_ids"])
        write_lines(data_dir / "validation_ids.txt", scope["validation_ids"])
        for group, ids in scope["test_by_group"].items():
            write_lines(data_dir / f"test_{group}_ids.txt", ids)

    def _write_selection(self, selection: Mapping[str, Any], selected_ids: list[str]) -> None:
        write_lines(self.output_dir / "data" / "selected_ids.txt", selected_ids)
        groups = selection.get("groups", {})
        current = list(groups.get("current", []))
        history = list(groups.get("history", []))
        if current:
            write_lines(self.output_dir / "selection" / "current_selected_ids.txt", current)
        if history:
            write_lines(self.output_dir / "selection" / "history_selected_ids.txt", history)
        replay = list(groups.get("replay", []))
        if replay:
            write_lines(self.output_dir / "selection" / "replay_ids.txt", replay)
        records = list(selection.get("records", []))
        if records:
            write_lines(
                self.output_dir / "selection" / "sample_scores.jsonl",
                [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records],
            )
        summary = {"selected_count": len(selected_ids), "group_counts": {key: len(value) for key, value in groups.items()}, "metadata": selection.get("metadata", {})}
        write_json(self.output_dir / "selection" / "summary.json", summary)

    def _resolve_initialization(self, names: list[str]) -> str:
        initialization = self.config["initialization"]
        source = initialization["source"]
        if source in {"pretrained", "explicit"}:
            checkpoint = initialization.get("checkpoint")
            if not checkpoint:
                raise ValueError(f"{source} initialization requires checkpoint")
            if source == "explicit" and not Path(checkpoint).is_file():
                raise FileNotFoundError(f"explicit checkpoint does not exist: {checkpoint}")
            return str(checkpoint)
        parent_dir = Path(self.config["task"]["parent_result"]).resolve()
        result_path = parent_dir / "task_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"parent TaskResult does not exist: {result_path}")
        parent = read_json(result_path)
        if parent.get("status") != "completed":
            raise ValueError(f"parent task is not completed: {parent_dir}")
        if parent.get("label_schema") != names:
            raise ValueError("parent checkpoint label schema is incompatible with the current task")
        key = str(initialization.get("checkpoint", "last"))
        artifact_key = {"last": "last_checkpoint", "best": "best_checkpoint"}.get(key)
        if artifact_key is None:
            raise ValueError("parent checkpoint must be last or best")
        checkpoint = parent_dir / parent["artifacts"][artifact_key]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"parent checkpoint does not exist: {checkpoint}")
        return str(checkpoint)

    def _evaluate(self, registry: Mapping[str, Any], scope: Mapping[str, Any], training: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        all_metrics: dict[str, Any] = {}
        elapsed = 0.0
        checkpoints = {
            "last": training["last_checkpoint"],
            "best": training["best_checkpoint"],
        }
        for checkpoint_name in self.config["evaluation"].get("evaluate_checkpoints", ["last", "best"]):
            checkpoint_metrics: dict[str, Any] = {}
            for group, sample_ids in scope["test_by_group"].items():
                print(f"[Task] evaluating: checkpoint={checkpoint_name}, group={group}, samples={len(sample_ids)}", flush=True)
                result = self.backend.evaluate(
                    {
                        "config": self.config,
                        "registry": registry,
                        "sample_ids": sample_ids,
                        "checkpoint": checkpoints[checkpoint_name],
                        "checkpoint_name": checkpoint_name,
                        "evaluation_group": group,
                        "output_dir": self.output_dir / "backend" / self.backend.output_namespace / "evaluations" / checkpoint_name / group,
                    }
                )
                elapsed += float(result.get("evaluation_seconds", 0.0))
                checkpoint_metrics[group] = result
                print(
                    f"[Task] evaluation: checkpoint={checkpoint_name}, group={group}, "
                    f"AP0.3={float(result.get('map30', 0.0)):.4f}, "
                    f"mAP50-95={float(result.get('map50_95', 0.0)):.4f}, "
                    f"mAP50={float(result.get('map50', 0.0)):.4f}, time={float(result.get('evaluation_seconds', 0.0)):.1f}s",
                    flush=True,
                )
            all_metrics[checkpoint_name] = checkpoint_metrics
        return all_metrics, elapsed

    def _result(
        self,
        registry: Mapping[str, Any],
        selection: Mapping[str, Any],
        training: Mapping[str, Any],
        metrics: Mapping[str, Any],
        cost: Mapping[str, Any],
        *,
        tensorboard_dir: Path,
    ) -> dict[str, Any]:
        backend_provenance = training.get("backend_provenance")
        if backend_provenance is None:
            backend_provenance = self.backend.provenance(self.config)
        artifacts = {
            "last_checkpoint": relative_to(training["last_checkpoint"], self.output_dir),
            "best_checkpoint": relative_to(training["best_checkpoint"], self.output_dir),
            "metrics": "metrics/evaluation.json",
            "cost": "metrics/cost.json",
            "selection": "selection/summary.json",
            "tensorboard": relative_to(tensorboard_dir, self.output_dir),
        }
        if training.get("validation_iou_metrics"):
            artifacts["validation_iou_metrics"] = relative_to(
                training["validation_iou_metrics"], self.output_dir
            )
        return {
            "schema_version": 1,
            "status": "completed",
            "task_label": self.config["task"]["label"],
            "output_dir": str(self.output_dir.resolve()),
            "parent_result": self.config["task"].get("parent_result"),
            "label_schema": registry["names"],
            "backend": backend_provenance,
            "primary_metric": self.config["evaluation"]["primary_metric"],
            "artifacts": artifacts,
            "selection": {"metadata": selection.get("metadata", {}), "group_counts": {key: len(value) for key, value in selection.get("groups", {}).items()}},
            "metrics": metrics,
            "cost": cost,
        }

    def _device_count(self) -> int:
        device = self.config["backend"].get("params", {}).get("device", "cpu")
        if isinstance(device, list):
            return len(device)
        if str(device).lower() == "cpu":
            return 0
        return 1
