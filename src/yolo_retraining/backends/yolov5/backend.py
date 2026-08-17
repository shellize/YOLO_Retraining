from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from ..base import DetectionBackend, require_artifact
from .dataset import write_backend_data_yaml, write_image_manifest
from .source import resolve_checkpoint, validate_yolov5_source
from .trainer import build_train_command, device_argument, estimated_optimizer_steps, read_training_history, run_command, validate_training_params


class Yolov5Backend(DetectionBackend):
    capabilities = frozenset({"static_training", "evaluation", "prediction", "distributed_training"})
    output_namespace = "yolov5"

    def validate_config(self, config: Mapping[str, Any]) -> None:
        if config["model"].get("backend") != "yolov5":
            raise ValueError("Yolov5Backend requires model.backend=yolov5")
        if config["model"].get("definition") != "yolov5s.yaml":
            raise ValueError("phase 1 original YOLOv5 backend supports only model.definition=yolov5s.yaml")
        if config["epoch_policy"].get("name") != "static":
            raise ValueError("phase 1 original YOLOv5 backend supports only the static epoch policy")
        validate_training_params(config)

    def provenance(self, config: Mapping[str, Any]) -> dict[str, Any]:
        source = validate_yolov5_source()
        params = config["backend"].get("params", {})
        best_metric = str(params.get("best_metric", "map50"))
        best_selection_metric = "map50" if best_metric == "map50" else "0.1*map50+0.9*map50_95"
        return {
            "name": "yolov5",
            "family": "original-yolov5",
            "definition": "yolov5s.yaml",
            "source_commit": source["commit"],
            "source_tag": source["tag"],
            "initial_weight": Path(str(config["initialization"].get("checkpoint", ""))).name,
            "best_selection_metric": best_selection_metric,
            "hyperparameter_file": str(params["hyp"]) if params.get("hyp") else "yolov5-default:hyp.scratch-low.yaml",
            "data_loader_adaptation": "read_only_incomplete_jpeg",
        }

    def train(self, request: Mapping[str, Any]) -> dict[str, Any]:
        config = request["config"]
        registry = request["registry"]
        selected_ids = list(request["selected_ids"])
        validation_ids = list(request["validation_ids"])
        standard_dir = Path(request["standard_dir"])
        raw_dir = Path(request["raw_dir"])
        data_dir = standard_dir / "data"
        train_manifest = write_image_manifest(registry, selected_ids, data_dir / "backend_train_images.txt")
        val_manifest = write_image_manifest(registry, validation_ids, data_dir / "backend_validation_images.txt")
        data_yaml = write_backend_data_yaml(train_manifest, val_manifest, registry["names"], data_dir / "backend_data.yaml")
        source = validate_yolov5_source()
        source_root = Path(source["root"])
        checkpoint = resolve_checkpoint(str(request["initial_checkpoint"]), source_root)
        command = build_train_command(config, source_root=source_root, checkpoint=checkpoint, data_yaml=data_yaml, output_dir=raw_dir)
        started = time.perf_counter()
        run_command(command, cwd=source_root, log_path=standard_dir / "logs" / "yolov5_train.log", progress_epochs=int(config["budget"]["value"]))
        elapsed = time.perf_counter() - started
        training_dir = raw_dir / "train"
        source_last = require_artifact(training_dir / "weights" / "last.pt", "YOLOv5 last checkpoint")
        source_best = training_dir / "weights" / "best.pt"
        if not source_best.is_file():
            source_best = source_last
        checkpoints = standard_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        last = checkpoints / "last.pt"
        best = checkpoints / "best.pt"
        shutil.copy2(source_last, last)
        shutil.copy2(source_best, best)
        params = config["backend"]["params"]
        epochs = int(config["budget"]["value"])
        return {
            "last_checkpoint": str(last),
            "best_checkpoint": str(best),
            "history": read_training_history(training_dir / "results.csv"),
            "training_seconds": elapsed,
            "images_read": len(selected_ids) * epochs,
            "optimizer_steps": estimated_optimizer_steps(len(selected_ids), int(params["batch"]), epochs),
            "data_yaml": str(data_yaml),
            "train_command": command,
            "backend_provenance": self.provenance(config),
        }

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        registry = request["registry"]
        sample_ids = list(request["sample_ids"])
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = write_image_manifest(registry, sample_ids, output_dir / "images.txt")
        data_yaml = write_backend_data_yaml(manifest, manifest, registry["names"], output_dir / "data.yaml")
        source = validate_yolov5_source()
        params = request["config"]["backend"]["params"]
        result_path = output_dir / "metrics.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("bridge.py")),
            "evaluate",
            "--root",
            source["root"],
            "--weights",
            str(Path(request["checkpoint"]).resolve()),
            "--data",
            str(data_yaml),
            "--project",
            str(output_dir),
            "--batch",
            str(int(params["batch"])),
            "--imgsz",
            str(int(params["imgsz"])),
            "--device",
            device_argument(params.get("device", "cpu")),
            "--workers",
            str(int(params.get("workers", 4))),
            "--output",
            str(result_path),
        ]
        if params.get("amp", True) and str(params.get("device", "cpu")).lower() != "cpu":
            command.append("--half")
        started = time.perf_counter()
        run_command(command, cwd=Path(source["root"]), log_path=output_dir / "evaluation.log")
        elapsed = time.perf_counter() - started
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return normalize_evaluation(payload, registry["names"], sample_count=len(sample_ids), evaluation_seconds=elapsed)

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        registry = request["registry"]
        sample_ids = list(request["sample_ids"])
        output_dir = Path(request.get("output_dir", Path.cwd() / ".yolov5_predict"))
        output_dir.mkdir(parents=True, exist_ok=True)
        images_path = output_dir / "images.json"
        result_path = output_dir / "predictions.json"
        images = [registry["records"][sample_id]["image_path"] for sample_id in sample_ids]
        images_path.write_text(json.dumps(images), encoding="utf-8")
        source = validate_yolov5_source()
        params = request["config"]["backend"]["params"]
        command = [
            sys.executable,
            str(Path(__file__).with_name("bridge.py")),
            "predict",
            "--root",
            source["root"],
            "--weights",
            str(Path(request["checkpoint"]).resolve()),
            "--images",
            str(images_path),
            "--imgsz",
            str(int(params["imgsz"])),
            "--device",
            device_argument(params.get("device", "cpu")),
            "--output",
            str(result_path),
        ]
        run_command(command, cwd=Path(source["root"]), log_path=output_dir / "prediction.log")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return {sample_id: payload[str(index)] for index, sample_id in enumerate(sample_ids)}


def normalize_evaluation(
    payload: Mapping[str, Any],
    names: list[str],
    *,
    sample_count: int,
    evaluation_seconds: float,
) -> dict[str, Any]:
    required = ("map50_95", "map50", "precision", "recall", "per_class_ap")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"YOLOv5 evaluation output is missing fields: {missing}")
    maps = list(payload["per_class_ap"])
    if len(maps) != len(names):
        raise ValueError(f"YOLOv5 returned {len(maps)} per-class AP values for {len(names)} classes")
    return {
        "map50_95": float(payload["map50_95"]),
        "map50": float(payload["map50"]),
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "per_class_ap": {name: float(maps[index]) for index, name in enumerate(names)},
        "sample_count": sample_count,
        "evaluation_seconds": evaluation_seconds,
    }
