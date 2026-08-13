from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from ..base import DetectionBackend, require_artifact
from .dataset import write_backend_data_yaml, write_image_manifest
from .trainer import estimated_optimizer_steps, read_training_history, training_arguments


class UltralyticsBackend(DetectionBackend):
    capabilities = frozenset({"static_training", "evaluation", "prediction", "distributed_training"})
    output_namespace = "ultralytics"

    def provenance(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return {"name": "ultralytics", "family": "ultralytics", "definition": str(config["model"].get("definition", ""))}

    def validate_config(self, config: Mapping[str, Any]) -> None:
        if config["model"].get("backend") != "ultralytics":
            raise ValueError("UltralyticsBackend requires model.backend=ultralytics")
        if config["epoch_policy"].get("name") != "static":
            raise ValueError("phase 1 Ultralytics backend supports only the static epoch policy")
        params = config["backend"].get("params", {})
        if int(params.get("batch", 0)) <= 0 or int(params.get("imgsz", 0)) <= 0:
            raise ValueError("backend.params.batch and imgsz must be positive")

    def train(self, request: Mapping[str, Any]) -> dict[str, Any]:
        from ultralytics import YOLO

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
        model = YOLO(str(request["initial_checkpoint"]))
        started = time.perf_counter()
        model.train(**training_arguments(config, data_yaml=data_yaml, output_dir=raw_dir))
        elapsed = time.perf_counter() - started
        training_dir = raw_dir / "train"
        source_last = require_artifact(training_dir / "weights" / "last.pt", "Ultralytics last checkpoint")
        source_best = training_dir / "weights" / "best.pt"
        if not source_best.is_file():
            source_best = source_last
        checkpoints = standard_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        last = checkpoints / "last.pt"
        best = checkpoints / "best.pt"
        shutil.copy2(source_last, last)
        shutil.copy2(source_best, best)
        history = read_training_history(training_dir / "results.csv")
        batch = int(config["backend"]["params"].get("batch", 16))
        epochs = int(config["budget"]["value"])
        return {
            "last_checkpoint": str(last),
            "best_checkpoint": str(best),
            "history": history,
            "training_seconds": elapsed,
            "images_read": len(selected_ids) * epochs,
            "optimizer_steps": estimated_optimizer_steps(len(selected_ids), batch, epochs),
            "data_yaml": str(data_yaml),
        }

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        from ultralytics import YOLO

        registry = request["registry"]
        sample_ids = list(request["sample_ids"])
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = write_image_manifest(registry, sample_ids, output_dir / "images.txt")
        data_yaml = write_backend_data_yaml(manifest, manifest, registry["names"], output_dir / "data.yaml")
        params = request["config"]["backend"].get("params", {})
        arguments = {
            "data": str(data_yaml),
            "split": "val",
            "imgsz": int(params.get("imgsz", 640)),
            "batch": int(params.get("batch", 16)),
            "device": params.get("device", "cpu"),
            "workers": int(params.get("workers", 4)),
            "project": str(output_dir),
            "name": "val",
            "exist_ok": True,
            "plots": False,
            "verbose": False,
        }
        started = time.perf_counter()
        metrics = YOLO(str(request["checkpoint"])).val(**arguments)
        elapsed = time.perf_counter() - started
        box = metrics.box
        return {
            "map50_95": float(box.map),
            "map50": float(box.map50),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class_ap": {registry["names"][index]: float(value) for index, value in enumerate(box.maps)},
            "sample_count": len(sample_ids),
            "evaluation_seconds": elapsed,
        }

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        from ultralytics import YOLO

        registry = request["registry"]
        sample_ids = list(request["sample_ids"])
        sources = [registry["records"][sample_id]["image_path"] for sample_id in sample_ids]
        results = YOLO(str(request["checkpoint"])).predict(source=sources, verbose=False, save=False)
        return {sample_id: result for sample_id, result in zip(sample_ids, results, strict=True)}
