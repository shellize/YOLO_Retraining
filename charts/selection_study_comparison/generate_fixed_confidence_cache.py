from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_retraining.backends.yolov5 import Yolov5Backend
from yolo_retraining.config import BUILTIN_DEFAULTS, deep_merge, load_config
from yolo_retraining.data import build_registry, catalog_from_data_config, resolve_scope

from compare_methods import METHODS, RUNS_ROOT, SHARED_STAGE0, task_result_for_stage, validate_sequence


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_sweep(task_result: Path) -> Path:
    return task_result.parent / "backend" / "yolov5" / "evaluations" / "best" / "test" / "confidence_sweep.json"


def main() -> int:
    sequence_config = load_config(PROJECT_ROOT / "configs" / "sequence" / "selection_study" / "positive_only__cum_cold.yaml")
    data_config = {
        "layout": sequence_config["data"]["layout"],
        "current": ["stage0"],
        "candidate": ["stage0", "stage1", "stage2", "stage3"],
        "validation": ["val"],
        "test": ["test"],
    }
    registry = build_registry(catalog_from_data_config(data_config))
    scope = resolve_scope(registry, data_config)
    task_config = deep_merge(BUILTIN_DEFAULTS, sequence_config["task_template"])
    task_config["backend"]["params"]["device"] = 0
    task_config["backend"]["params"]["workers"] = 16
    task_config["evaluation"]["save_prediction_artifacts"] = True
    task_config["evaluation"]["prediction_artifact_checkpoints"] = ["best"]
    task_config["evaluation"]["prediction_artifact_groups"] = ["test"]
    backend = Yolov5Backend()
    generated = 0
    skipped = 0
    test_ids = scope["test_by_group"]["test"]

    for method in METHODS:
        sequence = validate_sequence(method)
        for stage in range(4):
            task_result = task_result_for_stage(method, sequence, stage)
            existing = source_sweep(task_result)
            if existing.is_file():
                skipped += 1
                continue
            cache_dir = WORKSPACE / "fixed_confidence_cache" / method.key / f"stage{stage}"
            cached = cache_dir / "confidence_sweep.json"
            if cached.is_file():
                skipped += 1
                continue
            result = read_json(task_result)
            checkpoint = task_result.parent / result["artifacts"]["best_checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(f"best checkpoint missing: {checkpoint}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Fixed P/R] evaluating {method.label}, Stage {stage}: {checkpoint}", flush=True)
            backend.evaluate(
                {
                    "config": task_config,
                    "registry": registry,
                    "sample_ids": test_ids,
                    "checkpoint": str(checkpoint),
                    "checkpoint_name": "best",
                    "evaluation_group": "test",
                    "output_dir": cache_dir,
                }
            )
            generated += 1

    print(f"[Fixed P/R] generated={generated}, reused={skipped}, workspace={WORKSPACE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
