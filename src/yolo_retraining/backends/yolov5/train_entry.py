from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from yolo_retraining.backends.yolov5.process_start import configure_safe_start_method
from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier


# Configure multiprocessing before importing YOLOv5 or torch.  YOLOv5 moves
# the model to CUDA before it creates DataLoader workers, so Linux's default
# ``fork`` start method is not safe here.
configure_safe_start_method()


def _extract_source_root(arguments: list[str]) -> tuple[Path, list[str], str, Path | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yolo-source-root", type=Path)
    parser.add_argument("--best-metric", choices=("map30", "map50", "yolov5_fitness"), default="map30")
    parser.add_argument("--metrics-sidecar", type=Path)
    known, remaining = parser.parse_known_args(arguments)
    configured = known.yolo_source_root or os.environ.get("YOLO_RETRAINING_YOLOV5_ROOT")
    if not configured:
        parser.error("--yolo-source-root is required in the main process")
    sidecar = known.metrics_sidecar.resolve() if known.metrics_sidecar is not None else None
    return Path(configured).resolve(), remaining, known.best_metric, sidecar


SOURCE_ROOT, TRAIN_ARGUMENTS, BEST_METRIC, METRICS_SIDECAR = _extract_source_root(sys.argv[1:])
os.environ["YOLO_RETRAINING_YOLOV5_ROOT"] = str(SOURCE_ROOT)
sys.path.insert(0, str(SOURCE_ROOT))

# This runs at module import time so spawned multiprocessing workers install the
# same read-only verifier before YOLOv5 creates its label-cache process pool.
import utils.dataloaders as dataloaders  # noqa: E402

install_read_only_verifier(dataloaders)


def main() -> int:
    import train

    from yolo_retraining.backends.yolov5.iou_metrics import run_with_low_iou_metrics

    latest = {"map30": None, "call_index": 0}
    original_validation_run = train.validate.run

    def validation_run(*args, **kwargs):
        evaluation = run_with_low_iou_metrics(train.validate, original_validation_run, *args, **kwargs)
        latest["map30"] = float(evaluation.metrics["map30"])
        if METRICS_SIDECAR is not None:
            METRICS_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "call_index": latest["call_index"],
                **{key: evaluation.metrics[key] for key in ("map10", "map20", "map30", "map50", "map50_95")},
            }
            with METRICS_SIDECAR.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            latest["call_index"] += 1
        return evaluation.results, evaluation.per_class_map50_95, evaluation.timing

    train.validate.run = validation_run

    if BEST_METRIC == "map30":
        def map30_fitness(_results):
            if latest["map30"] is None:
                raise RuntimeError("AP0.3 was not captured from YOLOv5 validation")
            return float(latest["map30"])

        train.fitness = map30_fitness
    elif BEST_METRIC == "map50":
        # Keep the fixed upstream source untouched and override only the
        # function used by train.py when it selects best.pt.
        def map50_fitness(results):
            return float(results.reshape(-1, results.shape[-1])[0, 2])

        train.fitness = map50_fitness

    sys.argv = [str(SOURCE_ROOT / "train.py"), *TRAIN_ARGUMENTS]
    options = train.parse_opt()
    train.main(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
