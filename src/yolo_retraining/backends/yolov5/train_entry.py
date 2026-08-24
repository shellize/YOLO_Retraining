from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from yolo_retraining.backends.yolov5.process_start import configure_safe_start_method
from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier


# Configure multiprocessing before importing YOLOv5 or torch.  YOLOv5 moves
# the model to CUDA before it creates DataLoader workers, so Linux's default
# ``fork`` start method is not safe here.
configure_safe_start_method()


def _extract_source_root(arguments: list[str]) -> tuple[Path, list[str], str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yolo-source-root", type=Path)
    parser.add_argument("--best-metric", choices=("map50", "yolov5_fitness"), default="map50")
    known, remaining = parser.parse_known_args(arguments)
    configured = known.yolo_source_root or os.environ.get("YOLO_RETRAINING_YOLOV5_ROOT")
    if not configured:
        parser.error("--yolo-source-root is required in the main process")
    return Path(configured).resolve(), remaining, known.best_metric


SOURCE_ROOT, TRAIN_ARGUMENTS, BEST_METRIC = _extract_source_root(sys.argv[1:])
os.environ["YOLO_RETRAINING_YOLOV5_ROOT"] = str(SOURCE_ROOT)
sys.path.insert(0, str(SOURCE_ROOT))

# This runs at module import time so spawned multiprocessing workers install the
# same read-only verifier before YOLOv5 creates its label-cache process pool.
import utils.dataloaders as dataloaders  # noqa: E402

install_read_only_verifier(dataloaders)


def main() -> int:
    import train

    if BEST_METRIC == "map50":
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
