from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier


def _extract_source_root(arguments: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yolo-source-root", type=Path)
    known, remaining = parser.parse_known_args(arguments)
    configured = known.yolo_source_root or os.environ.get("YOLO_RETRAINING_YOLOV5_ROOT")
    if not configured:
        parser.error("--yolo-source-root is required in the main process")
    return Path(configured).resolve(), remaining


SOURCE_ROOT, TRAIN_ARGUMENTS = _extract_source_root(sys.argv[1:])
os.environ["YOLO_RETRAINING_YOLOV5_ROOT"] = str(SOURCE_ROOT)
sys.path.insert(0, str(SOURCE_ROOT))

# This runs at module import time so Windows multiprocessing workers install the
# same read-only verifier before YOLOv5 creates its label-cache process pool.
import utils.dataloaders as dataloaders  # noqa: E402

install_read_only_verifier(dataloaders)


def main() -> int:
    import train

    sys.argv = [str(SOURCE_ROOT / "train.py"), *TRAIN_ARGUMENTS]
    options = train.parse_opt()
    train.main(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
