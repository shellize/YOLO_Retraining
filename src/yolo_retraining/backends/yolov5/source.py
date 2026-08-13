from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


YOLOV5_COMMIT = "915bbf294bb74c859f0b41f1c23bc395014ea679"
YOLOV5_TAG = "v7.0"


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_yolov5_root() -> Path:
    configured = os.environ.get("YOLOV5_ROOT")
    root = Path(configured).expanduser() if configured else project_root() / ".third_party" / "yolov5"
    return root.resolve()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_yolov5_source(root: Path | None = None) -> dict[str, Any]:
    root = (root or resolve_yolov5_root()).resolve()
    required = (root / "train.py", root / "val.py", root / "models" / "yolov5s.yaml")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"YOLOv5 v7.0 source is missing at {root}; run requirements/bootstrap first or set YOLOV5_ROOT")
    try:
        commit = _git(root, "rev-parse", "HEAD")
        tracked_changes = _git(root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot verify YOLOv5 source revision at {root}: {error}") from error
    if commit != YOLOV5_COMMIT:
        raise RuntimeError(f"YOLOv5 source revision mismatch: expected {YOLOV5_COMMIT}, found {commit} at {root}")
    if tracked_changes:
        raise RuntimeError(f"YOLOv5 source contains tracked modifications at {root}")
    return {"root": str(root), "commit": commit, "tag": YOLOV5_TAG}


def resolve_checkpoint(value: str | Path, root: Path) -> Path:
    checkpoint = Path(value).expanduser()
    if checkpoint.is_absolute():
        resolved = checkpoint.resolve()
    elif checkpoint.is_file():
        resolved = checkpoint.resolve()
    else:
        resolved = (root / checkpoint).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"YOLOv5 checkpoint does not exist: {resolved}; run requirements/bootstrap to download yolov5s.pt")
    return resolved
