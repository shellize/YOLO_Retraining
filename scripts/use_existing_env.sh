#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${YOLO_RETRAINING_ENV_NAME:-yolo-retraining-v5}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' does not exist. Use scripts/bootstrap.sh explicitly." >&2
  exit 1
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$PROJECT_ROOT" --no-deps --no-build-isolation
YOLOV5_ROOT="$YOLO_ROOT" conda run --no-capture-output -n "$ENV_NAME" python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT"
