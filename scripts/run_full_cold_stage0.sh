#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"


conda run --no-capture-output -n yolo-retraining-v5 \
  python -m yolo_retraining.run task \
  --config configs/task/full_cold.yaml
