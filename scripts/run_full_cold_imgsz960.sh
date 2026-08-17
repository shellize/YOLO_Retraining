#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"

TASK_NAME="imgsz960"
DATA_LAYOUT="$PROJECT_ROOT/configs/data/self_improving_balance_test.yaml"

echo "[Experiment] starting $TASK_NAME full_cold_stage0123 on GPU $CUDA_VISIBLE_DEVICES"
conda run --no-capture-output -n yolo-retraining-v5 \
  python -m yolo_retraining.run task \
  --config "configs/task/full_cold_stage0123.yaml" \
  --set "task.name=$TASK_NAME" \
  --set "data.layout=$DATA_LAYOUT" \
  --set "backend.params.imgsz=960"
echo "[Experiment] completed $TASK_NAME full_cold_stage0123"
