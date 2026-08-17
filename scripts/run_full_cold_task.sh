#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TASK_NAME IMGSZ HYP_FILE" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"

TASK_NAME="$1"
IMGSZ="$2"
HYP_FILE="$(realpath "$3")"
CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
DATA_LAYOUT="$PROJECT_ROOT/configs/data/self_improving_balance_test.yaml"

if [[ ! -f "$HYP_FILE" ]]; then
  echo "[Experiment] hyperparameter file not found: $HYP_FILE" >&2
  exit 1
fi

echo "[Experiment] starting $TASK_NAME: imgsz=$IMGSZ hyp=$(basename "$HYP_FILE") GPU=$CUDA_VISIBLE_DEVICES"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -m yolo_retraining.run task \
  --config "configs/task/full_cold_stage0123.yaml" \
  --set "task.name=$TASK_NAME" \
  --set "data.layout=$DATA_LAYOUT" \
  --set "backend.params.imgsz=$IMGSZ" \
  --set "backend.params.hyp=$HYP_FILE"
echo "[Experiment] completed $TASK_NAME"
