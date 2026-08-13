#!/usr/bin/env bash
set -euo pipefail

# Formal Full Cold base training:
#   initialization = original YOLOv5s pretrained weights
#   train data     = stage0 only
#   selection      = full

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${YOLO_RETRAINING_ENV_NAME:-yolo-retraining-v5}"
CONFIG="$PROJECT_ROOT/configs/task/full_cold.yaml"

export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
# Select the physical GPU here. Inside the isolated process it is always cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK_NAME="${TASK_NAME:-$(date +%m%d-%H%M%S)}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/tasks}"

started_at="$(date --iso-8601=seconds)"
started_seconds="$(date +%s)"

report_exit() {
  status=$?
  finished_at="$(date --iso-8601=seconds)"
  elapsed_seconds=$(( $(date +%s) - started_seconds ))
  echo "Finished: $finished_at"
  echo "Elapsed: ${elapsed_seconds}s"
  echo "Exit status: $status"
}
trap report_exit EXIT

echo "Starting Full Cold stage0 training"
echo "Started: $started_at"
echo "Environment: $ENV_NAME"
echo "Visible physical GPU: $CUDA_VISIBLE_DEVICES"
echo "Backend device inside process: 0"
echo "Epochs: $EPOCHS"
echo "Batch: $BATCH"
echo "Output root: $OUTPUT_ROOT"

conda run --no-capture-output -n "$ENV_NAME" \
  python -m yolo_retraining.run task \
  --config "$CONFIG" \
  --set "task.name=$TASK_NAME" \
  --set "task.seed=$SEED" \
  --set "task.output_root=$OUTPUT_ROOT" \
  --set "budget.value=$EPOCHS" \
  --set "backend.params.batch=$BATCH" \
  --set "backend.params.device=0" \
  --set "backend.params.workers=$WORKERS"
