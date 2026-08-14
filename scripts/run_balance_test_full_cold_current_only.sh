#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
DATA_LAYOUT="$PROJECT_ROOT/configs/data/self_improving_balance_test.yaml"
EXPERIMENT_NAME="Balance_Test"

for baseline in full_cold current_only; do
  echo "[Baseline] starting $baseline on GPU $CUDA_VISIBLE_DEVICES"
  conda run --no-capture-output -n yolo-retraining-v5 \
    python -m yolo_retraining.run sequence \
    --config "configs/sequence/$baseline.yaml" \
    --set "data.layout=$DATA_LAYOUT" \
    --set "sequence.name=$EXPERIMENT_NAME"
  echo "[Baseline] completed $baseline"
done

echo "[Baseline] starting full_cold_stage0123 on GPU $CUDA_VISIBLE_DEVICES"
conda run --no-capture-output -n yolo-retraining-v5 \
  python -m yolo_retraining.run task \
  --config "configs/task/full_cold_stage0123.yaml" \
  --set "data.layout=$DATA_LAYOUT" \
  --set "task.name=$EXPERIMENT_NAME"
echo "[Baseline] completed full_cold_stage0123"
