#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=1
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
DATA_LAYOUT="$PROJECT_ROOT/configs/data/self_improving_balance_test.yaml"
EXPERIMENT_NAME="Balance_Test"

cd "$PROJECT_ROOT"

for baseline in full_warm random_replay; do
  echo "[Baseline] starting $baseline on GPU $CUDA_VISIBLE_DEVICES"
  conda run --no-capture-output -n yolo-retraining-v5 \
    python -m yolo_retraining.run sequence \
    --config "configs/sequence/$baseline.yaml" \
    --set "data.layout=$DATA_LAYOUT" \
    --set "sequence.name=$EXPERIMENT_NAME"
  echo "[Baseline] completed $baseline"
done
