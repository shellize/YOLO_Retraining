#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
CONDA_BIN="${YOLO_RETRAINING_CONDA_BIN:-/home/lijinhe/miniforge3/bin/conda}"
GPU_ID="${GPU_ID:-0}"

RAW_LAYOUT="$SCRIPT_DIR/layout_raw.yaml"
EASY_LAYOUT="$SCRIPT_DIR/layout_easy.yaml"
RAW_OUTPUT="$SCRIPT_DIR/results/difficult"
EASY_OUTPUT="$SCRIPT_DIR/results/easy"
REDUNDANCY_SCRIPT="$PROJECT_ROOT/data_analyse/dataset_redundancy/redundancy_analysis.py"
WEIGHTS="$PROJECT_ROOT/.third_party/yolov5/yolov5s.pt"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "conda executable not found: $CONDA_BIN" >&2
  exit 10
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "YOLOv5s weights not found: $WEIGHTS" >&2
  exit 11
fi

run_one() {
  local label="$1"
  local layout="$2"
  local output="$3"
  local group_id="$4"
  local manifest="$output/variants/dedup_tau_0p990/manifests/$group_id.txt"
  if [[ -f "$output/summary.json" && -f "$manifest" && -f "$output/variants/dedup_tau_0p990/layout.yaml" ]]; then
    echo "[stroller dedup] existing completed result; refusing to overwrite: $output"
    return 0
  fi
  if [[ -e "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[stroller dedup] non-empty incomplete output; refusing to overwrite: $output" >&2
    exit 12
  fi
  mkdir -p "$output"
  echo "[stroller dedup] start $label: layout=$layout output=$output gpu=$GPU_ID"
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python "$REDUNDANCY_SCRIPT" \
    --layout "$layout" \
    --output-dir "$output" \
    --weights "$WEIGHTS" \
    --device "$GPU_ID" \
    --imgsz 640 \
    --batch-size 16 \
    --thresholds 0.99 \
    --temporal-window 10 \
    --random-seeds 41,42 \
    --threshold-workers 1
}

cd "$PROJECT_ROOT"
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_one difficult "$RAW_LAYOUT" "$RAW_OUTPUT" stroller_raw
run_one easy "$EASY_LAYOUT" "$EASY_OUTPUT" stroller_easy_source
echo "[stroller dedup] both source analyses are ready"
