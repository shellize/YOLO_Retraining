#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
CONFIG_ROOT="$PROJECT_ROOT/configs/sequence/selection_study"

CONFIGS=(
  "$CONFIG_ROOT/positive_only__cum_cold.yaml"
  "$CONFIG_ROOT/positive_only__current_warm.yaml"
  "$CONFIG_ROOT/pred_positive__cum_cold.yaml"
  "$CONFIG_ROOT/pred_positive__current_warm.yaml"
  "$CONFIG_ROOT/error_hard__cum_cold.yaml"
  "$CONFIG_ROOT/error_hard__current_warm.yaml"
  "$CONFIG_ROOT/gradnorm_topk__cum_cold.yaml"
  "$CONFIG_ROOT/gradnorm_topk__current_warm.yaml"
  "$CONFIG_ROOT/random_topk__cum_cold.yaml"
  "$CONFIG_ROOT/random_topk__current_warm.yaml"
)

if ! command -v conda >/dev/null 2>&1; then
  echo "[Study] conda was not found in PATH" >&2
  exit 10
fi
if [[ ! -d "$YOLOV5_ROOT" ]]; then
  echo "[Study] YOLOv5 source directory is missing: $YOLOV5_ROOT" >&2
  exit 11
fi

echo "[Study] preflight: environment=$CONDA_ENV, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
conda run --no-capture-output -n "$CONDA_ENV" \
  python "$PROJECT_ROOT/scripts/preflight_selection_study.py" "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
  output="$(conda run --no-capture-output -n "$CONDA_ENV" \
    python "$PROJECT_ROOT/scripts/preflight_selection_study.py" --print-output "$config")"
  if [[ -f "$output/sequence_result.json" ]]; then
    echo "[Study] already completed, skipping: $output"
    continue
  fi
  echo "[Study] starting: $(basename "$config")"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python -m yolo_retraining.run sequence --config "$config"
  if [[ ! -f "$output/sequence_result.json" ]]; then
    echo "[Study] command returned without a completed sequence result: $output" >&2
    exit 12
  fi
  echo "[Study] completed: $output"
done

echo "[Study] all selection experiments completed"
