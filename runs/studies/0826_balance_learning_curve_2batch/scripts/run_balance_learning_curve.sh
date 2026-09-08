#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DEFAULT_STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
GPU_ID="${GPU_ID:-0}"
STUDY_ROOT="${BALANCE_LEARNING_CURVE_ROOT:-$DEFAULT_STUDY_ROOT}"
SEQUENCE_ROOT="$STUDY_ROOT/experiment/sequence"
REPORT_ROOT="$STUDY_ROOT/result/learning_curve"
LOG_ROOT="$STUDY_ROOT/logs"
CONFIG_ROOT="$STUDY_ROOT/config/sequence"
REPORT_GENERATOR="$PROJECT_ROOT/charts/balance_learning_curve_2batch/generate_report.py"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENTS=(ordered random_s41 random_s43)
CONFIGS=(
  "$CONFIG_ROOT/ordered.yaml"
  "$CONFIG_ROOT/random_s41.yaml"
  "$CONFIG_ROOT/random_s43.yaml"
)

if ! command -v conda >/dev/null 2>&1; then
  echo "[Learning curve] conda was not found in PATH" >&2
  exit 10
fi
if [[ ! -d "$YOLOV5_ROOT" ]]; then
  echo "[Learning curve] YOLOv5 source directory is missing: $YOLOV5_ROOT" >&2
  exit 11
fi

mkdir -p "$SEQUENCE_ROOT" "$REPORT_ROOT" "$LOG_ROOT"

for config in "${CONFIGS[@]}"; do
  conda run --no-capture-output -n "$CONDA_ENV" python -c \
    'import sys; from yolo_retraining.config import load_config; load_config(sys.argv[1], [f"sequence.output_root={sys.argv[2]}"])' \
    "$config" "$SEQUENCE_ROOT"
done

conda run --no-capture-output -n "$CONDA_ENV" \
  python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT"

sequence_dir() {
  local experiment="$1"
  printf '%s/%s\n' \
    "$SEQUENCE_ROOT" \
    "Balance_Learning_Curve_${experiment}__seq-full-cold__stage0-stage7__yolov5s__s42"
}

for index in "${!CONFIGS[@]}"; do
  experiment="${EXPERIMENTS[$index]}"
  config="${CONFIGS[$index]}"
  output="$(sequence_dir "$experiment")"
  if [[ -f "$output/sequence_result.json" ]]; then
    echo "[Learning curve] already completed, skipping: $output"
    continue
  fi
  if [[ -e "$output" ]]; then
    echo "[Learning curve] incomplete output exists; refusing to overwrite: $output" >&2
    exit 12
  fi
  echo "[Learning curve] starting $experiment on physical GPU $GPU_ID"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python -m yolo_retraining.run sequence \
    --config "$config" \
    --set "sequence.output_root=$SEQUENCE_ROOT" \
    2>&1 | tee "$LOG_ROOT/${experiment}.log"
  if [[ ! -f "$output/sequence_result.json" ]]; then
    echo "[Learning curve] command returned without a completed sequence result: $output" >&2
    exit 13
  fi
  echo "[Learning curve] completed: $output"
done

conda run --no-capture-output -n "$CONDA_ENV" \
  python "$REPORT_GENERATOR" \
  --output-dir "$REPORT_ROOT" \
  --sequence "ordered=$(sequence_dir ordered)" \
  --sequence "random_s41=$(sequence_dir random_s41)" \
  --sequence "random_s43=$(sequence_dir random_s43)" \
  2>&1 | tee "$LOG_ROOT/summary.log"

echo "[Learning curve] all experiments completed"
echo "[Learning curve] metrics: $REPORT_ROOT/metrics.csv"
echo "[Learning curve] mAP50 plot: $REPORT_ROOT/map50_learning_curve.png"
echo "[Learning curve] per-class plot: $REPORT_ROOT/per_class_ap_learning_curve.png"
