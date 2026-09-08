#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
RESULT_CONDA_ENV="${YOLO_RESULT_CONDA_ENV:-$CONDA_ENV}"
GPU_ID="${GPU_ID:-0}"
CONDA_BIN="${YOLO_RETRAINING_CONDA_BIN:-}"

VARIANT_ROOT="$STUDY_ROOT/experiment/variants"
SEQUENCE_ROOT="$STUDY_ROOT/experiment/sequence"
LOG_ROOT="$STUDY_ROOT/logs"
PREPARE_SCRIPT="$SCRIPT_DIR/prepare_splits.py"
CURVE_SCRIPT="$SCRIPT_DIR/generate_learning_curve.py"
CONFIG_ROOT="$STUDY_ROOT/config"

SEEDS=(41 42)
CONFIGS=(
  "$CONFIG_ROOT/difficult_stroller_window10_tau099_split_s41.yaml"
  "$CONFIG_ROOT/difficult_stroller_window10_tau099_split_s42.yaml"
  "$CONFIG_ROOT/difficult_stroller_window10_tau099_ordered.yaml"
)
RUN_LABELS=(split_s41 split_s42 ordered)
SEQUENCE_NAMES=(
  "DifficultStrollerWindow10Tau099_SplitS41__seq-full-cold__stage0-stage7__yolov5s__s42"
  "DifficultStrollerWindow10Tau099_SplitS42__seq-full-cold__stage0-stage7__yolov5s__s42"
  "DifficultStrollerWindow10Tau099_Ordered__seq-full-cold__stage0-stage7__yolov5s__s42"
)

START_TRAINING=0
if [[ "${1:-}" == "--start-training" ]]; then
  START_TRAINING=1
elif [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--start-training]" >&2
  exit 2
fi

if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
  for candidate in "$USER_HOME/miniforge3/bin/conda" "$USER_HOME/miniconda3/bin/conda" "$USER_HOME/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      CONDA_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "conda was not found; set YOLO_RETRAINING_CONDA_BIN" >&2
  exit 10
fi
if [[ ! -d "$PROJECT_ROOT/.third_party/yolov5" ]]; then
  echo "YOLOv5 source directory is missing: $PROJECT_ROOT/.third_party/yolov5" >&2
  exit 11
fi

mkdir -p "$VARIANT_ROOT" "$SEQUENCE_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_python() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python "$@"
}

echo "[Study] project_root=$PROJECT_ROOT"
echo "[Study] protocol=strollerdifficult; tau=0.99; window=10; split seeds=${SEEDS[*]}; ordered control=true"
echo "[Study] conda_env=$CONDA_ENV gpu=$GPU_ID start_training=$START_TRAINING"

run_python "$PREPARE_SCRIPT" >"$LOG_ROOT/prepare_splits.log"

for seed in "${SEEDS[@]}"; do
  variant="$VARIANT_ROOT/window10_tau0p990_split_s$seed"
  run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$variant/layout.yaml" >"$LOG_ROOT/validate_split_s$seed.log"
done
run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$VARIANT_ROOT/window10_tau0p990_ordered/layout.yaml" >"$LOG_ROOT/validate_ordered.log"

for config in "${CONFIGS[@]}"; do
  run_python -c 'import sys; from yolo_retraining.config import load_config; load_config(sys.argv[1])' "$config"
done
run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" >"$LOG_ROOT/doctor.log"

if [[ "$START_TRAINING" -eq 0 ]]; then
  echo "[Study] preparation and validation completed; training was not started."
  echo "[Study] use --start-training only after reviewing manifests and protocol.json files."
  exit 0
fi

sequence_completed() {
  local sequence_dir="$1"
  [[ -f "$sequence_dir/sequence_status.json" && -f "$sequence_dir/sequence_result.json" ]] || return 1
  run_python -c '
import json
import sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
raise SystemExit(0 if status.get("status") == "completed" and status.get("completed_tasks") == 8 and result.get("status") == "completed" else 1)
' "$sequence_dir/sequence_status.json" "$sequence_dir/sequence_result.json"
}

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  output="$SEQUENCE_ROOT/${SEQUENCE_NAMES[$index]}"
  log_path="$LOG_ROOT/sequence_${RUN_LABELS[$index]}.log"
  if sequence_completed "$output"; then
    echo "[Study] skipping completed ${RUN_LABELS[$index]}"
    continue
  fi
  if [[ -e "$output" ]]; then
    echo "sequence output already exists; refusing to overwrite: $output" >&2
    exit 12
  fi
  echo "[Study] starting ${RUN_LABELS[$index]} on GPU=$GPU_ID"
  set +e
  run_python -m yolo_retraining.run sequence --config "$config" 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "sequence failed with exit code $status: $config" >&2
    exit "$status"
  fi
  if ! sequence_completed "$output"; then
    echo "sequence did not produce a completed result: $output" >&2
    exit 13
  fi
done

set +e
"$CONDA_BIN" run --no-capture-output -n "$RESULT_CONDA_ENV" python "$CURVE_SCRIPT" 2>&1 | tee "$LOG_ROOT/generate_learning_curve.log"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" -ne 0 ]]; then
  echo "learning-curve generation failed with exit code $status" >&2
  exit "$status"
fi
echo "[Study] difficult-stroller learning curves completed"
