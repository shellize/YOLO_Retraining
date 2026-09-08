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
TASK_ROOT="$STUDY_ROOT/experiment/task"
LOG_ROOT="$STUDY_ROOT/logs"
CONFIG_ROOT="$STUDY_ROOT/config"
PREPARE_SCRIPT="$SCRIPT_DIR/prepare_nested.py"
EVALUATE_SCRIPT="$SCRIPT_DIR/evaluate_cross_groups.py"

SEEDS=(41 42)
MODELS=(m1 m2)

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

mkdir -p "$VARIANT_ROOT" "$TASK_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_python() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python "$@"
}

echo "[Study] project_root=$PROJECT_ROOT"
echo "[Study] protocol=m1 difficult versus m2 easy; split seeds=${SEEDS[*]}; m3=false"
echo "[Study] tests=test1,test2,testhard; conda_env=$CONDA_ENV gpu=$GPU_ID start_training=$START_TRAINING"

run_python "$PREPARE_SCRIPT" >"$LOG_ROOT/prepare_nested.log"

for seed in "${SEEDS[@]}"; do
  variant="$VARIANT_ROOT/split_s$seed"
  run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$variant/m1_layout.yaml" >"$LOG_ROOT/validate_m1_s$seed.log"
  run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$variant/m2_layout.yaml" >"$LOG_ROOT/validate_m2_s$seed.log"
  for group in test1 test2 testhard; do
    run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$variant/eval_layouts/$group.yaml" >"$LOG_ROOT/validate_${group}_s$seed.log"
  done
done

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_python -c 'import sys; from yolo_retraining.config import load_config; load_config(sys.argv[1])' "$CONFIG_ROOT/${model}_s${seed}.yaml"
  done
done
run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" >"$LOG_ROOT/doctor.log"

if [[ "$START_TRAINING" -eq 0 ]]; then
  echo "[Study] preparation and validation completed; training was not started."
  echo "[Study] use --start-training only after reviewing split summaries and sample counts."
  exit 0
fi

task_completed() {
  local task_dir="$1"
  [[ -f "$task_dir/task_status.json" && -f "$task_dir/task_result.json" ]] || return 1
  run_python -c '
import json
import sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
raise SystemExit(0 if status.get("status") == "completed" and result.get("status") == "completed" else 1)
' "$task_dir/task_status.json" "$task_dir/task_result.json"
}

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ "$model" == "m1" ]]; then
      label="m1-difficult"
      train_group="train1"
    else
      label="m2-easy"
      train_group="train2"
    fi
    output="$TASK_ROOT/${model}_s${seed}__${label}__${train_group}__yolov5s__s42"
    config="$CONFIG_ROOT/${model}_s${seed}.yaml"
    log_path="$LOG_ROOT/task_${model}_s${seed}.log"
    if task_completed "$output"; then
      echo "[Study] skipping completed ${model} split_s${seed}"
      continue
    fi
    if [[ -e "$output" ]]; then
      echo "task output already exists; refusing to overwrite: $output" >&2
      exit 12
    fi
    echo "[Study] starting ${model} split_s${seed} on GPU=$GPU_ID"
    set +e
    run_python -m yolo_retraining.run task --config "$config" 2>&1 | tee "$log_path"
    status="${PIPESTATUS[0]}"
    set -e
    if [[ "$status" -ne 0 ]]; then
      echo "task failed with exit code $status: $config" >&2
      exit "$status"
    fi
    if ! task_completed "$output"; then
      echo "task did not produce a completed result: $output" >&2
      exit 13
    fi
  done
done

set +e
"$CONDA_BIN" run --no-capture-output -n "$RESULT_CONDA_ENV" python "$EVALUATE_SCRIPT" 2>&1 | tee "$LOG_ROOT/evaluate_cross_groups.log"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" -ne 0 ]]; then
  echo "cross-group evaluation failed with exit code $status" >&2
  exit "$status"
fi
echo "[Study] m1/m2 cross-group evaluation completed"
