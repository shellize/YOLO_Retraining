#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
RESULT_CONDA_ENV="${YOLO_RESULT_CONDA_ENV:-$CONDA_ENV}"
GPU_ID="${GPU_ID:-0}"
CONDA_BIN="${YOLO_RETRAINING_CONDA_BIN:-}"

SEQUENCE_ROOT="$STUDY_ROOT/experiment/sequence"
VARIANT_ROOT="$STUDY_ROOT/experiment/variants"
LOG_ROOT="$STUDY_ROOT/logs"
CONFIG_ROOT="$STUDY_ROOT/config"

MAIN_CONFIG="$CONFIG_ROOT/global_dedup_tau099.yaml"
MAIN_OUTPUT="$SEQUENCE_ROOT/GlobalDedupTau099_PostSplit__seq-full-cold__stage0-stage7__yolov5s__s42"

SPLIT_SEEDS=(41 43)
NORMAL_CONFIGS=(
  "$CONFIG_ROOT/global_dedup_tau099_split_s41.yaml"
  "$CONFIG_ROOT/global_dedup_tau099_split_s43.yaml"
)
RETRY_CONFIGS=(
  "$CONFIG_ROOT/global_dedup_tau099_split_s41_retry.yaml"
  "$CONFIG_ROOT/global_dedup_tau099_split_s43_retry.yaml"
)
NORMAL_OUTPUTS=(
  "$SEQUENCE_ROOT/GlobalDedupTau099_PostSplit_RandomS41__seq-full-cold__stage0-stage7__yolov5s__s42"
  "$SEQUENCE_ROOT/GlobalDedupTau099_PostSplit_RandomS43__seq-full-cold__stage0-stage7__yolov5s__s42"
)
RETRY_OUTPUTS=(
  "$SEQUENCE_ROOT/GlobalDedupTau099_PostSplit_RandomS41_Rerun__seq-full-cold__stage0-stage7__yolov5s__s42"
  "$SEQUENCE_ROOT/GlobalDedupTau099_PostSplit_RandomS43_Rerun__seq-full-cold__stage0-stage7__yolov5s__s42"
)

if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
  for candidate in \
    "$USER_HOME/miniforge3/bin/conda" \
    "$USER_HOME/miniconda3/bin/conda" \
    "$USER_HOME/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      CONDA_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "[Global dedup controls] conda was not found; set YOLO_RETRAINING_CONDA_BIN" >&2
  exit 10
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$YOLOV5_ROOT" ]]; then
  echo "[Global dedup controls] YOLOv5 source directory is missing: $YOLOV5_ROOT" >&2
  exit 11
fi

mkdir -p "$SEQUENCE_ROOT" "$VARIANT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

run_python() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python "$@"
}

run_result_python() {
  "$CONDA_BIN" run --no-capture-output -n "$RESULT_CONDA_ENV" python "$@"
}

sequence_completed() {
  local sequence_dir="$1"
  [[ -f "$sequence_dir/sequence_status.json" ]] || return 1
  run_python -c '
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "completed" and payload.get("completed_tasks") == 8 else 1)
' "$sequence_dir/sequence_status.json"
}

run_sequence() {
  local config="$1"
  local output="$2"
  local log_path="$3"
  local label="$4"
  local status

  if sequence_completed "$output"; then
    echo "[Global dedup controls] skip completed sequence: $label"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "[Global dedup controls] incomplete output exists; refusing to overwrite: $output" >&2
    return 12
  fi

  echo "[Global dedup controls] start $label on physical GPU=$GPU_ID"
  set +e
  run_python -m yolo_retraining.run sequence \
    --config "$config" \
    --set "sequence.output_root=$SEQUENCE_ROOT" \
    2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "[Global dedup controls] sequence failed with exit code $status: $label" >&2
    return "$status"
  fi
  if ! sequence_completed "$output"; then
    echo "[Global dedup controls] sequence did not produce a completed result: $output" >&2
    return 13
  fi
  echo "[Global dedup controls] completed $label"
}

echo "[Global dedup controls] project_root=$PROJECT_ROOT"
echo "[Global dedup controls] study_root=$STUDY_ROOT"
echo "[Global dedup controls] physical_GPU=$GPU_ID"
echo "[Global dedup controls] protocol=global tau0.99 temporal-window=1; split controls=41,43"

for index in "${!SPLIT_SEEDS[@]}"; do
  seed="${SPLIT_SEEDS[$index]}"
  variant="$VARIANT_ROOT/dedup_tau_0p990_split_s$seed"
  run_python "$STUDY_ROOT/scripts/build_global_layout.py" \
    --seed "$seed" \
    --output-dir "$variant" \
    >"$LOG_ROOT/build_dedup_tau_0p990_split_s$seed.log"
done

for variant in \
  "$VARIANT_ROOT/dedup_tau_0p990" \
  "$VARIANT_ROOT/dedup_tau_0p990_split_s41" \
  "$VARIANT_ROOT/dedup_tau_0p990_split_s43"; do
  run_python data_analyse/custom_dataset/custom_dataset.py validate \
    --layout "$variant/layout.yaml" \
    >"$LOG_ROOT/$(basename "$variant")_layout_validation.log"
done

for config in \
  "$MAIN_CONFIG" \
  "${NORMAL_CONFIGS[@]}" \
  "${RETRY_CONFIGS[@]}"; do
  run_python -c \
    'import sys; from yolo_retraining.config import load_config; load_config(sys.argv[1], [f"sequence.output_root={sys.argv[2]}"])' \
    "$config" "$SEQUENCE_ROOT"
done

run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" \
  >"$LOG_ROOT/randomization_controls_server_doctor.log"

run_sequence \
  "$MAIN_CONFIG" \
  "$MAIN_OUTPUT" \
  "$LOG_ROOT/global_dedup_main_s42.log" \
  "global dedup split seed 42"

for index in "${!SPLIT_SEEDS[@]}"; do
  seed="${SPLIT_SEEDS[$index]}"
  normal_output="${NORMAL_OUTPUTS[$index]}"
  retry_output="${RETRY_OUTPUTS[$index]}"

  if sequence_completed "$normal_output"; then
    echo "[Global dedup controls] normal split seed $seed is already complete"
    continue
  fi

  if [[ -e "$normal_output" ]]; then
    echo "[Global dedup controls] normal split seed $seed is incomplete; preserving it and using retry output"
    run_sequence \
      "${RETRY_CONFIGS[$index]}" \
      "$retry_output" \
      "$LOG_ROOT/global_dedup_split_s${seed}_retry.log" \
      "random split seed $seed retry"
  else
    run_sequence \
      "${NORMAL_CONFIGS[$index]}" \
      "$normal_output" \
      "$LOG_ROOT/global_dedup_split_s${seed}.log" \
      "random split seed $seed"
  fi
done

set +e
run_result_python "$STUDY_ROOT/scripts/compare_randomized_learning_curves.py" \
  2>&1 | tee "$LOG_ROOT/randomization_comparison_server.log"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" -ne 0 ]]; then
  echo "[Global dedup controls] comparison failed with exit code $status" >&2
  exit "$status"
fi

echo "[Global dedup controls] completed main sequence, split controls, and comparison."
