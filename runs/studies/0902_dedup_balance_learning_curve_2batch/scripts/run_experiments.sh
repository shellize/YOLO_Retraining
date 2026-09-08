#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
GPU_ID=0
SEQUENCE_ROOT="$STUDY_ROOT/experiment/sequence"
LOG_ROOT="$STUDY_ROOT/logs"
CONDA_BIN="${YOLO_RETRAINING_CONDA_BIN:-}"

EXPERIMENTS=(
  dedup_tau_0p990
  dedup_tau_0p960_mixed_release_drop_pure_background
)
CONFIGS=(
  "$STUDY_ROOT/config/dedup_tau_0p990.yaml"
  "$STUDY_ROOT/config/dedup_tau_0p960_mixed_release_drop_pure_background.yaml"
)
OUTPUT_NAMES=(
  "Balance_LC2B_ordered_dedup_tau099__seq-full-cold__stage0-stage7__yolov5s__s42"
  "Balance_LC2B_ordered_tau096_mixed_release_drop_bg__seq-full-cold__stage0-stage7__yolov5s__s42"
)

mkdir -p "$SEQUENCE_ROOT" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

sequence_completed() {
  local result_file="$1/sequence_result.json"
  [[ -f "$result_file" ]] && grep -q '"status": "completed"' "$result_file"
}

if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" ]]; then
  user_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
  for candidate in \
    "$user_home/miniforge3/bin/conda" \
    "$user_home/miniconda3/bin/conda" \
    "$user_home/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      CONDA_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "[Dedup LC2B] conda was not found; set YOLO_RETRAINING_CONDA_BIN" >&2
  exit 10
fi
if [[ ! -d "$YOLOV5_ROOT" ]]; then
  echo "[Dedup LC2B] YOLOv5 source directory is missing: $YOLOV5_ROOT" >&2
  exit 11
fi

run_python() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python "$@"
}

echo "[Dedup LC2B] rebuild host-local manifests from cached embeddings"
run_python "$SCRIPT_DIR/build_manifests.py" >"$LOG_ROOT/build_manifests.log"

for experiment in "${EXPERIMENTS[@]}"; do
  layout="$STUDY_ROOT/experiment/variants/$experiment/layout.yaml"
  run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$layout" \
    >"$LOG_ROOT/${experiment}_layout_validation.log"
done

for config in "${CONFIGS[@]}"; do
  run_python -c \
    'import sys; from yolo_retraining.config import load_config; load_config(sys.argv[1], [f"sequence.output_root={sys.argv[2]}"])' \
    "$config" "$SEQUENCE_ROOT"
done

echo "[Dedup LC2B] environment preflight"
run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" >"$LOG_ROOT/doctor.log"

for index in "${!CONFIGS[@]}"; do
  experiment="${EXPERIMENTS[$index]}"
  config="${CONFIGS[$index]}"
  output="$SEQUENCE_ROOT/${OUTPUT_NAMES[$index]}"

  if sequence_completed "$output"; then
    echo "[Dedup LC2B] skip completed sequence: $experiment"
    continue
  fi
  if [[ -e "$output" ]]; then
    echo "[Dedup LC2B] incomplete output exists; refusing to overwrite: $output" >&2
    exit 12
  fi

  echo "[Dedup LC2B] start experiment=$experiment physical_GPU=$GPU_ID epochs=100"
  run_python -m yolo_retraining.run sequence \
    --config "$config" \
    --set "sequence.output_root=$SEQUENCE_ROOT" \
    2>&1 | tee "$LOG_ROOT/${experiment}.log"

  if ! sequence_completed "$output"; then
    echo "[Dedup LC2B] sequence returned without a completed result: $output" >&2
    exit 13
  fi
  echo "[Dedup LC2B] completed experiment=$experiment"
done

echo "[Dedup LC2B] both contrast sequences completed"
