#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
GPU_ID="${GPU_ID:-0}"
CONDA_BIN="${YOLO_RETRAINING_CONDA_BIN:-}"
SEQUENCE_ROOT="$STUDY_ROOT/experiment/sequence"
VARIANT_ROOT="$STUDY_ROOT/experiment/variants"
LOG_ROOT="$STUDY_ROOT/logs"

SPLIT_SEEDS=(41 43)
CONFIGS=(
  "$STUDY_ROOT/config/global_dedup_tau099_split_s41.yaml"
  "$STUDY_ROOT/config/global_dedup_tau099_split_s43.yaml"
)
VARIANTS=(
  dedup_tau_0p990_split_s41
  dedup_tau_0p990_split_s43
)
OUTPUT_NAMES=(
  "GlobalDedupTau099_PostSplit_RandomS41__seq-full-cold__stage0-stage7__yolov5s__s42"
  "GlobalDedupTau099_PostSplit_RandomS43__seq-full-cold__stage0-stage7__yolov5s__s42"
)

mkdir -p "$SEQUENCE_ROOT" "$VARIANT_ROOT" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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
  echo "[Randomization controls] conda was not found; set YOLO_RETRAINING_CONDA_BIN" >&2
  exit 10
fi
if [[ ! -d "$YOLOV5_ROOT" ]]; then
  echo "[Randomization controls] YOLOv5 source directory is missing: $YOLOV5_ROOT" >&2
  exit 11
fi

run_python() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python "$@"
}

sequence_completed() {
  local result_file="$1/sequence_result.json"
  [[ -f "$result_file" ]] && grep -q '"status": "completed"' "$result_file"
}

echo "[Randomization controls] project_root=$PROJECT_ROOT"
echo "[Randomization controls] physical_GPU=$GPU_ID"
echo "[Randomization controls] split_seeds=${SPLIT_SEEDS[*]} sequence_training_seed=42"

for index in "${!SPLIT_SEEDS[@]}"; do
  seed="${SPLIT_SEEDS[$index]}"
  variant="${VARIANTS[$index]}"
  output_dir="$VARIANT_ROOT/$variant"
  echo "[Randomization controls] build split seed=$seed output=$output_dir"
  run_python "$STUDY_ROOT/scripts/build_global_layout.py" \
    --seed "$seed" \
    --output-dir "$output_dir" \
    >"$LOG_ROOT/build_$variant.log"
done

for variant in "${VARIANTS[@]}"; do
  run_python data_analyse/custom_dataset/custom_dataset.py validate \
    --layout "$VARIANT_ROOT/$variant/layout.yaml" \
    >"$LOG_ROOT/${variant}_layout_validation.log"
done

for config in "${CONFIGS[@]}"; do
  run_python -c \
    'import sys; from yolo_retraining.config import load_config; load_config(sys.argv[1], [f"sequence.output_root={sys.argv[2]}"])' \
    "$config" "$SEQUENCE_ROOT"
done

run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" \
  >"$LOG_ROOT/randomization_controls_doctor.log"

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  output="$SEQUENCE_ROOT/${OUTPUT_NAMES[$index]}"
  variant="${VARIANTS[$index]}"

  if sequence_completed "$output"; then
    echo "[Randomization controls] skip completed sequence: $variant"
    continue
  fi
  if [[ -e "$output" ]]; then
    echo "[Randomization controls] incomplete output exists; refusing to overwrite: $output" >&2
    exit 12
  fi

  echo "[Randomization controls] start variant=$variant split_seed=${SPLIT_SEEDS[$index]} physical_GPU=$GPU_ID"
  run_python -m yolo_retraining.run sequence \
    --config "$config" \
    --set "sequence.output_root=$SEQUENCE_ROOT" \
    2>&1 | tee "$LOG_ROOT/$variant.log"

  if ! sequence_completed "$output"; then
    echo "[Randomization controls] sequence returned without a completed result: $output" >&2
    exit 13
  fi
  echo "[Randomization controls] completed variant=$variant"
done

run_python "$STUDY_ROOT/scripts/compare_randomized_learning_curves.py" \
  2>&1 | tee "$LOG_ROOT/randomization_comparison.log"

echo "[Randomization controls] both control sequences and comparison completed"
