#!/usr/bin/env bash
set -Eeuo pipefail

STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
GPU_ID=0
TRAIN_BATCH="${TRAIN_BATCH:-64}"
WORKERS="${WORKERS:-16}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-100}"
IMGSZ="${IMGSZ:-640}"
TASK_ROOT="$STUDY_ROOT/tasks"
LOG_ROOT="$STUDY_ROOT/logs"
BASE_CONFIG="$PROJECT_ROOT/configs/task/full_cold_stage0123.yaml"

VARIANTS=(
  dedup_tau_0p900
  dedup_tau_0p900_release_positive_clusters
  dedup_tau_0p900_release_positive_clusters_drop_pure_background
  dedup_tau_0p900_positive_representative
)

mkdir -p "$TASK_ROOT" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

task_dir() {
  local variant="$1"
  printf '%s/%s__full-cold-stage0123__stage0+stage1+stage2+stage3__yolov5s__s42' \
    "$TASK_ROOT" "$variant"
}

task_completed() {
  local result_file="$1/task_result.json"
  [[ -f "$result_file" ]] && grep -q '"status": "completed"' "$result_file"
}

echo "[Study] build and validate balance tau=0.90 manifests"
run_python "$STUDY_ROOT/build_manifests.py" >"$LOG_ROOT/build_manifests.log"

for variant in "${VARIANTS[@]}"; do
  layout="$STUDY_ROOT/variants/$variant/layout.yaml"
  run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$layout" \
    >"$LOG_ROOT/${variant}_layout_validation.log"
done

echo "[Study] environment preflight"
run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" \
  >"$LOG_ROOT/doctor.log"

for variant in "${VARIANTS[@]}"; do
  output_dir="$(task_dir "$variant")"
  if task_completed "$output_dir"; then
    echo "[Study] skip completed task: $variant"
    continue
  fi
  if [[ -e "$output_dir" ]]; then
    echo "[Study] incomplete output exists and will not be overwritten: $output_dir" >&2
    exit 1
  fi

  layout="$STUDY_ROOT/variants/$variant/layout.yaml"
  echo "[Study] start task=$variant GPU=$GPU_ID layout=$layout"
  run_python -m yolo_retraining.run task \
    --config "$BASE_CONFIG" \
    --set "task.name=$variant" \
    --set "task.output_root=$TASK_ROOT" \
    --set "data.layout=$layout" \
    --set "budget.value=$EPOCHS" \
    --set "backend.params.batch=$TRAIN_BATCH" \
    --set "backend.params.imgsz=$IMGSZ" \
    --set "backend.params.device=0" \
    --set "backend.params.workers=$WORKERS" \
    --set "backend.params.patience=$PATIENCE" \
    2>&1 | tee "$LOG_ROOT/${variant}.log"
  echo "[Study] completed task=$variant"
done

echo "[Study] all four new tau=0.90 tasks completed"
