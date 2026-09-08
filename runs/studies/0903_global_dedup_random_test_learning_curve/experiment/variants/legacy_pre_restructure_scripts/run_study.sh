#!/usr/bin/env bash
set -Eeuo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
PROJECT_ROOT="$(cd "$(dirname "$BASH_SOURCE")/../../../" && pwd)"
STUDY_ROOT="$PROJECT_ROOT/runs/studies/global_dedup_tau099_learning_curve"
CONDA_ENV="yolo-retraining-v5"
GPU_ID="0"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export YOLOV5_ROOT="$PROJECT_ROOT/.third_party/yolov5"
export PYTHONPATH="$PROJECT_ROOT/src"

BASE_LAYOUT="$PROJECT_ROOT/configs/data/self_improving.yaml"
EMBEDDINGS="$PROJECT_ROOT/data_analyse/dataset_redundancy/results/20260824-182608/embeddings.npz"
BUILDER="$STUDY_ROOT/build_global_layout.py"
VARIANT_DIR="$STUDY_ROOT/variants/dedup_tau_0p990"
LAYOUT="$VARIANT_DIR/layout.yaml"
CONFIG="$STUDY_ROOT/configs/global_dedup_tau099.yaml"
SEQUENCE_ROOT="$STUDY_ROOT/sequences"
LOG_ROOT="$STUDY_ROOT/logs"
OUTPUT_DIR="$SEQUENCE_ROOT/GlobalDedupTau099_PostSplit__seq-full-cold__stage0-stage7__yolov5s__s42"

mkdir -p "$STUDY_ROOT/configs" "$LOG_ROOT" "$SEQUENCE_ROOT"

for command in conda python nvidia-smi; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "[global-dedup] required command not found: $command" >&2
    exit 1
  }
done

echo "[global-dedup] project=$PROJECT_ROOT"
echo "[global-dedup] study=$STUDY_ROOT"
echo "[global-dedup] gpu=$GPU_ID"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "[global-dedup] environment doctor"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" \
  2>&1 | tee "$LOG_ROOT/doctor.log"

echo "[global-dedup] validate source layout"
conda run --no-capture-output -n "$CONDA_ENV" \
  python data_analyse/custom_dataset/custom_dataset.py validate --layout "$BASE_LAYOUT" \
  2>&1 | tee "$LOG_ROOT/source_layout_validation.log"

echo "[global-dedup] build post-dedup global layout"
conda run --no-capture-output -n "$CONDA_ENV" \
  python "$BUILDER" \
  --layout "$BASE_LAYOUT" \
  --embeddings "$EMBEDDINGS" \
  --output-dir "$VARIANT_DIR" \
  --seed 42 \
  --val-count 1000 \
  --test-count 1000 \
  --stages 8 \
  2>&1 | tee "$LOG_ROOT/build_global_layout.log"

echo "[global-dedup] validate generated layout"
conda run --no-capture-output -n "$CONDA_ENV" \
  python data_analyse/custom_dataset/custom_dataset.py validate --layout "$LAYOUT" \
  2>&1 | tee "$LOG_ROOT/generated_layout_validation.log"

if [[ ! -f "$CONFIG" ]]; then
  cat > "$CONFIG" <<'YAML'
include: ../../../../configs/sequence/learning_curve/_base_full_cold.yaml
sequence:
  name: GlobalDedupTau099_PostSplit
  output_root: ../sequences
data:
  layout: ../variants/dedup_tau_0p990/layout.yaml
YAML
fi

if [[ -f "$OUTPUT_DIR/sequence_result.json" ]] && grep -q '"status": "completed"' "$OUTPUT_DIR/sequence_result.json"; then
  echo "[global-dedup] sequence already completed: $OUTPUT_DIR"
  exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[global-dedup] refusing to overwrite incomplete sequence: $OUTPUT_DIR" >&2
  exit 3
fi

echo "[global-dedup] start 8-stage Full Cold sequence"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -m yolo_retraining.run sequence \
  --config "$CONFIG" \
  2>&1 | tee "$LOG_ROOT/sequence.log"

if [[ ! -f "$OUTPUT_DIR/sequence_result.json" ]]; then
  echo "[global-dedup] sequence did not produce a result: $OUTPUT_DIR" >&2
  exit 4
fi

echo "[global-dedup] completed: $OUTPUT_DIR"
