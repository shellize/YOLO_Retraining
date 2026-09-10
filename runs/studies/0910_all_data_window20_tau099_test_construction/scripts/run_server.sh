#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$STUDY_ROOT/../../.." && pwd)"
LOG_DIR="$STUDY_ROOT/logs"
mkdir -p "$LOG_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniforge3/bin/conda"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
  echo "conda executable not found" >&2
  exit 1
fi

if [[ -n "${YOLO_ANALYSIS_ENV:-}" ]]; then
  ANALYSIS_ENV="$YOLO_ANALYSIS_ENV"
elif "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq yolo-result-analysis; then
  ANALYSIS_ENV="yolo-result-analysis"
elif "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq yolo-retraining-v5; then
  ANALYSIS_ENV="yolo-retraining-v5"
else
  echo "neither yolo-result-analysis nor yolo-retraining-v5 conda environment exists" >&2
  exit 1
fi

VARIANT_DIR="$STUDY_ROOT/experiment/variants/window20_tau0p990_positive_representative"
TEMPORAL_DIR="$STUDY_ROOT/result/temporal_cluster_preview"
POST_DIR="$STUDY_ROOT/result/post_dedup_similarity"
for path in "$VARIANT_DIR" "$TEMPORAL_DIR" "$POST_DIR"; do
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite non-empty output: $path" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "analysis environment: $ANALYSIS_ENV"

"$CONDA_BIN" run --no-capture-output -n "$ANALYSIS_ENV" \
  python "$SCRIPT_DIR/build_all_data_dedup.py" \
  --config "$STUDY_ROOT/config/protocol.yaml" \
  2>&1 | tee "$LOG_DIR/build_all_data_dedup.log"

"$CONDA_BIN" run --no-capture-output -n "$ANALYSIS_ENV" \
  python "$SCRIPT_DIR/generate_audit_html.py" \
  --config "$STUDY_ROOT/config/protocol.yaml" \
  2>&1 | tee "$LOG_DIR/generate_audit_html.log"

echo "completed: $STUDY_ROOT"
