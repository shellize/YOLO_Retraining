#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
HYP_DIR="$PROJECT_ROOT/configs/hyp"
TASK_SUFFIX="__full-cold-stage0123__stage0+stage1+stage2+stage3__yolov5s__s42"

run_task() {
  bash "$PROJECT_ROOT/scripts/run_full_cold_task.sh" "$@"
}

task_dir() {
  printf '%s/runs/tasks/%s%s\n' "$PROJECT_ROOT" "$1" "$TASK_SUFFIX"
}

# Phase 1: one 1280-resolution probe. It uses the same augmentation as the
# existing imgsz=960 result, so this isolates the marginal resolution change.
TASK_1280="fullcold__gpu0__imgsz1280__mosaic1p0__scale0p5__resolution_probe"
run_task "$TASK_1280" 1280 "$HYP_DIR/hyp.scratch-low.mosaic1.scale05.yaml"

# Phase 2: compare Mosaic on the smaller, established 640 protocol.
TASK_MOSAIC1="fullcold__gpu0__imgsz640__mosaic1p0__scale0p5__mosaic_compare"
TASK_MOSAIC0="fullcold__gpu0__imgsz640__mosaic0p0__scale0p5__mosaic_compare"
run_task "$TASK_MOSAIC1" 640 "$HYP_DIR/hyp.scratch-low.mosaic1.scale05.yaml"
run_task "$TASK_MOSAIC0" 640 "$HYP_DIR/hyp.scratch-low.mosaic0.scale05.yaml"

MOSAIC1_DIR="$(task_dir "$TASK_MOSAIC1")"
MOSAIC0_DIR="$(task_dir "$TASK_MOSAIC0")"
BEST_MOSAIC="$(conda run --no-capture-output -n "$CONDA_ENV" python scripts/select_best_task.py \
  --history-metric 'metrics/mAP_0.5:0.95' \
  "mosaic1=$MOSAIC1_DIR" \
  "mosaic0=$MOSAIC0_DIR")"

echo "[Experiment] best Mosaic by validation mAP50:95: $BEST_MOSAIC"

# Phase 3: compare two scale values after the Mosaic decision. The default
# scale=0.5 is retained as the reference; scale=0.25 is the milder alternative.
case "$BEST_MOSAIC" in
  mosaic1)
    BEST_MOSAIC_TOKEN="mosaic1p0"
    HYP_SCALE025="$HYP_DIR/hyp.scratch-low.mosaic1.scale025.yaml"
    TASK_SCALE050="$TASK_MOSAIC1"
    ;;
  mosaic0)
    BEST_MOSAIC_TOKEN="mosaic0p0"
    HYP_SCALE025="$HYP_DIR/hyp.scratch-low.mosaic0.scale025.yaml"
    TASK_SCALE050="$TASK_MOSAIC0"
    ;;
  *)
    echo "[Experiment] unexpected Mosaic selection: $BEST_MOSAIC" >&2
    exit 1
    ;;
esac

TASK_SCALE025="fullcold__gpu0__imgsz640__${BEST_MOSAIC_TOKEN}__scale0p25__scale_compare"
run_task "$TASK_SCALE025" 640 "$HYP_SCALE025"
echo "[Experiment] reusing scale=0.5 result from $TASK_SCALE050"

echo "[Experiment] study completed"
echo "[Experiment] resolution probe: $(task_dir "$TASK_1280")"
echo "[Experiment] Mosaic comparison: $(task_dir "$TASK_MOSAIC1")"
echo "[Experiment] Mosaic comparison: $(task_dir "$TASK_MOSAIC0")"
echo "[Experiment] scale comparison: $(task_dir "$TASK_SCALE025")"
echo "[Experiment] scale comparison: $(task_dir "$TASK_SCALE050")"
