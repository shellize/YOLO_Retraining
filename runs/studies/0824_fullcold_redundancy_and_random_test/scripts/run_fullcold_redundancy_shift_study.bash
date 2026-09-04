#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DEFAULT_STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${YOLO_RETRAINING_CONDA_ENV:-yolo-retraining-v5}"
GPU_ID="${GPU_ID:-0}"
GPU_POLL_INTERVAL_SECONDS="${YOLO_RETRAINING_GPU_POLL_SECONDS:-600}"
GPU_REQUIRED_IDLE_CHECKS="${YOLO_RETRAINING_GPU_REQUIRED_IDLE_CHECKS:-2}"
GPU_IDLE_UTILIZATION_PCT="${YOLO_RETRAINING_GPU_IDLE_UTILIZATION_PCT:-5}"
GPU_IDLE_MEMORY_MIB="${YOLO_RETRAINING_GPU_IDLE_MEMORY_MIB:-1024}"
SKIP_GPU_WAIT="${YOLO_RETRAINING_SKIP_GPU_WAIT:-0}"
REDUNDANCY_DEVICE="${REDUNDANCY_DEVICE:-cpu}"
REDUNDANCY_THRESHOLD_WORKERS="${REDUNDANCY_THRESHOLD_WORKERS:-8}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
EMBED_BATCH="${EMBED_BATCH:-32}"
WORKERS="${WORKERS:-16}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
TOP_K="${TOP_K:-20}"
TEMPORAL_WINDOW=1
RANDOM_SPLIT_SEED="${RANDOM_SPLIT_SEED:-42}"
MATCHED_RANDOM_CONTROLS="${MATCHED_RANDOM_CONTROLS:-0}"
MATCHED_RANDOM_SEEDS="${MATCHED_RANDOM_SEEDS:-42}"
STUDY_ID="${STUDY_ID:-$(basename "$DEFAULT_STUDY_ROOT")}"
LEGACY_ASSET_ID="${LEGACY_ASSET_ID:-20260824-182608}"
BASE_LAYOUT="${BASE_LAYOUT:-$PROJECT_ROOT/configs/data/self_improving.yaml}"
YOLOV5_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"

DEDUP_CANDIDATE_THRESHOLDS=(0.900 0.910 0.920 0.930 0.940 0.950 0.960 0.970 0.980 0.990)
DEDUP_CANDIDATE_THRESHOLD_CSV="$(IFS=,; echo "${DEDUP_CANDIDATE_THRESHOLDS[*]}")"
DEDUP_TARGET_RETAINED_FRACTION="${DEDUP_TARGET_RETAINED_FRACTION:-0.42857142857142855}"

if [[ ! "$STUDY_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "STUDY_ID may contain only letters, numbers, dot, underscore, and dash: $STUDY_ID" >&2
  exit 2
fi

for command in conda python realpath tee nvidia-smi; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command not found: $command" >&2
    exit 1
  fi
done

if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one physical GPU index, received: $GPU_ID" >&2
  exit 2
fi
if ! [[ "$GPU_POLL_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "YOLO_RETRAINING_GPU_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$GPU_REQUIRED_IDLE_CHECKS" =~ ^[0-9]+$ ]] || (( GPU_REQUIRED_IDLE_CHECKS < 2 )); then
  echo "YOLO_RETRAINING_GPU_REQUIRED_IDLE_CHECKS must be an integer of at least 2" >&2
  exit 2
fi
if ! [[ "$GPU_IDLE_UTILIZATION_PCT" =~ ^[0-9]+$ && "$GPU_IDLE_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "GPU idle utilization and memory limits must be non-negative integers" >&2
  exit 2
fi
if ! [[ "$REDUNDANCY_THRESHOLD_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REDUNDANCY_THRESHOLD_WORKERS must be a positive integer" >&2
  exit 2
fi

BASE_LAYOUT="$(realpath "$BASE_LAYOUT")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export YOLOV5_ROOT

STUDY_ROOT="$PROJECT_ROOT/runs/studies/$STUDY_ID"
SEQUENCE_ROOT="$STUDY_ROOT/experiment/sequence"
LOG_ROOT="$STUDY_ROOT/logs"
REPORT_ROOT="$STUDY_ROOT/result/fullcold_comparison"
REDUNDANCY_ROOT="$PROJECT_ROOT/data_analyse/dataset_redundancy/results/$LEGACY_ASSET_ID"
RANDOM_SPLIT_ROOT="$PROJECT_ROOT/data_analyse/custom_dataset/results/random_frame_s${RANDOM_SPLIT_SEED}"
mkdir -p "$SEQUENCE_ROOT" "$LOG_ROOT" "$REPORT_ROOT"

on_error() {
  local exit_code=$?
  echo "[Study] failed at line $1 with exit code $exit_code" >&2
  echo "[Study] rerun with the same STUDY_ID=$STUDY_ID after resolving the error; completed sequences will be skipped" >&2
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

is_gpu_idle() {
  local snapshot utilization memory_used memory_total processes process_state
  if ! snapshot="$(
    nvidia-smi -i "$GPU_ID" \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
  )"; then
    echo "[GPU wait] $(date '+%Y-%m-%d %H:%M:%S') unable to query physical GPU $GPU_ID" >&2
    return 1
  fi
  IFS=',' read -r utilization memory_used memory_total <<< "$snapshot"
  if ! [[ "${utilization:-}" =~ ^[0-9]+$ && "${memory_used:-}" =~ ^[0-9]+$ && "${memory_total:-}" =~ ^[0-9]+$ ]]; then
    echo "[GPU wait] $(date '+%Y-%m-%d %H:%M:%S') unable to parse GPU snapshot: $snapshot" >&2
    return 1
  fi

  if ! processes="$(
    nvidia-smi -i "$GPU_ID" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null
  )"; then
    echo "[GPU wait] $(date '+%Y-%m-%d %H:%M:%S') unable to query GPU compute processes" >&2
    return 1
  fi
  processes="$(printf '%s' "$processes" | tr -d '[:space:]')"
  if [[ -n "$processes" ]]; then
    process_state="present"
  else
    process_state="none"
  fi

  printf '[GPU wait] %s GPU%s util=%s%% mem=%s/%sMiB compute_process=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$GPU_ID" "$utilization" "$memory_used" "$memory_total" "$process_state"

  [[ -z "$processes" ]] \
    && (( utilization <= GPU_IDLE_UTILIZATION_PCT )) \
    && (( memory_used <= GPU_IDLE_MEMORY_MIB ))
}

wait_for_consecutive_gpu_idle_checks() {
  local idle_checks=0
  echo "[GPU wait] polling physical GPU $GPU_ID every ${GPU_POLL_INTERVAL_SECONDS}s"
  echo "[GPU wait] launch requires ${GPU_REQUIRED_IDLE_CHECKS} consecutive idle checks"
  echo "[GPU wait] idle limits: utilization<=${GPU_IDLE_UTILIZATION_PCT}% memory<=${GPU_IDLE_MEMORY_MIB}MiB and no compute process"

  while (( idle_checks < GPU_REQUIRED_IDLE_CHECKS )); do
    if is_gpu_idle; then
      ((idle_checks += 1))
      echo "[GPU wait] idle check ${idle_checks}/${GPU_REQUIRED_IDLE_CHECKS} passed"
    else
      idle_checks=0
      echo "[GPU wait] GPU $GPU_ID is busy or unavailable; consecutive idle checks reset"
    fi

    if (( idle_checks < GPU_REQUIRED_IDLE_CHECKS )); then
      echo "[GPU wait] next check in ${GPU_POLL_INTERVAL_SECONDS}s"
      sleep "$GPU_POLL_INTERVAL_SECONDS"
    fi
  done

  echo "[GPU wait] GPU $GPU_ID passed ${GPU_REQUIRED_IDLE_CHECKS} consecutive idle checks; starting study"
}

threshold_label() {
  printf '%.3f' "$1" | tr '.' 'p'
}

sequence_dir() {
  local experiment_name="$1"
  printf '%s/%s__seq-full-cold__stage0-stage3__yolov5s__s42' "$SEQUENCE_ROOT" "$experiment_name"
}

sequence_completed() {
  local result_file="$1/sequence_result.json"
  [[ -f "$result_file" ]] && grep -q '"status": "completed"' "$result_file"
}

run_sequence() {
  local experiment_name="$1"
  local layout="$2"
  local output_dir
  output_dir="$(sequence_dir "$experiment_name")"
  if sequence_completed "$output_dir"; then
    echo "[Study] skip completed sequence: $experiment_name"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "[Study] incomplete output already exists and will not be overwritten: $output_dir" >&2
    exit 1
  fi
  echo "[Study] start sequence=$experiment_name layout=$layout GPU=$GPU_ID"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python -m yolo_retraining.run sequence \
    --config configs/sequence/full_cold.yaml \
    --set "sequence.name=$experiment_name" \
    --set "sequence.output_root=$SEQUENCE_ROOT" \
    --set "data.layout=$layout" \
    --set "task_template.budget.value=$EPOCHS" \
    --set "task_template.backend.params.batch=$TRAIN_BATCH" \
    --set "task_template.backend.params.imgsz=$IMGSZ" \
    --set "task_template.backend.params.device=0" \
    --set "task_template.backend.params.workers=$WORKERS" \
    2>&1 | tee "$LOG_ROOT/${experiment_name}.log"
  echo "[Study] completed sequence=$experiment_name"
}

echo "[Study] id=$STUDY_ID root=$STUDY_ROOT GPU=$GPU_ID"
echo "[Study] candidate_thresholds=$DEDUP_CANDIDATE_THRESHOLD_CSV target_retained=$DEDUP_TARGET_RETAINED_FRACTION temporal_window=$TEMPORAL_WINDOW"
echo "[Study] redundancy_device=$REDUNDANCY_DEVICE threshold_workers=$REDUNDANCY_THRESHOLD_WORKERS"

echo "[Study] environment preflight"
run_python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT" \
  2>&1 | tee "$LOG_ROOT/doctor.log"

echo "[Study] validate original layout"
run_python data_analyse/custom_dataset/custom_dataset.py validate --layout "$BASE_LAYOUT" \
  2>&1 | tee "$LOG_ROOT/base_layout_validation.log"

DEDUP_LAYOUTS_READY=1
for threshold in "${DEDUP_CANDIDATE_THRESHOLDS[@]}"; do
  label="$(threshold_label "$threshold")"
  if [[ ! -f "$REDUNDANCY_ROOT/variants/dedup_tau_${label}/layout.yaml" ]]; then
    DEDUP_LAYOUTS_READY=0
  fi
done

if [[ "$DEDUP_LAYOUTS_READY" -eq 1 && -f "$REDUNDANCY_ROOT/summary.json" ]]; then
  echo "[Study] reuse completed redundancy analysis: $REDUNDANCY_ROOT"
else
  echo "[Study] extract YOLOv5s features and build candidate adjacent-frame deduplicated layouts"
  REDUNDANCY_ARGS=(
    data_analyse/dataset_redundancy/redundancy_analysis.py
    --layout "$BASE_LAYOUT"
    --output-dir "$REDUNDANCY_ROOT"
    --device "$REDUNDANCY_DEVICE"
    --imgsz "$IMGSZ"
    --batch-size "$EMBED_BATCH"
    --thresholds "$DEDUP_CANDIDATE_THRESHOLD_CSV"
    --threshold-workers "$REDUNDANCY_THRESHOLD_WORKERS"
    --temporal-window "$TEMPORAL_WINDOW"
    --top-k "$TOP_K"
    --random-seeds 41,42,43
  )
  if [[ -f "$REDUNDANCY_ROOT/embeddings.npz" ]]; then
    REDUNDANCY_ARGS+=(--embeddings "$REDUNDANCY_ROOT/embeddings.npz")
  fi
  run_python "${REDUNDANCY_ARGS[@]}" 2>&1 | tee "$LOG_ROOT/redundancy_analysis.log"
fi

for threshold in "${DEDUP_CANDIDATE_THRESHOLDS[@]}"; do
  label="$(threshold_label "$threshold")"
  run_python data_analyse/custom_dataset/custom_dataset.py validate \
    --layout "$REDUNDANCY_ROOT/variants/dedup_tau_${label}/layout.yaml" \
    >"$LOG_ROOT/dedup_tau_${label}_validation.log"
done

echo "[Study] extract mixed annotation clusters from every dedup variant"
run_python data_analyse/dataset_redundancy/extract_mixed_clusters.py \
  --results-root "$REDUNDANCY_ROOT" \
  2>&1 | tee "$LOG_ROOT/mixed_clusters.log"

echo "[Study] select deduplication threshold closest to retained fraction 3/7"
SELECTED_DEDUP_THRESHOLD="$(
  run_python - \
    "$REDUNDANCY_ROOT/summary.json" \
    "$REPORT_ROOT/redundancy_threshold_selection.json" \
    "$DEDUP_TARGET_RETAINED_FRACTION" \
    "${DEDUP_CANDIDATE_THRESHOLDS[@]}" <<'PY'
import json
import sys
from pathlib import Path


summary_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
target = float(sys.argv[3])
thresholds = [float(value) for value in sys.argv[4:]]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary_thresholds = summary.get("thresholds")
if not isinstance(summary_thresholds, dict):
    raise SystemExit("redundancy summary has no thresholds mapping")


def get_result(threshold: float) -> dict:
    for key in (str(threshold), f"{threshold:.3f}", f"{threshold:g}"):
        result = summary_thresholds.get(key)
        if isinstance(result, dict):
            return result
    raise SystemExit(f"redundancy summary has no result for threshold {threshold:.3f}")


rows = []
for threshold in thresholds:
    result = get_result(threshold)
    rows.append(
        {
            "threshold": threshold,
            "retained_fraction": float(result["retained_fraction"]),
            "effective_train_images": int(result["effective_train_images"]),
            "original_train_images": int(result["original_train_images"]),
            "layout": result["layout"],
        }
    )

if rows[0]["retained_fraction"] > target:
    selected = rows[0]
    selection_reason = "lower_bound_0.900_already_above_target"
else:
    selected = min(rows, key=lambda row: (abs(row["retained_fraction"] - target), row["threshold"]))
    selection_reason = "closest_retained_fraction"

report = {
    "target_retained_fraction": target,
    "selection_reason": selection_reason,
    "selected": selected,
    "candidates": rows,
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"{selected['threshold']:.3f}")
PY
)"
DEDUP_THRESHOLDS=("$SELECTED_DEDUP_THRESHOLD")
THRESHOLD_CSV="$SELECTED_DEDUP_THRESHOLD"
SELECTED_DEDUP_LABEL="$(threshold_label "$SELECTED_DEDUP_THRESHOLD")"
echo "[Study] selected threshold=$SELECTED_DEDUP_THRESHOLD layout=$REDUNDANCY_ROOT/variants/dedup_tau_${SELECTED_DEDUP_LABEL}/layout.yaml"

if [[ -f "$RANDOM_SPLIT_ROOT/layout.yaml" ]]; then
  echo "[Study] reuse random-frame layout: $RANDOM_SPLIT_ROOT/layout.yaml"
else
  echo "[Study] create random-frame train/val/test reassignment with exact original group budgets"
  run_python data_analyse/custom_dataset/custom_dataset.py random-reassign \
    --layout "$BASE_LAYOUT" \
    --output-dir "$RANDOM_SPLIT_ROOT" \
    --seed "$RANDOM_SPLIT_SEED" \
    2>&1 | tee "$LOG_ROOT/random_frame_layout.log"
fi
run_python data_analyse/custom_dataset/custom_dataset.py validate \
  --layout "$RANDOM_SPLIT_ROOT/layout.yaml" \
  >"$LOG_ROOT/random_frame_validation.log"

SUMMARY_ARGS=()

if [[ "$SKIP_GPU_WAIT" == "1" ]]; then
  echo "[GPU wait] skipped by explicit operator confirmation; using current GPU $GPU_ID state"
else
  wait_for_consecutive_gpu_idle_checks
fi

run_sequence "full_all" "$BASE_LAYOUT"
SUMMARY_ARGS+=(--sequence "full_all=$(sequence_dir full_all)")

for threshold in "${DEDUP_THRESHOLDS[@]}"; do
  label="$(threshold_label "$threshold")"
  experiment_name="dedup_tau_${label}"
  layout="$REDUNDANCY_ROOT/variants/dedup_tau_${label}/layout.yaml"
  run_sequence "$experiment_name" "$layout"
  SUMMARY_ARGS+=(--sequence "$experiment_name=$(sequence_dir "$experiment_name")")
done

if [[ "$MATCHED_RANDOM_CONTROLS" == "1" ]]; then
  for threshold in "${DEDUP_THRESHOLDS[@]}"; do
    label="$(threshold_label "$threshold")"
    for seed in $MATCHED_RANDOM_SEEDS; do
      experiment_name="random_matched_tau_${label}_s${seed}"
      layout="$REDUNDANCY_ROOT/variants/random_matched_tau_${label}_s${seed}/layout.yaml"
      if [[ ! -f "$layout" ]]; then
        echo "matched random layout not found: $layout" >&2
        exit 1
      fi
      run_sequence "$experiment_name" "$layout"
      SUMMARY_ARGS+=(--sequence "$experiment_name=$(sequence_dir "$experiment_name")")
    done
  done
fi

RANDOM_EXPERIMENT="random_frame_s${RANDOM_SPLIT_SEED}"
run_sequence "$RANDOM_EXPERIMENT" "$RANDOM_SPLIT_ROOT/layout.yaml"
SUMMARY_ARGS+=(--sequence "$RANDOM_EXPERIMENT=$(sequence_dir "$RANDOM_EXPERIMENT")")

echo "[Study] summarize all completed sequences"
run_python scripts/summarize_fullcold_data_study.py \
  --output-dir "$REPORT_ROOT" \
  "${SUMMARY_ARGS[@]}" \
  2>&1 | tee "$LOG_ROOT/summary.log"

echo "[Study] all requested experiments completed"
echo "[Study] final comparison: $REPORT_ROOT/comparison_final_stage.csv"
echo "[Study] all-stage comparison: $REPORT_ROOT/comparison_all_stages.csv"
echo "[Study] redundancy summary: $REDUNDANCY_ROOT/summary.json"
