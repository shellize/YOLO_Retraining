#!/usr/bin/env bash
set -Eeuo pipefail

# Cross-study server poller. It waits for one physical GPU, prepares the new
# stroller manifests, then runs each requested workflow serially.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"

GPU_ID="${GPU_ID:-0}"
POLL_SECONDS="${POLL_SECONDS:-600}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"
GPU_IDLE_UTIL_MAX="${GPU_IDLE_UTIL_MAX:-5}"
GPU_IDLE_MEMORY_MAX_MB="${GPU_IDLE_MEMORY_MAX_MB:-512}"

DEDUP_RUNNER="$PROJECT_ROOT/data_analyse/stroller_protocol_0908/run_dedup.sh"
DIFFICULT_RUNNER="$PROJECT_ROOT/runs/studies/0908_difficult_stroller_window10_learning_curve/scripts/run_learning_curve.sh"
NESTED_RUNNER="$PROJECT_ROOT/runs/studies/0908_stroller_difficult_easy_nested_effect/scripts/run_nested_effect.sh"
GLOBAL_CONTROLS_RUNNER="$PROJECT_ROOT/runs/studies/0903_global_dedup_random_test_learning_curve/scripts/run_randomization_controls_server.sh"
STROLLER_CONTROLS_RUNNER="$SCRIPT_DIR/run_learning_curve.sh"

RAW_SOURCE="$PROJECT_ROOT/data/stroller_raw"
EASY_SOURCE="$PROJECT_ROOT/data/stroller"
LOG_ROOT="$STUDY_ROOT/logs"
POLL_LOG="$LOG_ROOT/gpu${GPU_ID}_stroller_studies_when_idle.log"
LOCK_PATH="$LOG_ROOT/gpu${GPU_ID}_stroller_studies.lock"

if [[ "${1:-}" != "--start-training" || "${2:-}" != "" ]]; then
  echo "usage: $0 --start-training" >&2
  echo "This guard is required because this poller eventually starts training." >&2
  exit 2
fi
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative physical GPU index: $GPU_ID" >&2
  exit 3
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer: $POLL_SECONDS" >&2
  exit 4
fi
if ! [[ "$REQUIRED_IDLE_CHECKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REQUIRED_IDLE_CHECKS must be a positive integer: $REQUIRED_IDLE_CHECKS" >&2
  exit 5
fi
if ! [[ "$GPU_IDLE_UTIL_MAX" =~ ^[0-9]+$ && "$GPU_IDLE_MEMORY_MAX_MB" =~ ^[0-9]+$ ]]; then
  echo "GPU idle thresholds must be non-negative integers" >&2
  exit 6
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found" >&2
  exit 10
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock was not found; refusing to start an unguarded duplicate poller" >&2
  exit 11
fi
for runner in \
  "$DEDUP_RUNNER" \
  "$DIFFICULT_RUNNER" \
  "$NESTED_RUNNER" \
  "$GLOBAL_CONTROLS_RUNNER" \
  "$STROLLER_CONTROLS_RUNNER"; do
  if [[ ! -f "$runner" ]]; then
    echo "runner is missing: $runner" >&2
    exit 12
  fi
done
for source_dir in "$RAW_SOURCE" "$EASY_SOURCE"; do
  if [[ ! -d "$source_dir/images" || ! -f "$source_dir/classes.txt" ]]; then
    echo "dataset source is incomplete: $source_dir" >&2
    exit 13
  fi
done
if [[ ! -d "$PROJECT_ROOT/.third_party/yolov5" ]]; then
  echo "YOLOv5 source directory is missing: $PROJECT_ROOT/.third_party/yolov5" >&2
  exit 14
fi

mkdir -p "$LOG_ROOT"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "another stroller-study poller already holds $LOCK_PATH" >&2
  exit 15
fi

export GPU_ID

log_line() {
  local message="$1"
  local line="[$(date '+%Y-%m-%d %H:%M:%S%z')] $message"
  echo "$line" | tee -a "$POLL_LOG"
}

gpu_is_idle() {
  local gpu_line gpu_util memory_used compute_pids compute_count
  gpu_line="$(nvidia-smi --id="$GPU_ID" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -z "$gpu_line" ]]; then
    log_line "GPU${GPU_ID}: unable to read utilization/memory"
    return 1
  fi

  gpu_util="$(awk -F',' '{gsub(/[[:space:]]/, "", $1); print $1}' <<<"$gpu_line")"
  memory_used="$(awk -F',' '{gsub(/[[:space:]]/, "", $2); print $2}' <<<"$gpu_line")"
  if [[ ! "$gpu_util" =~ ^[0-9]+$ || ! "$memory_used" =~ ^[0-9]+$ ]]; then
    log_line "GPU${GPU_ID}: unparsable nvidia-smi row: $gpu_line"
    return 1
  fi

  compute_pids="$(nvidia-smi --id="$GPU_ID" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  compute_count="$(awk '$1 ~ /^[0-9]+$/ {count += 1} END {print count + 0}' <<<"$compute_pids")"
  if (( compute_count == 0 && gpu_util <= GPU_IDLE_UTIL_MAX && memory_used <= GPU_IDLE_MEMORY_MAX_MB )); then
    log_line "GPU${GPU_ID}: idle util=${gpu_util}% memory=${memory_used}MiB compute_apps=${compute_count}"
    return 0
  fi

  log_line "GPU${GPU_ID}: busy util=${gpu_util}% memory=${memory_used}MiB compute_apps=${compute_count}"
  return 1
}

wait_for_gpu_idle() {
  while true; do
    local consecutive_idle=0
    while (( consecutive_idle < REQUIRED_IDLE_CHECKS )); do
      if gpu_is_idle; then
        consecutive_idle=$((consecutive_idle + 1))
        log_line "consecutive idle checks=${consecutive_idle}/${REQUIRED_IDLE_CHECKS}"
      else
        consecutive_idle=0
        log_line "consecutive idle checks reset to 0"
      fi
      if (( consecutive_idle < REQUIRED_IDLE_CHECKS )); then
        log_line "sleeping ${POLL_SECONDS}s before the next GPU${GPU_ID} check"
        sleep "$POLL_SECONDS"
      fi
    done

    if gpu_is_idle; then
      log_line "GPU${GPU_ID} passed the final pre-launch check"
      return 0
    fi
    log_line "GPU${GPU_ID} became busy during the final pre-launch check; restarting the idle window"
  done
}

run_child() {
  local label="$1"
  local script="$2"
  local log_path="$3"
  local status
  shift 3

  log_line "starting $label: $script $*"
  set +e
  bash "$script" "$@" 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    log_line "$label failed with exit code $status; later workflows will not start"
    return "$status"
  fi
  log_line "$label completed"
}

log_line "poller started; GPU${GPU_ID}, interval=${POLL_SECONDS}s, required_idle_checks=${REQUIRED_IDLE_CHECKS}, idle thresholds=${GPU_IDLE_UTIL_MAX}%/${GPU_IDLE_MEMORY_MAX_MB}MiB"
wait_for_gpu_idle

run_child \
  "0908 stroller raw/easy tau0.99 window10 dedup manifest preparation" \
  "$DEDUP_RUNNER" \
  "$LOG_ROOT/launch_0908_stroller_dedup.log"

run_child \
  "0908 difficult stroller window10 learning curve" \
  "$DIFFICULT_RUNNER" \
  "$LOG_ROOT/launch_0908_difficult_learning_curve.log" \
  --start-training

run_child \
  "0908 stroller difficult/easy nested effect" \
  "$NESTED_RUNNER" \
  "$LOG_ROOT/launch_0908_nested_effect.log" \
  --start-training

run_child \
  "0903 global dedup randomization and ordered controls" \
  "$GLOBAL_CONTROLS_RUNNER" \
  "$LOG_ROOT/launch_0903_randomization_controls.log"

run_child \
  "0907 stroller window10 random and ordered controls" \
  "$STROLLER_CONTROLS_RUNNER" \
  "$LOG_ROOT/launch_0907_stroller_controls.log" \
  --start-training

log_line "all requested stroller workflows and ordered controls completed"
