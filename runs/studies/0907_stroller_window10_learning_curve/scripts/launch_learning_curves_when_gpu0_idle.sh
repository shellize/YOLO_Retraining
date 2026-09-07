#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
GPU_ID="${GPU_ID:-0}"
POLL_SECONDS="${POLL_SECONDS:-600}"
GPU_IDLE_UTIL_MAX="${GPU_IDLE_UTIL_MAX:-5}"
GPU_IDLE_MEMORY_MAX_MB="${GPU_IDLE_MEMORY_MAX_MB:-512}"

GLOBAL_RUNNER="$PROJECT_ROOT/runs/studies/0903_global_dedup_random_test_learning_curve/scripts/run_randomization_controls_server.sh"
STROLLER_RUNNER="$SCRIPT_DIR/run_learning_curve.sh"
LOG_ROOT="$STUDY_ROOT/logs"
POLL_LOG="$LOG_ROOT/gpu0_learning_curves_when_idle.log"
LOCK_PATH="$LOG_ROOT/gpu0_learning_curves.lock"

if [[ "${1:-}" != "--start-training" || "${2:-}" != "" ]]; then
  echo "usage: $0 --start-training" >&2
  echo "This guard is required because this script eventually starts both training workflows." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found" >&2
  exit 10
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock was not found; refusing to start an unguarded duplicate poller" >&2
  exit 11
fi
if [[ ! -x "$GLOBAL_RUNNER" && ! -f "$GLOBAL_RUNNER" ]]; then
  echo "global dedup runner is missing: $GLOBAL_RUNNER" >&2
  exit 12
fi
if [[ ! -x "$STROLLER_RUNNER" && ! -f "$STROLLER_RUNNER" ]]; then
  echo "stroller runner is missing: $STROLLER_RUNNER" >&2
  exit 13
fi

mkdir -p "$LOG_ROOT"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "another GPU0 learning-curve poller already holds $LOCK_PATH" >&2
  exit 14
fi

log_line() {
  local message="$1"
  local line="[$(date '+%Y-%m-%d %H:%M:%S%z')] $message"
  echo "$line" | tee -a "$POLL_LOG"
}

gpu0_idle() {
  local gpu_line gpu_util memory_used compute_pids compute_count
  gpu_line="$(nvidia-smi --id="$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
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

  compute_pids="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  compute_count="$(awk '$1 ~ /^[0-9]+$/ {count += 1} END {print count + 0}' <<<"$compute_pids")"
  if (( compute_count == 0 && gpu_util <= GPU_IDLE_UTIL_MAX && memory_used <= GPU_IDLE_MEMORY_MAX_MB )); then
    log_line "GPU${GPU_ID}: idle util=${gpu_util}% memory=${memory_used}MiB compute_apps=${compute_count}"
    return 0
  fi

  log_line "GPU${GPU_ID}: busy util=${gpu_util}% memory=${memory_used}MiB compute_apps=${compute_count}"
  return 1
}

run_child() {
  local label="$1"
  local script="$2"
  local log_path="$3"
  local status

  log_line "starting $label: $script"
  set +e
  bash "$script" --start-training 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    log_line "$label failed with exit code $status; the next workflow will not start"
    return "$status"
  fi
  log_line "$label completed"
}

log_line "poller started; GPU${GPU_ID}, interval=${POLL_SECONDS}s, idle thresholds=${GPU_IDLE_UTIL_MAX}%/${GPU_IDLE_MEMORY_MAX_MB}MiB"
while true; do
  consecutive_idle=0
  while (( consecutive_idle < 2 )); do
    if gpu0_idle; then
      consecutive_idle=$((consecutive_idle + 1))
      log_line "consecutive idle checks=${consecutive_idle}/2"
    else
      consecutive_idle=0
      log_line "consecutive idle checks reset to 0"
    fi
    if (( consecutive_idle < 2 )); then
      log_line "sleeping ${POLL_SECONDS}s before the next GPU${GPU_ID} check"
      sleep "$POLL_SECONDS"
    fi
  done

  if gpu0_idle; then
    break
  fi
  log_line "GPU${GPU_ID} became busy during the final pre-launch check; restarting the two-check window"
done

run_child \
  "0903 global dedup randomization controls" \
  "$GLOBAL_RUNNER" \
  "$LOG_ROOT/launch_global_dedup_randomization.log"

run_child \
  "0907 stroller window10 learning curves" \
  "$STROLLER_RUNNER" \
  "$LOG_ROOT/launch_stroller_learning_curve.log"

log_line "both learning-curve workflows completed"
