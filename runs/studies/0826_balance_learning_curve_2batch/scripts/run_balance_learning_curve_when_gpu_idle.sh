#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DEFAULT_STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_INDEX="${GPU_ID:-0}"
POLL_SECONDS="${YOLO_RETRAINING_GPU_POLL_SECONDS:-60}"
REQUIRED_IDLE_SECONDS="${YOLO_RETRAINING_GPU_IDLE_DURATION_SECONDS:-600}"
IDLE_UTILIZATION_PCT="${YOLO_RETRAINING_GPU_IDLE_UTILIZATION_PCT:-5}"
IDLE_MEMORY_MIB="${YOLO_RETRAINING_GPU_IDLE_MEMORY_MIB:-1024}"
STUDY_ROOT="${BALANCE_LEARNING_CURVE_ROOT:-$DEFAULT_STUDY_ROOT}"
LOCK_PATH="$PROJECT_ROOT/runs/.gpu${GPU_INDEX}_balance_learning_curve.lock"
WAIT_LOG="$STUDY_ROOT/logs/gpu_wait.log"

mkdir -p "$PROJECT_ROOT/runs" "$STUDY_ROOT/logs"
exec > >(tee -a "$WAIT_LOG") 2>&1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU wait] nvidia-smi was not found; cannot inspect physical GPU $GPU_INDEX" >&2
  exit 20
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[GPU wait] flock was not found; refusing to launch without a duplicate-run guard" >&2
  exit 21
fi
if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[1-9][0-9]*$ && "$REQUIRED_IDLE_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "[GPU wait] GPU index, poll interval, and idle duration must be non-negative integer values" >&2
  exit 22
fi

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "[GPU wait] another learning-curve waiter/runner already holds $LOCK_PATH" >&2
  exit 23
fi

gpu_is_idle() {
  local snapshot utilization memory_used memory_total processes
  snapshot="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')" || return 1
  IFS=',' read -r utilization memory_used memory_total <<< "$snapshot"
  if ! [[ "${utilization:-}" =~ ^[0-9]+$ && "${memory_used:-}" =~ ^[0-9]+$ ]]; then
    echo "[GPU wait] unable to parse GPU $GPU_INDEX snapshot: $snapshot" >&2
    return 1
  fi
  processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  processes="$(printf '%s' "$processes" | tr -d '[:space:]')"
  printf '[GPU wait] %s GPU%s util=%s%% mem=%s/%sMiB compute_process=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$GPU_INDEX" "$utilization" "$memory_used" "${memory_total:-?}" \
    "$([[ -n "$processes" ]] && echo present || echo none)"
  [[ -z "$processes" ]] && (( utilization <= IDLE_UTILIZATION_PCT )) && (( memory_used <= IDLE_MEMORY_MIB ))
}

idle_since=""
while true; do
  now="$(date +%s)"
  if gpu_is_idle; then
    if [[ -z "$idle_since" ]]; then
      idle_since="$now"
      echo "[GPU wait] idle window started; requiring ${REQUIRED_IDLE_SECONDS}s continuously idle"
    fi
    elapsed=$((now - idle_since))
    if (( elapsed >= REQUIRED_IDLE_SECONDS )); then
      echo "[GPU wait] GPU $GPU_INDEX remained idle for ${elapsed}s; launch condition satisfied"
      break
    fi
    echo "[GPU wait] continuous idle ${elapsed}/${REQUIRED_IDLE_SECONDS}s"
  else
    if [[ -n "$idle_since" ]]; then
      echo "[GPU wait] GPU $GPU_INDEX became busy; idle timer reset"
    else
      echo "[GPU wait] GPU $GPU_INDEX is busy"
    fi
    idle_since=""
  fi
  sleep "$POLL_SECONDS"
done

echo "[GPU wait] launching Balance Test learning curves on physical GPU $GPU_INDEX"
export GPU_ID="$GPU_INDEX"
exec bash "$SCRIPT_DIR/run_balance_learning_curve.sh"
