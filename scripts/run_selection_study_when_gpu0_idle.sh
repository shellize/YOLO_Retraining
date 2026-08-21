#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU_INDEX=0
POLL_SECONDS="${YOLO_RETRAINING_GPU_POLL_SECONDS:-60}"
REQUIRED_IDLE_SECONDS="${YOLO_RETRAINING_GPU_IDLE_DURATION_SECONDS:-600}"
IDLE_UTILIZATION_PCT="${YOLO_RETRAINING_GPU_IDLE_UTILIZATION_PCT:-5}"
IDLE_MEMORY_MIB="${YOLO_RETRAINING_GPU_IDLE_MEMORY_MIB:-1024}"
WAIT_ONLY="${YOLO_RETRAINING_GPU_WAIT_ONLY:-0}"
LOCK_PATH="$PROJECT_ROOT/runs/.gpu0_selection_study.lock"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU wait] nvidia-smi was not found; cannot inspect physical GPU0" >&2
  exit 20
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[GPU wait] flock was not found; refusing to launch without a duplicate-run guard" >&2
  exit 21
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ && "$REQUIRED_IDLE_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "[GPU wait] poll and idle duration must be integer seconds" >&2
  exit 22
fi
if [[ ! -f "$PROJECT_ROOT/scripts/run_selection_study.sh" ]]; then
  echo "[GPU wait] launch script is missing" >&2
  exit 23
fi

mkdir -p "$PROJECT_ROOT/runs"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "[GPU wait] another GPU0 selection-study waiter/runner already holds $LOCK_PATH" >&2
  exit 24
fi

gpu_is_idle() {
  local snapshot utilization memory_used memory_total processes
  snapshot="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')" || return 1
  IFS=',' read -r utilization memory_used memory_total <<< "$snapshot"
  if ! [[ "${utilization:-}" =~ ^[0-9]+$ && "${memory_used:-}" =~ ^[0-9]+$ ]]; then
    echo "[GPU wait] unable to parse GPU0 snapshot: $snapshot" >&2
    return 1
  fi
  processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  processes="$(printf '%s' "$processes" | tr -d '[:space:]')"
  printf '[GPU wait] %s GPU0 util=%s%% mem=%s/%sMiB compute_process=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$utilization" "$memory_used" "${memory_total:-?}" "$([[ -n "$processes" ]] && echo present || echo none)"
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
      echo "[GPU wait] GPU0 remained idle for ${elapsed}s; launch condition satisfied"
      break
    fi
    echo "[GPU wait] continuous idle ${elapsed}/${REQUIRED_IDLE_SECONDS}s"
  else
    if [[ -n "$idle_since" ]]; then
      echo "[GPU wait] GPU0 became busy; idle timer reset"
    else
      echo "[GPU wait] GPU0 is busy"
    fi
    idle_since=""
  fi
  sleep "$POLL_SECONDS"
done

if [[ "$WAIT_ONLY" == "1" ]]; then
  echo "[GPU wait] wait-only verification completed; training was not launched"
  exit 0
fi

echo "[GPU wait] launching selection study on physical GPU0"
exec bash "$PROJECT_ROOT/scripts/run_selection_study.sh"
