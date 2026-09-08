#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STUDY_ROOT/../../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_INDEX=0
POLL_SECONDS=600
REQUIRED_IDLE_CHECKS=2
IDLE_UTILIZATION_PCT="${YOLO_RETRAINING_GPU_IDLE_UTILIZATION_PCT:-5}"
IDLE_MEMORY_MIB="${YOLO_RETRAINING_GPU_IDLE_MEMORY_MIB:-1024}"
LOCK_PATH="$PROJECT_ROOT/runs/.gpu0_balance_lc2b_dedup_contrast.lock"
WAIT_LOG="$STUDY_ROOT/logs/gpu0_wait.log"
LAUNCH_SCRIPT="$SCRIPT_DIR/run_experiments.sh"

mkdir -p "$PROJECT_ROOT/runs" "$STUDY_ROOT/logs"
exec > >(tee -a "$WAIT_LOG") 2>&1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU0 wait] nvidia-smi was not found" >&2
  exit 20
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "[GPU0 wait] flock was not found; refusing to run without a duplicate guard" >&2
  exit 21
fi
if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
  echo "[GPU0 wait] launch script is missing: $LAUNCH_SCRIPT" >&2
  exit 22
fi
if ! [[ "$IDLE_UTILIZATION_PCT" =~ ^[0-9]+$ && "$IDLE_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "[GPU0 wait] idle utilization and memory limits must be non-negative integers" >&2
  exit 23
fi

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "[GPU0 wait] another waiter/runner already holds $LOCK_PATH" >&2
  exit 24
fi

gpu0_is_idle() {
  local snapshot utilization memory_used memory_total processes process_state
  if ! snapshot="$(
    nvidia-smi -i "$GPU_INDEX" \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
  )"; then
    echo "[GPU0 wait] $(date '+%Y-%m-%d %H:%M:%S') unable to query GPU0" >&2
    return 1
  fi
  IFS=',' read -r utilization memory_used memory_total <<< "$snapshot"
  if ! [[ "${utilization:-}" =~ ^[0-9]+$ && "${memory_used:-}" =~ ^[0-9]+$ && "${memory_total:-}" =~ ^[0-9]+$ ]]; then
    echo "[GPU0 wait] $(date '+%Y-%m-%d %H:%M:%S') unable to parse snapshot: $snapshot" >&2
    return 1
  fi

  processes="$(
    nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true
  )"
  processes="$(printf '%s' "$processes" | tr -d '[:space:]')"
  if [[ -n "$processes" ]]; then process_state="present"; else process_state="none"; fi
  printf '[GPU0 wait] %s util=%s%% mem=%s/%sMiB compute_process=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$utilization" "$memory_used" "$memory_total" "$process_state"

  [[ -z "$processes" ]] \
    && (( utilization <= IDLE_UTILIZATION_PCT )) \
    && (( memory_used <= IDLE_MEMORY_MIB ))
}

echo "[GPU0 wait] polling physical GPU0 every 600s"
echo "[GPU0 wait] launch requires two consecutive idle checks"
echo "[GPU0 wait] idle limits: util<=${IDLE_UTILIZATION_PCT}% mem<=${IDLE_MEMORY_MIB}MiB and no compute process"

idle_checks=0
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  if gpu0_is_idle; then
    ((idle_checks += 1))
    echo "[GPU0 wait] idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS} passed"
  else
    if (( idle_checks > 0 )); then
      echo "[GPU0 wait] GPU0 became busy; consecutive count reset"
    else
      echo "[GPU0 wait] GPU0 is busy"
    fi
    idle_checks=0
  fi
  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    echo "[GPU0 wait] next check in 600s"
    sleep "$POLL_SECONDS"
  fi
done

echo "[GPU0 wait] GPU0 passed two consecutive ten-minute checks; launching both contrast sequences"
export GPU_ID=0
exec bash "$LAUNCH_SCRIPT"
