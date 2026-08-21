#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU_INDEX="${YOLO_RETRAINING_GPU_INDEX:-0}"
POLL_INTERVAL_SECONDS="${YOLO_RETRAINING_GPU_POLL_SECONDS:-600}"
REQUIRED_IDLE_CHECKS="${YOLO_RETRAINING_GPU_REQUIRED_IDLE_CHECKS:-2}"
IDLE_UTILIZATION_PCT="${YOLO_RETRAINING_GPU_IDLE_UTILIZATION_PCT:-5}"
IDLE_MEMORY_MIB="${YOLO_RETRAINING_GPU_IDLE_MEMORY_MIB:-1024}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU wait] nvidia-smi was not found; cannot inspect GPU $GPU_INDEX" >&2
  exit 1
fi

if [[ "$REQUIRED_IDLE_CHECKS" -lt 2 ]]; then
  echo "[GPU wait] REQUIRED_IDLE_CHECKS must be at least 2" >&2
  exit 2
fi

gpu_snapshot() {
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits | tr -d ' ' | head -n 1
}

has_compute_processes() {
  local processes
  processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "$(printf '%s' "$processes" | tr -d '[:space:]')" ]]
}

is_gpu_idle() {
  local snapshot utilization memory_used memory_total
  snapshot="$(gpu_snapshot)"
  IFS=',' read -r utilization memory_used memory_total <<< "$snapshot"

  if [[ -z "${utilization:-}" || -z "${memory_used:-}" ]]; then
    echo "[GPU wait] unable to parse nvidia-smi output: $snapshot" >&2
    return 1
  fi

  printf '%s|util=%s%%|mem=%s/%sMiB' "$(date '+%Y-%m-%d %H:%M:%S')" "$utilization" "$memory_used" "${memory_total:-?}"

  if has_compute_processes; then
    echo "|compute_process=present"
    return 1
  fi

  echo "|compute_process=none"
  (( utilization <= IDLE_UTILIZATION_PCT && memory_used <= IDLE_MEMORY_MIB ))
}

idle_checks=0
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  if is_gpu_idle; then
    ((idle_checks += 1))
    echo "[GPU wait] idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS} passed"
  else
    idle_checks=0
    echo "[GPU wait] GPU $GPU_INDEX is busy; consecutive idle checks reset"
  fi

  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    echo "[GPU wait] next check in ${POLL_INTERVAL_SECONDS}s"
    sleep "$POLL_INTERVAL_SECONDS"
  fi
done

echo "[GPU wait] GPU $GPU_INDEX passed ${REQUIRED_IDLE_CHECKS} consecutive idle checks; starting study"
exec "$PROJECT_ROOT/scripts/run_full_cold_imgsz1280_mosaic_scale_study.sh"
