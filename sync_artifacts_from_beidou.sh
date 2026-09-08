#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$PROJECT_ROOT/.artifact_sync"
CONFIG_FILE="${YOLO_ARTIFACT_SYNC_CONFIG:-$STATE_DIR/config.env}"
LOG_FILE="$STATE_DIR/sync.log"
DRY_RUN=0

usage() {
  echo "Usage: $0 [--dry-run]"
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$STATE_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "artifact sync config is missing: $CONFIG_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${YOLO_ARTIFACT_REMOTE_HOST:?set YOLO_ARTIFACT_REMOTE_HOST in $CONFIG_FILE}"
: "${YOLO_ARTIFACT_REMOTE_ROOT:?set YOLO_ARTIFACT_REMOTE_ROOT in $CONFIG_FILE}"
SSH_COMMAND="${YOLO_ARTIFACT_SSH_COMMAND:-ssh}"

if [[ ! -x "$SSH_COMMAND" ]] && ! command -v "$SSH_COMMAND" >/dev/null 2>&1; then
  echo "SSH command is not executable: $SSH_COMMAND" >&2
  exit 2
fi

exec 9>"$STATE_DIR/sync.lock"
if ! flock -n 9; then
  exit 0
fi

if [[ -f "$LOG_FILE" ]] && (( $(stat -c %s "$LOG_FILE") > 10485760 )); then
  mv -f "$LOG_FILE" "$LOG_FILE.1"
fi
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

tracked_artifact_count="$({
  git -C "$PROJECT_ROOT" ls-files -- \
    runs/tasks runs/sequences runs/studies reports charts data_analyse result_analysis
} | grep -Ev '/\.gitkeep$' \
  | grep -Ec '^(runs/(tasks|sequences)/|runs/studies/[^/]+/(experiment/(sequence|task)/|logs/|result/)|reports/|charts/.*\.(png|svg|csv|html)|charts/.*/summary\.json|charts/.*/fixed_confidence_cache/|charts/.*/(analysis|comparison_summary|complete_report|fixed_confidence_tables|metrics_tables)\.md|data_analyse/.*/results/|result_analysis/.*/results/|result_analysis/[^/]+/20[0-9]{6}[^/]*/)' || true)"

if (( tracked_artifact_count > 0 && DRY_RUN == 0 )); then
  log "refusing to mirror over $tracked_artifact_count artifact files that are still tracked by Git"
  log "finish the planned Git index migration first; use --dry-run for a read-only preview"
  exit 3
fi

SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=20)
if [[ -n "${YOLO_ARTIFACT_SSH_IDENTITY:-}" ]]; then
  if [[ ! -r "$YOLO_ARTIFACT_SSH_IDENTITY" ]]; then
    echo "SSH identity is not readable: $YOLO_ARTIFACT_SSH_IDENTITY" >&2
    exit 2
  fi
  SSH_ARGS+=(-i "$YOLO_ARTIFACT_SSH_IDENTITY" -o IdentitiesOnly=yes)
fi
if [[ -n "${YOLO_ARTIFACT_SSH_KNOWN_HOSTS:-}" ]]; then
  if [[ ! -r "$YOLO_ARTIFACT_SSH_KNOWN_HOSTS" ]]; then
    echo "SSH known-hosts file is not readable: $YOLO_ARTIFACT_SSH_KNOWN_HOSTS" >&2
    exit 2
  fi
  SSH_ARGS+=(-o "UserKnownHostsFile=$YOLO_ARTIFACT_SSH_KNOWN_HOSTS")
fi

printf -v RSYNC_RSH '%q ' "$SSH_COMMAND" "${SSH_ARGS[@]}"
RSYNC_RSH="${RSYNC_RSH% }"
RSYNC_ARGS=(
  -rt
  --partial
  --human-readable
  --stats
  --protect-args
  --exclude=.git/
  --exclude=.gitkeep
  --exclude=.rsync-partial/
  -e "$RSYNC_RSH"
)
if (( DRY_RUN == 1 )); then
  RSYNC_ARGS+=(--dry-run)
fi

remote_path_exists() {
  local remote_path="$1"
  local quoted_path
  printf -v quoted_path '%q' "$remote_path"
  "$SSH_COMMAND" "${SSH_ARGS[@]}" "$YOLO_ARTIFACT_REMOTE_HOST" "test -d $quoted_path"
}

sync_full_tree() {
  local relative_path="$1"
  local remote_path="$YOLO_ARTIFACT_REMOTE_ROOT/$relative_path"
  local local_path="$PROJECT_ROOT/$relative_path"
  local check_status=0

  remote_path_exists "$remote_path" || check_status=$?
  if (( check_status == 1 )); then
    log "skip missing remote directory: $relative_path"
    return 0
  elif (( check_status != 0 )); then
    log "remote directory check failed with exit code $check_status: $relative_path"
    return "$check_status"
  fi

  mkdir -p "$local_path"
  log "mirror $relative_path"
  rsync "${RSYNC_ARGS[@]}" \
    "$YOLO_ARTIFACT_REMOTE_HOST:$remote_path/" \
    "$local_path/"
}

sync_study_artifacts() {
  local relative_path="runs/studies"
  local remote_path="$YOLO_ARTIFACT_REMOTE_ROOT/$relative_path"
  local local_path="$PROJECT_ROOT/$relative_path"
  local check_status=0

  remote_path_exists "$remote_path" || check_status=$?
  if (( check_status == 1 )); then
    log "skip missing remote directory: $relative_path"
    return 0
  elif (( check_status != 0 )); then
    log "remote directory check failed with exit code $check_status: $relative_path"
    return "$check_status"
  fi

  mkdir -p "$local_path"
  log "mirror standard Study artifact directories"
  rsync "${RSYNC_ARGS[@]}" --prune-empty-dirs \
    --include='*/' \
    --include='*/experiment/sequence/***' \
    --include='*/experiment/task/***' \
    --include='*/logs/***' \
    --include='*/result/***' \
    --exclude='*' \
    "$YOLO_ARTIFACT_REMOTE_HOST:$remote_path/" \
    "$local_path/"
}

sync_analysis_results() {
  local relative_path="$1"
  local remote_path="$YOLO_ARTIFACT_REMOTE_ROOT/$relative_path"
  local local_path="$PROJECT_ROOT/$relative_path"
  local check_status=0

  remote_path_exists "$remote_path" || check_status=$?
  if (( check_status == 1 )); then
    log "skip missing remote directory: $relative_path"
    return 0
  elif (( check_status != 0 )); then
    log "remote directory check failed with exit code $check_status: $relative_path"
    return "$check_status"
  fi

  mkdir -p "$local_path"
  log "mirror $relative_path result directories"
  rsync "${RSYNC_ARGS[@]}" --prune-empty-dirs \
    --include='*/' \
    --include='**/results/***' \
    --exclude='*' \
    "$YOLO_ARTIFACT_REMOTE_HOST:$remote_path/" \
    "$local_path/"
}

sync_chart_artifacts() {
  local relative_path="charts"
  local remote_path="$YOLO_ARTIFACT_REMOTE_ROOT/$relative_path"
  local local_path="$PROJECT_ROOT/$relative_path"
  local check_status=0

  remote_path_exists "$remote_path" || check_status=$?
  if (( check_status == 1 )); then
    log "skip missing remote directory: $relative_path"
    return 0
  elif (( check_status != 0 )); then
    log "remote directory check failed with exit code $check_status: $relative_path"
    return "$check_status"
  fi

  mkdir -p "$local_path"
  log "mirror generated chart artifacts"
  rsync "${RSYNC_ARGS[@]}" --prune-empty-dirs \
    --include='*/' \
    --include='**/*.png' \
    --include='**/*.svg' \
    --include='**/*.csv' \
    --include='**/*.html' \
    --include='**/summary.json' \
    --include='**/fixed_confidence_cache/***' \
    --include='**/analysis.md' \
    --include='**/comparison_summary.md' \
    --include='**/complete_report.md' \
    --include='**/fixed_confidence_tables.md' \
    --include='**/metrics_tables.md' \
    --exclude='*' \
    "$YOLO_ARTIFACT_REMOTE_HOST:$remote_path/" \
    "$local_path/"
}

mode_suffix=""
if (( DRY_RUN == 1 )); then
  mode_suffix=" (dry-run)"
fi
log "artifact mirror started$mode_suffix"
sync_full_tree "runs/tasks"
sync_full_tree "runs/sequences"
sync_study_artifacts
sync_analysis_results "data_analyse"
sync_analysis_results "result_analysis"
sync_chart_artifacts
sync_full_tree "reports"
log "artifact mirror completed"
