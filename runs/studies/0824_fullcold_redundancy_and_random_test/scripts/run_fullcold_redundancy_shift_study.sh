#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v bash >/dev/null 2>&1; then
  echo "bash is required by the full-cold study runner" >&2
  exit 1
fi

exec bash "$SCRIPT_DIR/run_fullcold_redundancy_shift_study.bash" "$@"
