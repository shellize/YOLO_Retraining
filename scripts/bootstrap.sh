#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${YOLO_RETRAINING_ENV_NAME:-yolo-cl}"
TORCH_INDEX="${YOLO_RETRAINING_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
ALLOW_UPDATE="${YOLO_RETRAINING_ALLOW_ENV_UPDATE:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$ALLOW_UPDATE" != "1" ]]; then
    echo "Environment '$ENV_NAME' exists. Use use_existing_env.sh or set YOLO_RETRAINING_ALLOW_ENV_UPDATE=1." >&2
    exit 1
  fi
else
  conda env create -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install --index-url "$TORCH_INDEX" torch==2.6.0 torchvision==0.21.0
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements/runtime.lock"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements/dev.lock"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$PROJECT_ROOT" --no-deps --no-build-isolation
conda run --no-capture-output -n "$ENV_NAME" python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT"
