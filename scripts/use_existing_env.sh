#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${YOLO_RETRAINING_ENV_NAME:-yolo-cl}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' does not exist. Use scripts/bootstrap.sh explicitly." >&2
  exit 1
fi

conda run --no-capture-output -n "$ENV_NAME" python -c '
import sys, torch, torchvision, ultralytics
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split("+")[0] == "2.6.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.21.0", torchvision.__version__
assert ultralytics.__version__ == "8.4.102", ultralytics.__version__
'
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$PROJECT_ROOT" --no-deps --no-build-isolation
conda run --no-capture-output -n "$ENV_NAME" python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT"
