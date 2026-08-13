#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${YOLO_RETRAINING_ENV_NAME:-yolo-retraining-v5}"
TORCH_INDEX="${YOLO_RETRAINING_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
ALLOW_UPDATE="${YOLO_RETRAINING_ALLOW_ENV_UPDATE:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_COMMIT="915bbf294bb74c859f0b41f1c23bc395014ea679"
YOLO_ROOT="${YOLOV5_ROOT:-$PROJECT_ROOT/.third_party/yolov5}"
WHEEL_ROOT="$PROJECT_ROOT/.third_party/wheels"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$ALLOW_UPDATE" != "1" ]]; then
    echo "Environment '$ENV_NAME' exists. Set YOLO_RETRAINING_ALLOW_ENV_UPDATE=1 to verify/update project packages." >&2
    exit 1
  fi
else
  conda env create -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
fi

if [[ ! -e "$YOLO_ROOT" ]]; then
  mkdir -p "$(dirname "$YOLO_ROOT")"
  git clone https://github.com/ultralytics/yolov5.git "$YOLO_ROOT"
  git -C "$YOLO_ROOT" checkout --detach "$YOLO_COMMIT"
fi

ACTUAL_COMMIT="$(git -C "$YOLO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$YOLO_COMMIT" ]] || { echo "YOLOv5 revision mismatch at $YOLO_ROOT" >&2; exit 1; }
[[ -z "$(git -C "$YOLO_ROOT" status --porcelain --untracked-files=no)" ]] || { echo "YOLOv5 source has tracked modifications" >&2; exit 1; }

if [[ ! -s "$YOLO_ROOT/yolov5s.pt" ]]; then
  curl -fL "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt" -o "$YOLO_ROOT/yolov5s.pt"
fi

mkdir -p "$WHEEL_ROOT"
TORCH_WHEEL="$WHEEL_ROOT/torch-2.2.2+cu121-cp310-cp310-linux_x86_64.whl"
TORCHVISION_WHEEL="$WHEEL_ROOT/torchvision-0.17.2+cu121-cp310-cp310-linux_x86_64.whl"
TORCH_URL="${YOLO_RETRAINING_TORCH_WHEEL_URL:-$TORCH_INDEX/torch-2.2.2%2Bcu121-cp310-cp310-linux_x86_64.whl}"
TORCHVISION_URL="${YOLO_RETRAINING_TORCHVISION_WHEEL_URL:-$TORCH_INDEX/torchvision-0.17.2%2Bcu121-cp310-cp310-linux_x86_64.whl}"
if ! conda run -n "$ENV_NAME" python -m zipfile -t "$TORCH_WHEEL" >/dev/null 2>&1; then
  curl --fail --location --retry 20 --retry-all-errors --continue-at - --output "$TORCH_WHEEL" "$TORCH_URL"
  conda run -n "$ENV_NAME" python -m zipfile -t "$TORCH_WHEEL" >/dev/null
fi
if ! conda run -n "$ENV_NAME" python -m zipfile -t "$TORCHVISION_WHEEL" >/dev/null 2>&1; then
  curl --fail --location --retry 20 --retry-all-errors --continue-at - --output "$TORCHVISION_WHEEL" "$TORCHVISION_URL"
  conda run -n "$ENV_NAME" python -m zipfile -t "$TORCHVISION_WHEEL" >/dev/null
fi
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "$TORCH_WHEEL" "$TORCHVISION_WHEEL"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements/runtime.lock"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements/dev.lock"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$PROJECT_ROOT" --no-deps --no-build-isolation
YOLOV5_ROOT="$YOLO_ROOT" conda run --no-capture-output -n "$ENV_NAME" python -m yolo_retraining.doctor --project-root "$PROJECT_ROOT"
