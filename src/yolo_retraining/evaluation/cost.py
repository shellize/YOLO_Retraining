from __future__ import annotations

from typing import Any, Mapping


def build_cost(selection_seconds: float, training: Mapping[str, Any], evaluation_seconds: float, device_count: int) -> dict[str, Any]:
    training_seconds = float(training.get("training_seconds", 0.0))
    method_seconds = float(selection_seconds) + training_seconds
    return {
        "selection_seconds": float(selection_seconds),
        "training_seconds": training_seconds,
        "evaluation_seconds": float(evaluation_seconds),
        "method_seconds": method_seconds,
        "gpu_hours": method_seconds * max(0, int(device_count)) / 3600.0,
        "images_read": int(training.get("images_read", 0)),
        "optimizer_steps": int(training.get("optimizer_steps", 0)),
    }

