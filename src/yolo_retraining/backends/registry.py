from __future__ import annotations

from typing import Any, Mapping

from .base import DetectionBackend


def create_backend(config: Mapping[str, Any]) -> DetectionBackend:
    name = str(config["model"].get("backend", ""))
    if name == "yolov5":
        from .yolov5 import Yolov5Backend

        return Yolov5Backend()
    if name == "ultralytics":
        from .ultralytics import UltralyticsBackend

        return UltralyticsBackend()
    raise ValueError(f"unknown backend {name!r}; choices=['yolov5', 'ultralytics']")
