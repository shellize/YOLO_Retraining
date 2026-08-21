from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class DetectionBackend(ABC):
    capabilities: frozenset[str] = frozenset()
    output_namespace = "backend"

    def provenance(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return {"name": self.output_namespace}

    @abstractmethod
    def validate_config(self, config: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not implement predict")

    def analyze_samples(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} does not implement sample analysis")

    def gradient_scores(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} does not implement gradient scoring")


def require_artifact(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not produced: {path}")
    return path
