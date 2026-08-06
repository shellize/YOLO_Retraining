from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class SelectionPolicy(ABC):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

