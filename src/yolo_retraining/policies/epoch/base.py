from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class EpochPolicy(ABC):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def build_plan(self, selected_ids: list[str], *, epoch: int, seed: int) -> list[str]:
        raise NotImplementedError

