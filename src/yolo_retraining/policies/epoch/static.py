from __future__ import annotations

from .base import EpochPolicy


class StaticEpochPolicy(EpochPolicy):
    def __init__(self, params=None) -> None:
        super().__init__(params)
        if self.params:
            raise ValueError(f"static epoch policy accepts no params: {sorted(self.params)}")

    def build_plan(self, selected_ids: list[str], *, epoch: int, seed: int) -> list[str]:
        return list(selected_ids)

