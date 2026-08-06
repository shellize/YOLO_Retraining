from __future__ import annotations

from typing import Any, Mapping

from .base import SelectionPolicy


class FullSelection(SelectionPolicy):
    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if self.params:
            raise ValueError(f"full selection accepts no params: {sorted(self.params)}")
        selected = sorted(set(context["candidate_ids"]))
        return {"selected_ids": selected, "groups": {"candidate": selected}, "metadata": {"policy": "full"}}

