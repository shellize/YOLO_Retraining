from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SelectionPolicy


class RandomReplaySelection(SelectionPolicy):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        unknown = set(self.params) - {"replay_size"}
        if unknown:
            raise ValueError(f"unknown random_replay params: {sorted(unknown)}")
        self.replay_size = int(self.params.get("replay_size", 0))
        if self.replay_size <= 0:
            raise ValueError("random_replay requires a positive replay_size")

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        current = sorted(set(context["current_ids"]))
        history = sorted(set(context["candidate_ids"]) - set(current))
        count = min(self.replay_size, len(history))
        if count:
            rng = np.random.default_rng(int(context["seed"]))
            replay = sorted(str(item) for item in rng.choice(history, size=count, replace=False).tolist())
        else:
            replay = []
        selected = sorted(set(current) | set(replay))
        return {
            "selected_ids": selected,
            "groups": {"current": current, "replay": replay},
            "metadata": {"policy": "random_replay", "seed": int(context["seed"]), "requested_replay_size": self.replay_size, "actual_replay_size": len(replay)},
        }

