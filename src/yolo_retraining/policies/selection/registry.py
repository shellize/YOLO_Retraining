from __future__ import annotations

from typing import Any, Mapping

from .base import SelectionPolicy
from .full import FullSelection
from .random_replay import RandomReplaySelection
from .study import ErrorHardSelection, GradNormTopKSelection, PositiveOnlySelection, PredPositiveSelection, RandomTopKSelection


POLICIES: dict[str, type[SelectionPolicy]] = {
    "full": FullSelection,
    "random_replay": RandomReplaySelection,
    "positive_only": PositiveOnlySelection,
    "pred_positive": PredPositiveSelection,
    "error_hard": ErrorHardSelection,
    "gradnorm_topk": GradNormTopKSelection,
    "random_topk": RandomTopKSelection,
}


def create_selection_policy(config: Mapping[str, Any]) -> SelectionPolicy:
    name = str(config.get("name", ""))
    try:
        policy = POLICIES[name]
    except KeyError as error:
        raise ValueError(f"unknown selection policy {name!r}; choices={sorted(POLICIES)}") from error
    return policy(config.get("params", {}))
