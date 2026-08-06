from __future__ import annotations

from typing import Any, Mapping

from .base import EpochPolicy
from .static import StaticEpochPolicy


POLICIES: dict[str, type[EpochPolicy]] = {"static": StaticEpochPolicy}


def create_epoch_policy(config: Mapping[str, Any]) -> EpochPolicy:
    name = str(config.get("name", ""))
    try:
        policy = POLICIES[name]
    except KeyError as error:
        raise ValueError(f"unknown epoch policy {name!r}; choices={sorted(POLICIES)}") from error
    return policy(config.get("params", {}))

