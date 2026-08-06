from __future__ import annotations

from typing import Any, Mapping


def _collect(registry: Mapping[str, Any], group_ids: list[str], split: str) -> set[str]:
    result: set[str] = set()
    for group_id in group_ids:
        try:
            result.update(registry["groups"][group_id][split])
        except KeyError as error:
            raise ValueError(f"unknown data group or split: {group_id}.{split}") from error
    return result


def resolve_scope(registry: Mapping[str, Any], data_config: Mapping[str, Any]) -> dict[str, Any]:
    current_ids = _collect(registry, list(data_config["current"]), "train")
    candidate_ids = _collect(registry, list(data_config["candidate"]), "train")
    validation_ids = _collect(registry, list(data_config["validation"]), "val")
    test_by_group = {group_id: sorted(_collect(registry, [group_id], "test")) for group_id in data_config["test"]}
    test_ids = set().union(*(set(ids) for ids in test_by_group.values())) if test_by_group else set()
    if not current_ids.issubset(candidate_ids):
        missing = sorted(current_ids - candidate_ids)
        raise ValueError(f"current samples are not all eligible candidates: {missing[:5]}")
    train_ids = candidate_ids
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("train, validation, and test sample ids must be disjoint")
    return {
        "current_ids": sorted(current_ids),
        "candidate_ids": sorted(candidate_ids),
        "history_ids": sorted(candidate_ids - current_ids),
        "validation_ids": sorted(validation_ids),
        "test_by_group": test_by_group,
    }

