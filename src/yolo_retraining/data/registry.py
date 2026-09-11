from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .loaders import load_group


def catalog_from_data_config(data_config: Mapping[str, Any]) -> dict[str, str]:
    catalog = data_config.get("catalog")
    if isinstance(catalog, Mapping):
        return {str(key): str(value) for key, value in catalog.items()}
    layout = data_config.get("layout")
    if not isinstance(layout, str) or not layout:
        raise ValueError("data requires either catalog or layout")
    group_ids: set[str] = set()
    for scope in ("current", "candidate", "validation", "test"):
        group_ids.update(str(value) for value in data_config.get(scope, []))
    return {group_id: layout for group_id in sorted(group_ids)}


def build_registry(catalog: Mapping[str, str]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, list[str]]] = {}
    names: list[str] | None = None
    path_owners: dict[str, list[tuple[str, str]]] = {}
    sources: dict[str, str] = {}
    subset_of: dict[str, str] = {}
    for group_id, yaml_path in catalog.items():
        group = load_group(str(group_id), yaml_path)
        sources[str(group_id)] = group["yaml_path"]
        if group.get("subset_of") is not None:
            subset_of[str(group_id)] = str(group["subset_of"])
        if names is None:
            names = list(group["names"])
        elif names != group["names"]:
            raise ValueError(f"label schema mismatch for group {group_id!r}: {group['names']} != {names}")
        groups[str(group_id)] = {}
        for split, split_records in group["splits"].items():
            ids: list[str] = []
            for record in split_records:
                sample_id = record["sample_id"]
                if sample_id in records:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                normalized_path = str(Path(record["image_path"]).resolve()).casefold()
                path_owners.setdefault(normalized_path, []).append((str(group_id), split))
                records[sample_id] = record
                ids.append(sample_id)
            groups[str(group_id)][split] = ids
    if names is None:
        raise ValueError("catalog produced no dataset groups")
    for child, parent in subset_of.items():
        if parent not in groups:
            raise ValueError(f"test subset {child!r} references unknown parent group {parent!r}")
        child_paths = {
            str(Path(records[sample_id]["image_path"]).resolve()).casefold()
            for sample_id in groups[child]["test"]
        }
        parent_paths = {
            str(Path(records[sample_id]["image_path"]).resolve()).casefold()
            for sample_id in groups[parent]["test"]
        }
        if not child_paths.issubset(parent_paths):
            raise ValueError(f"test subset {child!r} contains images outside parent group {parent!r}")
    for image_path, owners in path_owners.items():
        for index, left in enumerate(owners):
            for right in owners[index + 1 :]:
                related_test_subset = (
                    left[1] == right[1] == "test"
                    and (subset_of.get(left[0]) == right[0] or subset_of.get(right[0]) == left[0])
                )
                if not related_test_subset:
                    raise ValueError(f"image appears in multiple dataset splits: {image_path} ({left} and {right})")
    return {
        "records": records,
        "groups": groups,
        "names": names,
        "sources": sources,
        "subset_of": subset_of,
    }


def records_for(registry: Mapping[str, Any], sample_ids: list[str] | set[str]) -> list[dict[str, Any]]:
    records = registry["records"]
    missing = set(sample_ids) - set(records)
    if missing:
        raise KeyError(f"unknown sample ids: {sorted(missing)[:5]}")
    return [records[sample_id] for sample_id in sample_ids]
