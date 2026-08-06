from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .loaders import load_group


def build_registry(catalog: Mapping[str, str]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, list[str]]] = {}
    names: list[str] | None = None
    path_owners: dict[str, tuple[str, str]] = {}
    sources: dict[str, str] = {}
    for group_id, yaml_path in catalog.items():
        group = load_group(str(group_id), yaml_path)
        sources[str(group_id)] = group["yaml_path"]
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
                owner = path_owners.get(normalized_path)
                if owner is not None:
                    raise ValueError(f"image appears in multiple dataset splits: {record['image_path']} ({owner} and {(group_id, split)})")
                path_owners[normalized_path] = (str(group_id), split)
                records[sample_id] = record
                ids.append(sample_id)
            groups[str(group_id)][split] = ids
    if names is None:
        raise ValueError("catalog produced no dataset groups")
    return {"records": records, "groups": groups, "names": names, "sources": sources}


def records_for(registry: Mapping[str, Any], sample_ids: list[str] | set[str]) -> list[dict[str, Any]]:
    records = registry["records"]
    missing = set(sample_ids) - set(records)
    if missing:
        raise KeyError(f"unknown sample ids: {sorted(missing)[:5]}")
    return [records[sample_id] for sample_id in sample_ids]

