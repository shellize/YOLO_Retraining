from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .manifests import read_image_manifest


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def _normalize_names(value: Any) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    if isinstance(value, Mapping):
        pairs = sorted((int(key), str(name)) for key, name in value.items())
        if [key for key, _ in pairs] != list(range(len(pairs))):
            raise ValueError("dataset names mapping must use contiguous class ids starting at zero")
        return [name for _, name in pairs]
    raise ValueError("dataset names must be a list or integer-keyed mapping")


def load_dataset_yaml(path: Path | str) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"dataset YAML root must be a mapping: {yaml_path}")
    for key in ("train", "val", "test", "names"):
        if key not in payload:
            raise ValueError(f"dataset YAML requires {key!r}: {yaml_path}")
    declared_root = Path(payload.get("path", yaml_path.parent)).expanduser()
    root = (yaml_path.parent / declared_root).resolve() if not declared_root.is_absolute() else declared_root.resolve()
    return {"yaml_path": str(yaml_path), "root": str(root), "names": _normalize_names(payload["names"]), "splits": {key: payload[key] for key in ("train", "val", "test")}}


def _load_layout_group(group_id: str, yaml_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    groups = payload.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"dataset layout requires a non-empty groups mapping: {yaml_path}")
    group_spec = groups.get(group_id)
    if not isinstance(group_spec, Mapping):
        raise ValueError(f"dataset layout has no logical group {group_id!r}: {yaml_path}")
    split = group_spec.get("split")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"layout group {group_id!r} split must be train, val, or test")
    images = group_spec.get("images")
    manifest = group_spec.get("manifest")
    if images is not None and manifest is not None:
        raise ValueError(f"layout group {group_id!r} must use exactly one of images or manifest")
    entry = manifest if manifest is not None else images
    if not isinstance(entry, (str, list)) or not entry:
        raise ValueError(f"layout group {group_id!r} requires a non-empty images or manifest entry")
    if manifest is not None:
        manifest_entries = manifest if isinstance(manifest, list) else [manifest]
        if not all(isinstance(value, str) and Path(value).suffix.lower() == ".txt" for value in manifest_entries):
            raise ValueError(f"layout group {group_id!r} manifest entries must be .txt paths")
    declared_root = Path(payload.get("path", yaml_path.parent)).expanduser()
    root = (yaml_path.parent / declared_root).resolve() if not declared_root.is_absolute() else declared_root.resolve()
    return {
        "yaml_path": str(yaml_path),
        "root": str(root),
        "names": _normalize_names(payload.get("names")),
        "splits": {name: entry if name == split else None for name in ("train", "val", "test")},
    }


def _images_from_entry(root: Path, entry: str | list[str]) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    files: list[Path] = []
    for raw in entries:
        path = Path(raw).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if path.is_dir():
            files.extend(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            files.extend(read_image_manifest(path, dataset_root=root))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
        else:
            raise FileNotFoundError(f"YOLO split entry does not exist or contains no supported images: {path}")
    invalid = [item for item in files if item.suffix.lower() not in IMAGE_SUFFIXES]
    if invalid:
        raise ValueError(f"manifest contains an unsupported image suffix: {invalid[0]}")
    normalized = [str(item).casefold() for item in files]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"YOLO split entry contains duplicate images: {entry!r}")
    unique = sorted(files, key=lambda item: item.as_posix())
    if not unique:
        raise ValueError(f"YOLO split contains no images: {entry!r}")
    return unique


def yolo_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"YOLO image path must contain an 'images' directory: {image_path}")
    index = len(lowered) - 1 - lowered[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_group(group_id: str, yaml_path: Path | str) -> dict[str, Any]:
    yaml_path = Path(yaml_path).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"dataset YAML root must be a mapping: {yaml_path}")
    spec = _load_layout_group(group_id, yaml_path, payload) if "groups" in payload else load_dataset_yaml(yaml_path)
    root = Path(spec["root"])
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split, entry in spec["splits"].items():
        records: list[dict[str, Any]] = []
        image_paths = [] if entry is None else _images_from_entry(root, entry)
        for image_path in image_paths:
            try:
                relative = image_path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(f"image must be located below dataset root {root}: {image_path}") from error
            sample_id = f"{group_id}::{relative}"
            records.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "label_path": str(yolo_label_path(image_path)),
                    "group": group_id,
                    "split": split,
                }
            )
        split_records[split] = records
    return {"group_id": group_id, "yaml_path": spec["yaml_path"], "root": spec["root"], "names": spec["names"], "splits": split_records}
