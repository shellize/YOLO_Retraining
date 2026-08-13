from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def write_image_manifest(registry: Mapping[str, Any], sample_ids: Sequence[str], path: Path) -> Path:
    records = registry["records"]
    missing = set(sample_ids) - set(records)
    if missing:
        raise KeyError(f"manifest references unknown sample ids: {sorted(missing)[:5]}")
    image_paths = [str(Path(records[sample_id]["image_path"]).resolve()) for sample_id in sample_ids]
    if not image_paths:
        raise ValueError(f"cannot write empty image manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image_path}\n" for image_path in image_paths), encoding="utf-8")
    return path


def write_backend_data_yaml(train_manifest: Path, val_manifest: Path, names: Sequence[str], path: Path) -> Path:
    payload = {
        "train": str(train_manifest.resolve()),
        "val": str(val_manifest.resolve()),
        "nc": len(names),
        "names": {index: name for index, name in enumerate(names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
