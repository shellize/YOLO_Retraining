from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def read_image_manifest(path: Path | str, *, dataset_root: Path | str | None = None) -> list[Path]:
    """Read a UTF-8 image manifest with paths relative to the manifest file."""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file() or manifest.suffix.lower() != ".txt":
        raise FileNotFoundError(f"image manifest must be an existing .txt file: {manifest}")
    root = None if dataset_root is None else Path(dataset_root).expanduser().resolve()
    images: list[Path] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value).expanduser()
        image = (manifest.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"manifest image does not exist at {manifest}:{line_number}: {image}")
        if root is not None:
            try:
                image.relative_to(root)
            except ValueError as error:
                raise ValueError(f"manifest image must be below dataset root {root}: {image}") from error
        key = str(image).casefold()
        if key in seen:
            raise ValueError(f"duplicate image in manifest {manifest}:{line_number}: {image}")
        seen.add(key)
        images.append(image)
    if not images:
        raise ValueError(f"image manifest contains no images: {manifest}")
    return images


def write_image_manifest(images: Iterable[Path | str], path: Path | str, *, relative: bool = True) -> Path:
    """Write a deterministic UTF-8 manifest without copying any source image."""

    manifest = Path(path).expanduser().resolve()
    resolved = sorted({Path(image).expanduser().resolve() for image in images}, key=lambda item: item.as_posix().casefold())
    if not resolved:
        raise ValueError(f"cannot write an empty image manifest: {manifest}")
    missing = [image for image in resolved if not image.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot write manifest with missing image: {missing[0]}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for image in resolved:
        value = os.path.relpath(image, manifest.parent) if relative else str(image)
        values.append(Path(value).as_posix())
    manifest.write_text("\n".join(values) + "\n", encoding="utf-8")
    return manifest


def find_manifest_overlaps(manifests: dict[str, Iterable[Path | str]]) -> dict[str, list[str]]:
    """Return physical images owned by more than one named manifest."""

    owners: dict[str, list[str]] = {}
    display: dict[str, str] = {}
    for name, images in manifests.items():
        for raw_image in images:
            image = Path(raw_image).expanduser().resolve()
            key = str(image).casefold()
            display[key] = str(image)
            owners.setdefault(key, []).append(str(name))
    return {display[key]: sorted(set(names)) for key, names in owners.items() if len(set(names)) > 1}
