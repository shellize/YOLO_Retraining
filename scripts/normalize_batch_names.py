from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def normalize(root: Path) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    operations: list[tuple[Path, Path]] = []
    for kind in ("images", "labels"):
        parent = root / kind
        if not parent.is_dir():
            raise FileNotFoundError(f"dataset is missing {kind} directory: {parent}")
        for index in range(1, 10):
            source = parent / f"0720_{index:02d}"
            target = parent / f"0720_{index}"
            if source.exists() and target.exists():
                raise FileExistsError(f"both old and normalized batch directories exist: {source}, {target}")
            if source.is_dir():
                operations.append((source, target))
            elif not target.is_dir():
                raise FileNotFoundError(f"neither old nor normalized batch directory exists: {source}, {target}")

    for source, target in operations:
        source.rename(target)
        print(f"renamed: {source.name} -> {target.name} ({source.parent.name})")

    data_yaml = root / "data.yaml"
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) if data_yaml.is_file() else {}
    payload = payload if isinstance(payload, dict) else {}
    names = payload.get("names")
    if names is None:
        raise ValueError(f"cannot preserve class names because data.yaml has no names: {data_yaml}")
    canonical = {
        "path": ".",
        "train": [f"images/0720_{index}" for index in range(5, 21)],
        "val": ["images/0720_3", "images/0720_4"],
        "test": ["images/0720_1", "images/0720_2"],
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(canonical, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"updated relative dataset YAML: {data_yaml}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize 0720_01..09 batch names and rewrite data.yaml relatively.")
    parser.add_argument("root", type=Path, help="dataset root containing images/, labels/, and data.yaml")
    args = parser.parse_args()
    normalize(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

