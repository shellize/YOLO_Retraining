from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "normalize_batch_names.py"
    spec = importlib.util.spec_from_file_location("normalize_batch_names", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalizer_renames_both_trees_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for kind in ("images", "labels"):
        for index in range(1, 21):
            name = f"0720_{index:02d}" if index < 10 else f"0720_{index}"
            directory = root / kind / name
            directory.mkdir(parents=True)
            (directory / "sample.txt").write_text("x", encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": "/stale", "names": {0: "object"}}), encoding="utf-8")
    module = _load_script()
    module.normalize(root)
    module.normalize(root)
    for kind in ("images", "labels"):
        assert sorted(path.name for path in (root / kind).iterdir()) == sorted(f"0720_{index}" for index in range(1, 21))
    payload = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert payload["path"] == "."
    assert payload["test"] == ["images/0720_1", "images/0720_2"]
    assert payload["val"] == ["images/0720_3", "images/0720_4"]
    assert payload["train"][0] == "images/0720_5"
    assert payload["train"][-1] == "images/0720_20"
