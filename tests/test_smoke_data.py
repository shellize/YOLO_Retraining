from pathlib import Path

import yaml
from PIL import Image

from smoke_data import prepare_real_smoke_dataset


def test_real_smoke_sampler_is_deterministic_and_does_not_copy_images(tmp_path: Path) -> None:
    root = tmp_path / "source"
    for index in range(120):
        batch = f"batch_{index % 3:02d}"
        image = root / "images" / batch / f"{index:03d}.jpg"
        label = root / "labels" / batch / f"{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16)).save(image)
        classes = {index % 4, (index + 1) % 4} if index < 40 else set()
        label.write_text("".join(f"{class_id} 0.5 0.5 0.4 0.4\n" for class_id in sorted(classes)), encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {index: f"c{index}" for index in range(4)}}), encoding="utf-8")
    first = prepare_real_smoke_dataset(root, tmp_path / "one")
    second = prepare_real_smoke_dataset(root, tmp_path / "two")
    for split, size in (("train", 64), ("val", 16), ("test", 16)):
        one = (tmp_path / "one" / f"{split}.txt").read_text(encoding="utf-8")
        two = (tmp_path / "two" / f"{split}.txt").read_text(encoding="utf-8")
        assert one == two
        assert len(one.splitlines()) == size
    all_paths = [sample.image for split in ("train", "val", "test") for sample in first[split]]
    assert len(all_paths) == len(set(all_paths)) == 96
    assert not list((tmp_path / "one").rglob("*.jpg"))
