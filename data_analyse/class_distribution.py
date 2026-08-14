"""Count image and object-class distribution for selected dataset batches.

Examples:
    python -m data_analyse.class_distribution --batches "1,2,5-8"
    python -m data_analyse.class_distribution --batches "1,2,5-8" --format json
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from .batch_args import add_batch_argument, batch_names
except ImportError:  # Supports direct execution from this directory.
    from batch_args import add_batch_argument, batch_names


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# Images and labels batch folders use date-prefixed names without leading
# zeros (0720_1, 0720_5, 0720_20), while the annotation archive keeps the
# canonical batch_XX names (batch_01, batch_05, batch_20).
DATASET_FOLDER_PREFIX = "0720"


def dataset_folder(batch: str) -> str:
    """Return the images/labels folder name for a canonical batch name.

    ``batch_05`` -> ``0720_5``, ``batch_10`` -> ``0720_10``
    """

    return f"{DATASET_FOLDER_PREFIX}_{int(batch.rsplit('_', 1)[-1])}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count images, labeled images, objects, and classes by batch."
    )
    add_batch_argument(parser)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"project data directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    return parser.parse_args()


def find_annotation_dir(data_root: Path, batch: str) -> Path:
    """Find archived XML first, with compatibility for the pre-archive layout."""

    candidates = (
        data_root / "self_improving" / "annotation_archive" / batch / "xml",
        data_root / "self_improving" / "labels" / dataset_folder(batch) / "xml",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"cannot find XML annotations for {batch}; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def analyze_batch(data_root: Path, batch: str) -> dict[str, object]:
    organized = data_root / "self_improving"
    image_dir = organized / "images" / dataset_folder(batch)
    xml_dir = find_annotation_dir(data_root, batch)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"cannot find images for {batch}: {image_dir}")

    images = sorted(image_dir.glob("*.jpg"))
    xml_files = sorted(xml_dir.glob("*.xml"))
    classes: Counter[str] = Counter()
    object_count = 0
    parse_errors: list[str] = []
    images_with_annotations = 0

    for xml_file in xml_files:
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError as exc:
            parse_errors.append(f"{xml_file}: {exc}")
            continue
        objects = root.findall("object")
        if objects:
            images_with_annotations += 1
        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            classes[label or "<missing>"] += 1
            object_count += 1

    return {
        "batch": batch,
        "images": len(images),
        "annotation_files": len(xml_files),
        "images_with_annotations": images_with_annotations,
        "images_without_annotations": len(images) - images_with_annotations,
        "objects": object_count,
        "classes": dict(sorted(classes.items())),
        "xml_parse_errors": parse_errors,
    }


def format_text(results: list[dict[str, object]]) -> str:
    lines = [
        "batch  images  labeled  empty  objects  classes",
        "-----  ------  -------  -----  -------  -------",
    ]
    total_images = total_labeled = total_empty = total_objects = 0
    total_classes: Counter[str] = Counter()
    errors: list[str] = []
    for result in results:
        classes = result["classes"]
        class_text = ", ".join(f"{name}={count}" for name, count in classes.items())
        lines.append(
            f"{result['batch']:5}  {result['images']:6}  "
            f"{result['images_with_annotations']:7}  "
            f"{result['images_without_annotations']:5}  "
            f"{result['objects']:7}  {class_text or '-'}"
        )
        total_images += result["images"]
        total_labeled += result["images_with_annotations"]
        total_empty += result["images_without_annotations"]
        total_objects += result["objects"]
        total_classes.update(classes)
        errors.extend(result["xml_parse_errors"])
    lines.append("-----  ------  -------  -----  -------  -------")
    lines.append(
        f"total  {total_images:6}  {total_labeled:7}  {total_empty:5}  "
        f"{total_objects:7}  "
        + ", ".join(f"{name}={count}" for name, count in sorted(total_classes.items()))
    )
    if errors:
        lines.append("\nXML parse errors:")
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    results = [analyze_batch(data_root, batch) for batch in batch_names(args.batches)]
    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_text(results))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

