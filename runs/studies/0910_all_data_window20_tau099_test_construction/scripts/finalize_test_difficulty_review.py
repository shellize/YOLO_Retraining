from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = STUDY_ROOT / "experiment" / "variants" / "group_stratified_s42_8_1_1"
DEFAULT_REVIEW = STUDY_ROOT / "result" / "final_test_positive_difficulty_review" / "manual_review.csv"
ALLOWED_STATUSES = {"", "difficult", "normal", "unsure"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the reviewed difficult subset and nested filtered test manifest.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def image_id(manifest_entry: str) -> str:
    normalized = manifest_entry.strip().replace("\\", "/")
    marker = "/data/self_improving/"
    if marker not in normalized:
        raise ValueError(f"manifest entry is outside data/self_improving: {manifest_entry}")
    return normalized.split(marker, 1)[1]


def read_review(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"rank", "image", "global_frame", "batch", "classes", "label_count", "status", "note"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"unexpected review CSV columns: {list(rows[0]) if rows else []}")
    identities = [row["image"].strip().replace("\\", "/") for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("review CSV contains duplicate images")
    invalid = sorted({row["status"].strip() for row in rows} - ALLOWED_STATUSES)
    if invalid:
        raise ValueError(f"unsupported review statuses: {invalid}")
    difficult = {
        row["image"].strip().replace("\\", "/")
        for row in rows
        if row["status"].strip() == "difficult"
    }
    return rows, difficult


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.expanduser().resolve()
    review_path = args.review_csv.expanduser().resolve()
    manifests_dir = split_dir / "manifests"
    test_path = manifests_dir / "test.txt"
    filtered_path = manifests_dir / "test_filtered.txt"
    difficult_path = manifests_dir / "test_difficult.txt"
    stored_review = split_dir / "test_difficulty_manual_review.csv"
    summary_path = split_dir / "test_difficulty_summary.json"
    for output in (filtered_path, difficult_path, stored_review, summary_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen output: {output}")

    entries = [line.strip() for line in test_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id = {image_id(entry): entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("test manifest contains duplicate images")
    review_rows, difficult = read_review(review_path)
    review_ids = {row["image"].strip().replace("\\", "/") for row in review_rows}
    unknown = sorted(review_ids - set(by_id))
    if unknown:
        raise ValueError(f"review CSV contains images outside test: {unknown[:5]}")
    if len(review_rows) != 296:
        raise ValueError(f"expected 296 reviewed positive images, found {len(review_rows)}")

    filtered_entries = [entry for entry in entries if image_id(entry) not in difficult]
    difficult_entries = [entry for entry in entries if image_id(entry) in difficult]
    filtered_path.write_text("\n".join(filtered_entries) + "\n", encoding="utf-8", newline="\n")
    difficult_path.write_text("\n".join(difficult_entries) + "\n", encoding="utf-8", newline="\n")
    shutil.copyfile(review_path, stored_review)

    layout_path = split_dir / "layout.yaml"
    layout = yaml.safe_load(layout_path.read_text(encoding="utf-8"))
    groups = layout["groups"]
    if "test_filtered" in groups:
        raise ValueError("layout already contains test_filtered")
    test_manifest = str(groups["test"]["manifest"])
    if not test_manifest.endswith("/test.txt"):
        raise ValueError(f"unexpected test manifest in layout: {test_manifest}")
    groups["test_filtered"] = {
        "split": "test",
        "subset_of": "test",
        "manifest": test_manifest[: -len("test.txt")] + "test_filtered.txt",
    }
    groups["test_difficult"] = {
        "split": "test",
        "subset_of": "test",
        "manifest": test_manifest[: -len("test.txt")] + "test_difficult.txt",
    }
    layout_path.write_text(
        yaml.safe_dump(layout, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n"
    )

    summary = {
        "status": "completed",
        "source_test_images": len(entries),
        "reviewed_positive_images": len(review_rows),
        "difficult_positive_images": len(difficult_entries),
        "filtered_test_images": len(filtered_entries),
        "filtered_test_positive_images": len(review_rows) - len(difficult_entries),
        "filtered_test_background_images": len(entries) - len(review_rows),
        "semantics": {
            "test": "complete frozen test",
            "test_filtered": "nested test excluding only images explicitly marked difficult",
            "test_difficult": "excluded difficult positive subset for audit",
        },
        "manifests": {
            "test": "manifests/test.txt",
            "test_filtered": "manifests/test_filtered.txt",
            "test_difficult": "manifests/test_difficult.txt",
        },
        "manual_review": stored_review.name,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
