from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = STUDY_ROOT.parents[2]
DEFAULT_CONFIG = STUDY_ROOT / "config" / "protocol.yaml"
ALLOWED_STATUSES = {
    "",
    "near_duplicate",
    "same_view_distinct_time",
    "different_scene",
    "unsure",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the reviewed post-dedup pool without modifying source images or labels."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--review-csv", type=Path, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_key(path: Path | str) -> str:
    parts = list(Path(path).parts)
    lowered = [part.casefold() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"image path has no images directory: {path}")
    index = len(lowered) - 1 - lowered[::-1].index("images")
    return Path(*parts[index:]).as_posix().casefold()


def global_frame_number(path: Path | str) -> int:
    token = Path(path).name.split("_", 1)[0]
    if not token.isdecimal():
        raise ValueError(f"image filename does not begin with a numeric global frame id: {path}")
    return int(token)


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [part.casefold() for part in parts]
    index = len(lowered) - 1 - lowered[::-1].index("images")
    return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")


def label_count(image_path: Path) -> tuple[int, bool]:
    path = label_path(image_path)
    if not path.is_file():
        return 0, True
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    review_path = args.review_csv.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    variants_dir = STUDY_ROOT / "experiment" / "variants"
    source_dir = variants_dir / str(config["variant_name"])
    output_dir = variants_dir / str(config["final_variant_name"])
    source_manifest = source_dir / "manifest.txt"
    canonical_pairs = (
        STUDY_ROOT
        / "result"
        / str(config["post_dedup_result_name"])
        / "top_similar_pairs.csv"
    )
    for required in (source_manifest, canonical_pairs, review_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen final dataset: {output_dir}")

    manifest_lines = [
        line.strip()
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_by_key: dict[str, tuple[str, Path]] = {}
    for line in manifest_lines:
        absolute = (source_dir / line).resolve()
        key = image_key(absolute)
        if key in manifest_by_key:
            raise ValueError(f"duplicate image identity in source manifest: {key}")
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        manifest_by_key[key] = (line, absolute)

    review_rows = read_csv(review_path)
    canonical_rows = {row["rank"]: row for row in read_csv(canonical_pairs)}
    required_columns = {
        "rank",
        "left",
        "right",
        "similarity",
        "same_batch",
        "sequence_gap",
        "global_frame_gap",
        "status",
        "note",
    }
    if not review_rows or not required_columns.issubset(review_rows[0]):
        raise ValueError(f"review CSV must contain columns: {sorted(required_columns)}")
    if len(review_rows) != len(canonical_rows):
        raise ValueError(
            f"review row count {len(review_rows)} does not match canonical audit {len(canonical_rows)}"
        )

    parent: dict[str, str] = {}
    evidence_by_key: dict[str, set[int]] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen_ranks: set[str] = set()
    status_counts: Counter[str] = Counter()
    near_duplicate_rows: list[dict[str, str]] = []
    for row in review_rows:
        rank = row["rank"].strip()
        if rank in seen_ranks:
            raise ValueError(f"duplicate review rank: {rank}")
        seen_ranks.add(rank)
        canonical = canonical_rows.get(rank)
        if canonical is None:
            raise ValueError(f"review rank is absent from canonical audit: {rank}")
        for field in ("left", "right"):
            if row[field].casefold() != canonical[field].casefold():
                raise ValueError(f"review row {rank} does not match canonical field {field}")
        status = row["status"].strip()
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported review status at rank {rank}: {status}")
        status_counts[status or "unreviewed"] += 1
        if status != "near_duplicate":
            continue
        left, right = image_key(row["left"]), image_key(row["right"])
        if left not in manifest_by_key or right not in manifest_by_key:
            raise ValueError(f"reviewed pair {rank} is not fully present in the source manifest")
        if left == right:
            raise ValueError(f"reviewed pair {rank} contains the same image twice")
        union(left, right)
        evidence_by_key.setdefault(left, set()).add(int(rank))
        evidence_by_key.setdefault(right, set()).add(int(rank))
        near_duplicate_rows.append(row)

    components_by_root: dict[str, set[str]] = {}
    for key in parent:
        components_by_root.setdefault(find(key), set()).add(key)
    components = sorted(
        components_by_root.values(),
        key=lambda members: min(global_frame_number(key) for key in members),
    )

    label_counts: dict[str, int] = {}
    missing_labels: list[str] = []
    for members in components:
        for key in members:
            count, missing = label_count(manifest_by_key[key][1])
            label_counts[key] = count
            if missing:
                missing_labels.append(key)
    if missing_labels:
        raise FileNotFoundError(f"reviewed images have missing label files: {missing_labels}")

    excluded: set[str] = set()
    decision_rows: list[dict[str, object]] = []
    for component_index, members in enumerate(components, start=1):
        representative = min(
            members,
            key=lambda key: (
                label_counts[key] == 0,
                global_frame_number(key),
                key,
            ),
        )
        evidence_ranks = sorted({rank for key in members for rank in evidence_by_key[key]})
        component_id = f"manual_near_duplicate_{component_index:03d}"
        for key in sorted(members, key=lambda item: (global_frame_number(item), item)):
            action = "keep" if key == representative else "exclude"
            if action == "exclude":
                excluded.add(key)
            decision_rows.append(
                {
                    "component_id": component_id,
                    "image": key,
                    "action": action,
                    "global_frame_number": global_frame_number(key),
                    "label_count": label_counts[key],
                    "kept_representative": representative,
                    "evidence_ranks": "|".join(str(rank) for rank in evidence_ranks),
                }
            )

    final_lines = [line for line in manifest_lines if image_key((source_dir / line).resolve()) not in excluded]
    if len(final_lines) != len(manifest_lines) - len(excluded):
        raise AssertionError("final manifest count does not match the exclusion set")

    per_batch: dict[str, dict[str, int]] = {}
    for line in final_lines:
        key = image_key((source_dir / line).resolve())
        absolute = manifest_by_key[key][1]
        batch = absolute.parent.name
        count, missing = label_count(absolute)
        if missing:
            raise FileNotFoundError(label_path(absolute))
        stats = per_batch.setdefault(
            batch,
            {"images": 0, "positive_images": 0, "background_images": 0},
        )
        stats["images"] += 1
        stats["positive_images" if count else "background_images"] += 1

    variants_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-", dir=variants_dir) as temp:
        temp_dir = Path(temp)
        final_manifest = temp_dir / "manifest.txt"
        final_manifest.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
        shutil.copyfile(review_path, temp_dir / "manual_review.csv")
        with (temp_dir / "review_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
            writer.writeheader()
            writer.writerows(decision_rows)

        selection_rule = (
            "Treat near_duplicate rows as undirected edges; retain one image per connected component, "
            "preferring a non-empty label, then the smaller global frame number, then the image path. "
            "Blank and all other review statuses remain in the dataset."
        )
        summary: dict[str, object] = {
            "status": "completed",
            "source_variant": source_dir.name,
            "source_images": len(manifest_lines),
            "review_rows": len(review_rows),
            "review_status_counts": dict(sorted(status_counts.items())),
            "near_duplicate_edges": len(near_duplicate_rows),
            "near_duplicate_components": len(components),
            "images_in_near_duplicate_components": len(parent),
            "manually_excluded_images": len(excluded),
            "excluded_positive_images": sum(label_counts[key] > 0 for key in excluded),
            "final_images": len(final_lines),
            "selection_rule": selection_rule,
            "per_batch": dict(sorted(per_batch.items())),
            "source_manifest_sha256": sha256(source_manifest),
            "manual_review_sha256": sha256(temp_dir / "manual_review.csv"),
            "final_manifest_sha256": sha256(final_manifest),
            "manifest": final_manifest.name,
            "manual_review": "manual_review.csv",
            "review_decisions": "review_decisions.csv",
        }
        protocol: dict[str, object] = {
            "purpose": "Final reviewed all-data pool before train/val/test construction",
            "source_variant": source_dir.name,
            "source_manifest": str(source_manifest.relative_to(STUDY_ROOT)),
            "canonical_pair_audit": str(canonical_pairs.relative_to(STUDY_ROOT)),
            "review_semantics": {
                "near_duplicate": "collapse by connected component",
                "unreviewed": "keep",
                "same_view_distinct_time": "keep",
                "different_scene": "keep",
                "unsure": "keep",
            },
            "representative_rule": selection_rule,
            "source_images_and_labels_mutated": False,
            "summary": summary,
        }
        write_json(temp_dir / "summary.json", summary)
        write_json(temp_dir / "protocol.json", protocol)
        temp_dir.replace(output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
