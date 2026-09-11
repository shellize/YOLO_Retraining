from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = "global_order_window20_tau0p990_label_aware_reviewed_final"
OUTPUT_NAME = "global_order_window20_tau0p990_label_aware_pair_reviewed_final"
AUDITS = (
    "positive_box_layout_pair_audit",
    "positive_box_layout_nonconsecutive_pair_audit",
)
ALLOWED = {"", "duplicate", "not_duplicate", "unsure"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two positive-pair reviews into the final manifest.")
    parser.add_argument("--source-dir", type=Path, default=STUDY_ROOT / "experiment" / "variants" / SOURCE_NAME)
    parser.add_argument("--output-dir", type=Path, default=STUDY_ROOT / "experiment" / "variants" / OUTPUT_NAME)
    parser.add_argument("--review", type=Path, action="append")
    return parser.parse_args()


def image_key(value: str | Path) -> str:
    normalized = str(value).replace("\\", "/")
    marker = "images/"
    index = normalized.casefold().rfind(marker)
    if index < 0:
        raise ValueError(f"path has no images directory: {value}")
    return normalized[index:].casefold()


def frame(value: str | Path) -> int:
    token = Path(value).name.split("_", 1)[0]
    if not token.isdecimal():
        raise ValueError(f"image has no numeric frame prefix: {value}")
    return int(token)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    source_dir, output_dir = args.source_dir.resolve(), args.output_dir.resolve()
    reviews = [path.resolve() for path in args.review] if args.review else [
        STUDY_ROOT / "result" / audit / "manual_review.csv" for audit in AUDITS
    ]
    canonicals = [STUDY_ROOT / "result" / audit / "candidate_pairs.csv" for audit in AUDITS]
    if len(reviews) != len(canonicals):
        raise ValueError(f"expected {len(canonicals)} review CSV files")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final variant: {output_dir}")

    manifest_lines = [line.strip() for line in (source_dir / "manifest.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    source_by_key = {image_key((source_dir / line).resolve()): line for line in manifest_lines}
    if len(source_by_key) != len(manifest_lines):
        raise ValueError("duplicate identities in source manifest")

    parent: dict[str, str] = {}
    evidence: dict[str, set[str]] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    review_summaries = []
    for audit, review_path, canonical_path in zip(AUDITS, reviews, canonicals):
        review_rows, canonical_rows = read_csv(review_path), read_csv(canonical_path)
        canonical = {row["rank"]: row for row in canonical_rows}
        if len(review_rows) != len(canonical_rows):
            raise ValueError(f"{audit}: review/canonical row count mismatch")
        counts: Counter[str] = Counter()
        for row in review_rows:
            rank, status = row["rank"].strip(), row["status"].strip()
            if status not in ALLOWED:
                raise ValueError(f"{audit} rank {rank}: unsupported status {status!r}")
            expected = canonical.get(rank)
            if expected is None or any(row[field].casefold() != expected[field].casefold() for field in ("left", "right")):
                raise ValueError(f"{audit} rank {rank}: review identity mismatch")
            counts[status or "unreviewed"] += 1
            if status != "duplicate":
                continue
            left, right = image_key(row["left"]), image_key(row["right"])
            if left not in source_by_key or right not in source_by_key:
                raise ValueError(f"{audit} rank {rank}: image absent from source manifest")
            union(left, right)
            evidence.setdefault(left, set()).add(f"{audit}:{rank}")
            evidence.setdefault(right, set()).add(f"{audit}:{rank}")
        review_summaries.append({"audit": audit, "review_rows": len(review_rows), "status_counts": dict(sorted(counts.items()))})

    grouped: dict[str, set[str]] = {}
    for key in parent:
        grouped.setdefault(find(key), set()).add(key)
    components = sorted(grouped.values(), key=lambda group: min(frame(key) for key in group))
    excluded: set[str] = set()
    decisions = []
    for number, members in enumerate(components, start=1):
        representative = min(members, key=lambda key: (frame(key), key))
        all_evidence = sorted({item for key in members for item in evidence.get(key, set())})
        component_id = f"positive_duplicate_{number:04d}"
        for key in sorted(members, key=lambda item: (frame(item), item)):
            action = "keep" if key == representative else "exclude"
            if action == "exclude":
                excluded.add(key)
            decisions.append({"component_id": component_id, "image": key, "action": action, "representative": representative, "evidence": "|".join(all_evidence)})

    final_lines = [line for line in manifest_lines if image_key((source_dir / line).resolve()) not in excluded]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent) as temp:
        target = Path(temp)
        (target / "manifest.txt").write_text("\n".join(final_lines) + "\n", encoding="utf-8", newline="\n")
        for audit, review in zip(AUDITS, reviews):
            shutil.copyfile(review, target / f"{audit}_manual_review.csv")
        with (target / "review_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("component_id", "image", "action", "representative", "evidence"), lineterminator="\n")
            writer.writeheader(); writer.writerows(decisions)
        summary = {
            "status": "completed", "source_variant": source_dir.name, "source_images": len(manifest_lines),
            "reviews": review_summaries, "duplicate_edges": sum(item["status_counts"].get("duplicate", 0) for item in review_summaries),
            "duplicate_components": len(components), "images_in_components": len(parent),
            "excluded_images": len(excluded), "final_images": len(final_lines),
            "representative_rule": "one connected component per duplicate-edge graph; retain the smallest global frame number",
            "source_images_and_labels_mutated": False,
        }
        (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "protocol.json").write_text(json.dumps({"purpose": "Final pool after both positive-pair reviews", **summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
