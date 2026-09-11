from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = STUDY_ROOT / "experiment" / "variants"
DEFAULT_INPUTS = (
    VARIANTS / "target_track_groups_reviewed_final",
    VARIANTS / "target_track_second_pass_groups_reviewed_final",
)
DEFAULT_OUTPUT = VARIANTS / "target_track_groups_two_pass_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine reviewed target events from both screening passes.")
    parser.add_argument("--input-dir", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def frame(image: str) -> int:
    return int(Path(image).name.split("_", 1)[0])


def main() -> int:
    args = parse_args()
    inputs = tuple(path.resolve() for path in args.input_dir) if args.input_dir else DEFAULT_INPUTS
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite combined groups: {output}")

    events: list[dict[str, object]] = []
    seen_event_names: set[str] = set()
    for pass_number, source in enumerate(inputs, start=1):
        for row in read_csv(source / "target_event_members.csv"):
            original = f"pass{pass_number}:{row['target_event_id']}"
            seen_event_names.add(original)
            events.append({"source_pass": pass_number, "original_event_id": original, **row})

    grouped_events: dict[str, list[dict[str, object]]] = {}
    for row in events:
        grouped_events.setdefault(str(row["original_event_id"]), []).append(row)
    ordered_events = sorted(grouped_events.items(), key=lambda item: min(int(row["sequence_index"]) for row in item[1]))
    event_rows = []
    for number, (original, members) in enumerate(ordered_events, start=1):
        event_id = f"target_event_{number:04d}"
        for row in sorted(members, key=lambda item: int(item["sequence_index"])):
            event_rows.append({"target_event_id": event_id, "source_pass": row["source_pass"], "source_track": row["source_track"], "kind": row["kind"], "image": row["image"], "box_index": row["box_index"], "sequence_index": row["sequence_index"]})

    parent: dict[str, str] = {}
    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]
    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    by_event: dict[str, list[str]] = {}
    for row in event_rows:
        by_event.setdefault(row["target_event_id"], []).append(row["image"])
    for members in by_event.values():
        for image in members[1:]:
            union(members[0], image)
    components: dict[str, set[str]] = {}
    for image in parent:
        components.setdefault(find(image), set()).add(image)
    ordered_components = sorted(components.values(), key=lambda members: min(frame(image) for image in members))
    image_rows = []
    for number, members in enumerate(ordered_components, start=1):
        group_id = f"split_group_{number:04d}"
        for image in sorted(members, key=lambda value: (frame(value), value)):
            event_ids = sorted({row["target_event_id"] for row in event_rows if row["image"] == image})
            image_rows.append({"split_group_id": group_id, "image": image, "target_event_ids": "|".join(event_ids)})

    output.mkdir(parents=True)
    def write(name: str, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    write("target_event_members.csv", ("target_event_id", "source_pass", "source_track", "kind", "image", "box_index", "sequence_index"), event_rows)
    write("image_split_groups.csv", ("split_group_id", "image", "target_event_ids"), image_rows)
    summary = {
        "status": "completed", "source_variants": [source.name for source in inputs],
        "accepted_target_events": len(ordered_events), "multi_image_split_groups": len(ordered_components),
        "images_bound_to_a_group": len(image_rows), "largest_split_group": max(len(component) for component in ordered_components),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
