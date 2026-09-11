from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = STUDY_ROOT / "result" / "target_track_candidate_audit" / "manual_review.csv"
DEFAULT_OVERRIDES = STUDY_ROOT / "config" / "target_track_manual_overrides.yaml"
DEFAULT_OUTPUT = STUDY_ROOT / "experiment" / "variants" / "target_track_groups_reviewed_final"
ALLOWED = {"", "same_track", "not_same_track", "unsure"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize reviewed target tracks and image-level split groups.")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def frame(image: str) -> int:
    return int(Path(image).name.split("_", 1)[0])


def parse_nodes(row: dict[str, str]) -> list[dict[str, object]]:
    images = row["images"].split("|")
    positions = [int(value) for value in row["sequence_indices"].split("|")]
    if len(images) != len(positions):
        raise ValueError(f'{row["track_id"]}: images/sequence positions mismatch')
    nodes = []
    for value, position in zip(images, positions):
        image, marker, box = value.rpartition("#box")
        if not marker or not box.isdecimal():
            raise ValueError(f'{row["track_id"]}: invalid node {value!r}')
        nodes.append({"image": image, "box_index": int(box), "sequence_index": position})
    return nodes


def main() -> int:
    args = parse_args()
    review_path, override_path, output = args.review.resolve(), args.overrides.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final track groups: {output}")
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overrides = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    by_id = {row["track_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate track IDs in review")
    for row in rows:
        if row["status"].strip() not in ALLOWED:
            raise ValueError(f'{row["track_id"]}: unsupported status {row["status"]!r}')
    if set(overrides) - set(by_id):
        raise ValueError(f"override tracks absent from review: {sorted(set(overrides) - set(by_id))}")

    accepted: list[dict[str, object]] = []
    override_audit = []
    for row in rows:
        track_id, status, nodes = row["track_id"], row["status"].strip(), parse_nodes(row)
        if track_id not in overrides:
            if status == "same_track":
                accepted.append({"source_track": track_id, "kind": "reviewed_same_track", "nodes": nodes})
            elif status == "unsure":
                raise ValueError(f"unsure track lacks a manual override: {track_id}")
            continue

        remaining = {(node["image"], node["box_index"]): node for node in nodes}
        image_to_node = {node["image"]: node for node in nodes}
        used_images: set[str] = set()
        for subgroup_index, specification in enumerate(overrides[track_id], start=1):
            if isinstance(specification, dict):
                start, end = specification["range"]
                if start not in image_to_node or end not in image_to_node:
                    raise ValueError(f"{track_id}: range endpoint absent from candidate track")
                low, high = sorted((image_to_node[start]["sequence_index"], image_to_node[end]["sequence_index"]))
                selected = [node for node in nodes if low <= node["sequence_index"] <= high]
            else:
                selected = []
                for image in specification:
                    if image not in image_to_node:
                        raise ValueError(f"{track_id}: override image absent from candidate track: {image}")
                    selected.append(image_to_node[image])
            selected_images = {node["image"] for node in selected}
            if used_images & selected_images:
                raise ValueError(f"{track_id}: override subgroups overlap")
            used_images.update(selected_images)
            for node in selected:
                remaining.pop((node["image"], node["box_index"]), None)
            if len(selected) >= 2:
                accepted.append({"source_track": track_id, "kind": "manual_subgroup", "nodes": selected})
            override_audit.append({"source_track": track_id, "subgroup": subgroup_index, "member_count": len(selected), "members": "|".join(node["image"] for node in selected)})
        if len(remaining) >= 2:
            accepted.append({"source_track": track_id, "kind": "manual_remainder", "nodes": list(remaining.values())})

    accepted.sort(key=lambda item: (min(node["sequence_index"] for node in item["nodes"]), item["source_track"], item["kind"]))
    target_rows = []
    for number, group in enumerate(accepted, start=1):
        group_id = f"target_event_{number:04d}"
        for node in sorted(group["nodes"], key=lambda item: item["sequence_index"]):
            target_rows.append({"target_event_id": group_id, "source_track": group["source_track"], "kind": group["kind"], **node})

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
    for row in target_rows:
        by_event.setdefault(row["target_event_id"], []).append(row["image"])
    for images in by_event.values():
        first = images[0]
        for image in images[1:]:
            union(first, image)
    components: dict[str, set[str]] = {}
    for image in parent:
        components.setdefault(find(image), set()).add(image)
    ordered_components = sorted(components.values(), key=lambda members: min(frame(image) for image in members))
    image_rows = []
    for number, members in enumerate(ordered_components, start=1):
        group_id = f"split_group_{number:04d}"
        for image in sorted(members, key=lambda value: (frame(value), value)):
            events = sorted({row["target_event_id"] for row in target_rows if row["image"] == image})
            image_rows.append({"split_group_id": group_id, "image": image, "target_event_ids": "|".join(events)})

    output.mkdir(parents=True)
    def write_csv(name: str, data: list[dict[str, object]], fields: tuple[str, ...]) -> None:
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(data)
    write_csv("target_event_members.csv", target_rows, ("target_event_id", "source_track", "kind", "image", "box_index", "sequence_index"))
    write_csv("image_split_groups.csv", image_rows, ("split_group_id", "image", "target_event_ids"))
    write_csv("manual_override_audit.csv", override_audit, ("source_track", "subgroup", "member_count", "members"))
    (output / "manual_review.csv").write_bytes(review_path.read_bytes())
    summary = {
        "status": "completed", "review_rows": len(rows),
        "review_status_counts": {status or "unreviewed": sum(row["status"].strip() == status for row in rows) for status in ("same_track", "not_same_track", "unsure", "")},
        "manual_override_tracks": len(overrides), "manual_override_rows": sum(len(value) for value in overrides.values()),
        "accepted_target_events": len(accepted), "multi_image_split_groups": len(ordered_components),
        "images_bound_to_a_group": len({row["image"] for row in image_rows}),
        "largest_split_group": max(len(component) for component in ordered_components),
        "unreviewed_and_not_same_semantics": "independent images unless overridden by the handwritten list",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
