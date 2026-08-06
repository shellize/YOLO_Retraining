from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


INVALID_PATH = re.compile(r'[<>:"/\\|?*\s]+')


def sanitize(value: Any, *, maximum: int = 80) -> str:
    cleaned = INVALID_PATH.sub("-", str(value)).strip("-. ")
    return (cleaned or "run")[:maximum]


def timestamp() -> str:
    return datetime.now().strftime("%m%d-%H%M%S")


def task_output_path(config: Mapping[str, Any]) -> Path:
    task = config["task"]
    prefix = sanitize(task.get("name") or timestamp())
    label = sanitize(task.get("label", "task"))
    current = sanitize("+".join(config["data"]["current"]))
    model = sanitize(Path(str(config["model"].get("definition", "model"))).stem)
    seed = int(task.get("seed", 42))
    return Path(task["output_root"]) / f"{prefix}__{label}__{current}__{model}__s{seed}"


def sequence_output_path(config: Mapping[str, Any]) -> Path:
    sequence = config["sequence"]
    prefix = sanitize(sequence.get("name") or timestamp())
    label = sanitize(sequence.get("label", "sequence"))
    arrivals = config["arrivals"]
    data_range = sanitize(f"{arrivals[0]['id']}-{arrivals[-1]['id']}")
    model = sanitize(Path(str(config["task_template"]["model"].get("definition", "model"))).stem)
    seed = int(sequence.get("seed", 42))
    return Path(sequence["output_root"]) / f"{prefix}__seq-{label}__{data_range}__{model}__s{seed}"


def create_output(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"output directory already exists; overwrite/resume are not supported: {path}")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_lines(path: Path, values: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            handle.write("")
            return path
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def relative_to(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def status_payload(status: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"), **extra}

