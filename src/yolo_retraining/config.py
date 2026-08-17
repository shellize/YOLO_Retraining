from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


BUILTIN_DEFAULTS: dict[str, Any] = {
    "task": {"name": None, "label": "experiment", "seed": 42, "parent_result": None},
    "select_policy": {"name": "full", "params": {}},
    "epoch_policy": {"name": "static", "params": {}},
    "budget": {"type": "epochs", "value": 100},
    "backend": {"params": {"batch": 64, "imgsz": 640, "device": 0, "workers": 16, "amp": True, "best_metric": "map50"}},
    "evaluation": {"primary_metric": "map50_95", "test_scope": "seen", "evaluate_checkpoints": ["last", "best"]},
}

TASK_KEYS = {"task", "data", "model", "initialization", "select_policy", "epoch_policy", "budget", "backend", "evaluation"}
SEQUENCE_KEYS = {"sequence", "data", "arrivals", "scope_rule", "initialization", "task_template", "task_overrides"}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def _absolute(base: Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _resolve_paths(payload: dict[str, Any], base: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(payload)
    task = resolved.get("task", {})
    if isinstance(task, dict):
        if task.get("output_root") is not None:
            task["output_root"] = _absolute(base, task["output_root"])
        if task.get("parent_result") is not None:
            task["parent_result"] = _absolute(base, task["parent_result"])
    sequence = resolved.get("sequence", {})
    if isinstance(sequence, dict) and sequence.get("output_root") is not None:
        sequence["output_root"] = _absolute(base, sequence["output_root"])
    data = resolved.get("data", {})
    if isinstance(data, dict):
        if isinstance(data.get("catalog"), dict):
            data["catalog"] = {str(key): _absolute(base, value) for key, value in data["catalog"].items()}
        if data.get("layout") is not None:
            data["layout"] = _absolute(base, data["layout"])
    arrivals = resolved.get("arrivals")
    if isinstance(arrivals, list):
        for arrival in arrivals:
            if isinstance(arrival, dict) and arrival.get("data") is not None:
                arrival["data"] = _absolute(base, arrival["data"])
    initialization = resolved.get("initialization", {})
    if isinstance(initialization, dict) and initialization.get("source") == "explicit":
        initialization["checkpoint"] = _absolute(base, initialization.get("checkpoint"))
    backend = resolved.get("backend", {})
    if isinstance(backend, dict) and isinstance(backend.get("params"), dict):
        if backend["params"].get("hyp") is not None:
            backend["params"]["hyp"] = _absolute(base, backend["params"]["hyp"])
    return resolved


def _load_with_includes(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"configuration include cycle: {chain}")
    payload = _load_yaml(path)
    includes = payload.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"include must be a path or list of paths: {path}")
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = deep_merge(merged, _load_with_includes(include_path, (*stack, path)))
    return deep_merge(merged, _resolve_paths(payload, path.parent))


def parse_scalar(value: str) -> Any:
    return yaml.safe_load(value)


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ValueError(f"invalid override key: {dotted_key!r}")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def load_config(path: Path | str, overrides: Iterable[str] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    loaded = _load_with_includes(path)
    is_sequence = "sequence" in loaded
    config = loaded if is_sequence else deep_merge(BUILTIN_DEFAULTS, loaded)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value: {override!r}")
        key, raw = override.split("=", 1)
        set_dotted(config, key, parse_scalar(raw))
    if is_sequence:
        validate_sequence_config(config)
    else:
        validate_task_config(config)
    config["_config_path"] = str(path)
    return config


def validate_task_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - TASK_KEYS - {"_config_path"}
    if unknown:
        raise ValueError(f"unknown task configuration fields: {sorted(unknown)}")
    for key in ("task", "data", "model", "initialization", "select_policy", "epoch_policy", "budget", "backend", "evaluation"):
        if key not in config or not isinstance(config[key], Mapping):
            raise ValueError(f"task configuration requires mapping field {key!r}")
    task = config["task"]
    if not task.get("output_root"):
        raise ValueError("task.output_root is required")
    data = config["data"]
    catalog = data.get("catalog")
    layout = data.get("layout")
    has_catalog = isinstance(catalog, Mapping) and bool(catalog)
    has_layout = isinstance(layout, str) and bool(layout)
    if has_catalog == has_layout:
        raise ValueError("data requires exactly one of a non-empty catalog or layout")
    available_groups = set(catalog) if has_catalog else _layout_groups(Path(layout))
    for scope in ("current", "candidate", "validation", "test"):
        groups = data.get(scope)
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"data.{scope} must be a non-empty list")
        missing = set(groups) - available_groups
        if missing:
            raise ValueError(f"data.{scope} references unknown groups: {sorted(missing)}")
    if not set(data["current"]).issubset(data["candidate"]):
        raise ValueError("data.current groups must be a subset of data.candidate groups")
    if config["budget"].get("type") != "epochs" or int(config["budget"].get("value", 0)) <= 0:
        raise ValueError("budget must be a positive epochs budget")
    if config["model"].get("backend") not in {"yolov5", "ultralytics"}:
        raise ValueError("phase 1 supports model.backend=yolov5 or ultralytics")
    source = config["initialization"].get("source")
    if source not in {"pretrained", "parent", "explicit"}:
        raise ValueError("initialization.source must be pretrained, parent, or explicit")
    if source == "parent" and not task.get("parent_result"):
        raise ValueError("parent initialization requires task.parent_result")
    if config["select_policy"].get("name") not in {"full", "random_replay"}:
        raise ValueError("phase 1 supports selection policies: full, random_replay")
    if config["epoch_policy"].get("name") != "static":
        raise ValueError("phase 1 supports only epoch_policy.name=static")


def validate_sequence_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - SEQUENCE_KEYS - {"_config_path"}
    if unknown:
        raise ValueError(f"unknown sequence configuration fields: {sorted(unknown)}")
    for key in ("sequence", "scope_rule", "initialization", "task_template"):
        if key not in config or not isinstance(config[key], Mapping):
            raise ValueError(f"sequence configuration requires mapping field {key!r}")
    arrivals = config.get("arrivals")
    if not isinstance(arrivals, list) or not arrivals:
        raise ValueError("arrivals must be a non-empty list")
    ids = [item.get("id") for item in arrivals if isinstance(item, Mapping)]
    if len(ids) != len(arrivals) or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("each arrival requires a unique non-empty id")
    layout_data = config.get("data")
    uses_layout = isinstance(layout_data, Mapping) and isinstance(layout_data.get("layout"), str)
    if uses_layout:
        available_groups = _layout_groups(Path(layout_data["layout"]))
        missing_arrivals = set(ids) - available_groups
        if missing_arrivals:
            raise ValueError(f"arrivals reference groups missing from data.layout: {sorted(missing_arrivals)}")
        for scope in ("validation", "test"):
            groups = layout_data.get(scope)
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"sequence data.{scope} must be a non-empty list")
            missing = set(groups) - available_groups
            if missing:
                raise ValueError(f"sequence data.{scope} references groups missing from layout: {sorted(missing)}")
    elif any(not isinstance(item.get("data"), str) for item in arrivals):
        raise ValueError("each arrival requires a data YAML path when sequence data.layout is absent")
    sequence = config["sequence"]
    if not sequence.get("output_root"):
        raise ValueError("sequence.output_root is required")
    if config["scope_rule"].get("candidate") not in {"all_seen", "current"}:
        raise ValueError("scope_rule.candidate must be all_seen or current")
    initialization = config["initialization"]
    for key in ("first", "subsequent"):
        if not isinstance(initialization.get(key), Mapping):
            raise ValueError(f"initialization.{key} must be a mapping")


def _layout_groups(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"data layout does not exist: {path}")
    payload = _load_yaml(path)
    groups = payload.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"data layout requires a non-empty groups mapping: {path}")
    return {str(group_id) for group_id in groups}


def dump_yaml(payload: Mapping[str, Any], path: Path) -> None:
    clean = {key: value for key, value in payload.items() if not key.startswith("_")}
    path.write_text(yaml.safe_dump(clean, sort_keys=False, allow_unicode=True), encoding="utf-8")
