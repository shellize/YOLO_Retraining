from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _log_scalars(writer: Any, value: Any, *, prefix: str, step: int) -> None:
    """Recursively write numeric mapping values as TensorBoard scalars."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _log_scalars(writer, child, prefix=child_prefix, step=step)
    elif _is_number(value):
        writer.add_scalar(prefix, float(value), global_step=step)


def write_task_tensorboard(
    output_dir: Path,
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> Path:
    """Write one task's local TensorBoard event file and return its directory."""
    from torch.utils.tensorboard import SummaryWriter

    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for index, row in enumerate(history):
            raw_epoch = row.get("epoch", index)
            try:
                epoch = int(raw_epoch)
            except (TypeError, ValueError):
                epoch = index
            for key, value in row.items():
                if key != "epoch" and _is_number(value):
                    writer.add_scalar(f"training/{key}", float(value), global_step=epoch)

        _log_scalars(writer, evaluation, prefix="evaluation", step=0)
        _log_scalars(writer, cost, prefix="cost", step=0)
        writer.add_text(
            "experiment/config",
            json.dumps(config, indent=2, ensure_ascii=False, default=str),
            global_step=0,
        )
        writer.flush()
    finally:
        writer.close()
    return log_dir
