from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_retraining.config import load_config
from yolo_retraining.engine.output import sequence_output_path


def completed_task(path: Path, description: str) -> None:
    result_path = path / "task_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"{description} TaskResult is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"{description} task is not completed: {path}")
    checkpoint = path / result["artifacts"]["best_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{description} best checkpoint is missing: {checkpoint}")


def inspect(config_path: Path) -> tuple[Path, str]:
    config = load_config(config_path)
    completed_task(Path(config["sequence"]["bootstrap_result"]), "bootstrap")
    for stage, override in config.get("task_overrides", {}).items():
        teacher = override.get("select_policy", {}).get("params", {}).get("teacher_result")
        if teacher:
            completed_task(Path(teacher), f"{stage} teacher")
    output = sequence_output_path(config)
    state = "ready"
    if output.exists():
        result_path = output / "sequence_result.json"
        if not result_path.is_file():
            raise FileExistsError(f"sequence output exists but is incomplete; move or inspect it before launch: {output}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise ValueError(f"existing sequence is not completed: {output}")
        state = "completed"
    return output, state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()
    for config_path in args.configs:
        output, state = inspect(config_path.resolve())
        if args.print_output:
            print(output)
        else:
            print(f"[Preflight] {config_path}: {state}; output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
