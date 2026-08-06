import json
from pathlib import Path

import pytest

from conftest import FakeBackend, task_config
from yolo_retraining.engine import TaskRunner


def test_task_runner_writes_completed_result(tmp_path: Path, catalog: dict[str, str]) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    output = TaskRunner(config, backend=FakeBackend()).run()
    result = json.loads((output / "task_result.json").read_text(encoding="utf-8"))
    status = json.loads((output / "task_status.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert status["status"] == "completed"
    assert (output / result["artifacts"]["last_checkpoint"]).is_file()
    assert not list(output.rglob("*.jpg"))


def test_task_runner_marks_failure_without_result(tmp_path: Path, catalog: dict[str, str]) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    runner = TaskRunner(config, backend=FakeBackend(fail_train=True))
    with pytest.raises(RuntimeError, match="simulated"):
        runner.run()
    status = json.loads((runner.output_dir / "task_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert not (runner.output_dir / "task_result.json").exists()
    assert (runner.output_dir / "logs" / "error.txt").is_file()

