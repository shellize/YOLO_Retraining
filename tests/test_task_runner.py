import json
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
    tensorboard_dir = output / result["artifacts"]["tensorboard"]
    assert tensorboard_dir.is_dir()
    assert list(tensorboard_dir.glob("events.out.tfevents.*"))
    event_data = EventAccumulator(str(tensorboard_dir)).Reload()
    assert "training/map50_95" in event_data.Tags()["scalars"]
    assert "evaluation/last/stage0/map50_95" in event_data.Tags()["scalars"]
    assert "experiment/config/text_summary" in event_data.Tags()["tensors"]
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
