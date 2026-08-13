import json
from pathlib import Path

import pytest

from conftest import FakeBackend
from yolo_retraining.engine import SequenceRunner


def sequence_config(tmp_path: Path, catalog: dict[str, str]) -> dict:
    return {
        "sequence": {"name": "unit", "label": "full-warm", "seed": 42, "output_root": str(tmp_path / "sequences")},
        "arrivals": [{"id": key, "data": catalog[key]} for key in ("stage0", "stage1", "stage2")],
        "scope_rule": {"candidate": "all_seen", "evaluation": "seen"},
        "initialization": {"first": {"source": "pretrained", "checkpoint": "yolov5s.pt"}, "subsequent": {"source": "parent", "checkpoint": "last"}},
        "task_template": {
            "model": {"backend": "yolov5", "definition": "yolov5s.yaml"},
            "select_policy": {"name": "full", "params": {}},
            "epoch_policy": {"name": "static", "params": {}},
            "budget": {"type": "epochs", "value": 1},
            "backend": {"params": {"batch": 2, "imgsz": 32, "device": "cpu", "workers": 0, "amp": True}},
            "evaluation": {"primary_metric": "map50_95", "test_scope": "seen", "evaluate_checkpoints": ["last", "best"]},
        },
        "task_overrides": {},
    }


def test_sequence_expands_all_seen_and_parent_results(monkeypatch, tmp_path: Path, catalog: dict[str, str]) -> None:
    monkeypatch.setattr("yolo_retraining.engine.task_runner.create_backend", lambda config: FakeBackend())
    output = SequenceRunner(sequence_config(tmp_path, catalog)).run()
    tasks = sorted((output / "tasks").iterdir())
    assert len(tasks) == 3
    second_config = (tasks[1] / "task.yaml").read_text(encoding="utf-8")
    assert "stage0" in second_config and "stage1" in second_config
    second_result = json.loads((tasks[1] / "task_result.json").read_text(encoding="utf-8"))
    assert second_result["parent_result"] == str(tasks[0].resolve())
    sequence_result = json.loads((output / "sequence_result.json").read_text(encoding="utf-8"))
    assert sequence_result["status"] == "completed"
    assert len(sequence_result["summary"]["matrix"]) == 3


def test_sequence_stops_after_failure(monkeypatch, tmp_path: Path, catalog: dict[str, str]) -> None:
    calls = {"count": 0}

    def factory(config):
        calls["count"] += 1
        return FakeBackend(fail_train=calls["count"] == 2)

    monkeypatch.setattr("yolo_retraining.engine.task_runner.create_backend", factory)
    runner = SequenceRunner(sequence_config(tmp_path, catalog))
    with pytest.raises(RuntimeError, match="simulated"):
        runner.run()
    tasks = sorted((runner.output_dir / "tasks").iterdir())
    assert len(tasks) == 2
    status = json.loads((runner.output_dir / "sequence_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"


@pytest.mark.parametrize(
    ("label", "candidate_rule", "selection", "subsequent", "expected_candidates", "expected_selected"),
    [
        ("full-cold", "all_seen", {"name": "full", "params": {}}, {"source": "pretrained", "checkpoint": "yolov5s.pt"}, 4, 4),
        ("full-warm", "all_seen", {"name": "full", "params": {}}, {"source": "parent", "checkpoint": "last"}, 4, 4),
        ("current-only", "current", {"name": "full", "params": {}}, {"source": "parent", "checkpoint": "last"}, 2, 2),
        ("random-replay", "all_seen", {"name": "random_replay", "params": {"replay_size": 1}}, {"source": "parent", "checkpoint": "last"}, 4, 3),
    ],
)
def test_all_sequence_baseline_semantics(
    monkeypatch,
    tmp_path: Path,
    catalog: dict[str, str],
    label: str,
    candidate_rule: str,
    selection: dict,
    subsequent: dict,
    expected_candidates: int,
    expected_selected: int,
) -> None:
    monkeypatch.setattr("yolo_retraining.engine.task_runner.create_backend", lambda config: FakeBackend())
    config = sequence_config(tmp_path / label, catalog)
    config["sequence"]["label"] = label
    config["arrivals"] = config["arrivals"][:2]
    config["scope_rule"]["candidate"] = candidate_rule
    config["task_template"]["select_policy"] = selection
    config["initialization"]["subsequent"] = subsequent
    output = SequenceRunner(config).run()
    second = sorted((output / "tasks").iterdir())[1]
    candidates = (second / "data" / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    selected = (second / "data" / "selected_ids.txt").read_text(encoding="utf-8").splitlines()
    assert len(candidates) == expected_candidates
    assert len(selected) == expected_selected
