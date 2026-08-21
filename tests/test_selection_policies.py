from pathlib import Path

import pytest

from yolo_retraining.policies import create_epoch_policy, create_selection_policy


def test_full_selection_and_static_epoch() -> None:
    selection = create_selection_policy({"name": "full", "params": {}}).select({"candidate_ids": ["b", "a"]})
    assert selection["selected_ids"] == ["a", "b"]
    assert create_epoch_policy({"name": "static", "params": {}}).build_plan(selection["selected_ids"], epoch=2, seed=42) == ["a", "b"]


def test_random_replay_is_deterministic_and_history_only() -> None:
    context = {"current_ids": ["current"], "candidate_ids": ["old0", "old1", "old2", "current"], "seed": 42}
    policy = create_selection_policy({"name": "random_replay", "params": {"replay_size": 2}})
    first = policy.select(context)
    second = policy.select(context)
    assert first == second
    assert first["groups"]["current"] == ["current"]
    assert set(first["groups"]["replay"]).issubset({"old0", "old1", "old2"})
    assert len(first["groups"]["replay"]) == 2


def test_random_replay_uses_all_short_history() -> None:
    result = create_selection_policy({"name": "random_replay", "params": {"replay_size": 5}}).select({"current_ids": ["new"], "candidate_ids": ["old", "new"], "seed": 1})
    assert result["groups"]["replay"] == ["old"]


def test_random_replay_requires_positive_size() -> None:
    with pytest.raises(ValueError, match="positive replay_size"):
        create_selection_policy({"name": "random_replay", "params": {"replay_size": 0}})


def test_positive_only_keeps_valid_nonempty_labels(tmp_path: Path) -> None:
    positive = tmp_path / "positive.txt"
    empty = tmp_path / "empty.txt"
    positive.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    empty.write_text("", encoding="utf-8")
    context = {
        "current_ids": ["positive", "empty"],
        "candidate_ids": ["positive", "empty"],
        "registry": {"records": {"positive": {"label_path": str(positive)}, "empty": {"label_path": str(empty)}}},
        "config": {"task": {"parent_result": None}},
    }
    result = create_selection_policy({"name": "positive_only", "params": {}}).select(context)
    assert result["selected_ids"] == ["positive"]
    assert result["metadata"]["current_selected_count"] == 1


def test_cumulative_positive_only_reuses_parent_selection(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    (parent / "data").mkdir(parents=True)
    (parent / "data" / "selected_ids.txt").write_text("stage0::a\n", encoding="utf-8")
    label = tmp_path / "current.txt"
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    context = {
        "current_ids": ["stage1::b"],
        "candidate_ids": ["stage0::a", "stage1::b"],
        "registry": {"records": {"stage1::b": {"label_path": str(label)}}},
        "config": {"task": {"parent_result": str(parent)}},
    }
    result = create_selection_policy({"name": "positive_only", "params": {"cumulative": True}}).select(context)
    assert result["selected_ids"] == ["stage0::a", "stage1::b"]
    assert result["groups"]["history"] == ["stage0::a"]
