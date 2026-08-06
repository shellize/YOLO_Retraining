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

