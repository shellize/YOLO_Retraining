from yolo_retraining.data import build_registry, resolve_scope


def test_scope_resolves_current_history_and_evaluation(catalog: dict[str, str]) -> None:
    registry = build_registry({key: catalog[key] for key in ("stage0", "stage1")})
    scope = resolve_scope(
        registry,
        {"current": ["stage1"], "candidate": ["stage0", "stage1"], "validation": ["stage0", "stage1"], "test": ["stage0", "stage1"]},
    )
    assert len(scope["current_ids"]) == 2
    assert len(scope["history_ids"]) == 2
    assert set(scope["test_by_group"]) == {"stage0", "stage1"}

