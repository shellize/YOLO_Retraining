import json
from pathlib import Path

from scripts.summarize_fullcold_data_study import collect_sequence, main


def _make_sequence(root: Path, map50_95: float, large_luggage_ap: float) -> Path:
    task = root / "tasks" / "000"
    task.mkdir(parents=True)
    task_result = {
        "output_dir": str(task),
        "metrics": {
            "best": {
                "test": {
                    "map50_95": map50_95,
                    "map50": map50_95 + 0.2,
                    "precision": 0.7,
                    "recall": 0.4,
                    "per_class_ap": {"large luggage": large_luggage_ap, "stroller": 0.1},
                }
            }
        },
        "cost": {
            "selected_count": 10,
            "candidate_count": 10,
            "images_read": 1000,
            "optimizer_steps": 100,
            "training_seconds": 12.0,
        },
    }
    (task / "task_result.json").write_text(json.dumps(task_result), encoding="utf-8")
    (root / "sequence_result.json").write_text(
        json.dumps({"status": "completed", "tasks": [str(task)]}),
        encoding="utf-8",
    )
    return root


def test_collect_sequence_includes_large_luggage_and_cost(tmp_path: Path) -> None:
    sequence = _make_sequence(tmp_path / "sequence", 0.2, 0.25)
    rows = collect_sequence("baseline", sequence)
    assert rows[0]["large_luggage_ap"] == 0.25
    assert rows[0]["images_read"] == 1000


def test_summary_writes_final_stage_deltas(tmp_path: Path) -> None:
    baseline = _make_sequence(tmp_path / "baseline", 0.2, 0.25)
    variant = _make_sequence(tmp_path / "variant", 0.3, 0.4)
    output = tmp_path / "report"
    assert main(
        [
            "--output-dir",
            str(output),
            "--sequence",
            f"full_all={baseline}",
            "--sequence",
            f"dedup={variant}",
        ]
    ) == 0
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    dedup = next(row for row in payload["final_stage"] if row["experiment"] == "dedup")
    assert abs(dedup["delta_vs_full_all.map50_95"] - 0.1) < 1e-9
    assert abs(dedup["delta_vs_full_all.large_luggage_ap"] - 0.15) < 1e-9
