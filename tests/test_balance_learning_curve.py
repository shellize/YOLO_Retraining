from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from charts.balance_learning_curve_2batch.generate_report import _plot, collect_sequence
from yolo_retraining.config import load_config


TRAIN_BATCHES = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20]


def _layout_batches(path: Path) -> tuple[list[list[int]], list[int], list[int]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    def batches(group: str) -> list[int]:
        return [int(Path(value).name.removeprefix("0720_")) for value in payload["groups"][group]["images"]]

    stages = [batches(f"stage{index}") for index in range(8)]
    return stages, batches("val"), batches("test")


def test_balance_learning_curve_configs_are_comparable_and_reproducible() -> None:
    root = Path(__file__).parents[1]
    random_s41 = TRAIN_BATCHES.copy()
    random_s43 = TRAIN_BATCHES.copy()
    random.Random(41).shuffle(random_s41)
    random.Random(43).shuffle(random_s43)
    cases = {
        "ordered": TRAIN_BATCHES,
        "random_s41": random_s41,
        "random_s43": random_s43,
    }
    for name, expected_order in cases.items():
        config = load_config(
            root
            / "runs"
            / "studies"
            / "0826_balance_learning_curve_2batch"
            / "config"
            / "sequence"
            / f"{name}.yaml"
        )
        assert [arrival["id"] for arrival in config["arrivals"]] == [f"stage{index}" for index in range(8)]
        assert config["sequence"]["seed"] == 42
        assert config["initialization"]["first"]["source"] == "pretrained"
        assert config["initialization"]["subsequent"]["source"] == "pretrained"
        assert config["task_template"]["backend"]["params"]["best_metric"] == "map50"
        stages, val_batches, test_batches = _layout_batches(Path(config["data"]["layout"]))
        assert all(len(stage) == 2 for stage in stages)
        assert [batch for stage in stages for batch in stage] == expected_order
        assert val_batches == [4, 15]
        assert test_batches == [2, 19]
        assert sorted(batch for stage in stages for batch in stage) == TRAIN_BATCHES


def test_learning_curve_summary_records_overall_and_per_class_ap50(tmp_path: Path) -> None:
    layout = tmp_path / "layout.yaml"
    groups = {
        "val": {"split": "val", "images": ["images/0720_4", "images/0720_15"]},
        "test": {"split": "test", "images": ["images/0720_2", "images/0720_19"]},
    }
    for index in range(8):
        groups[f"stage{index}"] = {
            "split": "train",
            "images": [f"images/0720_{index * 2 + 1}", f"images/0720_{index * 2 + 2}"],
        }
    layout.write_text(yaml.safe_dump({"groups": groups}), encoding="utf-8")

    sequence_dir = tmp_path / "sequence"
    sequence_dir.mkdir()
    (sequence_dir / "sequence.yaml").write_text(
        yaml.safe_dump(
            {
                "arrivals": [{"id": f"stage{index}"} for index in range(8)],
                "data": {"layout": str(layout)},
            }
        ),
        encoding="utf-8",
    )
    task_paths = []
    for index in range(8):
        task_dir = tmp_path / f"task{index}"
        task_dir.mkdir()
        task_paths.append(str(task_dir))
        score = 0.1 + index / 100
        (task_dir / "task_result.json").write_text(
            json.dumps(
                {
                    "output_dir": str(task_dir),
                    "metrics": {
                        "best": {
                            "test": {
                                "map50": score,
                                "map50_95": score / 2,
                                "precision": 0.7,
                                "recall": 0.6,
                                "per_class_ap50": {
                                    "large luggage": score,
                                    "stroller": score,
                                    "wheelchair": score,
                                    "flatbed truck": score,
                                },
                                "per_class_ap": {
                                    "large luggage": score / 2,
                                    "stroller": score / 2,
                                    "wheelchair": score / 2,
                                    "flatbed truck": score / 2,
                                },
                            }
                        }
                    },
                    "cost": {
                        "selected_count": (index + 1) * 1000,
                        "images_read": (index + 1) * 100000,
                        "optimizer_steps": (index + 1) * 100,
                        "training_seconds": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
    (sequence_dir / "sequence_result.json").write_text(
        json.dumps({"status": "completed", "tasks": task_paths}),
        encoding="utf-8",
    )

    rows = collect_sequence("ordered", sequence_dir)
    assert len(rows) == 8
    assert rows[-1]["map50"] == 0.17
    assert rows[-1]["ap50.wheelchair"] == 0.17
    assert rows[-1]["ap50_95.wheelchair"] == 0.085
    report = tmp_path / "report"
    report.mkdir()
    _plot(rows, report)
    assert (report / "map50_learning_curve.png").is_file()
    assert (report / "per_class_ap_learning_curve.png").is_file()
