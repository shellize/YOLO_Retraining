from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from yolo_retraining.backends.yolov5.artifacts import PredictionArtifactCollector, confidence_sweep


def test_confidence_sweep_counts_fixed_thresholds() -> None:
    result = confidence_sweep(
        np.array([[True] * 10, [False] * 10, [True] * 10], dtype=bool),
        np.array([0.9, 0.5, 0.2]),
        np.array([0, 0, 1]),
        np.array([0, 0, 1]),
        names={0: "a", 1: "b"},
        points=3,
        conf_floor=0.2,
    )

    assert np.allclose(result["thresholds"], [0.2, 0.6, 1.0])
    assert result["overall"]["tp"] == [2, 1, 0]
    assert result["overall"]["fp"] == [1, 0, 0]
    assert result["overall"]["fn"] == [1, 2, 3]
    assert result["best_f1"]["confidence"] in result["thresholds"]


def test_prediction_artifact_collector_writes_per_image_records(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    collector = PredictionArtifactCollector(output)
    collector.record_correct(torch.tensor([[True] * 13, [False] * 13]))
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2], [0, 1, 0.5, 0.5, 0.2, 0.2]])
    predictions = [torch.tensor([[1, 1, 2, 2, 0.9, 0], [2, 2, 3, 3, 0.2, 1]])]

    collector.on_val_batch_end(0, torch.zeros((1, 3, 8, 8)), targets, ["image.jpg"], None, predictions)
    stats = collector.raw_stats()
    collector.close()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["image_path"] == "image.jpg"
    assert record["iou_thresholds"][:3] == [0.1, 0.2, 0.3]
    assert record["target_boxes"][0]["xywh_normalized"] == [0.0625, 0.0625, 0.025, 0.025]
    assert record["predictions"][0]["xyxy_normalized"] == [0.125, 0.125, 0.25, 0.25]
    assert record["predictions"][0]["correct_iou"][0] is True
    assert stats[0].shape == (2, 13)
    assert stats[3].tolist() == [0, 1]
