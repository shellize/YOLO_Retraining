from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yolo_retraining.backends.yolov5.iou_metrics import IOU_THRESHOLDS, run_with_low_iou_metrics


def test_low_iou_adapter_preserves_standard_map_and_exposes_ap30() -> None:
    module = SimpleNamespace(torch=torch)

    def ap_per_class(tp, _conf, _pred_cls, _target_cls, **_kwargs):
        columns = np.asarray(tp, dtype=float).mean(axis=0, keepdims=True)
        return (
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.8]),
            np.array([0.7]),
            np.array([0.75]),
            columns,
            np.array([0]),
        )

    module.ap_per_class = ap_per_class

    def runner():
        thresholds = module.torch.linspace(0.5, 0.95, 10).cpu().numpy()
        assert np.allclose(thresholds, IOU_THRESHOLDS)
        tp = np.array(
            [
                [True, True, True, True, True, True, False, False, False, False, False, False, False],
                [True, True, False, False, False, False, False, False, False, False, False, False, False],
            ]
        )
        module.ap_per_class(
            tp,
            np.array([0.9, 0.8]),
            np.array([0, 0]),
            np.array([0, 0]),
            names={0: "object"},
        )
        return (0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0), np.array([0.0]), (1, 2, 3)

    evaluation = run_with_low_iou_metrics(module, runner)

    assert evaluation.metrics["map10"] == 1.0
    assert evaluation.metrics["map20"] == 1.0
    assert evaluation.metrics["map30"] == 0.5
    assert evaluation.metrics["map50"] == 0.5
    assert evaluation.metrics["map50_95"] == pytest.approx(0.15)
    assert evaluation.results[2:4] == pytest.approx((0.5, 0.15))
    assert evaluation.per_class_map50_95.tolist() == pytest.approx([0.15])
