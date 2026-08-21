import pytest

from yolo_retraining.evaluation.metrics import summarize_matrix


def test_stage_matrix_reports_forgetting_and_acquisition() -> None:
    rows = [
        {"metrics": {"stage0": {"map50_95": 0.50}, "stage1": {"map50_95": 0.20}}},
        {"metrics": {"stage0": {"map50_95": 0.40}, "stage1": {"map50_95": 0.60}}},
    ]
    summary = summarize_matrix(rows, stage_ids=["stage0", "stage1"])
    assert summary["final_forgetting"]["stage0"] == pytest.approx(0.1)
    assert summary["final_forgetting_ratio"]["stage0"] == pytest.approx(0.2)
    assert summary["acquisition_abs"]["stage1"] == pytest.approx(0.4)
    assert summary["acquisition_headroom_ratio"]["stage1"] == pytest.approx(0.5)
    assert summary["stage_specific_metrics_available"] is True
