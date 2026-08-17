from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "result_analysis"
    / "fp_fn_review"
    / "build_fp_fn_review.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_fp_fn_review", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_anylabeling_config_uses_relative_onnx_and_class_order(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "yolov5_anylabeling.yaml"
    classes = ["large luggage", "stroller", "wheelchair", "flatbed truck"]

    module._write_anylabeling_config(
        config_path,
        tmp_path / "task",
        classes,
        confidence_threshold=0.25,
        iou_threshold=0.6,
        max_det=300,
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["type"] == "yolov5"
    assert payload["engine"] == "ort"
    assert payload["model_path"] == "best.onnx"
    assert payload["classes"] == classes
    assert payload["conf_threshold"] == 0.25
    assert payload["iou_threshold"] == 0.6


def test_validate_onnx_checks_fixed_shape_and_yolov5_columns(tmp_path: Path) -> None:
    onnx = pytest.importorskip(
        "onnx", reason="ONNX validation requires the analysis environment"
    )
    from onnx import TensorProto, helper

    module = _load_module()
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 32, 32]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 1, 9]
    )
    value = helper.make_tensor("value", TensorProto.FLOAT, [1, 1, 9], [0.0] * 9)
    node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=value)
    graph = helper.make_graph([node], "fp-fn-review-test", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx_path = tmp_path / "best.onnx"
    onnx.save(model, onnx_path)

    result = module._validate_onnx(onnx_path, imgsz=32, class_count=4)

    assert result["checker"] == "passed"
    assert result["runtime_session"] == "passed"
    assert result["inputs"][0]["shape"] == [1, 3, 32, 32]
    assert result["outputs"][0]["shape"] == [1, 1, 9]


def test_export_onnx_uses_current_interpreter_and_yolov5_workdir(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    yolov5_root = tmp_path / "yolov5"
    yolov5_root.mkdir()
    (yolov5_root / "export.py").write_text("# test export", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    call = {}

    def fake_run(command, cwd, check):
        call.update({"command": command, "cwd": cwd, "check": check})
        checkpoint.with_suffix(".onnx").write_bytes(b"onnx")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._export_onnx(checkpoint, yolov5_root, imgsz=960, device="cpu")

    assert result == checkpoint.with_suffix(".onnx")
    assert call["command"][0] == sys.executable
    assert call["cwd"] == yolov5_root
    assert call["check"] is True
    assert call["command"][-4:] == ["--batch-size", "1", "--device", "cpu"]
    assert call["command"][call["command"].index("--imgsz") + 1] == "960"


def test_build_review_directory_uses_hardlinks_and_copies_checkpoint(
    tmp_path: Path,
) -> None:
    module = _load_module()
    image = tmp_path / "source.jpg"
    Image.new("RGB", (100, 80), color="white").save(image)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"weights")
    row = {
        "sample_id": "test::images/batch/source.jpg",
        "image_path": str(image),
        "ground_truth_count": 2,
        "prediction_count_at_conf": 2,
        "tp_at_conf": 1,
        "fp_at_conf": 1,
        "fn_at_conf": 1,
        "fp_by_class": "{}",
        "fn_by_class": "{}",
    }
    predictions = [
        {
            "sample_id": row["sample_id"],
            "prediction_index": 0,
            "class_id": 0,
            "class_name": "large luggage",
            "confidence": 0.9,
            "x1": 10,
            "y1": 10,
            "x2": 40,
            "y2": 40,
            "matched_gt_index": 0,
            "matched_iou": 0.8,
            "included_at_conf": True,
            "is_tp_at_conf": True,
            "is_fp_at_conf": False,
        },
        {
            "sample_id": row["sample_id"],
            "prediction_index": 1,
            "class_id": 1,
            "class_name": "stroller",
            "confidence": 0.7,
            "x1": 50,
            "y1": 10,
            "x2": 90,
            "y2": 40,
            "matched_gt_index": -1,
            "matched_iou": 0.0,
            "included_at_conf": True,
            "is_tp_at_conf": False,
            "is_fp_at_conf": True,
        },
    ]
    ground_truths = [
        {
            "sample_id": row["sample_id"],
            "gt_index": 0,
            "class_id": 0,
            "class_name": "large luggage",
            "x1": 11,
            "y1": 11,
            "x2": 41,
            "y2": 41,
            "matched_prediction_index": 0,
            "matched_iou": 0.8,
            "detected_at_conf": True,
        },
        {
            "sample_id": row["sample_id"],
            "gt_index": 1,
            "class_id": 2,
            "class_name": "wheelchair",
            "x1": 10,
            "y1": 50,
            "x2": 40,
            "y2": 75,
            "matched_prediction_index": -1,
            "matched_iou": 0.0,
            "detected_at_conf": False,
        },
    ]

    output = tmp_path / "review"
    summary = module._build_review_directory(
        output,
        checkpoint,
        [row],
        "hardlink",
        {
            "match_iou_threshold": 0.5,
            "class_names": ["large luggage", "stroller", "wheelchair"],
        },
        predictions,
        ground_truths,
    )

    fp_image = next((output / "false_positives").glob("*.jpg"))
    fn_image = next((output / "false_negatives").glob("*.jpg"))
    assert os.path.samefile(image, fp_image)
    assert os.path.samefile(image, fn_image)
    assert (output / "best.pt").read_bytes() == b"weights"
    assert summary["comparison_json_count"] == 2
    assert (output / "anylabeling_color_snippet.yaml").is_file()

    fp_payload = module.json.loads(fp_image.with_suffix(".json").read_text(encoding="utf-8"))
    assert fp_payload["imagePath"] == fp_image.name
    assert fp_payload["imageWidth"] == 100
    assert fp_payload["imageHeight"] == 80
    assert [shape["label"] for shape in fp_payload["shapes"]] == [
        "GT/large luggage",
        "GT/wheelchair",
        "TP/large luggage",
        "FP/stroller",
    ]
    assert fp_payload["description"] is None
    assert all(shape["description"] is None for shape in fp_payload["shapes"])
    assert all(shape["group_id"] is None for shape in fp_payload["shapes"])
    assert all(shape["attributes"] == {} for shape in fp_payload["shapes"])
    assert fp_payload["shapes"][2]["score"] == 0.9
    assert fp_payload["shapes"][1]["score"] is None

    color_payload = yaml.safe_load(
        (output / "anylabeling_color_snippet.yaml").read_text(encoding="utf-8")
    )
    assert color_payload["label_colors"]["GT/large luggage"] == [255, 255, 255]
    assert color_payload["label_colors"]["TP/large luggage"] == [0, 200, 0]
    assert color_payload["label_colors"]["FP/large luggage"] == [230, 25, 25]
