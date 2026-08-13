import os
from pathlib import Path

import pytest
import torch

from conftest import task_config
from yolo_retraining.engine import TaskRunner


pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("weights", ["yolov8n.pt", "yolo11n.pt"])
def test_real_one_epoch_full_cold(weights: str, tmp_path: Path, catalog: dict[str, str]) -> None:
    if os.environ.get("YOLO_RETRAINING_RUN_ULTRALYTICS_SMOKE") != "1":
        pytest.skip("set YOLO_RETRAINING_RUN_ULTRALYTICS_SMOKE=1 to run optional Ultralytics training")
    pytest.importorskip("ultralytics")
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    config["model"]["backend"] = "ultralytics"
    config["initialization"]["checkpoint"] = weights
    config["model"]["definition"] = weights.replace(".pt", ".yaml")
    config["backend"]["params"]["device"] = 0 if torch.cuda.is_available() else "cpu"
    output = TaskRunner(config).run()
    assert (output / "checkpoints" / "last.pt").is_file()
    assert (output / "checkpoints" / "best.pt").is_file()


@pytest.mark.ddp
def test_two_gpu_static_training(tmp_path: Path, catalog: dict[str, str]) -> None:
    if os.environ.get("YOLO_RETRAINING_RUN_DDP_SMOKE") != "1" or torch.cuda.device_count() < 2:
        pytest.skip("requires YOLO_RETRAINING_RUN_DDP_SMOKE=1 and two CUDA devices")
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    pytest.importorskip("ultralytics")
    config["model"] = {"backend": "ultralytics", "definition": "yolo11n.yaml"}
    config["initialization"] = {"source": "pretrained", "checkpoint": "yolo11n.pt"}
    config["backend"]["params"]["device"] = [0, 1]
    # DDP splits the global batch across ranks. With batch=2, each rank gets
    # one image, so imgsz=32 would leave BatchNorm with one value per channel.
    config["backend"]["params"]["imgsz"] = 64
    output = TaskRunner(config).run()
    assert (output / "checkpoints" / "last.pt").is_file()
