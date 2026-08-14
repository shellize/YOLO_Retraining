from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import task_config
from yolo_retraining.backends import create_backend
from yolo_retraining.backends.yolov5 import Yolov5Backend
from yolo_retraining.backends.yolov5.backend import normalize_evaluation
from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier
from yolo_retraining.backends.yolov5.trainer import build_train_command, read_training_history, run_command


def test_registry_creates_original_yolov5_backend(tmp_path: Path, catalog: dict[str, str]) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    backend = create_backend(config)
    assert isinstance(backend, Yolov5Backend)
    assert backend.output_namespace == "yolov5"


def test_single_gpu_train_command(tmp_path: Path, catalog: dict[str, str]) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    config["backend"]["params"].update(device=0, batch=4, imgsz=640, workers=2)
    command = build_train_command(
        config,
        source_root=tmp_path / "yolov5",
        checkpoint=tmp_path / "yolov5s.pt",
        data_yaml=tmp_path / "data.yaml",
        output_dir=tmp_path / "output",
    )
    assert command[0] == sys.executable
    assert command[1].endswith("train_entry.py")
    assert command[command.index("--yolo-source-root") + 1] == str(tmp_path / "yolov5")
    assert "torch.distributed.run" not in command
    assert command[command.index("--device") + 1] == "0"
    assert command[command.index("--weights") + 1].endswith("yolov5s.pt")
    assert command[command.index("--best-metric") + 1] == "map50"


def test_two_gpu_train_command(tmp_path: Path, catalog: dict[str, str]) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    config["backend"]["params"].update(device=[0, 1], batch=4)
    command = build_train_command(
        config,
        source_root=tmp_path / "yolov5",
        checkpoint=tmp_path / "last.pt",
        data_yaml=tmp_path / "data.yaml",
        output_dir=tmp_path / "output",
    )
    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"]
    assert command[command.index("--device") + 1] == "0,1"


def test_original_yolov5_fitness_can_be_selected(tmp_path: Path, catalog: dict[str, str]) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    config["backend"]["params"]["best_metric"] = "yolov5_fitness"
    command = build_train_command(
        config,
        source_root=tmp_path / "yolov5",
        checkpoint=tmp_path / "yolov5s.pt",
        data_yaml=tmp_path / "data.yaml",
        output_dir=tmp_path / "output",
    )
    assert command[command.index("--best-metric") + 1] == "yolov5_fitness"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"amp": False}, "amp=false"),
        ({"custom": 1}, "unknown YOLOv5 backend params"),
        ({"device": [0, 1], "batch": 3}, "divisible"),
        ({"best_metric": "map50_95"}, "best_metric must be map50 or yolov5_fitness"),
    ],
)
def test_yolov5_backend_rejects_unsupported_params(
    tmp_path: Path,
    catalog: dict[str, str],
    change: dict,
    message: str,
) -> None:
    config = task_config(tmp_path, {"stage0": catalog["stage0"]}, current=["stage0"], candidate=["stage0"])
    config["backend"]["params"].update(change)
    with pytest.raises(ValueError, match=message):
        Yolov5Backend().validate_config(config)


def test_evaluation_and_results_csv_conversion(tmp_path: Path) -> None:
    converted = normalize_evaluation(
        {"map50_95": 0.4, "map50": 0.7, "precision": 0.8, "recall": 0.6, "per_class_ap": [0.3, 0.5]},
        ["car", "bus"],
        sample_count=16,
        evaluation_seconds=1.25,
    )
    assert converted["per_class_ap"] == {"car": 0.3, "bus": 0.5}
    assert converted["sample_count"] == 16
    results = tmp_path / "results.csv"
    results.write_text("epoch,metrics/mAP_0.5,metrics/mAP_0.5:0.95\n0,0.7,0.4\n", encoding="utf-8")
    assert read_training_history(results) == [{"epoch": 0.0, "metrics/mAP_0.5": 0.7, "metrics/mAP_0.5:0.95": 0.4}]


def test_subprocess_failure_reports_log(tmp_path: Path) -> None:
    log_path = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="exit code 7"):
        run_command([sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path, log_path=log_path)
    assert log_path.is_file()


def test_training_subprocess_compacts_progress_and_repeated_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "training.log"
    program = (
        "import sys; "
        "sys.stdout.write('train: Scanning data...: 0%|          | 0/2 00:00\\n'); sys.stdout.flush(); "
        "sys.stdout.write('train: Scanning data...: 100%|##########| 2/2 00:00\\n'); sys.stdout.flush(); "
        "sys.stdout.write('val: Scanning data...: 0%|          | 0/1 00:00\\n'); sys.stdout.flush(); "
        "sys.stdout.write('val: Scanning data...: 100%|##########| 1/1 00:00\\n'); sys.stdout.flush(); "
        "sys.stdout.write('train: WARNING /data/a.jpg: incomplete JPEG accepted read-only\\n'); sys.stdout.flush(); "
        "sys.stdout.write('val: WARNING /data/b.jpg: incomplete JPEG accepted read-only\\n'); sys.stdout.flush(); "
        "sys.stdout.write('       0/2 1.0G 0.1 0.2 0.3 4 640: 0%|          | 0/2 00:00\\n'); sys.stdout.flush(); "
        "sys.stdout.write('       0/2 1.0G 0.1 0.2 0.3 4 640: 100%|##########| 2/2 00:01\\n'); sys.stdout.flush(); "
        "sys.stdout.write('       1/2 1.0G 0.1 0.2 0.3 4 640: 100%|##########| 2/2 00:01\\n'); sys.stdout.flush(); "
        "sys.stdout.write('all 16 20 0.8 0.7 0.6 0.4\\n'); sys.stdout.flush(); "
        "sys.stdout.write('3 epochs completed in 0.100 hours.\\n'); sys.stdout.flush()"
    )

    run_command([sys.executable, "-u", "-c", program], cwd=tmp_path, log_path=log_path, progress_epochs=3)

    terminal = capsys.readouterr().out
    compact = log_path.read_text(encoding="utf-8")
    assert "[YOLOv5] epoch 1/3 running" in terminal
    assert "[YOLOv5] epoch 2/3 running" in terminal
    assert "[YOLOv5] validation: all 16 20 0.8 0.7 0.6 0.4" in terminal
    assert "epochs completed" in terminal
    assert compact.count("train: Scanning data...") == 1
    assert "100%|##########| 2/2 00:00" in compact
    assert compact.count("val: Scanning data...") == 1
    assert "100%|##########| 1/1 00:00" in compact
    assert compact.count("train-epoch") == 0
    assert "0/2 1.0G 0.1 0.2 0.3 4 640: 100%|##########| 2/2 00:01" in compact
    assert "all 16 20 0.8 0.7 0.6 0.4" in compact
    assert "coalesced warning: incomplete JPEG accepted read-only; count=2; by_source=train=1, val=1" in compact
    assert compact.count("warning example:") == 2
    assert ": 0%|" not in compact


def test_training_subprocess_full_log_mode_keeps_detailed_output(tmp_path: Path) -> None:
    log_path = tmp_path / "training-full.log"
    program = (
        "import sys; "
        "sys.stdout.write('0/2 detailed batch progress\\r'); sys.stdout.flush(); "
        "sys.stdout.write('1/2 another detailed batch progress\\r'); sys.stdout.flush()"
    )

    run_command([sys.executable, "-u", "-c", program], cwd=tmp_path, log_path=log_path, log_mode="full")

    detailed = log_path.read_text(encoding="utf-8")
    assert "detailed batch progress" in detailed
    assert "another detailed batch progress" in detailed


def test_read_only_verifier_suppresses_yolov5_image_write(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"original")

    class Transposed:
        def save(self, path, *_args, **_kwargs):
            Path(path).write_bytes(b"modified")

    image_ops = SimpleNamespace(exif_transpose=lambda _image: Transposed())

    def original_verify(args):
        image_ops.exif_transpose(object()).save(args[0], "JPEG")
        return (args[0], None, None, None, 0, 1, 0, 0, "WARNING: corrupt JPEG restored and saved")

    dataloaders = SimpleNamespace(ImageOps=image_ops, verify_image_label=original_verify)
    install_read_only_verifier(dataloaders)
    result = dataloaders.verify_image_label((str(image_path), "label.txt", "train: "))
    assert image_path.read_bytes() == b"original"
    assert result[-1].endswith("incomplete JPEG accepted read-only")


def test_train_entry_worker_can_inherit_source_root() -> None:
    entry = Path(__file__).parents[1] / "src" / "yolo_retraining" / "backends" / "yolov5" / "train_entry.py"
    source_root = Path(__file__).parents[1] / ".third_party" / "yolov5"
    if not source_root.is_dir():
        pytest.skip("fixed YOLOv5 source is installed by bootstrap")
    environment = os.environ.copy()
    environment["YOLO_RETRAINING_YOLOV5_ROOT"] = str(source_root)
    result = __import__("subprocess").run(
        [sys.executable, str(entry), "--help"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
