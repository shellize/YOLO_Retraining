from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOWED_PARAMS = {"batch", "imgsz", "device", "workers", "cache", "optimizer", "patience", "amp", "best_metric"}
BEST_METRICS = {"map50", "yolov5_fitness"}


def device_argument(device: Any) -> str:
    if isinstance(device, list):
        if not device:
            raise ValueError("backend.params.device list cannot be empty")
        return ",".join(str(item) for item in device)
    return str(device)


def validate_training_params(config: Mapping[str, Any]) -> None:
    params = config["backend"].get("params", {})
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        raise ValueError(f"unknown YOLOv5 backend params: {sorted(unknown)}")
    batch = int(params.get("batch", 0))
    imgsz = int(params.get("imgsz", 0))
    workers = int(params.get("workers", 0))
    if batch <= 0 or imgsz <= 0 or workers < 0:
        raise ValueError("backend.params batch/imgsz must be positive and workers must be non-negative")
    if params.get("amp", True) is not True:
        raise ValueError("original YOLOv5 v7.0 controls AMP automatically; backend.params.amp=false is not supported")
    optimizer = str(params.get("optimizer", "SGD"))
    if optimizer not in {"SGD", "Adam", "AdamW"}:
        raise ValueError("backend.params.optimizer must be SGD, Adam, or AdamW")
    best_metric = str(params.get("best_metric", "map50"))
    if best_metric not in BEST_METRICS:
        raise ValueError("backend.params.best_metric must be map50 or yolov5_fitness")
    device = params.get("device", "cpu")
    if isinstance(device, list) and len(device) > 1 and batch % len(device) != 0:
        raise ValueError("backend.params.batch must be divisible by the number of DDP devices")


def build_train_command(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    checkpoint: Path,
    data_yaml: Path,
    output_dir: Path,
) -> list[str]:
    validate_training_params(config)
    params = config["backend"]["params"]
    device = params.get("device", "cpu")
    devices = device if isinstance(device, list) else [device]
    command = [sys.executable]
    if len(devices) > 1:
        command.extend(["-m", "torch.distributed.run", "--nproc_per_node", str(len(devices))])
    command.extend(
        [
            str(Path(__file__).with_name("train_entry.py")),
            "--yolo-source-root",
            str(source_root),
            "--weights",
            str(checkpoint),
            "--data",
            str(data_yaml),
            "--epochs",
            str(int(config["budget"]["value"])),
            "--batch-size",
            str(int(params["batch"])),
            "--imgsz",
            str(int(params["imgsz"])),
            "--device",
            device_argument(device),
            "--workers",
            str(int(params.get("workers", 4))),
            "--optimizer",
            str(params.get("optimizer", "SGD")),
            "--patience",
            str(int(params.get("patience", 100))),
            "--best-metric",
            str(params.get("best_metric", "map50")),
            "--seed",
            str(int(config["task"]["seed"])),
            "--project",
            str(output_dir),
            "--name",
            "train",
            "--exist-ok",
            "--noplots",
        ]
    )
    cache = params.get("cache", False)
    if cache:
        command.append("--cache")
        if isinstance(cache, str):
            command.append(cache)
    return command


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_BAR = re.compile(r"\b\d{1,3}%\|.*\|\s*\d+/\d+\b")
TRAIN_EPOCH_PROGRESS = re.compile(r"^\s*(\d+)/\d+\s+")
INCOMPLETE_JPEG_WARNING = "incomplete JPEG accepted read-only"
WARNING_SAMPLE_LIMIT = 3
LOG_MODES = frozenset({"compact", "full"})


class _CompactLogWriter:
    """Keep useful subprocess events while collapsing transient YOLOv5 output."""

    def __init__(self, log: Any, *, progress_epochs: int | None, shown_epochs: set[int]) -> None:
        self.log = log
        self.progress_epochs = progress_epochs
        self.shown_epochs = shown_epochs
        self.record: list[str] = []
        self.pending_progress: tuple[str, str] | None = None
        self.progress_records = 0
        self.progress_lines = 0
        self.warning_count = 0
        self.warning_counts: dict[str, int] = {}
        self.warning_samples: list[str] = []

    def feed(self, character: str) -> None:
        if character in {"\r", "\n"}:
            self._consume("".join(self.record), transient=character == "\r")
            self.record = []
        else:
            self.record.append(character)

    def finish(self) -> None:
        if self.record:
            self._consume("".join(self.record), transient=False)
        self._flush_progress()
        if self.progress_records > self.progress_lines:
            self._write(
                f"[YOLOv5] compacted progress updates: {self.progress_records} -> {self.progress_lines} lines"
            )
        if self.warning_count:
            by_source = ", ".join(
                f"{source}={count}" for source, count in sorted(self.warning_counts.items())
            )
            self._write(
                f"[YOLOv5] coalesced warning: {INCOMPLETE_JPEG_WARNING}; "
                f"count={self.warning_count}; by_source={by_source}"
            )
            for sample in self.warning_samples:
                self._write(f"[YOLOv5] warning example: {sample}")

    def _consume(self, raw_line: str, *, transient: bool) -> None:
        line = ANSI_ESCAPE.sub("", raw_line).strip()
        if not line:
            return
        _print_coarse_progress(line, self.progress_epochs, self.shown_epochs)
        if INCOMPLETE_JPEG_WARNING in line:
            self.warning_count += 1
            source = line.split(":", 1)[0].strip() or "unknown"
            self.warning_counts[source] = self.warning_counts.get(source, 0) + 1
            if len(self.warning_samples) < WARNING_SAMPLE_LIMIT:
                self.warning_samples.append(line)
            return
        progress_key = _progress_key(line)
        if progress_key is not None:
            self.progress_records += 1
            if progress_key == "generic":
                return
            if self.pending_progress is not None and self.pending_progress[0] != progress_key:
                self._flush_progress()
            self.pending_progress = (progress_key, line)
            return
        if transient:
            return
        self._flush_progress()
        self._write(line)

    def _flush_progress(self) -> None:
        if self.pending_progress is None:
            return
        _, line = self.pending_progress
        self._write(line)
        self.progress_lines += 1
        self.pending_progress = None

    def _write(self, line: str) -> None:
        self.log.write(f"{line}\n")


def _progress_key(line: str) -> str | None:
    if not PROGRESS_BAR.search(line):
        return None
    if "Scanning " in line:
        source = line.split(":", 1)[0].strip() or "unknown"
        return f"scan-{source}"
    if re.match(r"^\s*Class\s+Images\b", line):
        return "validation"
    match = TRAIN_EPOCH_PROGRESS.match(line)
    if match:
        return f"train-epoch-{match.group(1)}"
    return "generic"


def _resolve_log_mode(log_mode: str | None) -> str:
    resolved = (log_mode or os.environ.get("YOLO_RETRAINING_LOG_MODE", "compact")).strip().lower()
    if resolved not in LOG_MODES:
        choices = ", ".join(sorted(LOG_MODES))
        raise ValueError(f"YOLOv5 log mode must be one of: {choices}")
    return resolved


def _stream_full_output(stream: Any, log: Any, *, progress_epochs: int | None, shown_epochs: set[int]) -> None:
    record = ""
    for character in iter(lambda: stream.read(1), ""):
        log.write(character)
        if character in {"\r", "\n"}:
            _print_coarse_progress(record, progress_epochs, shown_epochs)
            record = ""
        else:
            record += character
    if record:
        _print_coarse_progress(record, progress_epochs, shown_epochs)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    progress_epochs: int | None = None,
    log_mode: str | None = None,
) -> None:
    """Run a detector subprocess and write either compact or full output.

    Compact mode is the default because YOLOv5 progress updates and immutable
    JPEG warnings are not useful as one log line per update. Set
    ``YOLO_RETRAINING_LOG_MODE=full`` (or pass ``log_mode="full"``) when the
    complete subprocess stream is needed for debugging.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_mode = _resolve_log_mode(log_mode)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            raise RuntimeError("failed to capture YOLOv5 subprocess output")
        shown_epochs: set[int] = set()
        if resolved_log_mode == "full":
            _stream_full_output(process.stdout, log, progress_epochs=progress_epochs, shown_epochs=shown_epochs)
        else:
            compact = _CompactLogWriter(log, progress_epochs=progress_epochs, shown_epochs=shown_epochs)
            for character in iter(lambda: process.stdout.read(1), ""):
                compact.feed(character)
            compact.finish()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"YOLOv5 subprocess failed with exit code {returncode}; see {log_path}")


def _print_coarse_progress(raw_line: str, epochs: int | None, shown_epochs: set[int]) -> None:
    line = ANSI_ESCAPE.sub("", raw_line).strip()
    if not line:
        return
    if epochs is not None:
        match = re.search(rf"(?<!\d)(\d+)/{epochs - 1}(?!\d)", line)
        if match:
            epoch = int(match.group(1))
            if epoch not in shown_epochs:
                shown_epochs.add(epoch)
                print(f"[YOLOv5] epoch {epoch + 1}/{epochs} running", flush=True)
    if re.match(r"^all\s+\d+\s+\d+\s+", line):
        print(f"[YOLOv5] validation: {line}", flush=True)
    elif "epochs completed in" in line:
        print(f"[YOLOv5] {line}", flush=True)


def read_training_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key.strip(): _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _number(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def estimated_optimizer_steps(sample_count: int, batch: int, epochs: int) -> int:
    return math.ceil(sample_count / max(1, batch)) * epochs
