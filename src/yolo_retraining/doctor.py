from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from yolo_retraining.backends.yolov5.source import resolve_yolov5_root, validate_yolov5_source


def collect_environment(project_root: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version.split()[0], "python_ok": sys.version_info[:2] == (3, 10)}
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "PyYAML",
        "opencv-python",
        "scipy",
        "pandas",
        "seaborn",
        "thop",
        "GitPython",
        "ipython",
        "setuptools",
    )
    for package in packages:
        try:
            report[f"package:{package}"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[f"package:{package}"] = None
    report["torch_ok"] = str(report.get("package:torch", "")).split("+")[0] == "2.2.2"
    report["torchvision_ok"] = str(report.get("package:torchvision", "")).split("+")[0] == "0.17.2"
    report["numpy_ok"] = report.get("package:numpy") == "1.26.4"
    try:
        import torch

        report.update(
            torch_cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpu_count=torch.cuda.device_count(),
            gpus=[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        )
    except Exception as error:  # pragma: no cover - environment dependent
        report["torch_error"] = repr(error)
    try:
        source = validate_yolov5_source()
        report.update(yolov5_root=source["root"], yolov5_commit=source["commit"], yolov5_tag=source["tag"], yolov5_source_ok=True)
    except Exception as error:
        report.update(yolov5_root=str(resolve_yolov5_root()), yolov5_source_ok=False, yolov5_error=repr(error))
    weights = Path(report["yolov5_root"]) / "yolov5s.pt"
    report["yolov5s_weights"] = str(weights)
    report["yolov5s_weights_ok"] = weights.is_file() and weights.stat().st_size > 0
    try:
        report["package:ultralytics"] = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        report["package:ultralytics"] = None
    if project_root is not None:
        import yolo_retraining

        package_path = Path(yolo_retraining.__file__).resolve()
        report["package_path"] = str(package_path)
        report["package_path_ok"] = project_root.resolve() in package_path.parents
    return report


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the original YOLOv5s yolo-retraining runtime.")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(arguments)
    report = collect_environment(args.project_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    required = (
        report.get("python_ok"),
        report.get("torch_ok"),
        report.get("torchvision_ok"),
        report.get("numpy_ok"),
        all(report.get(f"package:{package}") for package in ("Pillow", "PyYAML", "opencv-python", "scipy", "pandas", "seaborn", "thop", "GitPython", "ipython", "setuptools")),
        report.get("yolov5_source_ok"),
        report.get("yolov5s_weights_ok"),
    )
    if args.project_root is not None:
        required = (*required, report.get("package_path_ok"))
    return 0 if all(required) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
