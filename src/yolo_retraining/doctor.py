from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


def collect_environment(project_root: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version.split()[0], "python_ok": sys.version_info[:2] == (3, 11)}
    for package in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "PyYAML"):
        try:
            report[f"package:{package}"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[f"package:{package}"] = None
    report["torch_ok"] = str(report.get("package:torch", "")).split("+")[0] == "2.6.0"
    report["torchvision_ok"] = str(report.get("package:torchvision", "")).split("+")[0] == "0.21.0"
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
        import ultralytics

        report["ultralytics_ok"] = ultralytics.__version__ == "8.4.102"
    except Exception as error:  # pragma: no cover - environment dependent
        report["ultralytics_error"] = repr(error)
        report["ultralytics_ok"] = False
    if project_root is not None:
        import yolo_retraining

        package_path = Path(yolo_retraining.__file__).resolve()
        report["package_path"] = str(package_path)
        report["package_path_ok"] = project_root.resolve() in package_path.parents
    return report


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the yolo-retraining runtime.")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(arguments)
    report = collect_environment(args.project_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    required = (report.get("python_ok"), report.get("ultralytics_ok"), report.get("torch_ok"), report.get("torchvision_ok"))
    if args.project_root is not None:
        required = (*required, report.get("package_path_ok"))
    return 0 if all(required) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
