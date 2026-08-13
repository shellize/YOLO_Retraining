from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _activate_source(root: Path) -> None:
    sys.path.insert(0, str(root.resolve()))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _activate_source(args.root)
    import val
    import utils.dataloaders as dataloaders

    from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier

    install_read_only_verifier(dataloaders)

    results, maps, _ = val.run(
        data=str(args.data),
        weights=str(args.weights),
        batch_size=args.batch,
        imgsz=args.imgsz,
        task="val",
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name="val",
        exist_ok=True,
        half=args.half,
        plots=False,
        verbose=True,
    )
    precision, recall, map50, map50_95 = results[:4]
    return {
        "map50_95": float(map50_95),
        "map50": float(map50),
        "precision": float(precision),
        "recall": float(recall),
        "per_class_ap": [float(value) for value in maps],
    }


def inspect_model(args: argparse.Namespace) -> dict[str, Any]:
    _activate_source(args.root)
    from models.experimental import attempt_load

    model = attempt_load(str(args.weights), device="cpu", fuse=False)
    detect = model.model[-1]
    nc = int(detect.nc)
    no = int(detect.no)
    return {
        "model_class": type(model).__name__,
        "detect_class": type(detect).__name__,
        "anchor_based": hasattr(detect, "anchors") and int(detect.na) > 0,
        "anchors_per_scale": int(detect.na),
        "class_count": nc,
        "outputs_per_anchor": no,
        "has_objectness": no == nc + 5,
    }


def predict(args: argparse.Namespace) -> dict[str, Any]:
    _activate_source(args.root)
    from models.experimental import attempt_load

    model = attempt_load(str(args.weights), device=args.device).autoshape()
    paths = json.loads(args.images.read_text(encoding="utf-8"))
    results = model(paths, size=args.imgsz)
    return {
        str(index): tensor.detach().cpu().tolist()
        for index, tensor in enumerate(results.pred)
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--weights", type=Path, required=True)
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--project", type=Path, required=True)
    evaluate_parser.add_argument("--batch", type=int, required=True)
    evaluate_parser.add_argument("--imgsz", type=int, required=True)
    evaluate_parser.add_argument("--device", required=True)
    evaluate_parser.add_argument("--workers", type=int, required=True)
    evaluate_parser.add_argument("--half", action="store_true")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--root", type=Path, required=True)
    inspect_parser.add_argument("--weights", type=Path, required=True)
    inspect_parser.add_argument("--output", type=Path, required=True)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--root", type=Path, required=True)
    predict_parser.add_argument("--weights", type=Path, required=True)
    predict_parser.add_argument("--images", type=Path, required=True)
    predict_parser.add_argument("--imgsz", type=int, required=True)
    predict_parser.add_argument("--device", required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "evaluate":
        payload = evaluate(args)
    elif args.command == "inspect":
        payload = inspect_model(args)
    else:
        payload = predict(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
