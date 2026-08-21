from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _activate_source(root: Path) -> None:
    sys.path.insert(0, str(root.resolve()))


def _load_class_names(data_path: Path) -> dict[int, str]:
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, dict):
        return {int(class_id): str(class_name) for class_id, class_name in names.items()}
    return {index: str(class_name) for index, class_name in enumerate(names)}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _activate_source(args.root)
    import val
    import utils.dataloaders as dataloaders

    from yolo_retraining.backends.yolov5.artifacts import PredictionArtifactCollector, confidence_sweep
    from yolo_retraining.backends.yolov5.readonly_patch import install_read_only_verifier

    install_read_only_verifier(dataloaders)

    collector = None
    original_process_batch = val.process_batch
    original_ap_per_class = val.ap_per_class
    confidence_floor = 0.001
    captured_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    run_options = {
        "data": str(args.data),
        "weights": str(args.weights),
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "task": "val",
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": "val",
        "exist_ok": True,
        "half": args.half,
        "plots": False,
        "verbose": False,
        "conf_thres": confidence_floor,
    }
    artifact_dir = Path(args.artifacts_dir or args.project)
    if args.save_artifacts:
        collector = PredictionArtifactCollector(artifact_dir / "predictions.jsonl")
        callbacks = val.Callbacks()
        callbacks.register_action("on_val_batch_end", "collect_prediction_artifacts", collector.on_val_batch_end)

        def collect_process_batch(detections, labels, iouv):
            result = original_process_batch(detections, labels, iouv)
            collector.record_correct(result)
            return result

        def capture_ap_per_class(tp, conf, pred_cls, target_cls, *extra, **kwargs):
            captured_stats["stats"] = (
                np.asarray(tp).copy(),
                np.asarray(conf).copy(),
                np.asarray(pred_cls).copy(),
                np.asarray(target_cls).copy(),
            )
            return original_ap_per_class(tp, conf, pred_cls, target_cls, *extra, **kwargs)

        val.process_batch = collect_process_batch
        val.ap_per_class = capture_ap_per_class
        run_options["callbacks"] = callbacks

    try:
        results, maps, _ = val.run(**run_options)
    finally:
        if collector is not None:
            collector.close()
            val.process_batch = original_process_batch
            val.ap_per_class = original_ap_per_class

    if collector is not None:
        stats = captured_stats.get("stats", collector.raw_stats())
        sweep = confidence_sweep(
            *stats,
            names=_load_class_names(args.data),
            thresholds=args.confidence_thresholds,
            conf_floor=confidence_floor,
        )
        (artifact_dir / "confidence_sweep.json").write_text(
            json.dumps(sweep, indent=2),
            encoding="utf-8",
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


def gradient_score(args: argparse.Namespace) -> list[dict[str, Any]]:
    _activate_source(args.root)
    import torch
    import torch.nn as nn
    from models.experimental import attempt_load
    from utils.dataloaders import create_dataloader
    from utils.loss import ComputeLoss
    from utils.torch_utils import select_device

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model = attempt_load(str(args.weights), device=device, fuse=False).float()
    if not hasattr(model, "hyp"):
        default_hyp = args.root / "data" / "hyps" / "hyp.scratch-low.yaml"
        model.hyp = yaml.safe_load(default_hyp.read_text(encoding="utf-8"))
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    detection_head = model.model[-1]
    head_parameters = list(detection_head.parameters())
    for parameter in head_parameters:
        parameter.requires_grad_(True)
    stride = max(int(model.stride.max()), 32)
    dataloader, _ = create_dataloader(
        str(args.images),
        args.imgsz,
        1,
        stride,
        hyp=model.hyp,
        augment=False,
        rect=True,
        workers=0,
        shuffle=False,
        prefix="gradient-score: ",
    )
    compute_loss = ComputeLoss(model)
    records: list[dict[str, Any]] = []
    for images, targets, paths, _shapes in dataloader:
        images = images.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        model.zero_grad(set_to_none=True)
        predictions = model(images)
        loss, components = compute_loss(predictions, targets)
        loss.backward()
        squared_norm = torch.zeros(1, device=device)
        for parameter in head_parameters:
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().float().pow(2).sum()
        records.append(
            {
                "image_path": str(paths[0]),
                "gradient_norm": float(squared_norm.sqrt().item()),
                "loss": float(loss.detach().item()),
                "box_loss": float(components[0].item()),
                "objectness_loss": float(components[1].item()),
                "classification_loss": float(components[2].item()),
            }
        )
    return records


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
    evaluate_parser.add_argument("--save-artifacts", action="store_true")
    evaluate_parser.add_argument("--artifacts-dir", type=Path)
    evaluate_parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
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
    gradient_parser = commands.add_parser("gradient-score")
    gradient_parser.add_argument("--root", type=Path, required=True)
    gradient_parser.add_argument("--weights", type=Path, required=True)
    gradient_parser.add_argument("--images", type=Path, required=True)
    gradient_parser.add_argument("--imgsz", type=int, required=True)
    gradient_parser.add_argument("--device", required=True)
    gradient_parser.add_argument("--seed", type=int, required=True)
    gradient_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "evaluate":
        payload = evaluate(args)
    elif args.command == "inspect":
        payload = inspect_model(args)
    elif args.command == "predict":
        payload = predict(args)
    else:
        payload = gradient_score(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
