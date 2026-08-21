from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import SelectionPolicy


COMMON_PARAMS = {"cumulative"}
TEACHER_PARAMS = COMMON_PARAMS | {"teacher_result", "confidence", "iou"}


def _validate_params(params: Mapping[str, Any], allowed: set[str], policy: str) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unknown {policy} params: {sorted(unknown)}")


def _label_count(record: Mapping[str, Any]) -> int:
    label_path = Path(str(record["label_path"]))
    if not label_path.is_file():
        return 0
    count = 0
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}: expected 5 fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}") from error
        if class_id < 0 or any(not 0.0 <= value <= 1.0 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        count += 1
    return count


def _checkpoint_from_result(result_dir: Path) -> tuple[Path, str]:
    result_path = result_dir / "task_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"teacher TaskResult does not exist: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"teacher task is not completed: {result_dir}")
    checkpoint = result_dir / payload["artifacts"]["best_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"teacher best checkpoint does not exist: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return checkpoint.resolve(), digest


def _teacher_analysis(policy: SelectionPolicy, context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_dir = Path(str(policy.params.get("teacher_result", ""))).resolve()
    checkpoint, digest = _checkpoint_from_result(result_dir)
    confidence = float(policy.params.get("confidence", 0.25))
    iou = float(policy.params.get("iou", 0.5))
    if not 0.0 <= confidence <= 1.0 or not 0.0 < iou <= 1.0:
        raise ValueError("teacher confidence/iou must be in [0,1] and (0,1]")
    records = context["backend"].analyze_samples(
        {
            "config": context["config"],
            "registry": context["registry"],
            "sample_ids": list(context["current_ids"]),
            "checkpoint": str(checkpoint),
            "confidence": confidence,
            "iou": iou,
            "output_dir": Path(context["output_dir"]) / "selection" / "teacher_analysis",
        }
    )
    metadata = {
        "teacher_result": str(result_dir),
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": digest,
        "confidence": confidence,
        "iou": iou,
    }
    return records, metadata


def _prior_selection(context: Mapping[str, Any]) -> list[str]:
    parent_result = context["config"]["task"].get("parent_result")
    if not parent_result:
        raise ValueError("cumulative selection requires task.parent_result")
    selected_path = Path(str(parent_result)).resolve() / "data" / "selected_ids.txt"
    if not selected_path.is_file():
        raise FileNotFoundError(f"parent selected_ids does not exist: {selected_path}")
    prior = sorted({line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()})
    invalid = set(prior) - set(context["candidate_ids"])
    if invalid:
        raise ValueError(f"parent selection contains IDs outside the current candidate scope: {sorted(invalid)[:5]}")
    return prior


def _finish(policy: SelectionPolicy, context: Mapping[str, Any], current: list[str], records: list[dict[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    current = sorted(set(current))
    cumulative = bool(policy.params.get("cumulative", False))
    history = _prior_selection(context) if cumulative else []
    selected = sorted(set(history) | set(current))
    return {
        "selected_ids": selected,
        "groups": {"history": history, "current": current},
        "records": records,
        "metadata": {**dict(metadata), "cumulative": cumulative, "current_selected_count": len(current)},
    }


def _hard_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if int(record["fp"]) > 0 or int(record["fn"]) > 0]


class PositiveOnlySelection(SelectionPolicy):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_params(self.params, COMMON_PARAMS, "positive_only")

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        records = []
        current = []
        for sample_id in sorted(set(context["current_ids"])):
            count = _label_count(context["registry"]["records"][sample_id])
            selected = count > 0
            records.append({"sample_id": sample_id, "gt_count": count, "selected": selected, "reason": "has_gt" if selected else "zero_box"})
            if selected:
                current.append(sample_id)
        return _finish(self, context, current, records, {"policy": "positive_only"})


class PredPositiveSelection(SelectionPolicy):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_params(self.params, TEACHER_PARAMS, "pred_positive")

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        records, teacher = _teacher_analysis(self, context)
        current = [record["sample_id"] for record in records if int(record["prediction_count"]) > 0]
        for record in records:
            record["selected"] = int(record["prediction_count"]) > 0
            record["reason"] = "teacher_detection" if record["selected"] else "no_teacher_detection"
        return _finish(self, context, current, records, {"policy": "pred_positive", **teacher})


class ErrorHardSelection(SelectionPolicy):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_params(self.params, TEACHER_PARAMS, "error_hard")

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        records, teacher = _teacher_analysis(self, context)
        hard = _hard_records(records)
        hard_ids = {record["sample_id"] for record in hard}
        for record in records:
            record["selected"] = record["sample_id"] in hard_ids
            record["reason"] = "fp_or_fn" if record["selected"] else "no_error"
        return _finish(self, context, list(hard_ids), records, {"policy": "error_hard", **teacher})


class RandomTopKSelection(SelectionPolicy):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_params(self.params, TEACHER_PARAMS, "random_topk")

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        analysis, teacher = _teacher_analysis(self, context)
        budget = len(_hard_records(analysis))
        candidates = sorted(set(context["current_ids"]))
        rng = np.random.default_rng(int(context["seed"]))
        selected = sorted(str(value) for value in rng.choice(candidates, size=budget, replace=False).tolist()) if budget else []
        selected_set = set(selected)
        analysis_by_id = {record["sample_id"]: record for record in analysis}
        records = [
            {
                **analysis_by_id[sample_id],
                "selected": sample_id in selected_set,
                "reason": "random_topk" if sample_id in selected_set else "not_sampled",
            }
            for sample_id in candidates
        ]
        return _finish(self, context, selected, records, {"policy": "random_topk", "seed": int(context["seed"]), "selection_budget": budget, "budget_source": "error_hard", **teacher})


class GradNormTopKSelection(SelectionPolicy):
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_params(self.params, TEACHER_PARAMS, "gradnorm_topk")

    def select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        analysis, teacher = _teacher_analysis(self, context)
        budget = len(_hard_records(analysis))
        score_records = context["backend"].gradient_scores(
            {
                "config": context["config"],
                "registry": context["registry"],
                "sample_ids": list(context["current_ids"]),
                "checkpoint": teacher["teacher_checkpoint"],
                "output_dir": Path(context["output_dir"]) / "selection" / "gradient_scoring",
            }
        )
        analysis_by_id = {record["sample_id"]: record for record in analysis}
        score_records = [{**analysis_by_id[record["sample_id"]], **record} for record in score_records]
        ranked = sorted(score_records, key=lambda record: (-float(record["gradient_norm"]), str(record["sample_id"])))
        selected = [record["sample_id"] for record in ranked[:budget]]
        selected_set = set(selected)
        for rank, record in enumerate(ranked, start=1):
            record["rank"] = rank
            record["selected"] = record["sample_id"] in selected_set
            record["reason"] = "gradnorm_topk" if record["selected"] else "below_budget"
        return _finish(self, context, selected, ranked, {"policy": "gradnorm_topk", "selection_budget": budget, "budget_source": "error_hard", "gradient_scope": "detection_head", "augmentation": "disabled", **teacher})
