from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(thresholds) != 3 or thresholds != sorted(set(thresholds)):
        raise argparse.ArgumentTypeError("exactly three unique ascending thresholds are required")
    return thresholds


def validate_protocol(
    summary: dict[str, Any],
    thresholds: list[float],
    *,
    aggressive_max_retained: float,
    conservative_min_retained: float,
) -> dict[str, Any]:
    threshold_results = summary.get("thresholds")
    if not isinstance(threshold_results, dict):
        raise ValueError("redundancy summary has no thresholds mapping")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        result = threshold_results.get(str(threshold))
        if not isinstance(result, dict):
            raise ValueError(f"redundancy summary has no result for threshold {threshold}")
        retained = float(result["retained_fraction"])
        rows.append(
            {
                "threshold": threshold,
                "retained_fraction": retained,
                "effective_train_images": int(result["effective_train_images"]),
                "original_train_images": int(result["original_train_images"]),
                "layout": result["layout"],
            }
        )
    retained_values = [row["retained_fraction"] for row in rows]
    if retained_values != sorted(retained_values):
        raise ValueError(f"retained fractions must increase with threshold: {retained_values}")
    if retained_values[0] >= aggressive_max_retained:
        raise ValueError(
            f"aggressive threshold retained {retained_values[0]:.3%}, expected below {aggressive_max_retained:.1%}"
        )
    if retained_values[-1] < conservative_min_retained:
        raise ValueError(
            f"conservative threshold retained {retained_values[-1]:.3%}, expected at least {conservative_min_retained:.1%}"
        )
    return {
        "status": "valid",
        "aggressive_max_retained": aggressive_max_retained,
        "conservative_min_retained": conservative_min_retained,
        "thresholds": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the three-threshold redundancy protocol before training.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--thresholds", type=parse_thresholds, required=True)
    parser.add_argument("--aggressive-max-retained", type=float, default=0.5)
    parser.add_argument("--conservative-min-retained", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = validate_protocol(
        summary,
        args.thresholds,
        aggressive_max_retained=args.aggressive_max_retained,
        conservative_min_retained=args.conservative_min_retained,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
