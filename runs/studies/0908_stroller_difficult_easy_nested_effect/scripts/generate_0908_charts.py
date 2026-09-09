"""Generate comparison charts for the completed 0908 stroller studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[2]
DIFFICULT_CURVE = (
    PROJECT_ROOT
    / "runs/studies/0908_difficult_stroller_window10_learning_curve/result/learning_curve/metrics_by_split_seed.csv"
)
EASY_CURVE = (
    PROJECT_ROOT
    / "runs/studies/0907_stroller_window10_learning_curve/result/learning_curve/metrics_by_split_seed.csv"
)
NESTED_COMPARISON = STUDY_ROOT / "result/evaluation/comparison_by_split.csv"

DIFFICULT_OUTPUT = PROJECT_ROOT / "runs/studies/0908_difficult_stroller_window10_learning_curve/result/easy_vs_difficult"
NESTED_OUTPUT = STUDY_ROOT / "result/cross_evaluation"
SEEDS = (41, 42)
STAGES = tuple(range(8))
GROUPS = ("test1", "test2", "testhard")
METRICS = (
    ("map50_95", "mAP@0.5:0.95"),
    ("map50", "mAP@0.5"),
    ("precision", "Precision"),
    ("recall", "Recall"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing {key!r} in row: {row}")
    return float(value)


def integer(row: dict[str, str], key: str) -> int:
    return int(number(row, key))


def curve_rows(path: Path, *, has_split_mode: bool) -> dict[int, list[dict[str, Any]]]:
    rows = read_csv(path)
    selected: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    for row in rows:
        if has_split_mode and row.get("split_mode") != "random":
            continue
        seed = integer(row, "split_seed")
        if seed not in selected:
            continue
        selected[seed].append(
            {
                "stage": integer(row, "stage"),
                "x": integer(row, "cumulative_train_images"),
                "map50_95": number(row, "test_map50_95"),
                "map50": number(row, "test_map50"),
                "test_images": integer(row, "test_images"),
            }
        )
    for seed, values in selected.items():
        values.sort(key=lambda item: item["stage"])
        if [item["stage"] for item in values] != list(STAGES):
            raise ValueError(f"expected stages {STAGES} for seed {seed} in {path}: {values}")
    return selected


def aggregate_curve(curves: dict[int, list[dict[str, Any]]], metric: str) -> list[dict[str, Any]]:
    result = []
    for stage in STAGES:
        values = [curves[seed][stage][metric] for seed in SEEDS]
        x_values = [curves[seed][stage]["x"] for seed in SEEDS]
        result.append(
            {
                "stage": stage,
                "x_by_seed": x_values,
                "x_mean": sum(x_values) / len(x_values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return result


def validate_curve_pair(
    easy: dict[int, list[dict[str, Any]]],
    difficult: dict[int, list[dict[str, Any]]],
) -> None:
    for curves, label in ((easy, "easy"), (difficult, "difficult")):
        test_sizes = {item["test_images"] for seed in SEEDS for item in curves[seed]}
        if len(test_sizes) != 1:
            raise ValueError(f"{label} test size differs across rows: {test_sizes}")
        for seed in SEEDS:
            if len(curves[seed]) != len(STAGES):
                raise ValueError(f"{label} seed {seed} does not have eight curve points")


def write_curve_csv(
    easy: dict[int, list[dict[str, Any]]],
    difficult: dict[int, list[dict[str, Any]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "easy_train_images_s41",
        "easy_train_images_s42",
        "difficult_train_images_s41",
        "difficult_train_images_s42",
        "easy_map50_95_mean",
        "difficult_map50_95_mean",
        "difficult_minus_easy_map50_95_by_stage",
        "easy_map50_mean",
        "difficult_map50_mean",
        "difficult_minus_easy_map50_by_stage",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        easy_95 = aggregate_curve(easy, "map50_95")
        difficult_95 = aggregate_curve(difficult, "map50_95")
        easy_50 = aggregate_curve(easy, "map50")
        difficult_50 = aggregate_curve(difficult, "map50")
        for stage in STAGES:
            writer.writerow(
                {
                    "stage": stage,
                    "easy_train_images_s41": easy[41][stage]["x"],
                    "easy_train_images_s42": easy[42][stage]["x"],
                    "difficult_train_images_s41": difficult[41][stage]["x"],
                    "difficult_train_images_s42": difficult[42][stage]["x"],
                    "easy_map50_95_mean": easy_95[stage]["mean"],
                    "difficult_map50_95_mean": difficult_95[stage]["mean"],
                    "difficult_minus_easy_map50_95_by_stage": difficult_95[stage]["mean"] - easy_95[stage]["mean"],
                    "easy_map50_mean": easy_50[stage]["mean"],
                    "difficult_map50_mean": difficult_50[stage]["mean"],
                    "difficult_minus_easy_map50_by_stage": difficult_50[stage]["mean"] - easy_50[stage]["mean"],
                }
            )


def plot_easy_difficult(
    easy: dict[int, list[dict[str, Any]]],
    difficult: dict[int, list[dict[str, Any]]],
) -> dict[str, str]:
    DIFFICULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    easy_color = "#176B87"
    difficult_color = "#D95D39"
    seed_colors = {41: "#6B9FB2", 42: "#A5C7D1"}
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), sharex=True)
    fig.patch.set_facecolor("#F7F9FB")
    tick_labels = [
        f"S{stage}\n{int(easy[41][stage]['x'])}/{int(difficult[41][stage]['x'])}"
        for stage in STAGES
    ]

    for axis, metric, title in zip(axes, ("map50_95", "map50"), ("mAP@0.5:0.95", "mAP@0.5")):
        axis.set_facecolor("white")
        for curves, label, color in (
            (easy, "stroller_easy", easy_color),
            (difficult, "stroller_difficult", difficult_color),
        ):
            aggregate = aggregate_curve(curves, metric)
            axis.fill_between(
                STAGES,
                [item["min"] for item in aggregate],
                [item["max"] for item in aggregate],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            for seed in SEEDS:
                axis.plot(
                    STAGES,
                    [item[metric] for item in curves[seed]],
                    color=seed_colors[seed] if label == "stroller_easy" else "#E5A18E",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                )
            axis.plot(
                STAGES,
                [item["mean"] for item in aggregate],
                color=color,
                marker="o",
                linewidth=2.6,
                markersize=5.5,
                label=f"{label} mean",
            )
            final = aggregate[-1]["mean"]
            axis.annotate(
                f"{final:.3f}",
                (STAGES[-1], final),
                xytext=(-4, 9 if label == "stroller_easy" else -17),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9.5,
                fontweight="bold",
            )
        axis.set_title(title, fontweight="bold", pad=12)
        axis.set_xlabel("Training stage (easy/difficult cumulative images)")
        axis.set_xticks(STAGES)
        axis.set_xticklabels(tick_labels)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Test AP (best checkpoint)")
    fig.suptitle(
        "stroller_easy vs stroller_difficult — learning curves",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Random split seeds 41/42 | mean line with min–max band | easy test=80 images; difficult test=104 images",
        ha="center",
        color="#52616B",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93), w_pad=3.0)
    png = DIFFICULT_OUTPUT / "learning_curve_easy_vs_difficult.png"
    svg = DIFFICULT_OUTPUT / "learning_curve_easy_vs_difficult.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"png": png.name, "svg": svg.name}


def read_nested_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    selected = []
    for row in rows:
        seed = integer(row, "split_seed")
        group = row.get("test_group")
        if seed not in SEEDS or group not in GROUPS:
            continue
        selected.append(
            {
                "split_seed": seed,
                "test_group": group,
                "sample_count": integer(row, "sample_count"),
                **{
                    f"m1_{metric}": number(row, f"m1_{metric}")
                    for metric, _ in METRICS
                },
                **{
                    f"m2_{metric}": number(row, f"m2_{metric}")
                    for metric, _ in METRICS
                },
                **{
                    f"delta_{metric}": number(row, f"m2_minus_m1_{metric}")
                    for metric, _ in METRICS
                },
            }
        )
    expected = len(SEEDS) * len(GROUPS)
    if len(selected) != expected:
        raise ValueError(f"expected {expected} nested comparison rows, found {len(selected)}")
    return selected


def nested_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row["test_group"] == group]
        summary[group] = {
            "sample_count": {
                "min": min(row["sample_count"] for row in group_rows),
                "max": max(row["sample_count"] for row in group_rows),
            }
        }
        for metric, _ in METRICS:
            for prefix in ("m1", "m2", "delta"):
                values = [row[f"{prefix}_{metric}"] for row in group_rows]
                summary[group][f"{prefix}_{metric}"] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
    return summary


def bars_with_range(
    axis: Any,
    x: np.ndarray,
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    metric: str,
    prefix: str,
    offset: float,
    color: str,
    label: str,
) -> None:
    values = [row[f"{prefix}_{metric}"]["mean"] for row in rows]
    lower = [row[f"{prefix}_{metric}"]["mean"] - row[f"{prefix}_{metric}"]["min"] for row in rows]
    upper = [row[f"{prefix}_{metric}"]["max"] - row[f"{prefix}_{metric}"]["mean"] for row in rows]
    axis.bar(
        x + offset,
        values,
        width=0.32,
        color=color,
        alpha=0.9,
        label=label,
        yerr=np.array([lower, upper]),
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#38434A"},
    )
    for seed_index, seed in enumerate(SEEDS):
        seed_rows = [
            row
            for row in all_rows
            if row["test_group"] == rows[seed_index]["test_group"]
            and row["split_seed"] == seed
        ]
        if not seed_rows:
            continue
        axis.scatter(
            x[seed_index] + offset,
            seed_rows[0][f"{prefix}_{metric}"],
            color="#172033",
            edgecolors="white",
            linewidths=0.7,
            s=28,
            zorder=4,
        )

def plot_nested(rows: list[dict[str, Any]]) -> dict[str, str]:
    NESTED_OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped = nested_summary(rows)
    grouped_rows = [
        {
            "test_group": group,
            "sample_count": grouped[group]["sample_count"],
            **{
                f"{prefix}_{metric}": grouped[group][f"{prefix}_{metric}"]
                for metric, _ in METRICS
                for prefix in ("m1", "m2", "delta")
            },
        }
        for group in GROUPS
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    fig.patch.set_facecolor("#F7F9FB")
    x = np.arange(len(GROUPS), dtype=float)
    group_labels = []
    for row in grouped_rows:
        count = row["sample_count"]
        count_label = str(int(count["min"])) if count["min"] == count["max"] else f"{int(count['min'])}–{int(count['max'])}"
        group_labels.append(f"{row['test_group']}\n(n={count_label})")

    for axis, (metric, title) in zip(axes.flat, METRICS):
        bars_with_range(axis, x, grouped_rows, rows, metric, "m1", -0.17, "#176B87", "m1 difficult")
        bars_with_range(axis, x, grouped_rows, rows, metric, "m2", 0.17, "#D95D39", "m2 easy")
        axis.set_title(title, fontweight="bold", pad=12)
        axis.set_xticks(x)
        axis.set_xticklabels(group_labels)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "0908 nested effect — cross-evaluation metrics",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "m1 trained on train1 difficult; m2 trained on train2 easy projection | bars: seed mean ± min–max; dots: seeds 41/42",
        ha="center",
        color="#52616B",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94), h_pad=3.0, w_pad=2.4)
    png = NESTED_OUTPUT / "nested_cross_evaluation_metrics.png"
    svg = NESTED_OUTPUT / "nested_cross_evaluation_metrics.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), sharex=True)
    fig.patch.set_facecolor("#F7F9FB")
    delta_colors = {"map50_95": "#176B87", "map50": "#D95D39"}
    for axis, (metric, title) in zip(axes, METRICS[:2]):
        means = [row[f"delta_{metric}"]["mean"] for row in grouped_rows]
        lower = [row[f"delta_{metric}"]["mean"] - row[f"delta_{metric}"]["min"] for row in grouped_rows]
        upper = [row[f"delta_{metric}"]["max"] - row[f"delta_{metric}"]["mean"] for row in grouped_rows]
        axis.errorbar(
            x,
            means,
            yerr=np.array([lower, upper]),
            fmt="o-",
            color=delta_colors[metric],
            linewidth=2.4,
            markersize=6,
            capsize=5,
            label="m2 − m1",
        )
        for seed in SEEDS:
            ordered = []
            for group in GROUPS:
                match = next(row for row in rows if row["split_seed"] == seed and row["test_group"] == group)
                ordered.append(match[f"delta_{metric}"])
            axis.plot(x, ordered, linestyle=":", marker="x", color="#52616B", alpha=0.65, linewidth=1.2)
        axis.axhline(0, color="#38434A", linewidth=1.0)
        axis.set_title(f"Δ {title} (m2 − m1)", fontweight="bold", pad=12)
        axis.set_xticks(x)
        axis.set_xticklabels(group_labels)
        all_values = [row[f"delta_{metric}"] for row in rows]
        lower_domain = min(all_values + [0.0])
        upper_domain = max(all_values + [0.0])
        padding = max((upper_domain - lower_domain) * 0.18, 0.03)
        axis.set_ylim(lower_domain - padding, upper_domain + padding)
        axis.set_ylabel("Difference in score")
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "0908 nested effect — AP difference",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Positive values mean easy-only m2 is higher; negative values mean difficult-trained m1 is higher",
        ha="center",
        color="#52616B",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93), w_pad=3.0)
    delta_png = NESTED_OUTPUT / "nested_cross_evaluation_ap_delta.png"
    delta_svg = NESTED_OUTPUT / "nested_cross_evaluation_ap_delta.svg"
    fig.savefig(delta_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(delta_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "metrics_png": png.name,
        "metrics_svg": svg.name,
        "delta_png": delta_png.name,
        "delta_svg": delta_svg.name,
    }


def main() -> int:
    easy = curve_rows(EASY_CURVE, has_split_mode=False)
    difficult = curve_rows(DIFFICULT_CURVE, has_split_mode=True)
    validate_curve_pair(easy, difficult)
    curve_outputs = plot_easy_difficult(easy, difficult)
    write_curve_csv(easy, difficult, DIFFICULT_OUTPUT / "learning_curve_easy_vs_difficult_by_stage.csv")
    nested_rows = read_nested_rows(NESTED_COMPARISON)
    nested_outputs = plot_nested(nested_rows)

    summary = {
        "status": "completed",
        "sources": {
            "easy_curve": relative(EASY_CURVE),
            "difficult_curve": relative(DIFFICULT_CURVE),
            "nested_comparison": relative(NESTED_COMPARISON),
        },
        "curve_comparison": {
            "seeds": list(SEEDS),
            "stage_alignment": True,
            "easy_test_images": 80,
            "difficult_test_images": 104,
            "outputs": curve_outputs,
        },
        "nested_effect": {
            "seeds": list(SEEDS),
            "groups": list(GROUPS),
            "outputs": nested_outputs,
        },
    }
    (DIFFICULT_OUTPUT / "summary.json").write_text(
        json.dumps(summary["curve_comparison"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (NESTED_OUTPUT / "summary.json").write_text(
        json.dumps(summary["nested_effect"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
