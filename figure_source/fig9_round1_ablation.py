#!/usr/bin/env python3
"""Render the matched one-round relay-capacity ablation figure.

The two requested route labels have identical task-level plurality outcomes in
both the one- and two-round records. The figure therefore shows the shared
task-level profile once rather than presenting overlapping route-specific
series as independent replications.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
ONE_ROUND_PATH = ROOT / "llm_relay_round1_ablation" / "results" / "summary.json"
TWO_ROUND_PATH = ROOT / "llm_relay_benchmark" / "results" / "summary.json"
OUTPUT_BASENAME = ROOT / "images" / "fig9_round1_ablation"

MODELS = ("DeepSeek-V4-Flash", "GPT-4.1-mini")
CONDITIONS = ("solo", "pairs", "star", "triads", "five_cliques")
GAIN_CONDITIONS = CONDITIONS[1:]
CONDITION_LABELS = {
    "solo": "Solo",
    "pairs": "Pairs",
    "star": "Star",
    "triads": "Triads",
    "five_cliques": "5-cliques",
}
ROUND_STYLES = {
    "One round": {"offset": -0.07, "filled": True, "color": "#C5523C"},
    "Two rounds": {"offset": 0.07, "filled": False, "color": "#167D7F"},
}
CONDITION_COLORS = {
    "pairs": "#6D829C",
    "star": "#B85B48",
    "triads": "#3F907A",
    "five_cliques": "#A57824",
}
INK = "#242424"
MUTED = "#6C6C6C"
GRID = "#D9D9D9"


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def bootstrap_ci(values: list[float], *seed_parts: object) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_int("round1_ablation", *seed_parts))
    sampled = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return tuple(np.quantile(sampled, (0.025, 0.975)))


def records_for(summary: dict, model: str, condition: str) -> list[dict]:
    return sorted(
        (
            row
            for row in summary["records"]
            if row["model"] == model and row["condition"] == condition
        ),
        key=lambda row: row["task_id"],
    )


def task_access(summary: dict, condition: str) -> list[float]:
    values = summary["deterministic_access"][condition][
        "task_level_sufficient_agent_fraction"
    ]
    return [float(values[task_id]) for task_id in sorted(values)]


def task_accuracy(summary: dict, condition: str) -> list[float]:
    """Return the common task-level profile and guard against silent pooling."""
    profiles = [
        [
            float(row["plurality_correct"])
            for row in records_for(summary, model, condition)
        ]
        for model in MODELS
    ]
    if profiles[0] != profiles[1]:
        raise ValueError(
            "Route-specific plurality outcomes differ; render route-specific series."
        )
    return profiles[0]


def paired_accuracy_change(
    one_round: dict, two_round: dict, condition: str
) -> list[float]:
    return [
        after - before
        for before, after in zip(
            task_accuracy(one_round, condition),
            task_accuracy(two_round, condition),
        )
    ]


def draw_mean_ci(
    axis: plt.Axes,
    position: float,
    values: list[float],
    *,
    color: str,
    filled: bool,
    seed_parts: tuple[object, ...],
) -> None:
    mean = float(np.mean(values))
    lower, upper = bootstrap_ci(values, *seed_parts)
    axis.errorbar(
        position,
        mean,
        yerr=[[mean - lower], [upper - mean]],
        fmt="o",
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgewidth=1.15,
        markersize=6.0,
        capsize=2.6,
        linewidth=1.25,
        zorder=4,
    )


def plot_access(axis: plt.Axes, one_round: dict, two_round: dict) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    for index, condition in enumerate(CONDITIONS):
        first = task_access(one_round, condition)
        second = task_access(two_round, condition)
        for first_value, second_value in zip(first, second):
            axis.plot(
                (
                    index + ROUND_STYLES["One round"]["offset"],
                    index + ROUND_STYLES["Two rounds"]["offset"],
                ),
                (first_value, second_value),
                color="#CECECE",
                linewidth=0.6,
                alpha=0.65,
                zorder=1,
            )
        for label, values in (("One round", first), ("Two rounds", second)):
            style = ROUND_STYLES[label]
            draw_mean_ci(
                axis,
                index + style["offset"],
                values,
                color=style["color"],
                filled=style["filled"],
                seed_parts=("access", label, condition),
            )

    axis.set_title("Sufficient evidence after relay", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Agent fraction")
    axis.set_xticks(
        x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=20, ha="right"
    )
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=ROUND_STYLES["One round"]["color"],
                markeredgecolor=ROUND_STYLES["One round"]["color"],
                markersize=6.3,
                label="1 relay round",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=ROUND_STYLES["Two rounds"]["color"],
                markersize=6.3,
                label="2 relay rounds",
            ),
        ],
        loc="upper left",
        frameon=False,
        handletextpad=0.35,
    )


def plot_accuracy(axis: plt.Axes, one_round: dict, two_round: dict) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    for condition_index, condition in enumerate(CONDITIONS):
        for label, summary in (("One round", one_round), ("Two rounds", two_round)):
            style = ROUND_STYLES[label]
            values = task_accuracy(summary, condition)
            position = x[condition_index] + style["offset"]
            jitter = np.linspace(-0.026, 0.026, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=14,
                marker="o",
                color=style["color"],
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
            draw_mean_ci(
                axis,
                position,
                values,
                color=style["color"],
                filled=style["filled"],
                seed_parts=("accuracy", label, condition),
            )

    axis.set_title("Final group accuracy", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Plurality accuracy")
    axis.set_xticks(
        x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=20, ha="right"
    )
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_paired_gain(axis: plt.Axes, one_round: dict, two_round: dict) -> None:
    y = np.arange(len(GAIN_CONDITIONS), dtype=float)
    for index, condition in enumerate(GAIN_CONDITIONS):
        gains = paired_accuracy_change(one_round, two_round, condition)
        mean = float(np.mean(gains))
        lower, upper = bootstrap_ci(gains, "paired_gain", condition)
        color = CONDITION_COLORS[condition]
        axis.errorbar(
            mean,
            y[index],
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgewidth=1.0,
            markersize=5.8,
            capsize=2.6,
            linewidth=1.25,
            zorder=3,
        )
        axis.text(
            min(mean + 0.018, 0.415),
            y[index],
            f"+{mean * 100:.1f}",
            color=color,
            fontsize=7.2,
            fontweight="bold",
            va="center",
        )

    axis.axvline(0, color=INK, linewidth=0.75)
    axis.set_xlim(-0.04, 0.44)
    axis.set_xticks((0, 0.125, 0.25, 0.375), ("0", "12.5", "25", "37.5"))
    axis.set_yticks(y, [CONDITION_LABELS[key] for key in GAIN_CONDITIONS])
    axis.invert_yaxis()
    axis.set_xlabel("Two rounds - one (percentage points)")
    axis.set_title("Matched accuracy change", loc="left", pad=4, fontweight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def render(one_round: dict, two_round: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.1,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(5.7, 5.05), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.92, 1.0))
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])

    plot_access(axis_a, one_round, two_round)
    plot_accuracy(axis_b, one_round, two_round)
    plot_paired_gain(axis_c, one_round, two_round)

    label_positions = ((-0.08, 1.08), (-0.18, 1.08), (-0.27, 1.08))
    for label, axis, (x, y) in zip(
        ("a", "b", "c"),
        (axis_a, axis_b, axis_c),
        label_positions,
    ):
        axis.text(
            x,
            y,
            label,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
        )
    metadata = {
        "Title": "Matched one-round relay capacity ablation",
        "Creator": "fig9_round1_ablation.py",
        "CreationDate": datetime(2026, 7, 30, 12, 0, 0),
    }
    figure.savefig(
        OUTPUT_BASENAME.with_suffix(".pdf"),
        facecolor="white",
        pad_inches=0.03,
        metadata=metadata,
    )
    figure.savefig(
        OUTPUT_BASENAME.with_suffix(".png"),
        dpi=450,
        facecolor="white",
        pad_inches=0.03,
        metadata={"Title": "Matched one-round relay capacity ablation"},
    )
    plt.close(figure)


def main() -> None:
    render(load_json(ONE_ROUND_PATH), load_json(TWO_ROUND_PATH))


if __name__ == "__main__":
    main()
