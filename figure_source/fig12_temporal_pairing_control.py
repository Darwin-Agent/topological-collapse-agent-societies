#!/usr/bin/env python3
"""Render the prospective held-out temporal-pairing relay control figure.

The two requested route labels have identical task-level plurality outcomes.
The empirical panels therefore show their shared profile once and do not
present the routes as independent replications.
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
SUMMARY_PATH = ROOT / "figure_source" / "data" / "temporal_pairing_control_summary.json"
OUTPUT_BASENAME = ROOT / "images" / "fig_temporal_pairing_control"

MODELS = ("DeepSeek-V4-Flash", "GPT-4.1-mini")
CONDITIONS = ("repeated_pairs", "rotating_pairs")
CONDITION_LABELS = {
    "repeated_pairs": "Repeated pairs",
    "rotating_pairs": "Rotating pairs",
}
CONDITION_STYLES = {
    "repeated_pairs": {"color": "#C5523C", "marker": "o"},
    "rotating_pairs": {"color": "#167D7F", "marker": "s"},
}
INK = "#242424"
MUTED = "#696969"
GRID = "#D9D9D9"
PALETTE = ("#C5523C", "#C69430", "#3F907A", "#4C79A8")


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def bootstrap_ci(values: list[float], *seed_parts: object) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_int("temporal_pairing_figure", *seed_parts))
    sampled = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, (0.025, 0.975)))


def records_for(summary: dict, model: str, condition: str) -> list[dict]:
    return sorted(
        (
            row
            for row in summary["records"]
            if row["model"] == model and row["condition"] == condition
        ),
        key=lambda row: row["task_id"],
    )


def shared_task_accuracy(summary: dict, condition: str) -> list[float]:
    profiles = [
        [float(row["plurality_correct"]) for row in records_for(summary, model, condition)]
        for model in MODELS
    ]
    if profiles[0] != profiles[1]:
        raise ValueError(
            "Route-specific task outcomes differ; do not render a shared profile."
        )
    return profiles[0]


def task_access(summary: dict, condition: str) -> list[float]:
    values = summary["deterministic_access"][condition][
        "task_level_sufficient_agent_fraction"
    ]
    return [float(values[task_id]) for task_id in sorted(values)]


def draw_round(
    axis: plt.Axes,
    x_center: float,
    y_base: float,
    pairs: tuple[tuple[int, int], ...],
    *,
    highlight_recurrence: bool,
) -> None:
    pair_rows = (0.39, 0.15, -0.09, -0.33)
    for pair_index, (left_label, right_label) in enumerate(pairs):
        y = y_base + pair_rows[pair_index]
        color = PALETTE[pair_index]
        alpha = 0.72 if highlight_recurrence else 0.54
        axis.plot(
            (x_center - 0.13, x_center + 0.13),
            (y, y),
            color=color,
            linewidth=2.0,
            alpha=alpha,
            solid_capstyle="round",
            zorder=1,
        )
        axis.scatter(
            (x_center - 0.13, x_center + 0.13),
            (y, y),
            s=31,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=2,
        )
        axis.text(x_center - 0.13, y, str(left_label), ha="center", va="center", fontsize=5.2)
        axis.text(x_center + 0.13, y, str(right_label), ha="center", va="center", fontsize=5.2)


def draw_schedule(axis: plt.Axes) -> None:
    repeated = ((0, 1), (2, 3), (4, 5), (6, 7))
    rotating = (
        ((0, 1), (2, 3), (4, 5), (6, 7)),
        ((0, 2), (1, 3), (4, 6), (5, 7)),
        ((0, 3), (1, 2), (4, 7), (5, 6)),
        ((0, 4), (1, 5), (2, 6), (3, 7)),
    )
    for round_index in range(4):
        x = round_index + 0.58
        axis.text(x, 2.18, f"R{round_index + 1}", ha="center", fontsize=6.6, color=MUTED)
        draw_round(axis, x, 1.45, repeated, highlight_recurrence=True)
        draw_round(axis, x, 0.34, rotating[round_index], highlight_recurrence=False)

    axis.text(
        -0.05,
        1.45,
        "Repeated\npairs",
        ha="right",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=CONDITION_STYLES["repeated_pairs"]["color"],
    )
    axis.text(
        -0.05,
        0.34,
        "Rotating\npairs",
        ha="right",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=CONDITION_STYLES["rotating_pairs"]["color"],
    )
    axis.text(
        2.08,
        -0.18,
        "Matched: 4 dyads/round, 1 broadcast/agent/round, 4 rounds",
        ha="center",
        va="top",
        fontsize=5.7,
        color=MUTED,
    )
    axis.set_title("Matched schedule", loc="left", pad=4, fontweight="bold")
    axis.set_xlim(-0.56, 4.25)
    axis.set_ylim(-0.28, 2.42)
    axis.axis("off")


def draw_mean_ci(
    axis: plt.Axes,
    position: float,
    values: list[float],
    *,
    condition: str,
    seed_parts: tuple[object, ...],
) -> None:
    mean = float(np.mean(values))
    lower, upper = bootstrap_ci(values, *seed_parts)
    style = CONDITION_STYLES[condition]
    axis.errorbar(
        position,
        mean,
        yerr=[[mean - lower], [upper - mean]],
        fmt=style["marker"],
        color=style["color"],
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=6.0,
        capsize=2.5,
        linewidth=1.25,
        zorder=4,
    )


def plot_access(axis: plt.Axes, summary: dict) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    for index, condition in enumerate(CONDITIONS):
        values = task_access(summary, condition)
        style = CONDITION_STYLES[condition]
        jitter = np.linspace(-0.055, 0.055, len(values))
        axis.scatter(
            index + jitter,
            values,
            s=15,
            marker=style["marker"],
            color=style["color"],
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )
        draw_mean_ci(
            axis,
            index,
            values,
            condition=condition,
            seed_parts=("access", condition),
        )
        mean = float(np.mean(values))
        axis.text(
            index,
            max(mean + 0.085, 0.085),
            f"{mean:.2f}",
            ha="center",
            fontsize=7.2,
            color=style["color"],
            fontweight="bold",
        )

    axis.set_title("Evidence access", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Agents with sufficient evidence")
    axis.set_xticks(x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=15, ha="right")
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_accuracy(axis: plt.Axes, summary: dict) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    for index, condition in enumerate(CONDITIONS):
        values = shared_task_accuracy(summary, condition)
        style = CONDITION_STYLES[condition]
        jitter = np.linspace(-0.055, 0.055, len(values))
        axis.scatter(
            index + jitter,
            values,
            s=16,
            marker=style["marker"],
            color=style["color"],
            alpha=0.26,
            linewidths=0,
            zorder=1,
        )
        group_summary = summary["by_model_condition"][
            f"{MODELS[0]}::{condition}"
        ]
        mean = float(group_summary["group_plurality_accuracy"])
        lower, upper = group_summary["group_plurality_accuracy_ci95"]
        axis.errorbar(
            index,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt=style["marker"],
            color=style["color"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=6.0,
            capsize=2.5,
            linewidth=1.25,
            zorder=4,
        )
        axis.text(
            index,
            min(mean + 0.09, 1.035),
            f"{mean:.2f}",
            ha="center",
            fontsize=7.2,
            color=style["color"],
            fontweight="bold",
        )

    axis.set_title("Group accuracy", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Plurality accuracy")
    axis.set_xticks(x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=15, ha="right")
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
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=5.5,
                label="Shared route profile",
            )
        ],
        loc="lower right",
        frameon=False,
        handletextpad=0.35,
    )


def plot_paired_contrast(axis: plt.Axes, summary: dict) -> None:
    repeated = shared_task_accuracy(summary, "repeated_pairs")
    rotating = shared_task_accuracy(summary, "rotating_pairs")
    differences = [after - before for before, after in zip(repeated, rotating)]
    first = summary["paired_group_accuracy_contrasts"][
        f"{MODELS[0]}::rotating_pairs_minus_repeated_pairs"
    ]
    second = summary["paired_group_accuracy_contrasts"][
        f"{MODELS[1]}::rotating_pairs_minus_repeated_pairs"
    ]
    if first != second:
        raise ValueError("Route-specific paired contrasts differ; render both explicitly.")

    mean = float(first["mean_group_accuracy_difference"])
    lower, upper = first["ci95"]
    y = np.linspace(-0.10, 0.10, len(differences))
    axis.scatter(
        differences,
        y,
        s=18,
        color="#A8A8A8",
        alpha=0.55,
        linewidths=0,
        zorder=1,
    )
    axis.errorbar(
        mean,
        0.0,
        xerr=[[mean - lower], [upper - mean]],
        fmt="s",
        color=CONDITION_STYLES["rotating_pairs"]["color"],
        markerfacecolor=CONDITION_STYLES["rotating_pairs"]["color"],
        markeredgewidth=1.0,
        markersize=6.1,
        capsize=2.8,
        linewidth=1.35,
        zorder=3,
    )
    axis.text(
        mean,
        0.18,
        f"+{mean * 100:.1f} pp\n95% CI {lower * 100:.1f} to {upper * 100:.1f}",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=CONDITION_STYLES["rotating_pairs"]["color"],
        fontweight="bold",
        linespacing=1.15,
    )
    axis.axvline(0, color=INK, linewidth=0.75)
    axis.set_xlim(-0.08, 1.04)
    axis.set_ylim(-0.27, 0.47)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1.0), ("0", "25", "50", "75", "100"))
    axis.set_yticks(())
    axis.set_xlabel("Rotating - repeated (percentage points)")
    axis.set_title("Paired contrast (16 tasks)", loc="left", pad=4, fontweight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)


def render(summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.1,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(5.85, 5.00), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, wspace=0.30, hspace=0.34)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    draw_schedule(axis_a)
    plot_access(axis_b, summary)
    plot_accuracy(axis_c, summary)
    plot_paired_contrast(axis_d, summary)
    for label, axis, x, y in (
        ("a", axis_a, -0.15, 1.13),
        ("b", axis_b, -0.18, 1.13),
        ("c", axis_c, -0.18, 1.13),
        ("d", axis_d, -0.18, 1.13),
    ):
        axis.text(
            x,
            y,
            label,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )

    metadata = {
        "Title": "Prospective held-out temporal-pairing relay control",
        "Creator": "fig12_temporal_pairing_control.py",
        "CreationDate": datetime(2026, 7, 31, 12, 0, 0),
    }
    OUTPUT_BASENAME.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_BASENAME.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=metadata,
    )
    figure.savefig(
        OUTPUT_BASENAME.with_suffix(".png"),
        dpi=450,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Title": "Prospective held-out temporal-pairing relay control"},
    )
    plt.close(figure)


def main() -> None:
    render(load_summary())


if __name__ == "__main__":
    main()
