#!/usr/bin/env python3
"""Render the held-out distributed-evidence benchmark figure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "llm_task_benchmark" / "results" / "summary.json"
OUTPUT_BASENAME = ROOT / "images" / "fig7_distributed_evidence"

MODELS = ("DeepSeek-V4-Flash", "GPT-4.1-mini")
CONDITIONS = ("solo", "pairs", "star", "triads", "five_cliques")
CONDITION_LABELS = {
    "solo": "Solo",
    "pairs": "Pairs",
    "star": "Star",
    "triads": "Triads",
    "five_cliques": "5-cliques",
}
MODEL_STYLES = {
    "DeepSeek-V4-Flash": {"color": "#C54E3A", "marker": "o"},
    "GPT-4.1-mini": {"color": "#167D7F", "marker": "s"},
}
INK = "#242424"
GRID = "#D9D9D9"


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def bootstrap_ci(summary: dict, model: str, condition: str, metric: str) -> tuple[float, float]:
    record = summary["by_model_condition"][f"{model}::{condition}"]
    if metric == "plurality_correct":
        return tuple(record["group_plurality_accuracy_ci95"])
    values = [
        row[metric]
        for row in summary["records"]
        if row["model"] == model and row["condition"] == condition and row[metric] is not None
    ]
    seed_payload = f"{model}|{condition}|{metric}|fig7".encode()
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    )
    array = np.asarray(values, dtype=float)
    sampled = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return tuple(np.quantile(sampled, (0.025, 0.975)))


def plot_condition_metric(
    axis: plt.Axes,
    summary: dict,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    records = summary["records"]
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = (-0.17, 0.17)

    for model_index, model in enumerate(MODELS):
        style = MODEL_STYLES[model]
        for condition_index, condition in enumerate(CONDITIONS):
            values = np.asarray(
                [
                    row[metric]
                    for row in records
                    if row["model"] == model
                    and row["condition"] == condition
                    and row[metric] is not None
                ],
                dtype=float,
            )
            position = x[condition_index] + offsets[model_index]
            jitter = np.linspace(-0.052, 0.052, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=17,
                marker=style["marker"],
                color=style["color"],
                alpha=0.23,
                linewidths=0,
                zorder=1,
            )
            mean = float(values.mean())
            lower, upper = bootstrap_ci(summary, model, condition, metric)
            axis.errorbar(
                position,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=5.8,
                capsize=2.3,
                linewidth=1.2,
                zorder=3,
                label=model if condition_index == 0 else None,
            )

    axis.set_title(title, loc="left", pad=4, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=20, ha="right")
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_paired_gain(axis: plt.Axes, summary: dict) -> None:
    contrasts = summary["paired_group_accuracy_contrasts"]
    conditions = CONDITIONS[1:]
    y = np.arange(len(conditions), dtype=float)
    offsets = (-0.14, 0.14)

    for model_index, model in enumerate(MODELS):
        style = MODEL_STYLES[model]
        for condition_index, condition in enumerate(conditions):
            result = contrasts[f"{model}::{condition}_minus_solo"]
            mean = result["mean_group_accuracy_difference"]
            lower, upper = result["ci95"]
            axis.errorbar(
                mean,
                y[condition_index] + offsets[model_index],
                xerr=[[mean - lower], [upper - mean]],
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=5.8,
                capsize=2.3,
                linewidth=1.2,
                zorder=3,
                label=model if condition_index == 0 else None,
            )
            axis.text(
                min(mean + 0.035, 1.01),
                y[condition_index] + offsets[model_index],
                f"{mean * 100:.0f}",
                color=style["color"],
                fontsize=7.2,
                va="center",
            )

    axis.axvline(0, color=INK, linewidth=0.75)
    axis.set_xlim(-0.05, 1.08)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1.0), ("0", "25", "50", "75", "100"))
    axis.set_yticks(y, [CONDITION_LABELS[key] for key in conditions])
    axis.invert_yaxis()
    axis.set_xlabel("Gain over solo (percentage points)")
    axis.set_title("Paired gain over solo", loc="left", pad=4, fontweight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def draw_task_schematic(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Frozen distributed-evidence task", loc="left", pad=4, fontweight="bold")

    card_colors = ("#E6B4A7", "#D9C58C", "#A8CEBD", "#B7CBE6")
    card_x = (0.08, 0.27, 0.08, 0.27)
    card_y = (0.66, 0.66, 0.35, 0.35)
    exclusions = ("A B C", "D E F", "A D H", "B E H")
    for index, (x, y, exclusions_text) in enumerate(zip(card_x, card_y, exclusions), start=1):
        axis.add_patch(
            Rectangle(
                (x, y),
                0.15,
                0.17,
                facecolor=card_colors[(index - 1) % len(card_colors)],
                edgecolor=INK,
                linewidth=0.7,
            )
        )
        axis.text(
            x + 0.075,
            y + 0.128,
            f"Card {index}",
            ha="center",
            va="center",
            fontsize=5.35,
            fontweight="bold",
        )
        axis.text(
            x + 0.075,
            y + 0.076,
            "excludes",
            ha="center",
            va="center",
            fontsize=4.65,
        )
        axis.text(
            x + 0.075,
            y + 0.040,
            exclusions_text,
            ha="center",
            va="center",
            fontsize=5.25,
        )

    axis.text(0.25, 0.285, "Four illustrative cards of eight", ha="center", fontsize=6.1)
    agent_positions = (
        (0.07, 0.18),
        (0.18, 0.18),
        (0.29, 0.18),
        (0.40, 0.18),
        (0.07, 0.10),
        (0.18, 0.10),
        (0.29, 0.10),
        (0.40, 0.10),
    )
    for index, (x, y) in enumerate(agent_positions, start=1):
        axis.add_patch(Circle((x, y), 0.027, facecolor="white", edgecolor=INK, linewidth=0.8))
        axis.text(x, y, str(index), ha="center", va="center", fontsize=6.3)
    axis.text(0.25, 0.025, "Eight agents each receive one exclusion card", ha="center", fontsize=6.2)

    axis.add_patch(
        FancyArrowPatch(
            (0.48, 0.49),
            (0.62, 0.49),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=INK,
        )
    )
    axis.text(
        0.55,
        0.42,
        "lossless",
        ha="center",
        fontsize=6.1,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
    )
    axis.add_patch(Rectangle((0.65, 0.30), 0.27, 0.37, facecolor="#F4F4F4", edgecolor=INK, linewidth=0.8))
    axis.text(0.785, 0.59, "Pooled evidence", ha="center", va="center", fontsize=7.0, fontweight="bold")
    axis.text(0.785, 0.49, "A  B  C  D", ha="center", va="center", fontsize=6.5, color="#888888")
    axis.text(0.785, 0.42, "E  F  H", ha="center", va="center", fontsize=6.5, color="#888888")
    axis.text(0.785, 0.34, "G remains", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#C54E3A")
    axis.text(
        0.5,
        0.89,
        "One or two cards never identify the target; the full set does.",
        ha="center",
        va="center",
        fontsize=6.6,
    )


def render(summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.1,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(5.7, 4.50), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1),
        height_ratios=(1, 1),
        wspace=0.24,
        hspace=0.24,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    draw_task_schematic(axis_a)
    plot_condition_metric(
        axis_b,
        summary,
        "plurality_correct",
        "Held-out plurality accuracy",
        "Group accuracy",
    )
    plot_condition_metric(
        axis_c,
        summary,
        "solvable_agent_fraction",
        "Agents with sufficient evidence",
        "Agent fraction",
    )
    plot_paired_gain(axis_d, summary)

    for label, axis in zip(("a", "b", "c", "d"), (axis_a, axis_b, axis_c, axis_d)):
        axis.text(
            0.0,
            1.22,
            label,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )
    axis_b.legend(loc="lower right", frameon=False, handletextpad=0.35)

    # Matplotlib's PDF backend converts aware datetimes through the host locale.
    # Preserve the recorded UTC calendar date without introducing a local-date shift.
    created_at = datetime.combine(
        datetime.fromisoformat(summary["generated_at_utc"]).date(),
        time(hour=12),
    )
    pdf_metadata = {
        "Title": "Held-out distributed-evidence benchmark",
        "Creator": "fig7_distributed_evidence.py",
        "CreationDate": created_at,
    }
    png_metadata = {
        "Title": "Held-out distributed-evidence benchmark",
        "Software": "fig7_distributed_evidence.py",
    }
    OUTPUT_BASENAME.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_BASENAME.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=pdf_metadata,
    )
    figure.savefig(
        OUTPUT_BASENAME.with_suffix(".png"),
        dpi=450,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=png_metadata,
    )
    plt.close(figure)


def main() -> None:
    summary = load_summary()
    if summary.get("n_run_cells") != 160:
        raise RuntimeError("Fig. 7 requires the completed 160-cell held-out benchmark.")
    render(summary)


if __name__ == "__main__":
    main()
