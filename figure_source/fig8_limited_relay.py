#!/usr/bin/env python3
"""Render the bandwidth-limited evidence-relay benchmark figure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "llm_relay_benchmark" / "results" / "summary.json"
OUTPUT_BASENAME = ROOT / "images" / "fig8_limited_relay"

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
MUTED = "#6C6C6C"
GRID = "#D9D9D9"
ACCESS = "#6750A4"


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def bootstrap_ci(values: list[float], *seed_parts: object) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_int("fig8_bootstrap", *seed_parts))
    sampled = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return tuple(np.quantile(sampled, (0.025, 0.975)))


def records_for(summary: dict, model: str, condition: str) -> list[dict]:
    return [
        row
        for row in summary["records"]
        if row["model"] == model and row["condition"] == condition
    ]


def draw_relay_rule(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(
        "Two-round card relay",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    card_colors = ("#E6B4A7", "#D9C58C", "#A8CEBD", "#B7CBE6")
    card_positions = (
        (0.04, 0.71),
        (0.145, 0.71),
        (0.04, 0.56),
        (0.145, 0.56),
        (0.04, 0.41),
        (0.145, 0.41),
        (0.04, 0.26),
        (0.145, 0.26),
    )
    for index, (x, y) in enumerate(card_positions, start=1):
        axis.add_patch(
            Rectangle(
                (x, y),
                0.078,
                0.105,
                facecolor=card_colors[(index - 1) % len(card_colors)],
                edgecolor=INK,
                linewidth=0.65,
            )
        )
        axis.text(x + 0.039, y + 0.0525, f"C{index}", ha="center", va="center", fontsize=7.0)
    axis.text(
        0.125,
        0.135,
        "8 agents\n1 card each",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        linespacing=1.10,
    )

    axis.add_patch(
        FancyArrowPatch(
            (0.27, 0.54),
            (0.40, 0.54),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.00,
            color=INK,
        )
    )
    axis.text(
        0.335,
        0.85,
        "2 synchronous rounds",
        ha="center",
        fontsize=6.5,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
    )
    centre = (0.56, 0.54)
    leaves = ((0.47, 0.72), (0.65, 0.72), (0.44, 0.48), (0.68, 0.48), (0.56, 0.30))
    for x, y in leaves:
        axis.plot([centre[0], x], [centre[1], y], color="#A7A7A7", linewidth=0.85)
        axis.add_patch(Circle((x, y), 0.032, facecolor="white", edgecolor=INK, linewidth=0.70))
    axis.add_patch(Circle(centre, 0.045, facecolor="#F2F2F2", edgecolor=INK, linewidth=0.8))
    axis.text(0.56, 0.54, "group", ha="center", va="center", fontsize=5.6)
    axis.text(
        0.56,
        0.135,
        "membership protocol\nsets delivery",
        ha="center",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        linespacing=1.12,
    )

    axis.add_patch(
        FancyArrowPatch(
            (0.72, 0.54),
            (0.80, 0.54),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.00,
            color=INK,
        )
    )
    axis.add_patch(Rectangle((0.80, 0.37), 0.18, 0.30, facecolor="#F4F4F4", edgecolor=INK, linewidth=0.7))
    axis.text(0.89, 0.58, "Unique", ha="center", fontsize=7.2, fontweight="bold")
    axis.text(0.89, 0.50, "answer", ha="center", fontsize=7.2, fontweight="bold")
    axis.text(0.89, 0.43, ">=4 cards\nrequired", ha="center", va="center", fontsize=5.2, color=MUTED, linespacing=1.0)


def plot_access(axis: plt.Axes, summary: dict) -> None:
    diagnostics = summary["deterministic_access"]
    x = np.arange(len(CONDITIONS), dtype=float)
    for index, condition in enumerate(CONDITIONS):
        values = list(
            diagnostics[condition]["task_level_sufficient_agent_fraction"].values()
        )
        jitter = np.linspace(-0.052, 0.052, len(values))
        axis.scatter(
            index + jitter,
            values,
            s=15,
            color=ACCESS,
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )
        mean = float(np.mean(values))
        lower, upper = bootstrap_ci(values, condition, "access")
        axis.errorbar(
            index,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color=ACCESS,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5.8,
            capsize=2.3,
            linewidth=1.2,
            zorder=3,
        )
        axis.text(index, min(mean + 0.085, 1.035), f"{mean:.2f}", ha="center", fontsize=7.3, color=ACCESS)

    axis.set_title("Evidence access (16 tasks)", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Agents with sufficient evidence")
    axis.set_xticks(x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=20, ha="right")
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_accuracy(axis: plt.Axes, summary: dict) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = (-0.16, 0.16)
    for model_index, model in enumerate(MODELS):
        style = MODEL_STYLES[model]
        for condition_index, condition in enumerate(CONDITIONS):
            values = [
                float(row["plurality_correct"])
                for row in records_for(summary, model, condition)
            ]
            position = x[condition_index] + offsets[model_index]
            jitter = np.linspace(-0.052, 0.052, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=17,
                marker=style["marker"],
                color=style["color"],
                alpha=0.22,
                linewidths=0,
                zorder=1,
            )
            mean = float(np.mean(values))
            lower, upper = bootstrap_ci(values, model, condition, "accuracy")
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

    axis.set_title("Held-out group accuracy", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Group accuracy")
    axis.set_xticks(x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=20, ha="right")
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False, handletextpad=0.35)


def plot_paired_gain(axis: plt.Axes, summary: dict) -> None:
    conditions = CONDITIONS[1:]
    y = np.arange(len(conditions), dtype=float)
    offsets = (-0.14, 0.14)
    for model_index, model in enumerate(MODELS):
        style = MODEL_STYLES[model]
        for condition_index, condition in enumerate(conditions):
            result = summary["paired_group_accuracy_contrasts"][f"{model}::{condition}_minus_solo"]
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
                fontsize=7.3,
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


def plot_readout(axis: plt.Axes, summary: dict) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = (-0.16, 0.16)
    for model_index, model in enumerate(MODELS):
        style = MODEL_STYLES[model]
        for condition_index, condition in enumerate(CONDITIONS):
            values = [
                row["individual_accuracy"]
                for row in records_for(summary, model, condition)
            ]
            position = x[condition_index] + offsets[model_index]
            jitter = np.linspace(-0.052, 0.052, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=17,
                marker=style["marker"],
                color=style["color"],
                alpha=0.22,
                linewidths=0,
                zorder=1,
            )
            mean = float(np.mean(values))
            lower, upper = bootstrap_ci(values, model, condition, "individual")
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
            )
    axis.set_title("Final individual accuracy", loc="left", pad=4, fontweight="bold")
    axis.set_ylabel("Agent accuracy")
    axis.set_xticks(x, [CONDITION_LABELS[key] for key in CONDITIONS], rotation=20, ha="right")
    axis.set_ylim(-0.06, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


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
    figure = plt.figure(figsize=(5.7, 4.70), constrained_layout=True)
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

    draw_relay_rule(axis_a)
    plot_access(axis_b, summary)
    plot_accuracy(axis_c, summary)
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

    metadata = {
        "Title": "Bandwidth-limited distributed-evidence benchmark",
        "Creator": "fig8_limited_relay.py",
        "CreationDate": datetime(2026, 7, 30, 12, 0, 0),
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
        metadata={"Title": "Bandwidth-limited distributed-evidence benchmark"},
    )
    plt.close(figure)


def main() -> None:
    render(load_summary())


if __name__ == "__main__":
    main()
