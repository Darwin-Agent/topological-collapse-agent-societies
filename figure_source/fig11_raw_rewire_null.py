#!/usr/bin/env python3
"""Render the fixed-margin raw-file rewiring check for Extended Data Fig. 6."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "figure_source" / "data" / "raw_rewire_null.json"
OUTDIR = ROOT / "images"
OUTPUT_STEM = "fig_raw_rewire_null"
PDF_CREATION_DATE = datetime(2026, 7, 31, 2, 0, 0)

OBSERVED = "#C74B45"
NULL = "#2F6FC0"
INK = "#242424"
MUTED = "#6C6C6C"
GRID = "#D7D7D7"
LIGHT = "#F4F4F4"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.14, y: float = 1.10) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=9.5,
        ha="left",
        va="top",
        color=INK,
    )


def draw_constraints(axis: plt.Axes, data: dict) -> None:
    null_model = data["null_model"]
    diagnostics = data["chain_length_diagnostic"]["results"]
    primary = {
        "accepted_swaps_per_incidence": null_model["accepted_swaps_per_incidence"],
        "null_samples": null_model["null_samples"],
        **data["null_overlap"],
    }
    chains = sorted(
        [*diagnostics, primary],
        key=lambda row: row["accepted_swaps_per_incidence"],
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    panel_label(axis, "a", x=-0.05)
    axis.text(
        0.00,
        0.97,
        "Fixed margins in every rewire",
        fontsize=8.5,
        fontweight="bold",
        va="top",
        color=INK,
    )

    rows = (
        ("Node hyperdegree sequence", "exactly retained"),
        ("Thread-size sequence", "exactly retained"),
        ("Node-thread duplicate", "rejected"),
    )
    for row_index, (label, value) in enumerate(rows):
        y = 0.75 - row_index * 0.17
        axis.plot((0.01, 0.07), (y, y), color=NULL, linewidth=2.2, solid_capstyle="round")
        axis.text(0.11, y + 0.025, label, fontsize=7.1, color=INK, va="center")
        axis.text(0.11, y - 0.065, value, fontsize=5.9, color=MUTED, va="center")

    axis.text(
        0.00,
        0.29,
        "Chain-length sensitivity",
        fontsize=6.6,
        fontweight="bold",
        color=INK,
        va="center",
    )
    axis.text(
        0.00,
        0.245,
        "Mean overlap  \u2022  central 95% interval",
        fontsize=5.35,
        color=MUTED,
        va="center",
    )
    domain_min, domain_max = 0.076, 0.080
    left, right = 0.29, 0.97

    def map_overlap(value: float) -> float:
        return left + (value - domain_min) / (domain_max - domain_min) * (right - left)

    for row_index, chain in enumerate(chains):
        y = 0.18 - row_index * 0.075
        interval_low, interval_high = chain["central_95_percent_interval"]
        mean = chain["mean"]
        axis.text(
            0.00,
            y,
            f"{chain['accepted_swaps_per_incidence']}\u00d7  ({chain['null_samples']})",
            fontsize=5.8,
            color=INK,
            va="center",
        )
        axis.plot(
            (map_overlap(interval_low), map_overlap(interval_high)),
            (y, y),
            color=NULL,
            linewidth=1.5,
            solid_capstyle="round",
        )
        axis.scatter(
            map_overlap(mean),
            y,
            s=16,
            color=NULL,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        axis.text(
            right,
            y + 0.018,
            f"{mean:.3f}",
            fontsize=5.05,
            color=MUTED,
            ha="right",
            va="bottom",
        )


def draw_overlap(axis: plt.Axes, data: dict) -> None:
    observed = data["observed"]["mean_edge_overlap"]
    null = data["null_overlap"]
    null_values = np.array([row["mean_edge_overlap"] for row in null["samples"]])
    interval_low, interval_high = null["central_95_percent_interval"]
    generator = np.random.default_rng(42)
    jitter = generator.uniform(-0.135, 0.135, size=len(null_values))

    panel_label(axis, "b", x=-0.13)
    axis.axvspan(interval_low, interval_high, color=NULL, alpha=0.13, zorder=0)
    axis.axvline(null["mean"], color=NULL, linewidth=1.1, linestyle="--", zorder=1)
    axis.scatter(
        null_values,
        jitter,
        s=15,
        color=NULL,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.35,
        zorder=2,
    )
    axis.scatter(
        observed,
        0.72,
        s=57,
        marker="D",
        color=OBSERVED,
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    axis.hlines(0.72, null["mean"], observed, color=GRID, linewidth=1.0, zorder=1)
    axis.text(
        observed,
        0.94,
        f"Observed\n{observed:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=OBSERVED,
        fontweight="bold",
    )
    axis.text(
        null["mean"],
        -0.20,
        f"Null mean {null['mean']:.3f}",
        ha="center",
        va="top",
        fontsize=6.0,
        color=NULL,
    )
    axis.text(
        0.0685,
        1.17,
        f"0 / {len(null_values)} rewires reached observed",
        ha="left",
        va="center",
        fontsize=6.0,
        color=INK,
    )
    axis.set_xlim(0.068, 0.128)
    axis.set_ylim(-0.32, 1.32)
    axis.set_yticks(())
    axis.set_xlabel("Mean edge overlap (Jaccard)")
    axis.set_title("Observed recurrence exceeds fixed-margin rewires", loc="left", pad=3, fontsize=8.4)
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)


def build_figure() -> None:
    data = json.loads(DATA_PATH.read_text())
    configure_style()
    figure = plt.figure(figsize=(5.35, 2.62))
    grid = figure.add_gridspec(1, 2, width_ratios=(0.80, 1.34), wspace=0.48)
    draw_constraints(figure.add_subplot(grid[0, 0]), data)
    draw_overlap(figure.add_subplot(grid[0, 1]), data)
    OUTDIR.mkdir(exist_ok=True)
    metadata = {
        "Title": "raw fixed margin rewiring",
        "Creator": "fig11_raw_rewire_null.py",
        "CreationDate": PDF_CREATION_DATE,
    }
    figure.savefig(
        OUTDIR / f"{OUTPUT_STEM}.pdf",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
        metadata=metadata,
    )
    figure.savefig(
        OUTDIR / f"{OUTPUT_STEM}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
        metadata={"Title": "raw fixed margin rewiring"},
    )
    plt.close(figure)


if __name__ == "__main__":
    build_figure()
