#!/usr/bin/env python3
"""Render the non-primary raw-file consistency check for Extended Data Fig. 5."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "figure_source" / "data" / "raw_snapshot_check.json"
OUTDIR = ROOT / "images"
OUTPUT_STEM = "fig_snapshot_consistency"
PDF_CREATION_DATE = datetime(2026, 7, 31, 1, 0, 0)

ARCHIVE = "#C74B45"
CHECK = "#2F6FC0"
INK = "#242424"
MUTED = "#6C6C6C"
GRID = "#D7D7D7"
LIGHT = "#F5F5F5"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
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


def draw_ledger(axis: plt.Axes, data: dict) -> None:
    primary = data["archived_primary_input"]["result"]
    alternate = data["alternative_raw_pairing"]["result"]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    panel_label(axis, "a", x=-0.04)
    axis.text(
        0.00,
        0.96,
        "Matched rule, independent raw-file check",
        fontsize=8.5,
        fontweight="bold",
        va="top",
        color=INK,
    )

    rows = (
        (
            ARCHIVE,
            "Archived primary input",
            "frozen cross-platform summary",
            primary,
        ),
        (
            CHECK,
            "Alternative raw-file check",
            "lnajt posts + MoltNet comments",
            alternate,
        ),
    )
    for index, (color, title, source, result) in enumerate(rows):
        x0 = index * 0.505
        axis.add_patch(
            Rectangle(
                (x0, 0.25),
                0.49,
                0.50,
                facecolor="white",
                edgecolor=GRID,
                linewidth=0.8,
            )
        )
        axis.add_patch(Rectangle((x0, 0.25), 0.026, 0.50, facecolor=color, edgecolor="none"))
        axis.text(x0 + 0.060, 0.635, title, fontsize=7.3, fontweight="bold", color=INK)
        axis.text(x0 + 0.060, 0.515, source, fontsize=5.9, color=MUTED)
        axis.text(
            x0 + 0.060,
            0.365,
            f"{result['n_hyperedges']:,} hyperedges",
            fontsize=6.7,
            color=color,
            fontweight="bold",
        )
        axis.text(
            x0 + 0.060,
            0.290,
            f"{result['n_nodes']:,} participants",
            fontsize=6.7,
            color=color,
            fontweight="bold",
        )

    axis.add_patch(Rectangle((0.00, 0.035), 1.00, 0.110, facecolor=LIGHT, edgecolor="none"))
    axis.text(
        0.50,
        0.090,
        "Top 50,000 posts by recorded comment count | 60-min reply window | non-singletons",
        ha="center",
        va="center",
        fontsize=5.7,
        color=INK,
    )


def draw_metric(axis: plt.Axes, label: str, archive: float, alternate: float, y_limits: tuple[float, float], formatter) -> None:
    axis.plot((0, 1), (archive, alternate), color=GRID, linewidth=1.2, zorder=1)
    axis.scatter((0, 1), (archive, alternate), s=40, color=(ARCHIVE, CHECK), edgecolor="white", linewidth=0.7, zorder=2)
    margin = (y_limits[1] - y_limits[0]) * 0.055
    for x, value, color in ((0, archive, ARCHIVE), (1, alternate, CHECK)):
        axis.text(x, min(value + margin, y_limits[1] - margin * 0.2), formatter(value), ha="center", va="bottom", fontsize=7.0, color=color, fontweight="bold")
    axis.set_title(label, loc="left", pad=3, fontweight="bold", fontsize=7.0)
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(*y_limits)
    axis.set_xticks((0, 1), ("Archive", "Raw check"))
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def build_figure() -> None:
    data = json.loads(DATA_PATH.read_text())
    primary = data["archived_primary_input"]["result"]
    alternate = data["alternative_raw_pairing"]["result"]

    configure_style()
    figure = plt.figure(figsize=(5.35, 3.05))
    outer = gridspec.GridSpec(2, 1, figure=figure, height_ratios=(0.92, 1.00), hspace=0.26)
    draw_ledger(figure.add_subplot(outer[0, 0]), data)

    metrics = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 0], wspace=0.48)
    axes = [figure.add_subplot(metrics[0, index]) for index in range(3)]
    panel_label(axes[0], "b", x=-0.22, y=1.23)
    draw_metric(
        axes[0],
        "Higher-order\nfraction",
        primary["higher_order_fraction"],
        alternate["higher_order_fraction"],
        (0.94, 1.00),
        lambda value: f"{value * 100:.1f}%",
    )
    axes[0].set_yticks((0.94, 0.97, 1.00), ("94", "97", "100"))
    draw_metric(
        axes[1],
        "Hyperdegree\nGini",
        primary["hyperdegree_gini"],
        alternate["hyperdegree_gini"],
        (0.82, 0.90),
        lambda value: f"{value:.3f}",
    )
    axes[1].set_yticks((0.82, 0.86, 0.90))
    draw_metric(
        axes[2],
        "Mean edge\noverlap",
        primary["mean_edge_overlap"],
        alternate["mean_edge_overlap"],
        (0.09, 0.15),
        lambda value: f"{value:.3f}",
    )
    axes[2].set_yticks((0.09, 0.12, 0.15))
    axes[0].set_ylabel("Metric value")

    figure.text(
        0.50,
        0.006,
        "Directional check only: distinct public archives are paired by post ID; this does not re-estimate the primary analysis.",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=MUTED,
    )
    OUTDIR.mkdir(exist_ok=True)
    metadata = {
        "Title": "raw snapshot consistency",
        "Creator": "fig10_raw_snapshot_check.py",
        "CreationDate": PDF_CREATION_DATE,
    }
    figure.savefig(OUTDIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.03, metadata=metadata)
    figure.savefig(OUTDIR / f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03, metadata={"Title": "raw snapshot consistency"})
    plt.close(figure)


if __name__ == "__main__":
    build_figure()
