"""
Publication-quality visualizations for hypergraph topology comparison.

Generates figures for Nature Machine Intelligence submission:
  Fig 1: Hypergraph topology comparison (radar + bar)
  Fig 2: Edge size distribution (log-log)
  Fig 3: Degree distribution + triadic closure comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from .topology import TopologyReport

logger = logging.getLogger(__name__)

COLORS = {
    "moltbook": "#E24A33",
    "sociopatterns": "#348ABD",
    "null": "#988ED5",
    "arxiv": "#777777",
}

STYLE_DEFAULTS = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}


def apply_style():
    plt.rcParams.update(STYLE_DEFAULTS)


def fig_edge_size_distribution(
    reports: list[TopologyReport],
    colors: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 2: Log-log plot of hyperedge size distribution P(s).
    Key diagnostic: AI agent networks should show steep decay,
    concentrated at s=2, versus broader distributions for humans.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    if colors is None:
        color_cycle = list(COLORS.values())
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(reports))]

    for report, color in zip(reports, colors):
        dist = report.edge_size_distribution
        if not dist:
            continue
        sizes = sorted(dist.keys())
        counts = np.array([dist[s] for s in sizes], dtype=float)
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total

        ax.scatter(sizes, probs, s=30, color=color, alpha=0.7, zorder=3,
                   label=report.name)
        ax.plot(sizes, probs, color=color, alpha=0.4, linewidth=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hyperedge size $s$")
    ax.set_ylabel("$P(s)$")
    ax.set_title("Hyperedge Size Distribution")
    ax.legend(frameon=True, framealpha=0.9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved edge size distribution to %s", output_path)
    return fig


def fig_degree_distribution(
    reports: list[TopologyReport],
    hgs: list,
    colors: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Complementary CDF of hyperdegree distribution.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    if colors is None:
        color_cycle = list(COLORS.values())
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(reports))]

    for hg, report, color in zip(hgs, reports, colors):
        degrees = list(hg.node_degrees().values())
        if not degrees:
            continue
        degrees_sorted = np.sort(degrees)[::-1]
        ccdf = np.arange(1, len(degrees_sorted) + 1) / len(degrees_sorted)
        ax.plot(degrees_sorted, ccdf, color=color, alpha=0.7, label=report.name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hyperdegree $k_h$")
    ax.set_ylabel("CCDF $P(K \\geq k_h)$")
    ax.set_title("Hyperdegree Distribution (CCDF)")
    ax.legend(frameon=True, framealpha=0.9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved degree distribution to %s", output_path)
    return fig


def fig_topology_comparison_bar(
    reports: list[TopologyReport],
    colors: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 1: Grouped bar chart comparing key topology metrics.
    """
    apply_style()

    metrics = [
        ("frac_higher_order", "Higher-order\nfraction ($s \\geq 3$)"),
        ("triadic_closure_rate", "Triadic\nclosure"),
        ("hyperdegree_gini", "Degree\nGini"),
        ("mean_edge_overlap", "Edge\noverlap"),
    ]

    if colors is None:
        color_cycle = list(COLORS.values())
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(reports))]

    n_metrics = len(metrics)
    n_reports = len(reports)
    x = np.arange(n_metrics)
    width = 0.8 / n_reports

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (report, color) in enumerate(zip(reports, colors)):
        values = [getattr(report, attr) for attr, _ in metrics]
        offset = (i - n_reports / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=report.name,
                       color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Value")
    ax.set_title("Hypergraph Topology: AI Agents vs. Human Benchmarks")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved topology comparison to %s", output_path)
    return fig


def fig_radar(
    reports: list[TopologyReport],
    colors: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Radar/spider chart for multi-dimensional topology comparison.
    Values are min-max normalized across the compared networks.
    """
    apply_style()

    attrs = [
        ("frac_higher_order", "Higher-order %"),
        ("triadic_closure_rate", "Triadic closure"),
        ("hyperdegree_gini", "Degree inequality"),
        ("mean_edge_overlap", "Edge overlap"),
        ("edge_size_mean", "Mean edge size"),
    ]

    if colors is None:
        color_cycle = list(COLORS.values())
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(reports))]

    raw_data = {}
    for attr, _ in attrs:
        vals = [getattr(r, attr) for r in reports]
        vmin, vmax = min(vals), max(vals)
        span = vmax - vmin if vmax > vmin else 1.0
        raw_data[attr] = [(v - vmin) / span for v in vals]

    n = len(attrs)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (report, color) in enumerate(zip(reports, colors)):
        values = [raw_data[attr][i] for attr, _ in attrs]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, color=color, label=report.name)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in attrs], size=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Topology Radar: AI Agent vs. Human Networks", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved radar chart to %s", output_path)
    return fig
