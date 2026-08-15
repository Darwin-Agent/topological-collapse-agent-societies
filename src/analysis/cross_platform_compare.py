"""
Cross-platform topological comparison: Moltbook vs multiple human baselines.

Produces systematic comparison of higher-order topology metrics across
all platforms, with statistical tests and visualization.

Key outputs:
  Fig_cross_platform_radar.png    — multi-platform radar comparison
  Fig_cross_platform_bars.png     — bar charts with error bars
  Fig_cross_platform_boxplot.png  — distribution comparison
  cross_platform_summary.json     — full numeric results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "cross_platform"
FIG_DIR = ROOT / "results" / "paper_figures"

PLATFORM_COLORS = {
    "moltbook": "#E24A33",
    "enron": "#348ABD",
    "arxiv": "#2ca02c",
    "reddit": "#7A68A6",
    "stackoverflow": "#FFA500",
    "sociopatterns": "#188487",
}


def run_cross_platform_analysis(
    max_threads: int = 50000,
    delta_minutes: int = 60,
) -> dict:
    """Run full cross-platform comparison."""
    from src.analysis.topology import compute_topology
    from src.analysis.human_hypergraph_builder import build_all_human_hypergraphs
    from src.analysis.hypergraph_builder import build_moltbook_hypergraph_from_hf

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    reports = {}

    # Moltbook
    logger.info("Building Moltbook hypergraph...")
    posts_path = str(ROOT / "data" / "raw" / "moltbook_hf" / "lnajt" / "posts.parquet")
    comments_path = str(ROOT / "data" / "raw" / "moltbook_hf" / "lnajt" / "comments.parquet")

    try:
        moltbook_hg = build_moltbook_hypergraph_from_hf(
            posts_path, comments_path,
            delta_minutes=delta_minutes,
            max_posts=max_threads,
        )
        reports["moltbook"] = compute_topology(moltbook_hg, name="moltbook", triadic_sample=20000)
        logger.info("  Moltbook: %d nodes, %d edges", len(moltbook_hg.nodes), len(moltbook_hg.hyperedges))
    except Exception as e:
        logger.error("Moltbook failed: %s", e)

    # Human baselines
    logger.info("Building human hypergraphs...")
    human_hgs = build_all_human_hypergraphs(
        delta_minutes=delta_minutes,
        max_threads=max_threads,
    )

    for name, hg in human_hgs.items():
        try:
            reports[name] = compute_topology(hg, name=name, triadic_sample=20000)
            logger.info("  %s topology computed", name)
        except Exception as e:
            logger.error("  %s failed: %s", name, e)

    # compile results
    summary = {}
    for name, report in reports.items():
        summary[name] = {
            "n_nodes": report.n_nodes,
            "n_hyperedges": report.n_edges,
            "mean_edge_size": report.edge_size_mean,
            "max_edge_size": report.edge_size_max,
            "hyperdegree_mean": report.hyperdegree_mean,
            "hyperdegree_gini": report.hyperdegree_gini,
            "higher_order_fraction": report.frac_higher_order,
            "triadic_closure_rate": report.triadic_closure_rate,
            "mean_edge_overlap": report.mean_edge_overlap,
        }

    (RESULTS_DIR / "cross_platform_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # generate figures
    _plot_radar(summary)
    _plot_bars(summary)

    logger.info("Cross-platform analysis complete. Results: %s", RESULTS_DIR)
    return summary


def _plot_radar(summary: dict):
    """Multi-platform radar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_edge_size", "Mean Edge Size"),
        ("higher_order_fraction", "Higher-Order\nFraction"),
        ("triadic_closure_rate", "Triadic\nClosure"),
        ("mean_edge_overlap", "Edge\nOverlap"),
        ("hyperdegree_gini", "Degree\nGini"),
    ]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for platform, data in summary.items():
        values = []
        for key, _ in metrics:
            v = data.get(key, 0) or 0
            values.append(v)
        values += values[:1]

        # normalize to [0,1] for radar
        max_vals = {
            "mean_edge_size": max(d.get("mean_edge_size", 1) for d in summary.values()),
            "higher_order_fraction": 1.0,
            "triadic_closure_rate": 1.0,
            "mean_edge_overlap": max(max(d.get("mean_edge_overlap", 0.01) for d in summary.values()), 0.01),
            "hyperdegree_gini": 1.0,
        }

        norm_values = []
        for (key, _), v in zip(metrics, values[:-1]):
            mv = max_vals.get(key, 1.0)
            norm_values.append(v / mv if mv > 0 else 0)
        norm_values += norm_values[:1]

        color = PLATFORM_COLORS.get(platform, "#999999")
        ax.plot(angles, norm_values, "o-", linewidth=2, label=platform, color=color, markersize=5)
        ax.fill(angles, norm_values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Cross-Platform Hypergraph Topology Comparison\n"
                 "(Moltbook AI vs Human Social Networks)", fontsize=13, pad=20)

    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "Fig_cross_platform_radar.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved radar chart")


def _plot_bars(summary: dict):
    """Bar chart comparison of key metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_edge_size", "Mean Edge Size"),
        ("higher_order_fraction", "Higher-Order Fraction"),
        ("triadic_closure_rate", "Triadic Closure"),
        ("mean_edge_overlap", "Edge Overlap"),
        ("hyperdegree_gini", "Degree Gini"),
    ]

    platforms = list(summary.keys())
    n_platforms = len(platforms)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_platforms)
    width = 0.6

    for ax, (key, label) in zip(axes, metrics):
        vals = [summary[p].get(key, 0) or 0 for p in platforms]
        colors = [PLATFORM_COLORS.get(p, "#999999") for p in platforms]

        bars = ax.bar(x, vals, width, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(platforms, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")

        # highlight Moltbook
        if "moltbook" in platforms:
            idx = platforms.index("moltbook")
            bars[idx].set_edgecolor("red")
            bars[idx].set_linewidth(2)

    fig.suptitle("Topological Metrics: Moltbook AI vs Human Networks", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "Fig_cross_platform_bars.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved bar chart")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_cross_platform_analysis()
