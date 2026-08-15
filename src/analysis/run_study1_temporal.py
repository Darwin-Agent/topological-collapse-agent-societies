"""
Study 1 (temporal): How does the hypergraph topology evolve over time?

Splits the Moltbook data into weekly windows and tracks:
  - Edge size distribution shift
  - Triadic closure trend
  - Degree inequality trend
  - Number of active agents

This reveals whether the "topological collapse" was present from the
start or worsened over time (platform maturity effect).
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import Hypergraph
from src.analysis.topology import compute_topology


def build_weekly_hypergraphs(
    posts_path: str, comments_path: str,
    delta_minutes: int = 60,
) -> list[tuple[str, Hypergraph]]:
    """Split data into weekly windows and build one hypergraph per week."""

    logger.info("Loading posts...")
    posts = pd.read_parquet(posts_path, columns=["id", "author_id", "created_at", "comment_count"])
    posts = posts.dropna(subset=["id", "author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)
    posts = posts[posts["comment_count"] > 0]

    logger.info("Loading comments...")
    comments = pd.read_parquet(comments_path, columns=["id", "post_id", "author_id", "created_at"])
    comments = comments.dropna(subset=["post_id", "author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)

    posts["week"] = posts["created_at"].dt.isocalendar().week.astype(int)
    posts["year_week"] = posts["created_at"].dt.strftime("%Y-W%V")

    weeks = sorted(posts["year_week"].unique())
    logger.info("  Found %d weeks: %s", len(weeks), weeks)

    delta = timedelta(minutes=delta_minutes)
    comment_groups = comments.groupby("post_id")
    post_time = dict(zip(posts["id"], posts["created_at"]))
    post_author = dict(zip(posts["id"], posts["author_id"]))

    weekly_hgs = []
    for week_label in weeks:
        week_posts = posts[posts["year_week"] == week_label]
        if len(week_posts) < 100:
            continue

        sample = week_posts.sample(n=min(10000, len(week_posts)), random_state=42)

        hyperedges, timestamps, all_nodes = [], [], set()
        for pid in sample["id"]:
            author = post_author.get(pid)
            t0 = post_time.get(pid)
            if not author or not t0 or pid not in comment_groups.groups:
                continue
            group = comment_groups.get_group(pid)
            within = group[group["created_at"] <= t0 + delta]
            participants = set(within["author_id"].unique())
            participants.add(author)
            if len(participants) >= 2:
                hyperedges.append(frozenset(participants))
                timestamps.append(t0)
                all_nodes.update(participants)

        if len(hyperedges) < 50:
            continue

        hg = Hypergraph(nodes=all_nodes, hyperedges=hyperedges, timestamps=timestamps,
                        metadata={"week": week_label})
        weekly_hgs.append((week_label, hg))
        logger.info("  Week %s: %d edges, %d nodes", week_label, len(hyperedges), len(all_nodes))

    return weekly_hgs


def main():
    outdir = ROOT / "results" / "study1_temporal"
    outdir.mkdir(parents=True, exist_ok=True)

    posts_path = str(ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet")
    comments_path = str(ROOT / "data/raw/moltbook_hf/lnajt/comments.parquet")

    logger.info("=" * 60)
    logger.info("Study 1 Temporal: Weekly topology evolution")
    logger.info("=" * 60)

    weekly_hgs = build_weekly_hypergraphs(posts_path, comments_path)

    results = []
    for week_label, hg in weekly_hgs:
        report = compute_topology(hg, name=week_label, triadic_sample=10000)
        results.append({
            "week": week_label,
            "n_nodes": report.n_nodes,
            "n_edges": report.n_edges,
            "edge_size_mean": report.edge_size_mean,
            "frac_higher_order": report.frac_higher_order,
            "triadic_closure": report.triadic_closure_rate,
            "gini": report.hyperdegree_gini,
            "overlap": report.mean_edge_overlap,
        })

    (outdir / "temporal_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    # Plot temporal trends
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weeks = [r["week"] for r in results]
    metrics = [
        ("edge_size_mean", "Mean edge size"),
        ("triadic_closure", "Triadic closure"),
        ("gini", "Degree Gini"),
        ("frac_higher_order", "Higher-order %"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.flat, metrics):
        vals = [r[key] for r in results]
        ax.plot(weeks, vals, "o-", color="#E24A33", markersize=6)
        ax.set_title(title)
        ax.set_xlabel("Week")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)

    fig.suptitle("Moltbook Hypergraph Topology Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(outdir / "fig_temporal_evolution.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved temporal evolution figure")

    # Summary table
    logger.info("\n--- Temporal Summary ---")
    for r in results:
        logger.info("  %s: edges=%d, size=%.1f, closure=%.3f, gini=%.3f",
                    r["week"], r["n_edges"], r["edge_size_mean"],
                    r["triadic_closure"], r["gini"])

    logger.info("\nTemporal analysis complete! Results in %s", outdir)


if __name__ == "__main__":
    main()
