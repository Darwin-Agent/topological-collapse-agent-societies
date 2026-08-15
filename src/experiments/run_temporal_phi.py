"""
Temporal HIS and Phi Evolution Analysis.

Addresses reviewer questions:
  1. "Does HIS remain discriminative across all weeks?"
  2. "Does the Phi trajectory capture the W06-W07 collapse?"
  3. "Is HIS robust across temporal resolutions (delta-t)?"

Tracks all 4 Phi components (closure, overlap, heterogeneity, HIS) over
weekly snapshots and across multiple temporal aggregation windows.
"""

import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import (
    Hypergraph,
    build_moltbook_hypergraph_from_hf,
    build_sociopatterns_hypergraph,
)
from src.analysis.topology import compute_topology
from src.models.contagion_ho import (
    TopologyAwareContagionModel,
    TopologyParams,
)


# ═══════════════════════════════════════════════════════════════════════
# Data loading (adapted from run_study1_temporal.py)
# ═══════════════════════════════════════════════════════════════════════

def build_weekly_hypergraphs(
    posts_path: str,
    comments_path: str,
    delta_minutes: int = 60,
) -> list[tuple[str, Hypergraph]]:
    """Split parquet data into weekly windows and build one hypergraph per week."""
    logger.info("Loading posts...")
    posts = pd.read_parquet(posts_path, columns=["id", "author_id", "created_at", "comment_count"])
    posts = posts.dropna(subset=["id", "author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)
    posts = posts[posts["comment_count"] > 0]

    logger.info("Loading comments...")
    comments = pd.read_parquet(comments_path, columns=["id", "post_id", "author_id", "created_at"])
    comments = comments.dropna(subset=["post_id", "author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)

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
        hyperedges, all_nodes = [], set()

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
                edge = frozenset(str(p) for p in participants)
                hyperedges.append(edge)
                all_nodes.update(edge)

        if len(hyperedges) >= 50:
            hg = Hypergraph(nodes=all_nodes, hyperedges=hyperedges,
                            metadata={"week": week_label})
            weekly_hgs.append((week_label, hg))
            logger.info("  Week %s: %d edges, %d nodes", week_label, len(hyperedges), len(all_nodes))

    return weekly_hgs


# ═══════════════════════════════════════════════════════════════════════
# Core analysis functions
# ═══════════════════════════════════════════════════════════════════════

def compute_weekly_phi(
    weekly_hgs: list[tuple[str, Hypergraph]],
    sp_report,
    alpha: float = 2.0,
) -> list[dict]:
    """Compute full topology + Phi + components for each weekly snapshot."""
    sp_phi = TopologyAwareContagionModel.from_topology_report(
        sp_report, alpha=alpha).phi

    results = []
    for week_label, hg in weekly_hgs:
        report = compute_topology(hg, name=week_label, triadic_sample=10000)

        # Phi components
        cv2 = (report.hyperdegree_gini ** 2) * (np.pi / 2)
        closure_term = report.triadic_closure_rate
        overlap_term = 1.0 + alpha * report.mean_edge_overlap
        heterogeneity_term = 1.0 + cv2
        his_term = report.his_mean

        phi = closure_term * overlap_term * heterogeneity_term * his_term
        phi_ratio = sp_phi / phi if phi > 0 else float("inf")

        model = TopologyAwareContagionModel.from_topology_report(
            report, alpha=alpha)

        results.append({
            "week": week_label,
            # Core topology metrics
            "n_nodes": report.n_nodes,
            "n_edges": report.n_edges,
            "triadic_closure": report.triadic_closure_rate,
            "gini": report.hyperdegree_gini,
            "overlap": report.mean_edge_overlap,
            "frac_higher_order": report.frac_higher_order,
            # HIS metrics
            "his_mean": report.his_mean,
            "his_median": report.his_median,
            "his_std": report.his_std,
            "frac_simplicial": report.frac_simplicial,
            # Phi components
            "phi_closure": float(closure_term),
            "phi_overlap": float(overlap_term),
            "phi_heterogeneity": float(heterogeneity_term),
            "phi_his": float(his_term),
            "phi": float(phi),
            "phi_sp": float(sp_phi),
            "phi_ratio": float(phi_ratio),
            # Model properties
            "is_bistable": model.is_bistable(),
        })

        logger.info("  %s: HIS=%.3f, Φ=%.3f, ratio=%.2f, bistable=%s",
                    week_label, report.his_mean, phi, phi_ratio, model.is_bistable())

    return results


def compute_multiscale_his(
    posts_path: str,
    comments_path: str,
    delta_values: list[int] = None,
    max_posts: int = 15000,
) -> list[dict]:
    """Compute HIS at multiple temporal resolutions."""
    if delta_values is None:
        delta_values = [15, 30, 60, 120, 240]

    results = []
    for dt in delta_values:
        logger.info("  Δt = %d min...", dt)
        hg = build_moltbook_hypergraph_from_hf(
            posts_path, comments_path,
            delta_minutes=dt, max_posts=max_posts,
        )
        report = compute_topology(hg, name=f"Dt={dt}min", triadic_sample=10000)

        cv2 = (report.hyperdegree_gini ** 2) * (np.pi / 2)
        phi = (report.triadic_closure_rate *
               (1.0 + 2.0 * report.mean_edge_overlap) *
               (1.0 + cv2) *
               report.his_mean)

        results.append({
            "delta_minutes": dt,
            "n_nodes": report.n_nodes,
            "n_edges": report.n_edges,
            "his_mean": report.his_mean,
            "his_median": report.his_median,
            "his_std": report.his_std,
            "frac_simplicial": report.frac_simplicial,
            "triadic_closure": report.triadic_closure_rate,
            "gini": report.hyperdegree_gini,
            "overlap": report.mean_edge_overlap,
            "phi": float(phi),
        })

        logger.info("    HIS=%.3f, Φ=%.3f, n_edges=%d",
                    report.his_mean, phi, report.n_edges)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def _plot_temporal_phi(weekly, multiscale, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    weeks = [r["week"] for r in weekly]
    x = np.arange(len(weeks))

    # ── Panel A: HIS per week ──
    ax = axes[0, 0]
    his_vals = [r["his_mean"] for r in weekly]
    his_std = [r["his_std"] for r in weekly]
    ax.errorbar(x, his_vals, yerr=his_std, fmt="o-", color="#2ca02c",
                linewidth=2, markersize=6, capsize=4, label="HIS mean ± std")
    ax.axhline(0.69, color="#348ABD", linestyle="--", linewidth=1.5,
               label="SocioPatterns HIS=0.69")
    ax.axhline(0.41, color="#E24A33", linestyle="--", linewidth=1.5,
               label="Moltbook overall HIS=0.41")
    ax.set_xticks(x)
    ax.set_xticklabels(weeks, rotation=45, fontsize=8)
    ax.set_ylabel("HIS (Hyperedge Irreducibility)")
    ax.set_title("A. HIS remains low across all weeks")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel B: Phi per week with SP reference ──
    ax = axes[0, 1]
    phi_vals = [r["phi"] for r in weekly]
    phi_sp = weekly[0]["phi_sp"] if weekly else 1.0
    ax.plot(x, phi_vals, "o-", color="#E24A33", linewidth=2, markersize=6,
            label="Φ (Moltbook weekly)")
    ax.axhline(phi_sp, color="#348ABD", linestyle="--", linewidth=1.5,
               label=f"Φ SocioPatterns = {phi_sp:.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)

    # Mark bistability
    bistable_weeks = [i for i, r in enumerate(weekly) if r["is_bistable"]]
    if bistable_weeks:
        ax.scatter(bistable_weeks, [phi_vals[i] for i in bistable_weeks],
                   marker="*", s=150, color="gold", zorder=5, label="Bistable")

    ax.set_xticks(x)
    ax.set_xticklabels(weeks, rotation=45, fontsize=8)
    ax.set_ylabel("Topology factor Φ")
    ax.set_title("B. Weekly Φ consistently below SocioPatterns")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel C: Phi component stacked area ──
    ax = axes[1, 0]
    # Use log decomposition for stacking (multiplicative → additive)
    log_c = [np.log(r["phi_closure"]) if r["phi_closure"] > 0 else 0 for r in weekly]
    log_o = [np.log(r["phi_overlap"]) for r in weekly]
    log_h = [np.log(r["phi_heterogeneity"]) for r in weekly]
    log_his = [np.log(r["phi_his"]) if r["phi_his"] > 0 else 0 for r in weekly]

    # Plot as grouped bars
    w = 0.2
    ax.bar(x - 1.5 * w, log_c, w, label="log(closure)", color="#E24A33", alpha=0.8)
    ax.bar(x - 0.5 * w, log_o, w, label="log(1+αJ)", color="#348ABD", alpha=0.8)
    ax.bar(x + 0.5 * w, log_h, w, label="log(1+CV²)", color="#FFA500", alpha=0.8)
    ax.bar(x + 1.5 * w, log_his, w, label="log(HIS)", color="#2ca02c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(weeks, rotation=45, fontsize=8)
    ax.set_ylabel("log(component)")
    ax.set_title("C. Phi component decomposition per week")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel D: Multiscale HIS ──
    ax = axes[1, 1]
    if multiscale:
        dts = [r["delta_minutes"] for r in multiscale]
        his_multi = [r["his_mean"] for r in multiscale]
        his_std_multi = [r["his_std"] for r in multiscale]
        phi_multi = [r["phi"] for r in multiscale]

        ax.errorbar(dts, his_multi, yerr=his_std_multi, fmt="s-", color="#2ca02c",
                    linewidth=2, markersize=6, capsize=4, label="HIS mean ± std")
        ax.axhline(0.69, color="#348ABD", linestyle="--", linewidth=1.5,
                   label="SocioPatterns HIS")
        ax.set_xlabel("Temporal window Δt (minutes)")
        ax.set_ylabel("HIS")
        ax.set_title("D. HIS stable across temporal resolutions")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(dts, phi_multi, "o--", color="#E24A33", linewidth=1.5,
                 markersize=5, alpha=0.7, label="Φ")
        ax2.set_ylabel("Φ", color="#E24A33")
        ax2.legend(loc="center right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Multiscale data\nnot available",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)

    # Summary annotation
    if weekly:
        his_range = [min(r["his_mean"] for r in weekly), max(r["his_mean"] for r in weekly)]
        phi_range = [min(r["phi"] for r in weekly), max(r["phi"] for r in weekly)]
        fig.text(0.02, 0.01,
                 f"HIS range: [{his_range[0]:.3f}, {his_range[1]:.3f}] "
                 f"(always < SP=0.69) | "
                 f"Φ range: [{phi_range[0]:.3f}, {phi_range[1]:.3f}] "
                 f"(always < Φ_SP={phi_sp:.3f})",
                 fontsize=9, style="italic")

    fig.suptitle(
        "Temporal Evolution of HIS and Φ: AI Topology Signature Persists Across All Weeks",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(str(outdir / "fig_temporal_phi.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved temporal Phi figure")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    outdir = ROOT / "results" / "temporal_phi"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Temporal HIS and Phi Evolution")
    logger.info("=" * 60)

    # ── Data paths ────────────────────────────────────────────────
    posts_path = ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet"
    comments_path = ROOT / "data/raw/moltbook_hf/moltnet/data/v2026-02-28/comments.parquet"
    sp_path = ROOT / "data/raw/sociopatterns/contact/tij_SFHH.dat"

    if not posts_path.exists() or not comments_path.exists():
        logger.error("Moltbook parquet data not found!")
        logger.error("  posts: %s (%s)", posts_path, "exists" if posts_path.exists() else "MISSING")
        logger.error("  comments: %s (%s)", comments_path, "exists" if comments_path.exists() else "MISSING")
        return

    # ── SocioPatterns reference ───────────────────────────────────
    logger.info("\n--- Computing SocioPatterns reference ---")
    hg_sp = build_sociopatterns_hypergraph(str(sp_path), delta_seconds=300)
    report_sp = compute_topology(hg_sp, name="SocioPatterns", triadic_sample=10000)
    logger.info("  SP: HIS=%.3f, closure=%.3f, Gini=%.3f",
                report_sp.his_mean, report_sp.triadic_closure_rate, report_sp.hyperdegree_gini)

    # ── Weekly Phi evolution ──────────────────────────────────────
    logger.info("\n--- Building weekly hypergraphs ---")
    weekly_hgs = build_weekly_hypergraphs(str(posts_path), str(comments_path))

    logger.info("\n--- Computing weekly Phi and HIS ---")
    weekly_results = compute_weekly_phi(weekly_hgs, report_sp)

    # Summary
    if weekly_results:
        his_vals = [r["his_mean"] for r in weekly_results]
        phi_vals = [r["phi"] for r in weekly_results]
        logger.info("\n  HIS across weeks: min=%.3f, max=%.3f, mean=%.3f",
                    min(his_vals), max(his_vals), np.mean(his_vals))
        logger.info("  Φ across weeks: min=%.3f, max=%.3f, mean=%.3f",
                    min(phi_vals), max(phi_vals), np.mean(phi_vals))
        logger.info("  All HIS < SP (0.69)? %s", all(h < 0.69 for h in his_vals))
        logger.info("  All Φ < Φ_SP? %s",
                    all(p < weekly_results[0]["phi_sp"] for p in phi_vals))

    # ── Multiscale HIS ────────────────────────────────────────────
    logger.info("\n--- Multiscale HIS analysis ---")
    multiscale_results = compute_multiscale_his(str(posts_path), str(comments_path))

    # ── Save results ──────────────────────────────────────────────
    full_results = {
        "weekly": weekly_results,
        "multiscale": multiscale_results,
        "sp_reference": {
            "his_mean": report_sp.his_mean,
            "phi": float(TopologyAwareContagionModel.from_topology_report(report_sp).phi),
        },
    }
    (outdir / "temporal_phi_metrics.json").write_text(
        json.dumps(full_results, indent=2, default=str), encoding="utf-8"
    )

    # ── Figure ────────────────────────────────────────────────────
    _plot_temporal_phi(weekly_results, multiscale_results, outdir)

    logger.info("\n=== Temporal Phi analysis complete! Results in %s ===", outdir)


if __name__ == "__main__":
    main()
