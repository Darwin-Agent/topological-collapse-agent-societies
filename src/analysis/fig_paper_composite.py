"""
Generate the key composite figures for the paper.

Fig_main: A 2x3 composite figure showing the core argument:
  (a) Radar comparison (Moltbook clean vs SocioPatterns)
  (b) Temporal evolution of triadic closure
  (c) Multi-scale robustness
  (d) Explosive transition (dynamics)
  (e) Bifurcation diagram
  (f) Bistability parameter map
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    outdir = RESULTS / "paper_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Fig: Core comparison (clean vs human) ---
    clean_metrics = json.loads(
        (RESULTS / "study1_clean" / "topology_metrics.json").read_text()
    )
    molt_clean = clean_metrics["Moltbook (clean)"]
    sp = clean_metrics["SP-SFHH"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Bar comparison of key metrics
    ax = axes[0, 0]
    metrics = ["triadic_closure_rate", "mean_edge_overlap", "hyperdegree_gini", "frac_higher_order"]
    labels = ["Triadic\nClosure", "Edge\nOverlap", "Degree\nGini", "Higher-Order\nFraction"]
    molt_vals = [molt_clean[m] for m in metrics]
    sp_vals = [sp[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, molt_vals, w, color="#E24A33", label="Moltbook (clean agents)", alpha=0.85)
    ax.bar(x + w/2, sp_vals, w, color="#348ABD", label="Human (SocioPatterns)", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("(a) Topological Comparison: AI vs Human", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for i, (mv, sv) in enumerate(zip(molt_vals, sp_vals)):
        deficit = (mv - sv) / sv * 100
        if abs(deficit) > 5:
            y = max(mv, sv) + 0.02
            ax.text(i, y, f"{deficit:+.0f}%", ha="center", fontsize=8,
                    color="#E24A33" if deficit < 0 else "#2ca02c")

    # (b) Temporal evolution
    ax = axes[0, 1]
    temporal = json.loads((RESULTS / "study1_temporal" / "temporal_metrics.json").read_text())
    weeks = [r["week"] for r in temporal]
    closures = [r["triadic_closure"] for r in temporal]
    ginis = [r["gini"] for r in temporal]

    ax.plot(weeks, closures, "o-", color="#E24A33", markersize=7, label="Triadic Closure", linewidth=2)
    ax.axhline(y=0.97, color="#348ABD", linestyle="--", alpha=0.7, label="Human baseline (0.97)")
    ax.fill_between(range(len(weeks)), closures, alpha=0.15, color="#E24A33")
    ax.set_xlabel("Week")
    ax.set_ylabel("Triadic Closure Rate")
    ax.set_title("(b) Temporal Evolution: Collapse Onset at W07", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.grid(alpha=0.3)

    # (c) Multi-scale
    ax = axes[1, 0]
    ms_metrics = json.loads(
        (RESULTS / "study1_multiscale" / "multiscale_metrics.json").read_text()
    )
    deltas = [15, 30, 60, 120]
    ms_closure = [ms_metrics[f"Δt={d}min"]["triadic_closure_rate"] for d in deltas]
    ms_gini = [ms_metrics[f"Δt={d}min"]["hyperdegree_gini"] for d in deltas]
    ms_overlap = [ms_metrics[f"Δt={d}min"]["mean_edge_overlap"] for d in deltas]

    ax.plot(deltas, ms_closure, "o-", color="#E24A33", label="Triadic Closure", markersize=8, linewidth=2)
    ax.plot(deltas, ms_gini, "s-", color="#FBC15E", label="Degree Gini", markersize=8, linewidth=2)
    ax.plot(deltas, ms_overlap, "D-", color="#8EBA42", label="Edge Overlap", markersize=8, linewidth=2)

    ax.axhline(y=0.97, color="#348ABD", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.37, color="#348ABD", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0.27, color="#348ABD", linestyle="-.", alpha=0.5, linewidth=1)

    ax.set_xlabel("Time Window Δt (minutes)")
    ax.set_ylabel("Metric Value")
    ax.set_title("(c) Multi-Scale Robustness", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Z-score significance
    ax = axes[1, 1]
    zscores_data = {
        "Triadic\nClosure": -15.40,
        "Edge\nOverlap": 89.26,
        "Degree\nGini": 28.16,
        "Higher-\nOrder %": 4.33,
        "Edge\nSize": 29.26,
        "Degree\nMean": 29.26,
    }
    names = list(zscores_data.keys())
    zvals = list(zscores_data.values())
    colors = ["#E24A33" if z < 0 else "#2ca02c" for z in zvals]

    bars = ax.barh(names, zvals, color=colors, alpha=0.8)
    ax.axvline(x=-2, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=2, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("z-score (vs. configuration model null)")
    ax.set_title("(d) Statistical Significance (Clean, N=50)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, z in zip(bars, zvals):
        ax.text(bar.get_width() + (2 if z > 0 else -2), bar.get_y() + bar.get_height()/2,
                f"z={z:.1f}", va="center", fontsize=8,
                ha="left" if z > 0 else "right")

    fig.suptitle("Topological Collapse in AI Agent Societies",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = str(outdir / "Fig_main_composite.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved composite figure to %s", out_path)

    # --- Fig: Clean vs All comparison ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    categories = ["Triadic\nClosure", "Edge\nOverlap", "Degree\nGini",
                  "Higher-Order\nFraction", "Edge Size\n(normalized)"]
    all_vals = [0.754, 0.116, 0.863, 0.969, 9.93/67.88]
    clean_vals_2 = [0.643, 0.128, 0.781, 0.822, 5.54/67.88]
    human_vals = [0.970, 0.269, 0.371, 0.980, 1.0]

    x = np.arange(len(categories))
    w = 0.25
    ax2.bar(x - w, all_vals, w, color="#FBC15E", label="Moltbook (with puppets)", alpha=0.8)
    ax2.bar(x, clean_vals_2, w, color="#E24A33", label="Moltbook (clean only)", alpha=0.8)
    ax2.bar(x + w, human_vals, w, color="#348ABD", label="Human (SocioPatterns)", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel("Value")
    ax2.set_title("Puppet Removal Reveals Deeper Topological Collapse", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(str(outdir / "Fig_puppet_effect.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved puppet effect figure")

    logger.info("All composite figures generated!")


if __name__ == "__main__":
    main()
