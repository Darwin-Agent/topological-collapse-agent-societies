"""
Study 1 (multi-scale): Sensitivity analysis across time windows.

Runs hypergraph construction with Δt ∈ {15, 30, 60, 120} minutes
to verify that topological deficit is robust, not an artifact of
window choice.
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import build_moltbook_hypergraph_from_hf
from src.analysis.topology import compute_topology, compare_reports


DELTAS = [15, 30, 60, 120]
MAX_POSTS = 20000


def main():
    outdir = ROOT / "results" / "study1_multiscale"
    outdir.mkdir(parents=True, exist_ok=True)

    posts_path = str(ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet")
    comments_path = str(ROOT / "data/raw/moltbook_hf/lnajt/comments.parquet")

    logger.info("=" * 60)
    logger.info("Study 1 Multi-scale: Δt = %s minutes", DELTAS)
    logger.info("=" * 60)

    reports = {}
    for dt in DELTAS:
        logger.info("\n--- Building hypergraph with Δt=%d min ---", dt)
        hg = build_moltbook_hypergraph_from_hf(
            posts_path, comments_path,
            delta_minutes=dt, max_posts=MAX_POSTS,
        )
        report = compute_topology(hg, name=f"Δt={dt}min", triadic_sample=20000)
        reports[f"Δt={dt}min"] = report

    comparison = compare_reports(*reports.values())
    print("\n" + comparison)
    (outdir / "multiscale_comparison.txt").write_text(comparison, encoding="utf-8")

    metrics_json = {n: r.to_dict() for n, r in reports.items()}
    (outdir / "multiscale_metrics.json").write_text(
        json.dumps(metrics_json, indent=2, default=str), encoding="utf-8"
    )

    # Summary: how each metric changes with Δt
    logger.info("\n--- Trend Summary ---")
    for attr in ["edge_size_mean", "frac_higher_order", "triadic_closure_rate",
                 "hyperdegree_gini", "mean_edge_overlap"]:
        vals = [getattr(reports[f"Δt={dt}min"], attr) for dt in DELTAS]
        logger.info("  %-25s: %s", attr, " → ".join(f"{v:.4f}" for v in vals))

    logger.info("\nMulti-scale analysis complete! Results in %s", outdir)


if __name__ == "__main__":
    main()
