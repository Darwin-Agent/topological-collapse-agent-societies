"""
Study 1: Topological deficit in AI agent societies.

Run the complete analysis pipeline:
  1. Build hypergraphs (Moltbook + SocioPatterns)
  2. Compute topological metrics
  3. Run null model tests
  4. Generate comparison figures

Usage:
    python -m src.analysis.run_study1 --mode quick     # Small sample, fast test
    python -m src.analysis.run_study1 --mode medium    # 50K posts, moderate
    python -m src.analysis.run_study1 --mode full      # All data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import (
    build_moltbook_hypergraph_from_hf,
    build_moltbook_hypergraph_from_db,
    build_sociopatterns_hypergraph,
)
from src.analysis.topology import compute_topology, compare_reports
from src.analysis.null_model import null_model_ensemble, compute_zscores, format_zscores
from src.analysis.visualize import (
    fig_edge_size_distribution,
    fig_degree_distribution,
    fig_topology_comparison_bar,
    fig_radar,
)


MODE_CONFIG = {
    "quick": {
        "max_posts": 1000,
        "null_samples": 10,
        "triadic_sample": 5000,
        "description": "Quick test run (1K posts, 10 null samples)",
    },
    "medium": {
        "max_posts": 50000,
        "null_samples": 50,
        "triadic_sample": 30000,
        "description": "Medium run (50K posts, 50 null samples)",
    },
    "full": {
        "max_posts": 0,
        "null_samples": 100,
        "triadic_sample": 50000,
        "description": "Full analysis (all data, 100 null samples)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Study 1: Topological audit")
    parser.add_argument("--mode", choices=["quick", "medium", "full"],
                        default="quick", help="Analysis scale")
    parser.add_argument("--delta-minutes", type=int, default=60,
                        help="Time window for Moltbook hyperedge construction")
    parser.add_argument("--source", choices=["hf", "db", "both"], default="hf",
                        help="Moltbook data source (hf=HuggingFace, db=scraper)")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "study1"),
                        help="Output directory for figures and reports")
    args = parser.parse_args()

    config = MODE_CONFIG[args.mode]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Study 1: %s", config["description"])
    logger.info("=" * 60)

    # ── Step 1: Build hypergraphs ──────────────────────────────────
    hypergraphs = {}

    # Moltbook (HuggingFace)
    if args.source in ("hf", "both"):
        posts_path = str(ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet")
        comments_path = str(ROOT / "data/raw/moltbook_hf/lnajt/comments.parquet")
        if Path(posts_path).exists() and Path(comments_path).exists():
            hg_moltbook = build_moltbook_hypergraph_from_hf(
                posts_path, comments_path,
                delta_minutes=args.delta_minutes,
                max_posts=config["max_posts"],
            )
            hypergraphs["Moltbook (HF)"] = hg_moltbook
        else:
            logger.warning("HF data not found, skipping")

    # Moltbook (Custom scraper DB)
    if args.source in ("db", "both"):
        db_path = str(ROOT / "data/raw/moltbook/moltbook.db")
        if Path(db_path).exists():
            hg_db = build_moltbook_hypergraph_from_db(
                db_path,
                delta_minutes=args.delta_minutes,
                max_posts=config["max_posts"],
            )
            hypergraphs["Moltbook (DB)"] = hg_db
        else:
            logger.warning("Scraper DB not found, skipping")

    # SocioPatterns (multiple datasets)
    sp_dir = ROOT / "data/raw/sociopatterns/contact"
    sp_files = {
        "SP-SFHH": "tij_SFHH.dat",
        "SP-LyonSchool": "tij_LyonSchool.dat",
        "SP-Thiers13": "tij_Thiers13.dat",
    }
    for sp_name, sp_file in sp_files.items():
        sp_path = sp_dir / sp_file
        if sp_path.exists():
            hg_sp = build_sociopatterns_hypergraph(str(sp_path), delta_seconds=300)
            hypergraphs[sp_name] = hg_sp
            break  # use first available for quick mode
        else:
            logger.warning("SocioPatterns %s not found", sp_file)

    if not hypergraphs:
        logger.error("No data sources available. Exiting.")
        sys.exit(1)

    logger.info("\nBuilt %d hypergraphs:", len(hypergraphs))
    for name, hg in hypergraphs.items():
        logger.info("  %s: %s", name, hg.summary().split("\n")[0])

    # ── Step 2: Compute topology ──────────────────────────────────
    reports = {}
    for name, hg in hypergraphs.items():
        report = compute_topology(hg, name=name,
                                  triadic_sample=config["triadic_sample"])
        reports[name] = report

    # Print comparison table
    comparison = compare_reports(*reports.values())
    print("\n" + comparison)

    report_path = outdir / "topology_comparison.txt"
    report_path.write_text(comparison, encoding="utf-8")
    logger.info("Saved comparison table to %s", report_path)

    # Save JSON
    json_path = outdir / "topology_metrics.json"
    json_data = {name: r.to_dict() for name, r in reports.items()}
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")

    # ── Step 3: Null model test (on Moltbook) ─────────────────────
    moltbook_key = next((k for k in reports if "Moltbook" in k), None)
    if moltbook_key:
        logger.info("\nRunning null model tests for %s...", moltbook_key)
        hg_target = hypergraphs[moltbook_key]
        null_reports = null_model_ensemble(
            hg_target,
            n_samples=config["null_samples"],
            name_prefix="null",
            triadic_sample=config["triadic_sample"] // 2,
        )
        zscores = compute_zscores(reports[moltbook_key], null_reports)
        zscore_text = format_zscores(zscores, name=moltbook_key)
        print("\n" + zscore_text)
        (outdir / "null_model_zscores.txt").write_text(zscore_text, encoding="utf-8")

    # ── Step 4: Generate figures ──────────────────────────────────
    logger.info("\nGenerating figures...")
    report_list = list(reports.values())
    hg_list = [hypergraphs[r.name] for r in report_list]
    color_list = []
    for r in report_list:
        if "Moltbook" in r.name:
            color_list.append(COLORS["moltbook"])
        elif "SP" in r.name:
            color_list.append(COLORS["sociopatterns"])
        else:
            color_list.append(COLORS["null"])

    fig_edge_size_distribution(
        report_list, colors=color_list,
        output_path=str(outdir / "fig2_edge_size_distribution.png"),
    )
    fig_degree_distribution(
        report_list, hg_list, colors=color_list,
        output_path=str(outdir / "fig3_degree_distribution.png"),
    )
    fig_topology_comparison_bar(
        report_list, colors=color_list,
        output_path=str(outdir / "fig1_topology_comparison.png"),
    )
    fig_radar(
        report_list, colors=color_list,
        output_path=str(outdir / "fig1_radar.png"),
    )

    logger.info("\n" + "=" * 60)
    logger.info("Study 1 complete! Results in %s", outdir)
    logger.info("=" * 60)


COLORS = {
    "moltbook": "#E24A33",
    "sociopatterns": "#348ABD",
    "null": "#988ED5",
}

if __name__ == "__main__":
    main()
