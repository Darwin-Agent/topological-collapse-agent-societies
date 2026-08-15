"""
MoltNet agent profile analysis: enrich Study 0 findings with agent metadata.

Uses iNLP-Lab/MoltNet's agents.parquet (149K agent profiles) to:
  1. Characterize clean vs puppet agents (karma, follower distributions)
  2. Identify behavior clusters among clean agents
  3. Compute agent-level features for hypergraph weighting
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def main():
    outdir = ROOT / "results" / "study0_profiles"
    outdir.mkdir(parents=True, exist_ok=True)

    agents_path = ROOT / "data/raw/moltbook_hf/moltnet/data/v2026-02-28/agents.parquet"
    clean_ids_path = ROOT / "data/processed/clean_agent_ids.json"
    flagged_ids_path = ROOT / "data/processed/flagged_agent_ids.json"

    logger.info("Loading agent profiles from MoltNet...")
    agents = pd.read_parquet(str(agents_path))
    logger.info("  %d agent profiles, columns: %s", len(agents), list(agents.columns))
    logger.info("  Sample:\n%s", agents.head(3).to_string())

    clean_ids = set(json.loads(clean_ids_path.read_text()))
    flagged_ids = set(json.loads(flagged_ids_path.read_text()))

    if "id" in agents.columns:
        id_col = "id"
    elif "author_id" in agents.columns:
        id_col = "author_id"
    else:
        id_col = agents.columns[0]
        logger.warning("Using first column '%s' as agent ID", id_col)

    agents["is_clean"] = agents[id_col].isin(clean_ids)
    agents["is_flagged"] = agents[id_col].isin(flagged_ids)

    n_clean = agents["is_clean"].sum()
    n_flagged = agents["is_flagged"].sum()
    n_neither = len(agents) - n_clean - n_flagged
    logger.info("Matched: %d clean, %d flagged, %d unmatched (not in HF data)",
                n_clean, n_flagged, n_neither)

    numeric_cols = agents.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("\nNumeric columns available: %s", numeric_cols)

    if len(numeric_cols) > 0:
        logger.info("\n=== Clean vs Flagged Agent Comparison ===")
        for col in numeric_cols[:10]:
            clean_vals = agents.loc[agents["is_clean"], col].dropna()
            flagged_vals = agents.loc[agents["is_flagged"], col].dropna()
            if len(clean_vals) > 0 and len(flagged_vals) > 0:
                logger.info("  %s:", col)
                logger.info("    Clean:   mean=%.2f, median=%.2f, std=%.2f (n=%d)",
                            clean_vals.mean(), clean_vals.median(), clean_vals.std(), len(clean_vals))
                logger.info("    Flagged: mean=%.2f, median=%.2f, std=%.2f (n=%d)",
                            flagged_vals.mean(), flagged_vals.median(), flagged_vals.std(), len(flagged_vals))

    # Visualize key distributions
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_cols = [c for c in ["karma", "follower_count", "post_count", "comment_count"]
                 if c in agents.columns]

    if plot_cols:
        fig, axes = plt.subplots(1, len(plot_cols), figsize=(5 * len(plot_cols), 5))
        if len(plot_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, plot_cols):
            clean_data = agents.loc[agents["is_clean"], col].dropna()
            flagged_data = agents.loc[agents["is_flagged"], col].dropna()

            if len(clean_data) > 0:
                ax.hist(clean_data.clip(upper=clean_data.quantile(0.95)),
                        bins=50, alpha=0.6, label="Clean", color="#2ca02c")
            if len(flagged_data) > 0:
                ax.hist(flagged_data.clip(upper=flagged_data.quantile(0.95) if len(flagged_data) > 0 else 1),
                        bins=50, alpha=0.6, label="Flagged", color="#E24A33")
            ax.set_title(col)
            ax.legend()
            ax.set_ylabel("Count")

        fig.suptitle("Agent Profile: Clean vs Flagged", fontsize=14)
        fig.tight_layout()
        fig.savefig(str(outdir / "fig_agent_profiles.png"), dpi=300, bbox_inches="tight")
        logger.info("Saved profile comparison figure")

    # Save summary
    summary = {
        "total_profiles": len(agents),
        "matched_clean": int(n_clean),
        "matched_flagged": int(n_flagged),
        "unmatched": int(n_neither),
        "columns": list(agents.columns),
    }
    (outdir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    logger.info("\nProfile analysis complete! Results in %s", outdir)


if __name__ == "__main__":
    main()
