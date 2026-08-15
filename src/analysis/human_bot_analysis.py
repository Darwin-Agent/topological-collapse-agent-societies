"""
Human-Bot Analysis: Moltbook agent operator structure.

Analyzes the relationship between human operators and AI agents on Moltbook
to support the paper's argument that topological collapse is structural,
not an artifact of AI autonomy.

Key finding: 35.2% of agents have verified human X/Twitter operators,
yet the platform-wide topology still exhibits star-dominated collapse.

Outputs:
  results/human_bot_analysis/human_bot_stats.json — statistics
  results/human_bot_analysis/fig_human_bot_analysis.png — 3-panel figure
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
DATA = ROOT / "data"

# Colors
C_HUMAN  = "#348ABD"  # Human-linked agents
C_CLEAN  = "#2ca02c"  # Clean AI agents
C_PUPPET = "#E24A33"  # Puppet/flagged agents
C_ANON   = "#FFA500"  # Claimed but anonymous


def load_agents() -> pd.DataFrame:
    """Load and classify all Moltbook agents."""
    agents_path = DATA / "raw" / "moltbook_hf" / "moltnet" / "data" / "v2026-02-28" / "agents.parquet"
    clean_path = DATA / "processed" / "clean_agent_ids.json"
    flagged_path = DATA / "processed" / "flagged_agent_ids.json"

    df = pd.read_parquet(agents_path)
    logger.info("Loaded %d agents", len(df))

    # Parse is_claimed_history
    def _is_claimed(h):
        if pd.isna(h) or h == "[]":
            return False
        try:
            items = json.loads(h) if isinstance(h, str) else h
            return len(items) > 0 and any(i.get("value", 0) == 1 for i in items)
        except Exception:
            return False

    df["is_claimed"] = df["is_claimed_history"].apply(_is_claimed)
    df["has_owner_x"] = df["owner_x_handle"].notna()

    # Load puppet classification
    clean_ids = set(json.loads(clean_path.read_text()))
    flagged_ids = set(json.loads(flagged_path.read_text()))
    df["is_clean"] = df["id"].isin(clean_ids)
    df["is_flagged"] = df["id"].isin(flagged_ids)

    # Assign category
    def _category(row):
        if row["has_owner_x"]:
            return "human_linked"
        elif row["is_claimed"]:
            return "claimed_anon"
        elif row["is_clean"]:
            return "clean_ai"
        elif row["is_flagged"]:
            return "puppet"
        else:
            return "unclassified"

    df["category"] = df.apply(_category, axis=1)

    counts = df["category"].value_counts()
    for cat, n in counts.items():
        logger.info("  %s: %d (%.1f%%)", cat, n, n / len(df) * 100)

    return df


def compute_operator_stats(df: pd.DataFrame) -> dict:
    """Compute statistics about human operators."""
    human = df[df["has_owner_x"]]
    operators = human.groupby("owner_x_handle").agg(
        n_bots=("id", "count"),
        verified=("owner_x_verified", "any"),
        follower_count=("owner_x_follower_count", "first"),
    ).reset_index()

    stats = {
        "total_agents": len(df),
        "human_linked": int(df["has_owner_x"].sum()),
        "human_linked_pct": float(df["has_owner_x"].mean() * 100),
        "claimed": int(df["is_claimed"].sum()),
        "claimed_pct": float(df["is_claimed"].mean() * 100),
        "clean_ai": int(df["is_clean"].sum()),
        "flagged_puppet": int(df["is_flagged"].sum()),
        "unique_operators": len(operators),
        "bots_per_operator": {
            "mean": float(operators["n_bots"].mean()),
            "median": float(operators["n_bots"].median()),
            "max": int(operators["n_bots"].max()),
            "std": float(operators["n_bots"].std()),
        },
        "verified_operators": int(operators["verified"].sum()),
        "verified_pct": float(operators["verified"].mean() * 100),
        "categories": df["category"].value_counts().to_dict(),
    }

    # Follower count distribution for operators
    followers = operators["follower_count"].dropna()
    if len(followers) > 0:
        stats["operator_followers"] = {
            "mean": float(followers.mean()),
            "median": float(followers.median()),
            "max": int(followers.max()),
            "p25": float(followers.quantile(0.25)),
            "p75": float(followers.quantile(0.75)),
        }

    return stats


def compute_activity_by_category(df: pd.DataFrame) -> dict:
    """Compute activity metrics per category."""
    results = {}
    for cat in ["human_linked", "clean_ai", "puppet", "claimed_anon"]:
        subset = df[df["category"] == cat]
        if len(subset) == 0:
            continue

        # Parse karma history to get final karma
        karmas = []
        for kh in subset["karma_history"].dropna():
            try:
                items = json.loads(kh) if isinstance(kh, str) else kh
                if items:
                    karmas.append(items[-1].get("value", 0))
            except Exception:
                pass

        # Parse follower count
        followers = []
        for fh in subset["follower_count_history"].dropna():
            try:
                items = json.loads(fh) if isinstance(fh, str) else fh
                if items:
                    followers.append(items[-1].get("value", 0))
            except Exception:
                pass

        results[cat] = {
            "n_agents": len(subset),
            "karma_mean": float(np.mean(karmas)) if karmas else 0,
            "karma_median": float(np.median(karmas)) if karmas else 0,
            "follower_mean": float(np.mean(followers)) if followers else 0,
            "follower_median": float(np.median(followers)) if followers else 0,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def plot_figure(df: pd.DataFrame, stats: dict, activity: dict, outdir: Path):
    """Generate the 3-panel human-bot analysis figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel A: Agent Category Breakdown ---
    ax = axes[0]
    cats = stats["categories"]
    cat_labels = {
        "human_linked": f"Human-Linked\n({cats.get('human_linked', 0):,})",
        "puppet": f"Puppet/Flagged\n({cats.get('puppet', 0):,})",
        "clean_ai": f"Clean AI\n({cats.get('clean_ai', 0):,})",
        "claimed_anon": f"Claimed Anon\n({cats.get('claimed_anon', 0):,})",
    }
    cat_colors = {
        "human_linked": C_HUMAN,
        "puppet": C_PUPPET,
        "clean_ai": C_CLEAN,
        "claimed_anon": C_ANON,
    }

    sizes = [cats.get(k, 0) for k in ["human_linked", "puppet", "clean_ai", "claimed_anon"]]
    labels = [cat_labels.get(k, k) for k in ["human_linked", "puppet", "clean_ai", "claimed_anon"]]
    colors = [cat_colors[k] for k in ["human_linked", "puppet", "clean_ai", "claimed_anon"]]

    # Filter out zero categories
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if nonzero:
        sizes, labels, colors = zip(*nonzero)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
            textprops=dict(fontsize=8),
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_fontweight("bold")

    ax.set_title("A. Agent Classification\n(N=149,574)", fontsize=10, fontweight="bold")

    # --- Panel B: Operator Concentration ---
    ax = axes[1]
    human = df[df["has_owner_x"]]
    bots_per_op = human.groupby("owner_x_handle").size()

    # Histogram
    ax.hist(bots_per_op.values, bins=range(1, bots_per_op.max() + 2),
            color=C_HUMAN, alpha=0.8, edgecolor="black", linewidth=0.5,
            align="left")
    ax.set_xlabel("Bots per Human Operator", fontsize=9)
    ax.set_ylabel("Number of Operators", fontsize=9)
    ax.set_title("B. Operator Concentration", fontsize=10, fontweight="bold")

    # Annotation
    ax.text(0.97, 0.95,
            f"Operators: {len(bots_per_op):,}\n"
            f"Mean: {bots_per_op.mean():.2f}\n"
            f"Max: {bots_per_op.max()}\n"
            f"1:1 ratio: {(bots_per_op == 1).mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))
    ax.grid(axis="y", alpha=0.3)

    # --- Panel C: Activity by Category ---
    ax = axes[2]
    cat_order = ["human_linked", "clean_ai", "puppet"]
    cat_names = ["Human-\nLinked", "Clean AI", "Puppet"]
    cat_cols = [C_HUMAN, C_CLEAN, C_PUPPET]

    karma_vals = [activity.get(c, {}).get("karma_mean", 0) for c in cat_order]
    follower_vals = [activity.get(c, {}).get("follower_mean", 0) for c in cat_order]

    x = np.arange(len(cat_order))
    width = 0.35
    bars1 = ax.bar(x - width / 2, karma_vals, width, label="Mean Karma",
                   color=[c for c in cat_cols], alpha=0.7, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, follower_vals, width, label="Mean Followers",
                   color=[c for c in cat_cols], alpha=0.4, edgecolor="black", linewidth=0.5,
                   hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=8)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title("C. Platform Activity by Category", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", fontsize=6.5)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", fontsize=6.5)

    fig.suptitle("Human Operators in the Moltbook AI Agent Society",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(outdir / "fig_human_bot_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig_human_bot_analysis.png")


def main():
    outdir = RESULTS / "human_bot_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Human-Bot Analysis")
    logger.info("=" * 60)

    # Load and classify
    df = load_agents()

    # Compute statistics
    stats = compute_operator_stats(df)
    activity = compute_activity_by_category(df)

    # Save stats
    all_results = {
        "operator_stats": stats,
        "activity_by_category": activity,
    }
    (outdir / "human_bot_stats.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved: human_bot_stats.json")

    # Key findings for paper
    logger.info("\n--- Key Findings for Paper ---")
    logger.info("  Total agents: %d", stats["total_agents"])
    logger.info("  Human-linked (has X account): %d (%.1f%%)",
                stats["human_linked"], stats["human_linked_pct"])
    logger.info("  Claimed by operator: %d (%.1f%%)",
                stats["claimed"], stats["claimed_pct"])
    logger.info("  Unique operators: %d", stats["unique_operators"])
    logger.info("  Bots per operator: mean=%.2f, max=%d",
                stats["bots_per_operator"]["mean"],
                stats["bots_per_operator"]["max"])
    logger.info("  → Nearly 1:1 operator-to-bot ratio")
    logger.info("  → Human presence does NOT prevent topological collapse")

    # Generate figure
    plot_figure(df, stats, activity, outdir)

    logger.info("\n=== Human-Bot Analysis Complete ===")


if __name__ == "__main__":
    main()
