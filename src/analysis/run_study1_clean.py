"""
Study 1 (clean): Re-run topological analysis using only clean agents
(after puppet/bot removal from Study 0).

This is the paper-ready version of Study 1.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import (
    build_sociopatterns_hypergraph, Hypergraph,
)
from src.analysis.topology import compute_topology, compare_reports
from src.analysis.null_model import null_model_ensemble, compute_zscores, format_zscores
from src.analysis.visualize import (
    fig_edge_size_distribution, fig_degree_distribution,
    fig_topology_comparison_bar, fig_radar,
)


def build_moltbook_clean_hypergraph(
    posts_path: str, comments_path: str,
    clean_ids_path: str,
    delta_minutes: int = 60,
    max_posts: int = 0,
) -> Hypergraph:
    """Build hypergraph using only clean (non-puppet) agents."""
    from datetime import timedelta

    clean_ids = set(json.loads(Path(clean_ids_path).read_text()))
    logger.info("Loaded %d clean agent IDs", len(clean_ids))

    logger.info("Loading posts...")
    posts = pd.read_parquet(posts_path, columns=["id", "author_id", "created_at", "comment_count"])
    posts = posts.dropna(subset=["id", "author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)

    # Filter: keep only posts by clean agents
    posts = posts[posts["author_id"].isin(clean_ids)]
    if max_posts > 0:
        posts = posts.nlargest(max_posts, "comment_count")
    post_ids = set(posts["id"])
    logger.info("  %d posts by clean agents", len(posts))

    logger.info("Loading comments...")
    comments = pd.read_parquet(comments_path, columns=["id", "post_id", "author_id", "created_at"])
    comments = comments.dropna(subset=["post_id", "author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)
    comments = comments[comments["post_id"].isin(post_ids)]
    # Keep only comments by clean agents
    comments = comments[comments["author_id"].isin(clean_ids)]
    logger.info("  %d comments by clean agents", len(comments))

    delta = timedelta(minutes=delta_minutes)
    post_time = dict(zip(posts["id"], posts["created_at"]))
    post_author = dict(zip(posts["id"], posts["author_id"]))
    comment_groups = comments.groupby("post_id")

    hyperedges, timestamps, all_nodes = [], [], set()
    for pid in posts["id"]:
        author = post_author[pid]
        t0 = post_time[pid]
        if pid not in comment_groups.groups:
            continue
        group = comment_groups.get_group(pid)
        within = group[group["created_at"] <= t0 + delta]
        participants = set(within["author_id"].unique())
        participants.add(author)
        if len(participants) < 2:
            continue
        edge = frozenset(participants)
        hyperedges.append(edge)
        timestamps.append(t0)
        all_nodes.update(participants)

    hg = Hypergraph(nodes=all_nodes, hyperedges=hyperedges, timestamps=timestamps,
                    metadata={"source": "moltbook_clean", "delta_minutes": delta_minutes,
                              "n_clean_agents": len(clean_ids)})
    logger.info("Clean Moltbook hypergraph: %s", hg.summary())
    return hg


def main():
    outdir = ROOT / "results" / "study1_clean"
    outdir.mkdir(parents=True, exist_ok=True)

    posts_path = str(ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet")
    comments_path = str(ROOT / "data/raw/moltbook_hf/lnajt/comments.parquet")
    clean_ids_path = str(ROOT / "data/processed/clean_agent_ids.json")

    logger.info("=" * 60)
    logger.info("Study 1 (Clean agents): 50K posts, 50 null samples")
    logger.info("=" * 60)

    hg_clean = build_moltbook_clean_hypergraph(
        posts_path, comments_path, clean_ids_path,
        delta_minutes=60, max_posts=50000,
    )

    sp_path = ROOT / "data/raw/sociopatterns/contact/tij_SFHH.dat"
    hg_sp = build_sociopatterns_hypergraph(str(sp_path), delta_seconds=300)

    hypergraphs = {"Moltbook (clean)": hg_clean, "SP-SFHH": hg_sp}

    reports = {}
    for name, hg in hypergraphs.items():
        reports[name] = compute_topology(hg, name=name, triadic_sample=30000)

    comparison = compare_reports(*reports.values())
    print("\n" + comparison)
    (outdir / "topology_comparison.txt").write_text(comparison, encoding="utf-8")
    json_path = outdir / "topology_metrics.json"
    json_path.write_text(json.dumps({n: r.to_dict() for n, r in reports.items()}, indent=2, default=str))

    # Null model
    moltbook_report = reports["Moltbook (clean)"]
    null_reports = null_model_ensemble(hg_clean, n_samples=50, triadic_sample=15000)
    zscores = compute_zscores(moltbook_report, null_reports)
    zs_text = format_zscores(zscores, name="Moltbook (clean)")
    print("\n" + zs_text)
    (outdir / "null_model_zscores.txt").write_text(zs_text, encoding="utf-8")

    # Figures
    report_list = list(reports.values())
    hg_list = [hypergraphs[r.name] for r in report_list]
    colors = ["#2ca02c", "#348ABD"]
    fig_edge_size_distribution(report_list, colors=colors,
                               output_path=str(outdir / "fig2_edge_size_clean.png"))
    fig_topology_comparison_bar(report_list, colors=colors,
                                output_path=str(outdir / "fig1_topology_clean.png"))
    fig_radar(report_list, colors=colors, output_path=str(outdir / "fig1_radar_clean.png"))

    logger.info("Study 1 (clean) complete! Results in %s", outdir)


if __name__ == "__main__":
    main()
