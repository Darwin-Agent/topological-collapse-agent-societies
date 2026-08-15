"""
Study 0: Puppet cluster detection and data denoising.

Wiz security audit revealed 500K+ fake accounts registered by ~17K human
operators (avg 88 agents per operator). We identify puppet clusters via:

  1. Temporal burstiness: agents registered in rapid bursts by same operator
  2. Content similarity: near-duplicate posts/comments across agents
  3. Activity synchrony: agents that post within seconds of each other

Output: a clean agent set for downstream hypergraph analysis.

Usage:
    python -m src.analysis.puppet_detection --source hf
    python -m src.analysis.puppet_detection --source hf --output data/processed/clean_agents.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
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


def load_hf_data(posts_path: str, comments_path: str):
    logger.info("Loading posts...")
    posts = pd.read_parquet(posts_path,
                            columns=["id", "author_id", "created_at", "body", "comment_count"])
    posts = posts.dropna(subset=["author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)

    logger.info("Loading comments...")
    comments = pd.read_parquet(comments_path,
                               columns=["id", "post_id", "author_id", "created_at", "body"])
    comments = comments.dropna(subset=["author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)

    logger.info("  Posts: %d, Comments: %d", len(posts), len(comments))
    return posts, comments


# ── Detector 1: Activity volume outliers ─────────────────────────────

def detect_volume_outliers(posts: pd.DataFrame, comments: pd.DataFrame,
                           post_threshold: int = 500,
                           comment_threshold: int = 5000) -> set:
    """Flag agents with abnormally high activity (likely automated spam)."""
    post_counts = posts["author_id"].value_counts()
    comment_counts = comments["author_id"].value_counts()

    spam_posters = set(post_counts[post_counts > post_threshold].index)
    spam_commenters = set(comment_counts[comment_counts > comment_threshold].index)
    flagged = spam_posters | spam_commenters

    logger.info("Volume outliers: %d agents (post>%d: %d, comment>%d: %d)",
                len(flagged), post_threshold, len(spam_posters),
                comment_threshold, len(spam_commenters))
    return flagged


# ── Detector 2: Temporal burst registration ──────────────────────────

def detect_burst_clusters(posts: pd.DataFrame,
                          window_seconds: int = 10,
                          min_cluster_size: int = 5) -> set:
    """
    Detect agents whose first activity appears in tight temporal bursts.
    If N agents all make their first post within W seconds, they likely
    share an operator.
    """
    first_activity = posts.groupby("author_id")["created_at"].min().reset_index()
    first_activity = first_activity.sort_values("created_at").reset_index(drop=True)

    flagged = set()
    window = timedelta(seconds=window_seconds)
    n = len(first_activity)
    i = 0

    while i < n:
        j = i + 1
        while j < n and (first_activity.iloc[j]["created_at"] - first_activity.iloc[i]["created_at"]) <= window:
            j += 1
        if j - i >= min_cluster_size:
            cluster_agents = set(first_activity.iloc[i:j]["author_id"])
            flagged.update(cluster_agents)
        i = j if j > i + 1 else i + 1

    logger.info("Burst clusters (window=%ds, min_size=%d): %d agents flagged",
                window_seconds, min_cluster_size, len(flagged))
    return flagged


# ── Detector 3: Content duplication ──────────────────────────────────

def detect_content_duplicates(posts: pd.DataFrame,
                              min_duplicates: int = 10) -> set:
    """
    Flag agents that post identical or near-identical content repeatedly.
    Uses exact body hash for efficiency.
    """
    posts_with_body = posts.dropna(subset=["body"])
    posts_with_body = posts_with_body[posts_with_body["body"].str.len() > 20]

    body_hash = posts_with_body["body"].str.strip().str.lower()
    posts_with_body = posts_with_body.assign(body_hash=body_hash)

    dup_bodies = body_hash.value_counts()
    dup_bodies = dup_bodies[dup_bodies >= min_duplicates]
    dup_set = set(dup_bodies.index)

    flagged_agents = set()
    for _, row in posts_with_body.iterrows():
        if row["body_hash"] in dup_set:
            flagged_agents.add(row["author_id"])

    logger.info("Content duplicates (min_dup=%d): %d duplicate templates, %d agents flagged",
                min_duplicates, len(dup_set), len(flagged_agents))
    return flagged_agents


# ── Detector 4: Synchronized activity ────────────────────────────────

def detect_synchronized_agents(comments: pd.DataFrame,
                               window_seconds: int = 5,
                               min_sync_events: int = 20,
                               sample_size: int = 500000) -> set:
    """
    Find agent pairs that repeatedly comment within W seconds of each other
    on different posts — suggests coordinated puppet behavior.
    """
    if len(comments) > sample_size:
        sample = comments.sample(n=sample_size, random_state=42)
    else:
        sample = comments

    sample = sample.sort_values("created_at")
    window = timedelta(seconds=window_seconds)

    pair_counts: Counter = Counter()
    by_post = sample.groupby("post_id")

    for _, group in by_post:
        if len(group) < 2:
            continue
        group = group.sort_values("created_at")
        agents = group["author_id"].values
        times = group["created_at"].values

        for i in range(len(group)):
            for j in range(i + 1, min(i + 10, len(group))):
                if (times[j] - times[i]) > np.timedelta64(window_seconds, 's'):
                    break
                if agents[i] != agents[j]:
                    pair = tuple(sorted([agents[i], agents[j]]))
                    pair_counts[pair] += 1

    flagged = set()
    for (a, b), count in pair_counts.items():
        if count >= min_sync_events:
            flagged.add(a)
            flagged.add(b)

    logger.info("Synchronized agents (window=%ds, min_events=%d): %d agents flagged",
                window_seconds, min_sync_events, len(flagged))
    return flagged


# ── Main pipeline ────────────────────────────────────────────────────

def run_puppet_detection(posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
    """Run all detectors and return results."""
    all_agents = set(posts["author_id"].unique()) | set(comments["author_id"].unique())
    logger.info("Total unique agents: %d", len(all_agents))

    results = {}

    results["volume"] = detect_volume_outliers(posts, comments)
    results["burst"] = detect_burst_clusters(posts)
    results["content"] = detect_content_duplicates(posts)
    results["sync"] = detect_synchronized_agents(comments)

    all_flagged = set()
    for flagged in results.values():
        all_flagged.update(flagged)

    clean_agents = all_agents - all_flagged

    logger.info("\n=== PUPPET DETECTION SUMMARY ===")
    logger.info("Total agents:   %d", len(all_agents))
    logger.info("Flagged:        %d (%.1f%%)",
                len(all_flagged), len(all_flagged) / len(all_agents) * 100)
    for name, flagged in results.items():
        logger.info("  %-12s: %d agents", name, len(flagged))
    logger.info("Clean agents:   %d (%.1f%%)",
                len(clean_agents), len(clean_agents) / len(all_agents) * 100)

    overlap_vol_burst = results["volume"] & results["burst"]
    overlap_vol_content = results["volume"] & results["content"]
    logger.info("Overlap volume∩burst: %d", len(overlap_vol_burst))
    logger.info("Overlap volume∩content: %d", len(overlap_vol_content))

    return {
        "all_agents": len(all_agents),
        "flagged_total": len(all_flagged),
        "flagged_by_detector": {k: len(v) for k, v in results.items()},
        "clean_agents": len(clean_agents),
        "clean_agent_ids": list(clean_agents),
        "flagged_agent_ids": list(all_flagged),
    }


def main():
    parser = argparse.ArgumentParser(description="Study 0: Puppet detection")
    parser.add_argument("--source", choices=["hf"], default="hf")
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "puppet_detection.json"))
    args = parser.parse_args()

    posts_path = str(ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet")
    comments_path = str(ROOT / "data/raw/moltbook_hf/lnajt/comments.parquet")

    posts, comments = load_hf_data(posts_path, comments_path)
    results = run_puppet_detection(posts, comments)

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    save_data = {k: v for k, v in results.items()
                 if k not in ("clean_agent_ids", "flagged_agent_ids")}
    save_data["clean_agent_count"] = results["clean_agents"]
    outpath.write_text(json.dumps(save_data, indent=2, default=str), encoding="utf-8")

    ids_path = outpath.with_name("clean_agent_ids.json")
    ids_path.write_text(json.dumps(results["clean_agent_ids"]), encoding="utf-8")

    flagged_path = outpath.with_name("flagged_agent_ids.json")
    flagged_path.write_text(json.dumps(results["flagged_agent_ids"]), encoding="utf-8")

    logger.info("Saved results to %s", outpath.parent)


if __name__ == "__main__":
    main()
