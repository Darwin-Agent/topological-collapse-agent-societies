#!/usr/bin/env python3
"""
puppet_sensitivity.py — Sensitivity analysis of puppet detection to the
signal-count threshold (reviewer problem 5).

Reuses the four detectors in src/analysis/puppet_detection.py, but instead of
flagging by the union (>=1 signal), it counts how many independent signals each
agent triggers and reports the puppet fraction AND post-removal topology at each
threshold: >=1, >=2, >=3, >=4 signals.

Real data only (Moltbook HF parquet). No fabricated numbers.

Output: results/human_bot_analysis/puppet_sensitivity.json
Usage:  /tmp/figenv/bin/python -m src.analysis.puppet_sensitivity
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.puppet_detection import (
    detect_volume_outliers, detect_burst_clusters,
    detect_content_duplicates, detect_synchronized_agents,
)

POSTS = ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet"
COMMENTS = ROOT / "data/raw/moltbook_hf/moltnet/data/v2026-02-28/comments.parquet"
OUT = ROOT / "results/human_bot_analysis/puppet_sensitivity.json"


def _gini(counts):
    import numpy as np
    v = np.sort(np.asarray(counts, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n)


def main():
    print("Loading Moltbook raw data (real)...")
    posts = pd.read_parquet(POSTS, columns=["id", "author_id", "created_at", "body"])
    posts = posts.dropna(subset=["author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)

    comments = pd.read_parquet(COMMENTS, columns=["id", "post_id", "author_id", "created_at"])
    comments = comments.dropna(subset=["author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)
    # detector 3 expects a 'body' column on comments; it only uses posts here,
    # so we pass posts to content detector (matches original pipeline).
    print(f"  posts={len(posts)}  comments={len(comments)}")

    all_agents = set(posts["author_id"].unique()) | set(comments["author_id"].unique())
    n_all = len(all_agents)

    print("Running 4 detectors (real)...")
    sig = {
        "volume": detect_volume_outliers(posts, comments),
        "burst": detect_burst_clusters(posts),
        "content": detect_content_duplicates(posts),
        "sync": detect_synchronized_agents(comments),
    }
    for k, v in sig.items():
        print(f"  {k}: {len(v)} agents")

    # Per-agent signal count
    signal_count = Counter()
    for flagged in sig.values():
        for a in flagged:
            signal_count[a] += 1

    # Post degree (hyperdegree proxy): number of threads an agent commented in
    thread_members = comments.groupby("post_id")["author_id"].nunique()  # size per thread
    agent_threads = comments.groupby("author_id")["post_id"].nunique()    # degree per agent

    def topology_after_removal(flagged_set):
        """Recompute coarse topology metrics on comment-thread hypergraph
        after removing flagged agents."""
        keep = comments[~comments["author_id"].isin(flagged_set)]
        # hyperedge = thread; members = distinct authors
        grp = keep.groupby("post_id")["author_id"].nunique()
        grp = grp[grp >= 2]  # hyperedges need >=2
        mean_edge = float(grp.mean()) if len(grp) else 0.0
        deg = keep.groupby("author_id")["post_id"].nunique()
        gini = _gini(deg.values) if len(deg) else 0.0
        return {"n_hyperedges": int(len(grp)), "mean_edge_size": round(mean_edge, 3),
                "degree_gini": round(gini, 3), "n_agents_remaining": int(deg.shape[0])}

    print("\nSensitivity across thresholds:")
    rows = {}
    for thr in [1, 2, 3, 4]:
        flagged = {a for a, c in signal_count.items() if c >= thr}
        frac = len(flagged) / n_all
        topo = topology_after_removal(flagged)
        rows[f">={thr}"] = {
            "n_flagged": len(flagged),
            "flagged_frac": round(frac, 4),
            "post_removal_topology": topo,
        }
        print(f"  >={thr} signals: flagged={len(flagged)} ({frac*100:.1f}%)  "
              f"remaining Gini={topo['degree_gini']} mean_edge={topo['mean_edge_size']}")

    # Baseline (no removal) topology for reference
    baseline = topology_after_removal(set())

    result = {
        "n_all_agents": n_all,
        "per_detector_counts": {k: len(v) for k, v in sig.items()},
        "signal_count_distribution": dict(Counter(signal_count.values())),
        "baseline_topology_no_removal": baseline,
        "by_threshold": rows,
        "note": ("Original pipeline flags by the UNION of detectors (>=1 signal); "
                 "this analysis reports every threshold so the >=2 operating point "
                 "used in the paper can be read directly."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")
    print(f"\n>=1 signal = {rows['>=1']['flagged_frac']*100:.1f}%  "
          f">=2 = {rows['>=2']['flagged_frac']*100:.1f}%  "
          f">=3 = {rows['>=3']['flagged_frac']*100:.1f}%")


if __name__ == "__main__":
    main()
