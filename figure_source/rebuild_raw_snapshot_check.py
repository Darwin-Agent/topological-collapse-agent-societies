#!/usr/bin/env python3
"""Rebuild the optional raw-file consistency check without making API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values)
    n_values = len(sorted_values)
    indices = np.arange(1, n_values + 1)
    return float(
        2 * np.sum(indices * sorted_values) / (n_values * np.sum(sorted_values))
        - (n_values + 1) / n_values
    )


def stable_mean_edge_overlap(hyperedges: list[frozenset[str]], sample_size: int = 10000) -> float:
    node_to_edges: dict[str, list[int]] = defaultdict(list)
    for edge_index, edge in enumerate(hyperedges):
        for member in sorted(edge):
            node_to_edges[member].append(edge_index)
    nodes = tuple(sorted(node_to_edges))
    generator = np.random.default_rng(42)
    overlaps = []
    attempts = 0
    while len(overlaps) < sample_size and attempts < sample_size * 5:
        attempts += 1
        edge_ids = node_to_edges[nodes[generator.integers(len(nodes))]]
        if len(edge_ids) < 2:
            continue
        first, second = generator.choice(edge_ids, size=2, replace=False)
        left, right = hyperedges[first], hyperedges[second]
        overlaps.append(len(left & right) / len(left | right))
    return float(np.mean(overlaps))


def build_hyperedges(posts_path: Path, comments_path: Path, max_posts: int, reply_window_minutes: int) -> tuple[list[frozenset[str]], dict]:
    posts = pd.read_parquet(posts_path, columns=["id", "author_id", "created_at", "comment_count"])
    n_post_records = len(posts)
    posts = posts.dropna(subset=["id", "author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)
    selected = posts.nlargest(max_posts, "comment_count")

    comments = pd.read_parquet(comments_path, columns=["post_id", "author_id", "created_at"])
    n_comment_records = len(comments)
    comments = comments.dropna(subset=["post_id", "author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)

    selected_ids = set(selected["id"])
    comments = comments[comments["post_id"].isin(selected_ids)]
    post_times = dict(zip(selected["id"], selected["created_at"]))
    post_authors = dict(zip(selected["id"], selected["author_id"]))
    groups = comments.groupby("post_id")
    window = pd.Timedelta(minutes=reply_window_minutes)
    hyperedges = []
    for post_id in selected["id"]:
        if post_id not in groups.groups:
            continue
        replies = groups.get_group(post_id)
        participants = set(replies.loc[replies["created_at"] <= post_times[post_id] + window, "author_id"])
        participants.add(post_authors[post_id])
        if len(participants) >= 2:
            hyperedges.append(frozenset(str(participant) for participant in participants))
    return hyperedges, {"n_post_records": n_post_records, "n_comment_records": n_comment_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--comments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-posts", type=int, default=50000)
    parser.add_argument("--reply-window-minutes", type=int, default=60)
    arguments = parser.parse_args()

    hyperedges, record_counts = build_hyperedges(
        arguments.posts,
        arguments.comments,
        arguments.max_posts,
        arguments.reply_window_minutes,
    )
    sizes = np.asarray([len(edge) for edge in hyperedges], dtype=int)
    degree_counts: Counter[str] = Counter(member for edge in hyperedges for member in edge)
    degrees = np.asarray(list(degree_counts.values()), dtype=float)
    result = {
        "source_files": [
            {
                "relative_path": str(arguments.posts),
                "n_records": record_counts["n_post_records"],
                "sha256": sha256(arguments.posts),
            },
            {
                "relative_path": str(arguments.comments),
                "n_records": record_counts["n_comment_records"],
                "sha256": sha256(arguments.comments),
            },
        ],
        "selection": {
            "max_posts_by_recorded_comment_count": arguments.max_posts,
            "reply_window_minutes": arguments.reply_window_minutes,
            "non_singleton_hyperedges_only": True,
        },
        "result": {
            "n_nodes": len(degree_counts),
            "n_hyperedges": len(hyperedges),
            "mean_edge_size": float(sizes.mean()),
            "higher_order_fraction": float((sizes >= 3).mean()),
            "hyperdegree_gini": gini(degrees),
            "mean_edge_overlap": stable_mean_edge_overlap(hyperedges),
        },
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
