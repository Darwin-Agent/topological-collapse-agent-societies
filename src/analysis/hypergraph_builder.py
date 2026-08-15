"""
Construct hypergraphs from interaction data.

Two data sources supported:
  1. Moltbook post-comment threads  →  "co-participation" hypergraph
  2. SocioPatterns face-to-face contacts →  "temporal aggregation" hypergraph

Hyperedge construction follows Battiston et al. (NHB 2025) Box 1:
  For each "interaction context" (post thread / temporal window),
  all participating agents form a single hyperedge.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Hypergraph:
    """Minimal hypergraph representation for analysis."""
    nodes: set = field(default_factory=set)
    hyperedges: list[frozenset] = field(default_factory=list)
    timestamps: list[Optional[datetime]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.hyperedges)

    def edge_sizes(self) -> np.ndarray:
        return np.array([len(e) for e in self.hyperedges])

    def node_degrees(self) -> dict[str, int]:
        """Hyperdegree: number of hyperedges each node participates in."""
        deg = defaultdict(int)
        for e in self.hyperedges:
            for n in e:
                deg[n] += 1
        return dict(deg)

    def remove_singletons(self) -> "Hypergraph":
        """Return a new hypergraph with |e| < 2 edges removed."""
        edges = [e for e in self.hyperedges if len(e) >= 2]
        ts = [t for e, t in zip(self.hyperedges, self.timestamps) if len(e) >= 2]
        nodes = set()
        for e in edges:
            nodes.update(e)
        return Hypergraph(nodes=nodes, hyperedges=edges, timestamps=ts,
                          metadata=self.metadata.copy())

    def filter_by_size(self, min_size: int = 2, max_size: int = 0) -> "Hypergraph":
        edges, ts = [], []
        for e, t in zip(self.hyperedges, self.timestamps):
            if len(e) >= min_size and (max_size == 0 or len(e) <= max_size):
                edges.append(e)
                ts.append(t)
        nodes = set()
        for e in edges:
            nodes.update(e)
        return Hypergraph(nodes=nodes, hyperedges=edges, timestamps=ts,
                          metadata=self.metadata.copy())

    def summary(self) -> str:
        sizes = self.edge_sizes()
        if len(sizes) == 0:
            return f"Hypergraph: {self.n_nodes} nodes, 0 edges"
        return (
            f"Hypergraph: {self.n_nodes:,} nodes, {self.n_edges:,} edges\n"
            f"  Edge size: mean={sizes.mean():.2f}, median={np.median(sizes):.1f}, "
            f"max={sizes.max()}, min={sizes.min()}\n"
            f"  s=2 (dyadic): {(sizes == 2).sum():,} ({(sizes == 2).mean()*100:.1f}%)\n"
            f"  s>=3 (higher-order): {(sizes >= 3).sum():,} ({(sizes >= 3).mean()*100:.1f}%)"
        )


# ─── Moltbook: post-thread → hyperedge ──────────────────────────────────

def build_moltbook_hypergraph_from_hf(
    posts_path: str,
    comments_path: str,
    delta_minutes: int = 60,
    max_posts: int = 0,
) -> Hypergraph:
    """
    Build hypergraph from HuggingFace lnajt/moltbook parquet files.

    For each post, collect all agents who commented within delta_minutes
    of the post creation. The post author + commenters form a hyperedge.

    Args:
        posts_path: path to posts.parquet
        comments_path: path to comments.parquet
        delta_minutes: time window for co-participation
        max_posts: limit number of posts (0 = all, useful for testing)
    """
    import pandas as pd

    logger.info("Loading posts from %s ...", posts_path)
    posts = pd.read_parquet(posts_path, columns=["id", "author_id", "created_at", "comment_count"])
    posts = posts.dropna(subset=["id", "author_id", "created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)

    if max_posts > 0:
        posts = posts.nlargest(max_posts, "comment_count")

    post_ids = set(posts["id"])
    logger.info("  %d posts loaded", len(posts))

    logger.info("Loading comments from %s ...", comments_path)
    comments = pd.read_parquet(comments_path, columns=["id", "post_id", "author_id", "created_at"])
    comments = comments.dropna(subset=["post_id", "author_id", "created_at"])
    comments["created_at"] = pd.to_datetime(comments["created_at"], format="ISO8601", utc=True)
    comments = comments[comments["post_id"].isin(post_ids)]
    logger.info("  %d comments loaded (filtered to selected posts)", len(comments))

    post_time = dict(zip(posts["id"], posts["created_at"]))
    post_author = dict(zip(posts["id"], posts["author_id"]))

    delta = timedelta(minutes=delta_minutes)
    comment_groups = comments.groupby("post_id")

    hyperedges = []
    timestamps = []
    all_nodes = set()
    skipped = 0

    for pid in posts["id"]:
        author = post_author[pid]
        t0 = post_time[pid]

        if pid not in comment_groups.groups:
            continue

        group = comment_groups.get_group(pid)
        within_window = group[group["created_at"] <= t0 + delta]
        participants = set(within_window["author_id"].unique())
        participants.add(author)

        if len(participants) < 2:
            skipped += 1
            continue

        edge = frozenset(participants)
        hyperedges.append(edge)
        timestamps.append(t0)
        all_nodes.update(participants)

    hg = Hypergraph(
        nodes=all_nodes,
        hyperedges=hyperedges,
        timestamps=timestamps,
        metadata={
            "source": "moltbook_hf",
            "delta_minutes": delta_minutes,
            "n_posts_input": len(posts),
            "n_posts_skipped": skipped,
        },
    )
    logger.info("Built Moltbook hypergraph: %s", hg.summary())
    return hg


def build_moltbook_hypergraph_from_db(
    db_path: str,
    delta_minutes: int = 60,
    max_posts: int = 0,
    min_comments: int = 1,
) -> Hypergraph:
    """
    Build hypergraph from our custom scraper SQLite database.
    Uses the full comment tree (no truncation).
    """
    import sqlite3

    logger.info("Loading from SQLite: %s", db_path)
    conn = sqlite3.connect(db_path)

    if max_posts > 0:
        posts = conn.execute("""
            SELECT id, author_id, created_at FROM posts
            WHERE comment_count >= ? AND comments_scraped = 1
            ORDER BY comment_count DESC LIMIT ?
        """, (min_comments, max_posts)).fetchall()
    else:
        posts = conn.execute("""
            SELECT id, author_id, created_at FROM posts
            WHERE comment_count >= ? AND comments_scraped = 1
        """, (min_comments,)).fetchall()

    logger.info("  %d posts with scraped comments", len(posts))

    hyperedges = []
    timestamps = []
    all_nodes = set()
    skipped = 0

    for pid, author_id, created_at_str in posts:
        if not created_at_str or not author_id:
            skipped += 1
            continue

        try:
            t0 = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            skipped += 1
            continue

        cutoff = (t0 + timedelta(minutes=delta_minutes)).isoformat()

        commenters = conn.execute("""
            SELECT DISTINCT author_id FROM comments
            WHERE post_id = ? AND created_at <= ? AND author_id IS NOT NULL
        """, (pid, cutoff)).fetchall()

        participants = {author_id}
        for (aid,) in commenters:
            participants.add(aid)

        if len(participants) < 2:
            skipped += 1
            continue

        edge = frozenset(participants)
        hyperedges.append(edge)
        timestamps.append(t0)
        all_nodes.update(participants)

    conn.close()

    hg = Hypergraph(
        nodes=all_nodes,
        hyperedges=hyperedges,
        timestamps=timestamps,
        metadata={
            "source": "moltbook_db",
            "delta_minutes": delta_minutes,
            "n_posts_input": len(posts),
            "n_posts_skipped": skipped,
        },
    )
    logger.info("Built Moltbook (DB) hypergraph: %s", hg.summary())
    return hg


# ─── SocioPatterns: temporal aggregation → hyperedge ─────────────────────

def build_sociopatterns_hypergraph(
    dat_path: str,
    delta_seconds: int = 300,
) -> Hypergraph:
    """
    Build hypergraph from SocioPatterns face-to-face contact data.

    Format: each line is "timestamp node_i node_j" (20-second resolution).
    We aggregate contacts in non-overlapping windows of delta_seconds:
    all nodes active in a window form one hyperedge.

    This follows the standard approach from Battiston et al.
    """
    logger.info("Loading SocioPatterns from %s (window=%ds)", dat_path, delta_seconds)

    contacts = []
    with open(dat_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                t, i, j = int(parts[0]), parts[1], parts[2]
                contacts.append((t, i, j))

    if not contacts:
        raise ValueError(f"No contacts found in {dat_path}")

    contacts.sort(key=lambda x: x[0])
    t_min = contacts[0][0]
    t_max = contacts[-1][0]
    logger.info("  %d contact events, time range: %d → %d", len(contacts), t_min, t_max)

    hyperedges = []
    timestamps = []
    all_nodes = set()

    window_start = t_min
    window_participants: set[str] = set()

    ci = 0
    while window_start <= t_max:
        window_end = window_start + delta_seconds
        window_participants.clear()

        while ci < len(contacts) and contacts[ci][0] < window_end:
            _, i, j = contacts[ci]
            window_participants.add(i)
            window_participants.add(j)
            ci += 1

        if len(window_participants) >= 2:
            edge = frozenset(window_participants)
            hyperedges.append(edge)
            timestamps.append(datetime.fromtimestamp(window_start, tz=timezone.utc))
            all_nodes.update(window_participants)

        window_start = window_end

    hg = Hypergraph(
        nodes=all_nodes,
        hyperedges=hyperedges,
        timestamps=timestamps,
        metadata={
            "source": "sociopatterns",
            "file": dat_path,
            "delta_seconds": delta_seconds,
            "n_contacts": len(contacts),
        },
    )
    logger.info("Built SocioPatterns hypergraph: %s", hg.summary())
    return hg
