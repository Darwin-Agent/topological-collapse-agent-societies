"""
Build hypergraphs from human baseline datasets using a unified interface.

Each dataset is converted to hyperedges where:
  - thread/email/paper = one hyperedge
  - participants within a time window = members of that hyperedge

This mirrors the Moltbook hypergraph construction in hypergraph_builder.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / "human_baselines"


# reuse Hypergraph dataclass from existing builder
from src.analysis.hypergraph_builder import Hypergraph


def build_from_thread_parquet(
    parquet_path: str | Path,
    name: str,
    delta_minutes: int = 60,
    max_threads: int = 0,
    min_participants: int = 2,
) -> Hypergraph:
    """
    Build a hypergraph from thread-based parquet data.

    Expected columns: thread_id, user_id, timestamp
    Each thread becomes one hyperedge (participants within delta_minutes).
    """
    df = pd.read_parquet(str(parquet_path))
    logger.info("Loaded %s: %d records", name, len(df))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.Timestamp.now(tz="UTC")

    # group by thread
    threads = df.groupby("thread_id")
    if max_threads > 0:
        thread_ids = list(threads.groups.keys())[:max_threads]
    else:
        thread_ids = list(threads.groups.keys())

    hyperedges = []
    node_set = set()

    for tid in thread_ids:
        group = threads.get_group(tid)
        if delta_minutes > 0 and "timestamp" in group.columns:
            # filter by time window from first post
            t0 = group["timestamp"].min()
            if pd.notna(t0):
                cutoff = t0 + pd.Timedelta(minutes=delta_minutes)
                group = group[group["timestamp"] <= cutoff]

        participants = set(group["user_id"].unique())
        if len(participants) >= min_participants:
            edge = frozenset(str(u) for u in participants)
            hyperedges.append(edge)
            node_set.update(edge)

    logger.info("  %s hypergraph: %d nodes, %d hyperedges", name, len(node_set), len(hyperedges))

    return Hypergraph(nodes=node_set, hyperedges=hyperedges, metadata={"name": name})


def build_all_human_hypergraphs(
    delta_minutes: int = 60,
    max_threads: int = 50000,
) -> dict[str, Hypergraph]:
    """Build hypergraphs for all available human baseline datasets."""
    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest found. Run download_human_baselines.py first.")
        return {}

    manifest = json.loads(manifest_path.read_text())
    hypergraphs = {}

    for name, path in manifest.items():
        path = Path(path)
        if not path.exists():
            logger.warning("Skipping %s: path not found", name)
            continue

        if path.suffix == ".parquet":
            try:
                hg = build_from_thread_parquet(
                    path, name=name,
                    delta_minutes=delta_minutes,
                    max_threads=max_threads,
                )
                hypergraphs[name] = hg
            except Exception as e:
                logger.error("Failed to build %s hypergraph: %s", name, e)
        elif path.is_dir():
            logger.info("Skipping directory-based dataset: %s", name)

    return hypergraphs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    hgs = build_all_human_hypergraphs()
    for name, hg in hgs.items():
        logger.info("  %s: %d nodes, %d edges", name, len(hg.nodes), len(hg.hyperedges))
