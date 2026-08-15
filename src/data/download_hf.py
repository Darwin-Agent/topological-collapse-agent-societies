"""
Download existing Moltbook datasets from HuggingFace.

Datasets:
  1. lnajt/moltbook      -- Largest: ~15M rows, posts + comments with parent_id (2.7 GB)
  2. iNLP-Lab/MoltNet     -- Richest: 1M posts + 3.1M comments + 149K agents + 18K submolts (1.6 GB)

Usage:
    python -m src.data.download_hf                 # Download all
    python -m src.data.download_hf --dataset lnajt  # Download specific
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "moltbook_hf"


def download_lnajt():
    """
    lnajt/moltbook: 15M rows, complete comment threading (parent_id).
    Best for hypergraph construction due to completeness.
    """
    from huggingface_hub import hf_hub_download

    dest = DATA_DIR / "lnajt"
    dest.mkdir(parents=True, exist_ok=True)

    for fname in ["posts.parquet", "comments.parquet"]:
        target = dest / fname
        if target.exists():
            logger.info("Already exists: %s", target)
            continue
        logger.info("Downloading lnajt/moltbook/%s ...", fname)
        hf_hub_download(
            repo_id="lnajt/moltbook",
            filename=fname,
            repo_type="dataset",
            local_dir=str(dest),
        )
        logger.info("Saved: %s", target)

    logger.info("lnajt/moltbook download complete → %s", dest)


def download_moltnet():
    """
    iNLP-Lab/MoltNet: Integrated from 10 crawls. Jan 27 - Feb 28.
    Has agents.parquet with karma_history, follower_count_history, etc.
    """
    from huggingface_hub import snapshot_download

    dest = DATA_DIR / "moltnet"
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "data").exists() and any((dest / "data").iterdir()):
        logger.info("MoltNet already downloaded at %s", dest)
        return

    logger.info("Downloading iNLP-Lab/MoltNet (full snapshot) ...")
    snapshot_download(
        repo_id="iNLP-Lab/Moltbook-MoltNet",
        repo_type="dataset",
        local_dir=str(dest),
    )
    logger.info("MoltNet download complete → %s", dest)


def validate_lnajt():
    """Quick validation of lnajt dataset fields for hypergraph construction."""
    import pandas as pd

    dest = DATA_DIR / "lnajt"
    if not (dest / "comments.parquet").exists():
        logger.warning("lnajt not downloaded yet")
        return

    logger.info("Validating lnajt/moltbook fields...")
    comments = pd.read_parquet(dest / "comments.parquet")
    posts = pd.read_parquet(dest / "posts.parquet")

    print(f"\n=== lnajt/moltbook Validation ===")
    print(f"Posts: {len(posts):,}")
    print(f"Comments: {len(comments):,}")
    print(f"\nPost columns: {list(posts.columns)}")
    print(f"Comment columns: {list(comments.columns)}")
    print(f"\nComment parent_id non-null: {comments['parent_id'].notna().sum():,} "
          f"({comments['parent_id'].notna().mean():.1%})")
    print(f"Comment created_at sample: {comments['created_at'].iloc[0]}")
    print(f"Comment author_id non-null: {comments['author_id'].notna().sum():,} "
          f"({comments['author_id'].notna().mean():.1%})")

    # Depth analysis (via parent_id chain)
    top_level = comments["parent_id"].isna().sum()
    nested = comments["parent_id"].notna().sum()
    print(f"\nTop-level comments (no parent): {top_level:,} ({top_level/len(comments):.1%})")
    print(f"Nested replies (has parent): {nested:,} ({nested/len(comments):.1%})")

    # Time range
    print(f"\nTime range: {comments['created_at'].min()} → {comments['created_at'].max()}")
    print(f"Posts time range: {posts['created_at'].min()} → {posts['created_at'].max()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lnajt", "moltnet", "all"], default="all")
    parser.add_argument("--validate", action="store_true", help="Validate after download")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("lnajt", "all"):
        download_lnajt()
    if args.dataset in ("moltnet", "all"):
        download_moltnet()
    if args.validate:
        validate_lnajt()


if __name__ == "__main__":
    main()
