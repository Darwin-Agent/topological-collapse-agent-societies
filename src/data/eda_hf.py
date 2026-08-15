"""
Exploratory data analysis on lnajt/moltbook HuggingFace dataset.
Validates field completeness for hypergraph construction and produces
key statistics on comment threading, depth distribution, and hyperedge size.

Usage:
    python -m src.data.eda_hf
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "moltbook_hf" / "lnajt"


def load_data():
    print("Loading lnajt/moltbook data...")
    posts = pd.read_parquet(DATA_DIR / "posts.parquet")
    comments = pd.read_parquet(DATA_DIR / "comments.parquet")
    print(f"  Posts: {len(posts):,}")
    print(f"  Comments: {len(comments):,}")
    return posts, comments


def field_validation(posts, comments):
    print("\n" + "=" * 60)
    print("FIELD VALIDATION FOR HYPERGRAPH CONSTRUCTION")
    print("=" * 60)

    print("\n--- Posts ---")
    print(f"Columns: {list(posts.columns)}")
    for col in ["id", "author_id", "created_at", "submolt", "comment_count"]:
        if col in posts.columns:
            non_null = posts[col].notna().sum()
            print(f"  {col}: {non_null:,} non-null ({non_null/len(posts):.1%})")

    print(f"\n  Time range: {posts['created_at'].min()} → {posts['created_at'].max()}")
    print(f"  Unique authors: {posts['author_id'].nunique():,}")
    print(f"  Unique submolts: {posts['submolt'].nunique():,}")

    print("\n--- Comments ---")
    print(f"Columns: {list(comments.columns)}")
    for col in ["id", "post_id", "parent_id", "author_id", "created_at"]:
        if col in comments.columns:
            non_null = comments[col].notna().sum()
            print(f"  {col}: {non_null:,} non-null ({non_null/len(comments):.1%})")

    print(f"\n  Time range: {comments['created_at'].min()} → {comments['created_at'].max()}")
    print(f"  Unique comment authors: {comments['author_id'].nunique():,}")


def comment_depth_analysis(comments):
    """Analyze comment threading structure — key for hypergraph feasibility."""
    print("\n" + "=" * 60)
    print("COMMENT DEPTH / THREADING ANALYSIS")
    print("=" * 60)

    top_level = comments["parent_id"].isna().sum()
    nested = comments["parent_id"].notna().sum()
    print(f"\n  Top-level (depth 0, no parent): {top_level:,} ({top_level/len(comments):.1%})")
    print(f"  Nested (has parent_id):          {nested:,} ({nested/len(comments):.1%})")

    # Comments per post distribution
    cpc = comments.groupby("post_id").size()
    print(f"\n  Comments per post:")
    print(f"    mean:   {cpc.mean():.1f}")
    print(f"    median: {cpc.median():.0f}")
    print(f"    p90:    {cpc.quantile(0.9):.0f}")
    print(f"    p99:    {cpc.quantile(0.99):.0f}")
    print(f"    max:    {cpc.max():,}")

    # Posts with >100 comments (the truncation boundary in giordano dataset)
    over_100 = (cpc > 100).sum()
    over_500 = (cpc > 500).sum()
    print(f"\n  Posts with >100 comments: {over_100:,} ({over_100/len(cpc):.2%})")
    print(f"  Posts with >500 comments: {over_500:,}")

    # Replies per comment (who got replied to)
    if nested > 0:
        reply_counts = comments[comments["parent_id"].notna()].groupby("parent_id").size()
        comments_with_replies = reply_counts.shape[0]
        total_comments = len(comments)
        pct_zero_reply = 1 - comments_with_replies / total_comments
        print(f"\n  Comments receiving >= 1 reply: {comments_with_replies:,} ({comments_with_replies/total_comments:.1%})")
        print(f"  Comments with ZERO replies:    {pct_zero_reply:.1%}  ← key metric (Holtz: 93.5%)")


def hyperedge_size_estimation(comments, posts, delta_t_minutes=60):
    """
    Estimate hyperedge sizes using Battiston's method:
    If k distinct agents comment on post P within time window delta_t,
    then {P.author, commenter_1, ..., commenter_k} form a (k+1)-hyperedge.
    """
    print("\n" + "=" * 60)
    print(f"HYPEREDGE SIZE ESTIMATION (delta_t = {delta_t_minutes} min)")
    print("=" * 60)

    # Parse timestamps
    comments_ts = comments[["post_id", "author_id", "created_at"]].copy()
    comments_ts["created_at"] = pd.to_datetime(comments_ts["created_at"], errors="coerce")
    comments_ts = comments_ts.dropna(subset=["created_at", "author_id", "post_id"])

    # Get post authors
    post_authors = posts.set_index("id")["author_id"].to_dict()

    # For each post, find unique commenters in rolling windows
    hyperedge_sizes = []
    sampled_posts = comments_ts["post_id"].value_counts()
    # Sample top 50K posts for speed
    target_posts = sampled_posts.head(50000).index.tolist()

    print(f"  Analyzing {len(target_posts):,} posts...")

    for i, post_id in enumerate(target_posts):
        post_comments = comments_ts[comments_ts["post_id"] == post_id].sort_values("created_at")
        if len(post_comments) < 2:
            hyperedge_sizes.append(1 + len(post_comments))
            continue

        # Simplification: treat all commenters on a post as one hyperedge
        # (Battiston method: within delta_t window)
        unique_commenters = post_comments["author_id"].nunique()
        post_author = post_authors.get(post_id)
        if post_author:
            s = unique_commenters + 1  # +1 for post author
        else:
            s = unique_commenters
        hyperedge_sizes.append(s)

        if i % 10000 == 0 and i > 0:
            print(f"    processed {i:,} posts...")

    sizes = pd.Series(hyperedge_sizes)
    print(f"\n  Hyperedge size distribution (s = #agents in hyperedge):")
    print(f"    s=1 (isolated posts):  {(sizes == 1).sum():,} ({(sizes == 1).mean():.1%})")
    print(f"    s=2 (dyadic):          {(sizes == 2).sum():,} ({(sizes == 2).mean():.1%})")
    print(f"    s=3 (triadic):         {(sizes == 3).sum():,} ({(sizes == 3).mean():.1%})")
    print(f"    s=4-5:                 {((sizes >= 4) & (sizes <= 5)).sum():,}")
    print(f"    s=6-10:                {((sizes >= 6) & (sizes <= 10)).sum():,}")
    print(f"    s=11-50:               {((sizes >= 11) & (sizes <= 50)).sum():,}")
    print(f"    s=51-100:              {((sizes >= 51) & (sizes <= 100)).sum():,}")
    print(f"    s>100:                 {(sizes > 100).sum():,}")
    print(f"\n    mean: {sizes.mean():.2f}")
    print(f"    median: {sizes.median():.0f}")
    print(f"    max: {sizes.max():,}")

    # Key question: what fraction are purely dyadic (s<=2)?
    dyadic_or_less = (sizes <= 2).mean()
    print(f"\n  *** CRITICAL: {dyadic_or_less:.1%} of hyperedges are s<=2 (dyadic or isolated) ***")
    print(f"  *** Only {1 - dyadic_or_less:.1%} involve genuine higher-order (3+) interactions ***")


def reciprocity_analysis(comments):
    """Measure reciprocity: if A replies to B, does B ever reply to A?"""
    print("\n" + "=" * 60)
    print("RECIPROCITY ANALYSIS")
    print("=" * 60)

    # Build directed reply graph from parent_id
    nested = comments[comments["parent_id"].notna()][["author_id", "parent_id", "post_id"]].copy()

    # Map parent_id -> parent author
    comment_authors = comments.set_index("id")["author_id"].to_dict()
    nested["parent_author"] = nested["parent_id"].map(comment_authors)
    nested = nested.dropna(subset=["parent_author", "author_id"])
    # Remove self-replies
    nested = nested[nested["author_id"] != nested["parent_author"]]

    print(f"  Valid directed reply edges: {len(nested):,}")

    # Directed edges: author_id -> parent_author
    edges = set(zip(nested["author_id"], nested["parent_author"]))
    reverse_edges = set((b, a) for a, b in edges)
    reciprocal = edges & reverse_edges

    print(f"  Unique directed edges: {len(edges):,}")
    print(f"  Reciprocal pairs: {len(reciprocal):,}")
    if len(edges) > 0:
        r = len(reciprocal) / len(edges)
        print(f"  Reciprocity rate: {r:.3f}  (Holtz reported 0.197)")


def main():
    posts, comments = load_data()
    field_validation(posts, comments)
    comment_depth_analysis(comments)
    reciprocity_analysis(comments)
    hyperedge_size_estimation(comments, posts)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
