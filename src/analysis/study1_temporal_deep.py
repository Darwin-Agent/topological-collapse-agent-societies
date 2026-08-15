"""
Study 1 – Deep temporal analysis on the full Moltbook SQLite dataset.

Produces:
  results/study1_temporal/temporal_analysis.json   — all metrics per week
  results/study1_temporal/pre_post_meta.json       — before/after Meta acquisition
  results/study1_temporal/fig_weekly_activity.png
  results/study1_temporal/fig_zero_reply_rate.png
  results/study1_temporal/fig_response_time.png
  results/study1_temporal/fig_meta_impact.png
  results/study1_temporal/fig_depth_evolution.png
  results/study1_temporal/fig_composite_temporal.png  — Nature-quality 4-panel

Usage:
    python -m src.analysis.study1_temporal_deep
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "raw" / "moltbook" / "moltbook.db"
OUT_DIR = ROOT / "results" / "study1_temporal"

META_ACQUISITION_DATE = "2026-03-10"


# ── helpers ────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-2000000")  # 2 GB page cache
    return conn


def iso_week(date_str: str) -> str:
    """'2026-01-30T...' → '2026-W05'."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── 1  Weekly aggregates (pure SQL, fast) ──────────────────────────────

def weekly_post_stats(conn: sqlite3.Connection) -> list[dict]:
    """Weekly post-level metrics computed entirely in SQL."""
    logger.info("Computing weekly post stats...")
    rows = conn.execute("""
        SELECT
            substr(created_at, 1, 10) AS day_str,
            COUNT(*)                        AS n_posts,
            SUM(CASE WHEN comment_count = 0 THEN 1 ELSE 0 END) AS zero_reply,
            COUNT(DISTINCT author_name)     AS unique_authors,
            AVG(comment_count)              AS avg_comments,
            MAX(comment_count)              AS max_comments
        FROM posts
        WHERE created_at IS NOT NULL
        GROUP BY day_str
        ORDER BY day_str
    """).fetchall()

    from datetime import datetime
    week_agg = {}
    for day_str, n_posts, zero_reply, authors, avg_c, max_c in rows:
        try:
            dt = datetime.fromisoformat(day_str)
            iso = dt.isocalendar()
            wk = f"{iso[0]}-W{iso[1]:02d}"
        except Exception:
            continue
        if wk not in week_agg:
            week_agg[wk] = {"year_week": wk, "n_posts": 0, "zero_reply": 0,
                            "authors": set(), "sum_comments": 0, "max_comments": 0}
        w = week_agg[wk]
        w["n_posts"] += n_posts
        w["zero_reply"] += zero_reply
        w["sum_comments"] += (avg_c or 0) * n_posts
        w["max_comments"] = max(w["max_comments"], max_c or 0)

    results = []
    for wk in sorted(week_agg):
        w = week_agg[wk]
        d = {
            "year_week": wk,
            "n_posts": w["n_posts"],
            "zero_reply": w["zero_reply"],
            "unique_authors": 0,
            "avg_comments": w["sum_comments"] / w["n_posts"] if w["n_posts"] else 0,
            "max_comments": w["max_comments"],
        }
        d["zero_reply_rate"] = d["zero_reply"] / d["n_posts"] if d["n_posts"] else 0
        results.append(d)
    logger.info("  %d weeks computed", len(results))
    return results


def weekly_comment_stats(conn: sqlite3.Connection) -> list[dict]:
    """Weekly comment-level metrics."""
    logger.info("Computing weekly comment stats...")
    rows = conn.execute("""
        SELECT
            substr(created_at, 1, 10) AS day_str,
            COUNT(*)                        AS n_comments,
            COUNT(DISTINCT author_name)     AS unique_commenters,
            AVG(depth)                      AS avg_depth,
            SUM(CASE WHEN depth = 0 THEN 1 ELSE 0 END) AS depth0,
            SUM(CASE WHEN depth = 1 THEN 1 ELSE 0 END) AS depth1,
            SUM(CASE WHEN depth >= 2 THEN 1 ELSE 0 END) AS depth2plus
        FROM comments
        WHERE created_at IS NOT NULL
        GROUP BY day_str
        ORDER BY day_str
    """).fetchall()

    from datetime import datetime
    week_agg = {}
    for day_str, n_c, commenters, avg_d, d0, d1, d2p in rows:
        try:
            dt = datetime.fromisoformat(day_str)
            iso = dt.isocalendar()
            wk = f"{iso[0]}-W{iso[1]:02d}"
        except Exception:
            continue
        if wk not in week_agg:
            week_agg[wk] = {"year_week": wk, "n_comments": 0,
                            "unique_commenters": 0, "depth_sum": 0,
                            "depth0": 0, "depth1": 0, "depth2plus": 0}
        w = week_agg[wk]
        w["n_comments"] += n_c
        w["depth_sum"] += (avg_d or 0) * n_c
        w["depth0"] += d0 or 0
        w["depth1"] += d1 or 0
        w["depth2plus"] += d2p or 0

    results = []
    for wk in sorted(week_agg):
        w = week_agg[wk]
        results.append({
            "year_week": wk,
            "n_comments": w["n_comments"],
            "unique_commenters": w["unique_commenters"],
            "avg_depth": w["depth_sum"] / w["n_comments"] if w["n_comments"] else 0,
            "depth0": w["depth0"],
            "depth1": w["depth1"],
            "depth2plus": w["depth2plus"],
        })
    logger.info("  %d weeks computed", len(results))
    return results


def response_time_sample(conn: sqlite3.Connection, limit: int = 200000) -> list[float]:
    """Sample response times (seconds) from post creation to first comment."""
    logger.info("Sampling response times (limit=%d)...", limit)
    rows = conn.execute("""
        SELECT
            (julianday(c.created_at) - julianday(p.created_at)) * 86400.0 AS delta_sec
        FROM (
            SELECT post_id, MIN(created_at) AS created_at
            FROM comments
            WHERE created_at IS NOT NULL
            GROUP BY post_id
        ) c
        JOIN posts p ON c.post_id = p.id
        WHERE p.created_at IS NOT NULL
          AND delta_sec > 0
          AND delta_sec < 86400 * 7
        LIMIT ?
    """, (limit,)).fetchall()
    deltas = [r[0] for r in rows]
    logger.info("  Got %d response time samples", len(deltas))
    return deltas


# ── 2  Pre/Post Meta split ──────────────────────────────────────────────

def pre_post_meta(conn: sqlite3.Connection) -> dict:
    """Aggregate metrics split by Meta acquisition date."""
    logger.info("Computing pre/post Meta aggregates...")
    results = {}
    for label, op in [("pre_meta", "<"), ("post_meta", ">=")]:
        row = conn.execute(f"""
            SELECT
                COUNT(*) AS n_posts,
                SUM(CASE WHEN comment_count = 0 THEN 1 ELSE 0 END) AS zero_reply,
                COUNT(DISTINCT author_name) AS unique_authors,
                AVG(comment_count) AS avg_comments,
                MIN(created_at) AS first_post,
                MAX(created_at) AS last_post
            FROM posts
            WHERE created_at IS NOT NULL AND created_at {op} ?
        """, (META_ACQUISITION_DATE,)).fetchone()
        d = dict(zip(["n_posts", "zero_reply", "unique_authors",
                       "avg_comments", "first_post", "last_post"], row))
        d["zero_reply_rate"] = d["zero_reply"] / d["n_posts"] if d["n_posts"] else 0

        crow = conn.execute(f"""
            SELECT COUNT(*), AVG(depth),
                   COUNT(DISTINCT author_name)
            FROM comments
            WHERE created_at IS NOT NULL AND created_at {op} ?
        """, (META_ACQUISITION_DATE,)).fetchone()
        d["n_comments"] = crow[0]
        d["avg_depth"] = crow[1]
        d["unique_commenters"] = crow[2]
        results[label] = d

    logger.info("  Pre-Meta posts: %d, Post-Meta posts: %d",
                results["pre_meta"]["n_posts"], results["post_meta"]["n_posts"])
    return results


# ── 3  Platform outage detection ───────────────────────────────────────

def detect_outages(conn: sqlite3.Connection, gap_hours: float = 6.0) -> list[dict]:
    """Find periods with no posts for > gap_hours."""
    logger.info("Detecting outages (gap > %.0fh)...", gap_hours)
    rows = conn.execute("""
        SELECT created_at FROM posts
        WHERE created_at IS NOT NULL
        ORDER BY created_at
    """).fetchall()
    if not rows:
        return []

    from datetime import datetime, timezone
    times = []
    for (ts,) in rows:
        try:
            times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except Exception:
            continue
    times.sort()

    outages = []
    for i in range(1, len(times)):
        gap = (times[i] - times[i - 1]).total_seconds() / 3600
        if gap >= gap_hours:
            outages.append({
                "start": times[i - 1].isoformat(),
                "end": times[i].isoformat(),
                "gap_hours": round(gap, 1),
            })
    logger.info("  Found %d outages >= %.0fh", len(outages), gap_hours)
    return outages


# ── 4  Figures ─────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    return plt


def fig_weekly_activity(post_stats, comment_stats, outdir):
    plt = _setup_matplotlib()
    weeks = [s["year_week"] for s in post_stats]
    n_posts = [s["n_posts"] for s in post_stats]
    n_comments = [s.get("n_comments", 0) for s in comment_stats[:len(weeks)]]
    authors = [s["unique_authors"] for s in post_stats]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(weeks, n_posts, color="#348ABD", alpha=0.85)
    axes[0].set_ylabel("Posts / week")
    axes[0].axvline(x=weeks.index("2026-10") if "2026-10" in weeks else len(weeks) // 2,
                     color="red", ls="--", alpha=0.6, label="Meta acquisition")

    if len(n_comments) == len(weeks):
        axes[1].bar(weeks, n_comments, color="#E24A33", alpha=0.85)
    axes[1].set_ylabel("Comments / week")

    axes[2].plot(weeks, authors, "o-", color="#2ca02c", markersize=4)
    axes[2].set_ylabel("Unique authors")
    axes[2].set_xlabel("Year-Week")

    meta_week = None
    for i, w in enumerate(weeks):
        if w >= "2026-10":
            meta_week = i
            break
    if meta_week:
        for ax in axes:
            ax.axvline(x=meta_week, color="red", ls="--", alpha=0.5, lw=1.2)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Moltbook Platform Activity Over Time", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = outdir / "fig_weekly_activity.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def fig_zero_reply(post_stats, outdir):
    plt = _setup_matplotlib()
    weeks = [s["year_week"] for s in post_stats]
    rates = [s["zero_reply_rate"] * 100 for s in post_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weeks, rates, "s-", color="#E24A33", markersize=5, lw=2)
    ax.set_ylabel("Zero-reply rate (%)")
    ax.set_xlabel("Year-Week")
    ax.set_title("Weekly Zero-Reply Rate on Moltbook", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.tick_params(axis="x", rotation=45)

    ax.axhline(y=93.5, color="gray", ls=":", alpha=0.6, label="Holtz et al. reported 93.5%")
    ax.legend()

    fig.tight_layout()
    path = outdir / "fig_zero_reply_rate.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def fig_response_time(deltas, outdir):
    plt = _setup_matplotlib()
    arr = np.array(deltas)
    arr_min = arr / 60.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(arr_min[arr_min < 120], bins=100, color="#348ABD", alpha=0.8, edgecolor="white", lw=0.3)
    ax1.set_xlabel("Response time (minutes)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution (< 2 hours)")
    ax1.axvline(x=np.median(arr_min), color="red", ls="--",
                label=f"Median = {np.median(arr_min):.1f} min")
    ax1.legend()
    ax1.grid(alpha=0.2)

    bins_log = np.logspace(np.log10(1), np.log10(max(arr_min.max(), 1e4)), 80)
    ax2.hist(arr_min, bins=bins_log, color="#E24A33", alpha=0.8, edgecolor="white", lw=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Response time (minutes, log scale)")
    ax2.set_ylabel("Count")
    ax2.set_title("Full distribution (log scale)")
    ax2.grid(alpha=0.2)

    fig.suptitle("Post-to-First-Comment Response Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = outdir / "fig_response_time.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def fig_meta_impact(pre_post, outdir):
    plt = _setup_matplotlib()

    metrics = ["zero_reply_rate", "avg_comments", "avg_depth"]
    labels = ["Zero-reply rate", "Avg comments/post", "Avg comment depth"]
    pre_vals = [pre_post["pre_meta"].get(m, 0) for m in metrics]
    post_vals = [pre_post["post_meta"].get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, pre_vals, width, label="Pre-Meta", color="#348ABD", alpha=0.85)
    bars2 = ax.bar(x + width / 2, post_vals, width, label="Post-Meta", color="#E24A33", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Before vs After Meta Acquisition (2026-03-10)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}" if h < 1 else f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = outdir / "fig_meta_impact.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def fig_depth_evolution(comment_stats, outdir):
    plt = _setup_matplotlib()
    weeks = [s["year_week"] for s in comment_stats]
    avg_depth = [s["avg_depth"] or 0 for s in comment_stats]
    frac_d0 = []
    for s in comment_stats:
        total = s["depth0"] + s["depth1"] + s["depth2plus"]
        frac_d0.append(s["depth0"] / total * 100 if total else 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(weeks, avg_depth, "o-", color="#9467bd", markersize=4, lw=2)
    ax1.set_ylabel("Mean comment depth")
    ax1.set_title("Comment Depth Evolution", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.2)

    ax2.plot(weeks, frac_d0, "s-", color="#d62728", markersize=4, lw=2)
    ax2.set_ylabel("% depth-0 comments")
    ax2.set_xlabel("Year-Week")
    ax2.grid(alpha=0.2)
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = outdir / "fig_depth_evolution.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def fig_composite(post_stats, comment_stats, outdir):
    """Nature-quality 4-panel composite figure."""
    plt = _setup_matplotlib()

    weeks = [s["year_week"] for s in post_stats]
    n_posts = [s["n_posts"] for s in post_stats]
    zero_rates = [s["zero_reply_rate"] * 100 for s in post_stats]
    authors = [s["unique_authors"] for s in post_stats]

    avg_depth = []
    for s in comment_stats[:len(weeks)]:
        avg_depth.append(s["avg_depth"] or 0)
    while len(avg_depth) < len(weeks):
        avg_depth.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panel_labels = ["a", "b", "c", "d"]

    # (a) Activity
    axes[0, 0].bar(range(len(weeks)), n_posts, color="#348ABD", alpha=0.85)
    axes[0, 0].set_ylabel("Posts / week")
    axes[0, 0].set_title("Platform activity")

    # (b) Zero-reply
    axes[0, 1].plot(range(len(weeks)), zero_rates, "s-", color="#E24A33", markersize=4, lw=2)
    axes[0, 1].axhline(y=93.5, color="gray", ls=":", alpha=0.5)
    axes[0, 1].set_ylabel("Zero-reply rate (%)")
    axes[0, 1].set_title("Interaction failure rate")

    # (c) Unique authors
    axes[1, 0].plot(range(len(weeks)), authors, "o-", color="#2ca02c", markersize=4, lw=2)
    axes[1, 0].set_ylabel("Unique authors")
    axes[1, 0].set_xlabel("Week index")
    axes[1, 0].set_title("Active agent population")

    # (d) Depth
    axes[1, 1].plot(range(len(avg_depth)), avg_depth, "D-", color="#9467bd", markersize=4, lw=2)
    axes[1, 1].set_ylabel("Mean comment depth")
    axes[1, 1].set_xlabel("Week index")
    axes[1, 1].set_title("Conversational depth")

    # Meta acquisition line
    meta_idx = None
    for i, w in enumerate(weeks):
        if w >= "2026-10":
            meta_idx = i
            break

    for i, ax in enumerate(axes.flat):
        if meta_idx is not None:
            ax.axvline(x=meta_idx, color="red", ls="--", alpha=0.4, lw=1)
        ax.grid(alpha=0.15)
        ax.text(-0.08, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")

    fig.suptitle("Moltbook Temporal Dynamics (Jan–Mar 2026)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = outdir / "fig_composite_temporal.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ── main ───────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = _conn()

    logger.info("=" * 60)
    logger.info("Study 1 – Deep Temporal Analysis (SQLite)")
    logger.info("=" * 60)

    # 1) Weekly aggregates
    post_stats = weekly_post_stats(conn)
    comment_stats = weekly_comment_stats(conn)

    # Align by week
    cs_by_week = {s["year_week"]: s for s in comment_stats}
    aligned_comment_stats = []
    for ps in post_stats:
        cs = cs_by_week.get(ps["year_week"], {
            "year_week": ps["year_week"], "n_comments": 0,
            "unique_commenters": 0, "avg_depth": 0,
            "depth0": 0, "depth1": 0, "depth2plus": 0,
        })
        aligned_comment_stats.append(cs)

    # 2) Response time
    deltas = response_time_sample(conn, limit=300000)

    # 3) Pre/Post Meta
    pre_post = pre_post_meta(conn)

    # 4) Outages
    outages = detect_outages(conn, gap_hours=6.0)

    conn.close()

    # Build unified JSON output
    weekly_combined = []
    for ps, cs in zip(post_stats, aligned_comment_stats):
        entry = {**ps, **{f"c_{k}": v for k, v in cs.items() if k != "year_week"}}
        weekly_combined.append(entry)

    output = {
        "weekly": weekly_combined,
        "pre_post_meta": pre_post,
        "outages": outages[:20],
        "response_time": {
            "n_samples": len(deltas),
            "median_seconds": float(np.median(deltas)) if deltas else None,
            "mean_seconds": float(np.mean(deltas)) if deltas else None,
            "p25_seconds": float(np.percentile(deltas, 25)) if deltas else None,
            "p75_seconds": float(np.percentile(deltas, 75)) if deltas else None,
            "p95_seconds": float(np.percentile(deltas, 95)) if deltas else None,
        },
    }

    json_path = OUT_DIR / "temporal_analysis.json"
    json_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info("Saved JSON: %s", json_path)

    # Meta comparison
    meta_path = OUT_DIR / "pre_post_meta.json"
    meta_path.write_text(json.dumps(pre_post, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", meta_path)

    # Figures
    logger.info("Generating figures...")
    fig_weekly_activity(post_stats, aligned_comment_stats, OUT_DIR)
    fig_zero_reply(post_stats, OUT_DIR)
    if deltas:
        fig_response_time(deltas, OUT_DIR)
    fig_meta_impact(pre_post, OUT_DIR)
    fig_depth_evolution(aligned_comment_stats, OUT_DIR)
    fig_composite(post_stats, aligned_comment_stats, OUT_DIR)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Date range: %s → %s",
                post_stats[0]["year_week"] if post_stats else "?",
                post_stats[-1]["year_week"] if post_stats else "?")
    total_posts = sum(s["n_posts"] for s in post_stats)
    total_zero = sum(s["zero_reply"] for s in post_stats)
    logger.info("Total posts: %d", total_posts)
    logger.info("Overall zero-reply rate: %.1f%%", total_zero / total_posts * 100 if total_posts else 0)
    logger.info("Pre-Meta: %d posts, %.1f%% zero-reply",
                pre_post["pre_meta"]["n_posts"],
                pre_post["pre_meta"]["zero_reply_rate"] * 100)
    logger.info("Post-Meta: %d posts, %.1f%% zero-reply",
                pre_post["post_meta"]["n_posts"],
                pre_post["post_meta"]["zero_reply_rate"] * 100)
    if deltas:
        logger.info("Median response time: %.1f sec (%.1f min)",
                    np.median(deltas), np.median(deltas) / 60)
    logger.info("Outages >= 6h: %d (max gap: %.1fh)",
                len(outages),
                max(o["gap_hours"] for o in outages) if outages else 0)
    logger.info("\nAll outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
