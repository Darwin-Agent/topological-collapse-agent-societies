"""
Moltbook scraper: fetches posts (with full comment trees) and agent profiles.

Usage:
    python -m src.data.scraper posts          # Scrape all posts (offset pagination)
    python -m src.data.scraper comments       # Scrape comments for all posts
    python -m src.data.scraper comments --min-comments 10  # Only posts with >=10 comments
    python -m src.data.scraper agents         # Enrich agent profiles
    python -m src.data.scraper status         # Show database stats
    python -m src.data.scraper full           # Run all stages sequentially

Supports checkpoint/resume: safe to interrupt and restart.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.moltbook_client import MoltbookClient
from src.data.database import Database

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "moltbook" / "moltbook.db")


def scrape_posts(client: MoltbookClient, db: Database, batch_size: int = 100):
    """Scrape all posts using cursor pagination. Supports resume via saved cursor."""
    saved_cursor = db.get_state("posts_cursor")
    cursor = saved_cursor
    total_new = 0
    page = 0

    logger.info("Starting post scrape (%s, existing: %d posts)",
                "resuming" if saved_cursor else "from beginning", db.post_count())

    try:
        while True:
            posts, next_cursor, has_more = client.get_posts_page(
                sort="new", limit=batch_size, cursor=cursor
            )
            if not posts:
                break

            for post in posts:
                db.upsert_post(post)
                total_new += 1

            page += 1
            db.commit()

            if next_cursor:
                db.set_state("posts_cursor", next_cursor)
                cursor = next_cursor

            if page % 10 == 0:
                logger.info("Page %d: +%d posts this run (DB total: %d)",
                            page, total_new, db.post_count())

            if not has_more or not next_cursor:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted. Progress saved.")
    finally:
        db.commit()
        logger.info("Post scrape done. New: %d, DB total: %d", total_new, db.post_count())


def scrape_comments(client: MoltbookClient, db: Database,
                    min_comments: int = 1, max_comments: int = 0):
    """
    Scrape full comment trees for posts. Skips posts with 0 comments.
    Prioritizes high-interaction posts (DESC by comment_count).
    Supports resume via comments_scraped flag on each post.
    """
    total_pending = db.count_posts_needing_comments(min_comments=min_comments)
    already_done = db.count_comments_scraped_posts()
    logger.info("Comment scrape: %d posts pending, %d already done "
                "(min_comments=%d, max_comments=%s)",
                total_pending, already_done, min_comments,
                max_comments if max_comments > 0 else "unlimited")

    BATCH = 5000
    global_i = 0
    total_comments_added = 0
    start_time = time.time()

    while True:
        posts = db.get_posts_needing_comments(
            min_comments=min_comments, max_comments=max_comments, limit=BATCH
        )
        if not posts:
            break

        for post_id, expected_count in posts:
            global_i += 1
            try:
                comment_n = 0
                page_n = 0
                for comment in client.iter_comments(post_id, sort="new"):
                    db.upsert_comment_tree(comment, post_id, parent_id=None,
                                           depth=comment.get("depth", 0))
                    comment_n += 1
                    page_n += 1

                    if expected_count > 500 and page_n % 200 == 0:
                        db.commit()
                        logger.info(
                            "  ... Post %s in progress: %d/%d top-level comments fetched",
                            post_id[:8], page_n, expected_count
                        )

                db.mark_comments_scraped(post_id)
                total_comments_added += comment_n

                if global_i % 20 == 0:
                    db.commit()

                elapsed = time.time() - start_time
                rate = global_i / elapsed if elapsed > 0 else 0

                should_log = (
                    global_i % 200 == 0 or
                    expected_count > 50 or
                    global_i <= 10
                )
                if should_log:
                    logger.info(
                        "[%d/%d] Post %s: expected=%d, fetched=%d | "
                        "Total: %d comments | %.1f posts/min",
                        global_i, total_pending, post_id[:8],
                        expected_count, comment_n,
                        total_comments_added, rate * 60
                    )

            except KeyboardInterrupt:
                db.commit()
                logger.info("Interrupted. Progress saved at post %d.", global_i)
                return
            except Exception as e:
                logger.error("Failed on post %s (expected=%d): %s",
                             post_id[:8], expected_count, e)
                db.commit()
                continue

        db.commit()

    db.commit()
    elapsed = time.time() - start_time
    logger.info("Comment scrape complete. Posts processed: %d, "
                "Comments added: %d, Time: %.1f min",
                global_i, total_comments_added, elapsed / 60)


def scrape_agents(client: MoltbookClient, db: Database):
    """Enrich agent profiles for all agents seen in posts/comments."""
    names = db.get_unique_agent_names()
    logger.info("Found %d agents to profile (already have %d)",
                len(names), db.agent_count())

    for i, (name,) in enumerate(names):
        try:
            data = client.get_agent_profile(name)
            agent = data.get("agent", data)
            db.upsert_agent(agent)
            if i % 100 == 0:
                db.commit()
                logger.info("[%d/%d] Profiled agent: %s", i + 1, len(names), name)
        except Exception as e:
            logger.warning("Failed to profile agent %s: %s", name, e)
            continue

    db.commit()
    logger.info("Agent enrichment complete. Total: %d", db.agent_count())


def show_status(db: Database):
    stats = db.stats()
    print("\n=== Moltbook Database Status ===")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

    depth_dist = db.conn.execute("""
        SELECT depth, COUNT(*) FROM comments GROUP BY depth ORDER BY depth
    """).fetchall()
    if depth_dist:
        print("\n  Comment depth distribution:")
        for d, c in depth_dist:
            print(f"    depth {d}: {c:,}")

    top_posts = db.conn.execute("""
        SELECT id, comment_count, comments_scraped
        FROM posts ORDER BY comment_count DESC LIMIT 5
    """).fetchall()
    if top_posts:
        print("\n  Top posts by comment_count:")
        for pid, cc, scraped in top_posts:
            status = "done" if scraped else "pending"
            print(f"    {pid[:12]}... comments={cc:,} [{status}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="Moltbook scraper")
    parser.add_argument("command", choices=["posts", "comments", "agents", "status", "full"],
                        help="Scrape stage to run")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--min-comments", type=int, default=1,
                        help="Minimum comment count for comment scraping")
    parser.add_argument("--max-comments", type=int, default=0,
                        help="Maximum comment count (0=unlimited)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Posts per API request")
    args = parser.parse_args()

    api_key = os.getenv("MOLTBOOK_API_KEY")
    if not api_key and args.command != "status":
        logger.error("MOLTBOOK_API_KEY not set. Check .env file.")
        sys.exit(1)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = Database(args.db)

    if args.command == "status":
        show_status(db)
        db.close()
        return

    client = MoltbookClient(api_key)

    try:
        if args.command == "posts":
            scrape_posts(client, db, args.batch_size)
        elif args.command == "comments":
            scrape_comments(client, db, min_comments=args.min_comments,
                            max_comments=args.max_comments)
        elif args.command == "agents":
            scrape_agents(client, db)
        elif args.command == "full":
            logger.info("=== Full scrape: posts → comments → agents ===")
            scrape_posts(client, db, args.batch_size)
            scrape_comments(client, db)
            scrape_agents(client, db)
            show_status(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
