"""
Async high-speed comment scraper for high-interaction posts (>100 comments).

Uses asyncio + aiohttp with multiple concurrent workers to bypass
the single-threaded bottleneck. Each worker handles one post at a time,
paginating through its comments sequentially, but multiple posts are
processed in parallel.

Usage:
    python -m src.data.scraper_async --workers 5 --min-comments 101
    python -m src.data.scraper_async --workers 3 --min-comments 50 --max-comments 500
"""

import os
import ssl
import sys
import time
import asyncio
import argparse
import logging
from pathlib import Path

import aiohttp
import certifi
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.database import Database

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.moltbook.com/api/v1"
DEFAULT_DB = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "moltbook" / "moltbook.db")

# Global rate limiter: token bucket shared across all workers
REQUEST_INTERVAL = 0.35  # target ~3 req/s total; conservative to avoid 500s


class AsyncRateLimiter:
    """Token bucket rate limiter for concurrent workers."""
    def __init__(self, rate: float):
        self.rate = rate
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self.rate - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


class AsyncStats:
    """Thread-safe counters for progress tracking."""
    def __init__(self):
        self.posts_done = 0
        self.comments_added = 0
        self.errors = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()

    async def record(self, comments: int):
        async with self._lock:
            self.posts_done += 1
            self.comments_added += comments

    async def record_error(self):
        async with self._lock:
            self.errors += 1

    @property
    def elapsed(self):
        return time.time() - self.start_time

    @property
    def rate(self):
        e = self.elapsed
        return self.posts_done / e * 60 if e > 0 else 0


async def fetch_json(session: aiohttp.ClientSession, url: str,
                     params: dict, limiter: AsyncRateLimiter,
                     max_retries: int = 8) -> dict:
    for attempt in range(max_retries):
        await limiter.acquire()
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning("429 rate limited, sleeping %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                if resp.status >= 500:
                    wait = min(2 ** attempt + 5, 60)
                    if attempt < 2:
                        pass  # suppress log for first retries
                    else:
                        logger.warning("Server %d (attempt %d), wait %ds", resp.status, attempt + 1, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            wait = min(2 ** attempt + 3, 60)
            logger.warning("Request error (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def save_comment_tree(db: Database, comment: dict, post_id: str,
                      parent_id=None, depth=0):
    """Recursively save comment + nested replies."""
    db.upsert_comment(comment, post_id, parent_id, depth)
    for reply in comment.get("replies", []) or []:
        save_comment_tree(db, reply, post_id,
                          parent_id=comment["id"], depth=depth + 1)


async def scrape_one_post(session: aiohttp.ClientSession, post_id: str,
                          expected: int, db: Database,
                          limiter: AsyncRateLimiter, stats: AsyncStats,
                          db_lock: asyncio.Lock):
    """Scrape all comments for a single post (sequential pagination)."""
    url = f"{BASE_URL}/posts/{post_id}/comments"
    cursor = None
    total_fetched = 0

    try:
        while True:
            params = {"sort": "new"}
            if cursor:
                params["cursor"] = cursor

            data = await fetch_json(session, url, params, limiter)
            comments = data.get("comments", [])
            if not comments:
                break

            async with db_lock:
                for c in comments:
                    save_comment_tree(db, c, post_id, parent_id=None,
                                      depth=c.get("depth", 0))
                total_fetched += len(comments)

            if not data.get("has_more", False):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break

        async with db_lock:
            db.mark_comments_scraped(post_id)
            if stats.posts_done % 5 == 0:
                db.commit()

        await stats.record(total_fetched)

        if stats.posts_done % 20 == 0 or expected > 1000:
            logger.info(
                "[%d done] Post %s: expected=%d, fetched=%d | "
                "Total: %d comments, %d errors | %.1f posts/min",
                stats.posts_done, post_id[:8], expected, total_fetched,
                stats.comments_added, stats.errors, stats.rate
            )

    except Exception as e:
        logger.error("Failed post %s (expected=%d): %s", post_id[:8], expected, e)
        await stats.record_error()
        async with db_lock:
            db.commit()


async def worker(queue: asyncio.Queue, session: aiohttp.ClientSession,
                 db: Database, limiter: AsyncRateLimiter,
                 stats: AsyncStats, db_lock: asyncio.Lock,
                 worker_id: int):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        post_id, expected = item
        await scrape_one_post(session, post_id, expected, db, limiter, stats, db_lock)
        queue.task_done()


async def run(args):
    api_key = os.getenv("MOLTBOOK_API_KEY")
    if not api_key:
        logger.error("MOLTBOOK_API_KEY not set")
        sys.exit(1)

    db = Database(args.db)
    posts_raw = db.get_posts_needing_comments(
        min_comments=args.min_comments,
        max_comments=args.max_comments,
        limit=0
    )
    # Sort ASC: process smaller posts first (faster, less likely to 500)
    posts = sorted(posts_raw, key=lambda x: x[1])
    already_done = db.count_comments_scraped_posts()
    logger.info("Async scraper: %d posts to scrape, %d already done "
                "(min=%d, max=%s, workers=%d)",
                len(posts), already_done, args.min_comments,
                args.max_comments or "unlimited", args.workers)

    if not posts:
        logger.info("Nothing to scrape!")
        db.close()
        return

    est_calls = sum(max(1, cc // 35 + 1) for _, cc in posts)
    est_hours = est_calls * REQUEST_INTERVAL / 3600
    logger.info("Estimated: %d API calls, %.1f hours with %d workers",
                est_calls, est_hours, args.workers)

    limiter = AsyncRateLimiter(REQUEST_INTERVAL)
    stats = AsyncStats()
    db_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()

    for pid, cc in posts:
        await queue.put((pid, cc))
    for _ in range(args.workers):
        await queue.put(None)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        workers = [
            asyncio.create_task(
                worker(queue, session, db, limiter, stats, db_lock, i)
            )
            for i in range(args.workers)
        ]

        progress_task = asyncio.create_task(
            _progress_logger(stats, len(posts), queue)
        )

        await asyncio.gather(*workers)
        progress_task.cancel()

    db.commit()
    db.close()

    logger.info("=" * 60)
    logger.info("DONE. Posts: %d, Comments: %d, Errors: %d, Time: %.1f min",
                stats.posts_done, stats.comments_added, stats.errors,
                stats.elapsed / 60)
    logger.info("=" * 60)


async def _progress_logger(stats: AsyncStats, total: int,
                           queue: asyncio.Queue):
    try:
        while True:
            await asyncio.sleep(60)
            remaining = queue.qsize()
            logger.info(
                "=== PROGRESS: %d/%d posts (%.1f%%) | "
                "%d comments | %d errors | "
                "%.1f posts/min | ~%.1f hours remaining ===",
                stats.posts_done, total,
                stats.posts_done / total * 100 if total else 0,
                stats.comments_added, stats.errors,
                stats.rate,
                remaining / stats.rate if stats.rate > 0 else 0,
            )
    except asyncio.CancelledError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Async Moltbook comment scraper")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--min-comments", type=int, default=101)
    parser.add_argument("--max-comments", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent workers (each handles one post at a time)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
