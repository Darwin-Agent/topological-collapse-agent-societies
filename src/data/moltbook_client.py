"""
Moltbook REST API client with rate-limiting, retry, and cursor/offset pagination.
"""

import time
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Iterator

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.moltbook.com/api/v1"

# ~90 req/min, will back off automatically on 429
DEFAULT_MIN_INTERVAL = 0.65  # seconds between requests


class RateLimiter:
    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL):
        self.min_interval = min_interval
        self._last_request = 0.0

    def wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()


class MoltbookClient:
    def __init__(self, api_key: str, min_interval: float = DEFAULT_MIN_INTERVAL):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self.limiter = RateLimiter(min_interval)

    def _get(self, endpoint: str, params: Optional[dict] = None,
             max_retries: int = 5) -> dict:
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        for attempt in range(max_retries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning("Rate limited. Sleeping %ds (attempt %d)",
                                   retry_after, attempt + 1)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                logger.warning("Timeout on %s (attempt %d)", url, attempt + 1)
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError:
                logger.warning("Connection error on %s (attempt %d)", url, attempt + 1)
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as e:
                if resp.status_code >= 500:
                    logger.warning("Server error %d (attempt %d)", resp.status_code, attempt + 1)
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"Failed after {max_retries} retries: {url}")

    # ── Posts ──────────────────────────────────────────────────────────

    def get_posts_page(self, sort: str = "new", limit: int = 100,
                       cursor: Optional[str] = None) -> tuple:
        """Fetch one page of posts. Returns (posts, next_cursor, has_more)."""
        params = {"sort": sort, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = self._get("/posts", params)
        return (
            data.get("posts", []),
            data.get("next_cursor"),
            data.get("has_more", False),
        )

    def iter_posts(self, sort: str = "new", limit: int = 100,
                   start_cursor: Optional[str] = None) -> Iterator[dict]:
        """Yield all posts using cursor pagination."""
        cursor = start_cursor
        while True:
            posts, next_cursor, has_more = self.get_posts_page(sort, limit, cursor)
            if not posts:
                break
            yield from posts
            if not has_more or not next_cursor:
                break
            cursor = next_cursor

    def get_post(self, post_id: str) -> dict:
        return self._get(f"/posts/{post_id}")

    # ── Comments ──────────────────────────────────────────────────────

    def iter_comments(self, post_id: str, sort: str = "new") -> Iterator[dict]:
        """
        Yield all top-level comments for a post, with nested replies
        included in each comment's 'replies' field.
        Uses cursor pagination.
        """
        cursor = None
        while True:
            params = {"sort": sort}
            if cursor:
                params["cursor"] = cursor
            data = self._get(f"/posts/{post_id}/comments", params)
            comments = data.get("comments", [])
            if not comments:
                break
            yield from comments
            if not data.get("has_more", False):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break

    # ── Agents ────────────────────────────────────────────────────────

    def get_agent_profile(self, name: str) -> dict:
        return self._get("/agents/profile", params={"name": name})

    # ── Submolts ──────────────────────────────────────────────────────

    def iter_submolts(self) -> Iterator[dict]:
        data = self._get("/submolts")
        yield from data.get("submolts", [])
