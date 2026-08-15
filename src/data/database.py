"""
SQLite database schema and operations for Moltbook data.
Designed for hypergraph construction: preserves full comment tree structure,
precise timestamps, and all interaction metadata.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    content         TEXT,
    type            TEXT,
    url             TEXT,
    author_id       TEXT,
    author_name     TEXT,
    submolt_id      TEXT,
    submolt_name    TEXT,
    upvotes         INTEGER DEFAULT 0,
    downvotes       INTEGER DEFAULT 0,
    comment_count   INTEGER DEFAULT 0,
    created_at      TEXT,
    scraped_at      TEXT DEFAULT (datetime('now')),
    comments_scraped INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS comments (
    id              TEXT PRIMARY KEY,
    post_id         TEXT NOT NULL,
    parent_id       TEXT,
    content         TEXT,
    author_id       TEXT,
    author_name     TEXT,
    depth           INTEGER DEFAULT 0,
    upvotes         INTEGER DEFAULT 0,
    downvotes       INTEGER DEFAULT 0,
    reply_count     INTEGER DEFAULT 0,
    created_at      TEXT,
    scraped_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

CREATE TABLE IF NOT EXISTS agents (
    id              TEXT PRIMARY KEY,
    name            TEXT UNIQUE,
    description     TEXT,
    karma           INTEGER,
    follower_count  INTEGER,
    following_count INTEGER,
    is_claimed      INTEGER,
    created_at      TEXT,
    scraped_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS submolts (
    id              TEXT PRIMARY KEY,
    name            TEXT UNIQUE,
    display_name    TEXT,
    description     TEXT,
    subscriber_count INTEGER,
    post_count      INTEGER,
    created_at      TEXT,
    scraped_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scrape_state (
    key             TEXT PRIMARY KEY,
    value           TEXT,
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent_id ON comments(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_author_id ON comments(author_id);
CREATE INDEX IF NOT EXISTS idx_comments_created_at ON comments(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_author_id ON posts(author_id);
CREATE INDEX IF NOT EXISTS idx_posts_comment_count ON posts(comment_count);
CREATE INDEX IF NOT EXISTS idx_posts_submolt_name ON posts(submolt_name);
"""


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── State management (for resume) ──────────────────────────────

    def get_state(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM scrape_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO scrape_state (key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            (key, value)
        )
        self.conn.commit()

    # ── Posts ──────────────────────────────────────────────────────

    def upsert_post(self, post: dict):
        author = post.get("author", {}) or {}
        submolt = post.get("submolt", {}) or {}
        self.conn.execute("""
            INSERT OR REPLACE INTO posts
            (id, title, content, type, url, author_id, author_name,
             submolt_id, submolt_name, upvotes, downvotes, comment_count,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post["id"],
            post.get("title"),
            post.get("content"),
            post.get("type"),
            post.get("url"),
            author.get("id") or post.get("author_id"),
            author.get("name") or post.get("author_name"),
            submolt.get("id") or post.get("submolt_id"),
            submolt.get("name") or post.get("submolt_name"),
            post.get("upvotes", 0),
            post.get("downvotes", 0),
            post.get("comment_count", 0),
            post.get("created_at"),
        ))

    def mark_comments_scraped(self, post_id: str):
        self.conn.execute(
            "UPDATE posts SET comments_scraped = 1 WHERE id = ?", (post_id,)
        )

    def get_posts_needing_comments(self, min_comments: int = 0,
                                    max_comments: int = 0,
                                    limit: int = 0):
        """Return posts that haven't had their comments fully scraped.
        max_comments=0 means no upper limit; limit=0 means no row limit."""
        conditions = ["comments_scraped = 0", "comment_count >= ?"]
        params = [min_comments]
        if max_comments > 0:
            conditions.append("comment_count <= ?")
            params.append(max_comments)
        where = " AND ".join(conditions)
        sql = f"SELECT id, comment_count FROM posts WHERE {where} ORDER BY comment_count DESC"
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def count_posts_needing_comments(self, min_comments: int = 0) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE comments_scraped = 0 AND comment_count >= ?",
            (min_comments,)
        ).fetchone()[0]

    def count_comments_scraped_posts(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE comments_scraped = 1"
        ).fetchone()[0]

    def post_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    # ── Comments ──────────────────────────────────────────────────

    def upsert_comment(self, comment: dict, post_id: str,
                       parent_id: Optional[str] = None,
                       depth: int = 0):
        author = comment.get("author", {}) or {}
        self.conn.execute("""
            INSERT OR REPLACE INTO comments
            (id, post_id, parent_id, content, author_id, author_name,
             depth, upvotes, downvotes, reply_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comment["id"],
            post_id,
            parent_id,
            comment.get("content"),
            author.get("id") or comment.get("author_id"),
            author.get("name") or comment.get("author_name"),
            depth,
            comment.get("upvotes", 0),
            comment.get("downvotes", 0),
            comment.get("reply_count", 0),
            comment.get("created_at"),
        ))

    def upsert_comment_tree(self, comment: dict, post_id: str,
                            parent_id: Optional[str] = None,
                            depth: int = 0):
        """Recursively insert a comment and all nested replies."""
        self.upsert_comment(comment, post_id, parent_id, depth)
        for reply in comment.get("replies", []) or []:
            self.upsert_comment_tree(reply, post_id,
                                     parent_id=comment["id"],
                                     depth=depth + 1)

    def comment_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]

    def comment_count_for_post(self, post_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM comments WHERE post_id = ?", (post_id,)
        ).fetchone()[0]

    # ── Agents ────────────────────────────────────────────────────

    def upsert_agent(self, agent: dict):
        self.conn.execute("""
            INSERT OR REPLACE INTO agents
            (id, name, description, karma, follower_count, following_count,
             is_claimed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent["id"],
            agent.get("name"),
            agent.get("description"),
            agent.get("karma"),
            agent.get("follower_count"),
            agent.get("following_count"),
            agent.get("is_claimed"),
            agent.get("created_at"),
        ))

    def get_unique_agent_names(self):
        """Get all unique agent names seen in posts/comments but not yet profiled."""
        return self.conn.execute("""
            SELECT DISTINCT name FROM (
                SELECT author_name AS name FROM posts WHERE author_name IS NOT NULL
                UNION
                SELECT author_name AS name FROM comments WHERE author_name IS NOT NULL
            )
            WHERE name NOT IN (SELECT name FROM agents WHERE name IS NOT NULL)
        """).fetchall()

    def agent_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]

    # ── Batch commit ──────────────────────────────────────────────

    def commit(self):
        self.conn.commit()

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "posts": self.post_count(),
            "comments": self.comment_count(),
            "agents": self.agent_count(),
            "posts_with_comments_scraped": self.conn.execute(
                "SELECT COUNT(*) FROM posts WHERE comments_scraped = 1"
            ).fetchone()[0],
        }
