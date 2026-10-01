"""
One-shot schema setup, run first in the pipeline. Merged from the former
init_source_registry.py + init_social_tables.py.

Creates: source_registry, social_mentions, ticker_density_snapshots.
Idempotent — CREATE IF NOT EXISTS + INSERT OR REPLACE, safe to run every cycle.
"""
from __future__ import annotations

from common import connect_db


def init_source_registry() -> None:
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS source_registry (
            source_name TEXT PRIMARY KEY,
            source_type TEXT,
            source_category TEXT,
            base_url TEXT,
            is_active INTEGER DEFAULT 1,
            requires_auth INTEGER DEFAULT 0,
            credibility_score REAL,
            notes TEXT
        );
        """
    )
    sources = [
        ("SEC", "rss", "official", "https://www.sec.gov", 1, 0, 1.00, "Official regulatory source"),
        ("FDA", "rss", "official", "https://www.fda.gov", 1, 0, 1.00, "Official regulatory source"),
        ("PR Newswire", "rss", "newswire", "https://www.prnewswire.com", 1, 0, 0.80, "Corporate press releases"),
        ("GlobeNewswire", "rss", "newswire", "https://www.globenewswire.com", 1, 0, 0.78, "Corporate press releases (good small/mid cap coverage)"),
        ("BusinessWire", "rss", "newswire", "https://www.businesswire.com", 0, 0, 0.80, "Potential future source"),
        ("ACCESS Wire", "rss", "newswire", "https://www.accesswire.com", 0, 0, 0.75, "Potential future source"),
        ("Finviz", "screener", "platform", "https://finviz.com", 1, 1, 0.70, "Ticker reference and screener data"),
        ("TradingView", "news", "platform", "https://www.tradingview.com", 0, 1, 0.75, "Potential future source"),
        ("Stocktwits", "social", "social", "https://stocktwits.com", 0, 0, 0.55, "Potential social discussion source"),
        ("Reddit", "social", "social", "https://www.reddit.com", 0, 0, 0.45, "Potential social discussion source"),
        ("X", "social", "social", "https://x.com", 0, 1, 0.45, "Harder API/access source"),
    ]
    cur.executemany(
        """
        INSERT OR REPLACE INTO source_registry (
            source_name, source_type, source_category, base_url,
            is_active, requires_auth, credibility_score, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        sources,
    )
    conn.commit()
    conn.close()


def init_social_tables() -> None:
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS social_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            ticker TEXT,
            message_text TEXT,
            author TEXT,
            post_time TEXT,
            collect_time TEXT,
            url TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            message_hash TEXT UNIQUE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_density_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            calculated_at TEXT,
            mentions_1min INTEGER,
            mentions_5min INTEGER,
            mentions_15min INTEGER,
            mentions_30min INTEGER,
            mentions_1hr INTEGER,
            density_score REAL,
            density_level TEXT,
            notes TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def init_db() -> None:
    init_source_registry()
    init_social_tables()
    print("Schema ready: source_registry, social_mentions, ticker_density_snapshots.")


if __name__ == "__main__":
    init_db()
