"""
Dashboard-editable registry of the sources the app collects from.

One DB table (news_sources) holds both the RSS news wires (with a feed_url) and
the code-based collectors (Yahoo per-ticker, Stocktwits, Reddit), each with an
`enabled` flag:
  - ingest_news reads the enabled RSS rows.
  - each collector checks is_enabled(<its name>) and no-ops when disabled.
  - the dashboard 'Sources' tab lists them and lets you add / enable / disable.

Seeded once from DEFAULTS when the table is empty; after that it's yours to edit
(deleting a default sticks — we don't re-seed a non-empty table).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

# (name, source_type, feed_url) — collectors have feed_url = None
DEFAULTS = [
    ("SEC", "rss", "https://www.sec.gov/news/pressreleases.rss"),
    ("PR Newswire", "rss", "https://www.prnewswire.com/rss/news-releases-list.rss"),
    ("GlobeNewswire", "rss",
     "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies"),
    ("GlobeNewswire US", "rss",
     "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/US"),
    ("FDA", "rss", "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"),
    ("Yahoo per-ticker", "collector", None),
    ("Stocktwits", "collector", None),
    ("Reddit", "collector", None),
]

COLLECTOR_NOTES = {
    "Yahoo per-ticker": "Per-ticker Yahoo news for every universe ticker",
    "Stocktwits": "Social mentions / crowd mood",
    "Reddit": "Finance-subreddit mentions",
}


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL;")
    return c


def init_news_sources() -> None:
    c = _conn()
    cur = c.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            source_type TEXT DEFAULT 'rss',
            feed_url TEXT,
            enabled INTEGER DEFAULT 1,
            added_at TEXT,
            consecutive_empty INTEGER DEFAULT 0
        );
        """
    )
    # Migrate older tables that predate the empty-feed counter.
    cols = {r[1] for r in cur.execute("PRAGMA table_info(news_sources);").fetchall()}
    if "consecutive_empty" not in cols:
        cur.execute("ALTER TABLE news_sources ADD COLUMN consecutive_empty INTEGER DEFAULT 0;")
    # Seed only when empty, so deletions/edits stick.
    if cur.execute("SELECT COUNT(*) FROM news_sources;").fetchone()[0] == 0:
        now = datetime.now(NY_TZ).isoformat()
        cur.executemany(
            "INSERT INTO news_sources (name, source_type, feed_url, enabled, added_at) VALUES (?,?,?,1,?);",
            [(n, t, u, now) for (n, t, u) in DEFAULTS],
        )
    c.commit()
    c.close()


def get_enabled_rss() -> list[dict]:
    """Enabled RSS feeds, shaped like ingest_news expects."""
    init_news_sources()
    c = _conn()
    rows = c.execute(
        """SELECT name, feed_url FROM news_sources
           WHERE enabled=1 AND source_type='rss' AND feed_url IS NOT NULL AND feed_url!=''
           ORDER BY name;"""
    ).fetchall()
    c.close()
    return [{"source": r[0], "source_type": "rss", "feed_url": r[1]} for r in rows]


def record_ingest_result(name: str, n_entries: int) -> int:
    """Track the empty-feed streak for a source. Resets to 0 when the feed brought
    in items; otherwise increments. Returns the current consecutive-empty count so
    the caller can warn when a feed looks dead/misconfigured."""
    c = _conn()
    try:
        if n_entries > 0:
            c.execute("UPDATE news_sources SET consecutive_empty=0 WHERE name=?;", (name,))
            c.commit()
            return 0
        c.execute(
            "UPDATE news_sources SET consecutive_empty=COALESCE(consecutive_empty,0)+1 WHERE name=?;",
            (name,),
        )
        c.commit()
        row = c.execute("SELECT consecutive_empty FROM news_sources WHERE name=?;", (name,)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        c.close()


def is_enabled(name: str) -> bool:
    """Whether a source is enabled. Unknown name -> True (fail-open, so a
    not-yet-seeded collector still runs)."""
    init_news_sources()
    c = _conn()
    row = c.execute("SELECT enabled FROM news_sources WHERE name=?;", (name,)).fetchone()
    c.close()
    return True if row is None else bool(row[0])


def get_all():
    import pandas as pd
    init_news_sources()
    c = _conn()
    df = pd.read_sql_query(
        "SELECT name, source_type, feed_url, enabled FROM news_sources ORDER BY source_type, name;", c
    )
    c.close()
    return df


def set_enabled(name: str, enabled: bool) -> None:
    c = _conn()
    c.execute("UPDATE news_sources SET enabled=? WHERE name=?;", (1 if enabled else 0, name))
    c.commit()
    c.close()


def add_source(name: str, feed_url: str, source_type: str = "rss") -> tuple[bool, str]:
    name, feed_url = (name or "").strip(), (feed_url or "").strip()
    if not name or not feed_url:
        return False, "Name and feed URL are both required."
    c = _conn()
    if c.execute("SELECT 1 FROM news_sources WHERE name=?;", (name,)).fetchone():
        c.close()
        return False, f"A source named '{name}' already exists."
    c.execute(
        "INSERT INTO news_sources (name, source_type, feed_url, enabled, added_at) VALUES (?,?,?,1,?);",
        (name, source_type, feed_url, datetime.now(NY_TZ).isoformat()),
    )
    c.commit()
    c.close()
    return True, f"Added '{name}'."


def delete_source(name: str) -> None:
    c = _conn()
    c.execute("DELETE FROM news_sources WHERE name=?;", (name,))
    c.commit()
    c.close()
