"""
Crowd-mood summarizer — reads Stocktwits chatter and writes a one-line mood per
candidate ticker into `ticker_crowd_mood` (shown on the dashboard).

Usage-safe by design (keeps Gemini free-tier calls tiny):
  - Only tickers that already have recent social_mentions (the collector only
    gathers those for positive-news growing candidates) — usually 0-3.
  - All of them summarized in ONE batched Gemini request.
  - Cooldown: a ticker summarized within CROWD_MOOD_COOLDOWN_MIN is skipped, so a
    persistent candidate costs one call, not one per cycle.
  - If Gemini is unavailable/over quota, this stage no-ops and the dashboard
    falls back to the bullish/bearish ratio it computes from stored tags.

Config (.env):
    CROWD_MOOD_LOOKBACK_HOURS   how recent comments must be (default 6)
    CROWD_MOOD_COOLDOWN_MIN     don't re-summarize within this (default 45)
    CROWD_MOOD_MAX_TICKERS      cap per run (default 10)
    CROWD_MOOD_MAX_COMMENTS     comments per ticker sent to Gemini (default 25)
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import llm_sentiment as llm


from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

load_dotenv(BASE_DIR / ".env")

LOOKBACK_HOURS = int(os.getenv("CROWD_MOOD_LOOKBACK_HOURS", "6"))
COOLDOWN_MIN = int(os.getenv("CROWD_MOOD_COOLDOWN_MIN", "45"))
MAX_TICKERS = int(os.getenv("CROWD_MOOD_MAX_TICKERS", "10"))
MAX_COMMENTS = int(os.getenv("CROWD_MOOD_MAX_COMMENTS", "25"))


def ny_now() -> datetime:
    return datetime.now(NY_TZ)


def init_table() -> None:
    conn = connect_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_crowd_mood (
            ticker TEXT PRIMARY KEY,
            mood TEXT,
            summary TEXT,
            n_messages INTEGER,
            calculated_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def select_tickers_needing_summary() -> list[tuple[str, int]]:
    """Tickers with recent chatter not summarized within the cooldown."""
    now = ny_now()
    recent_cutoff = (now - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    cooldown_cutoff = (now - timedelta(minutes=COOLDOWN_MIN)).isoformat()

    conn = connect_db()
    rows = conn.execute(
        """
        SELECT sm.ticker, COUNT(*) AS n
        FROM social_mentions sm
        LEFT JOIN ticker_crowd_mood m ON sm.ticker = m.ticker
        WHERE sm.post_time >= ?
          AND (m.calculated_at IS NULL OR m.calculated_at < ?)
        GROUP BY sm.ticker
        ORDER BY n DESC
        LIMIT ?;
        """,
        (recent_cutoff, cooldown_cutoff, MAX_TICKERS),
    ).fetchall()
    conn.close()
    return [(r[0], int(r[1])) for r in rows]


def fetch_comments(ticker: str) -> list[str]:
    recent_cutoff = (ny_now() - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    conn = connect_db()
    rows = conn.execute(
        """
        SELECT message_text FROM social_mentions
        WHERE ticker = ? AND post_time >= ?
        ORDER BY post_time DESC
        LIMIT ?;
        """,
        (ticker, recent_cutoff, MAX_COMMENTS),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows if r[0]]


def save_mood(ticker: str, mood: str, summary: str, n_messages: int) -> None:
    conn = connect_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO ticker_crowd_mood (ticker, mood, summary, n_messages, calculated_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (ticker, mood, summary, n_messages, ny_now().isoformat()),
    )
    conn.commit()
    conn.close()


def run_crowd_mood() -> None:
    init_table()

    candidates = select_tickers_needing_summary()
    if not candidates:
        print("No tickers need a crowd-mood summary (no fresh chatter, or all within cooldown).")
        return

    # Build one batch: index -> (ticker, n, comments)
    items = []
    meta = {}
    for idx, (ticker, n) in enumerate(candidates):
        comments = fetch_comments(ticker)
        if not comments:
            continue
        items.append({"id": idx, "ticker": ticker, "comments": comments})
        meta[idx] = (ticker, n)

    if not items:
        print("Candidates had no comment text to summarize.")
        return

    print(f"Summarizing crowd mood for {len(items)} ticker(s): {', '.join(m[0] for m in meta.values())}")

    try:
        results = llm.summarize_crowd_moods(items)  # ONE request
    except llm.LLMUnavailable as e:
        print(f"  Gemini unavailable ({e}); skipping — dashboard will use the bull/bear ratio.")
        return

    saved = 0
    for idx, (ticker, n) in meta.items():
        r = results.get(idx)
        if not r:
            continue
        save_mood(ticker, r["mood"], r["summary"], n)
        print(f"  {ticker}: {r['mood']} — {r['summary']}")
        saved += 1

    print(f"Saved {saved} crowd-mood summaries.")


if __name__ == "__main__":
    run_crowd_mood()
