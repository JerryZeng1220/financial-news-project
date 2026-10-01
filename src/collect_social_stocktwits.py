"""
Social-mention collector — Stocktwits (the real Phase 3 data source).

Replaces the disabled add_sample_social_mentions.py with live data. It fetches
recent Stocktwits messages for a small set of *candidate* tickers and inserts
them into `social_mentions`, which calculate_density.py then turns into
ticker_density_snapshots (no change needed downstream).

Why only candidates (not every growing ticker)
-----------------------------------------------
Discussion density only matters for tickers that already have a positive, recent
news catalyst (those are the would-be alerts). Collecting for just those keeps us
well under Stocktwits' unauthenticated rate budget (~200 requests/hour) and is
exactly the signal we care about. Candidate = ticker in active_ticker_universe
with a recent news_sentiment >= SOCIAL_MIN_SENTIMENT, capped at SOCIAL_MAX_TICKERS.

Source
------
Public, keyless endpoint:
    https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json
(Stocktwits' premium real-time "Firestream" needs auth; we don't use it.)

Configuration (.env / environment)
----------------------------------
    SOCIAL_MIN_SENTIMENT   min recent news score to make a ticker a candidate (default 0.0)
    SOCIAL_NEWS_WINDOW_MIN  how recent the news must be (default 1440 = 24h)
    SOCIAL_MAX_TICKERS     cap on tickers per run (default 15)
    SOCIAL_PAGE_DELAY      seconds between ticker requests (default 1.5)
    USE_CURL_IMPERSONATE   "true" routes requests through curl_cffi (recommended)
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv


from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

load_dotenv(BASE_DIR / ".env")

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

SOCIAL_MIN_SENTIMENT = float(os.getenv("SOCIAL_MIN_SENTIMENT", "0.0"))
SOCIAL_NEWS_WINDOW_MIN = int(os.getenv("SOCIAL_NEWS_WINDOW_MIN", "1440"))
SOCIAL_MAX_TICKERS = int(os.getenv("SOCIAL_MAX_TICKERS", "15"))
SOCIAL_PAGE_DELAY = float(os.getenv("SOCIAL_PAGE_DELAY", "1.5"))

import sqlite3


def ny_now() -> datetime:
    return datetime.now(NY_TZ)


def ny_iso(dt: datetime) -> str:
    return dt.isoformat()


def make_hash(source: str, ticker: str, message_text: str, post_time: str) -> str:
    raw = f"{source}|{ticker}|{message_text}|{post_time}".lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ----------------------------
# Candidate selection
# ----------------------------

def get_candidate_tickers() -> list[str]:
    """
    Tickers that are currently 'growing' (in active_ticker_universe) AND have a
    recent, non-negative news sentiment — i.e. the ones worth measuring buzz for.
    """
    cutoff = (ny_now() - timedelta(minutes=SOCIAL_NEWS_WINDOW_MIN)).isoformat()

    query = """
        SELECT ns.ticker, MAX(ns.sentiment_score) AS best_score
        FROM news_sentiment ns
        JOIN active_ticker_universe atu ON ns.ticker = atu.ticker
        JOIN clean_news cn ON ns.clean_news_id = cn.id
        WHERE ns.sentiment_score >= ?
          AND cn.publish_time IS NOT NULL
          AND cn.publish_time >= ?
        GROUP BY ns.ticker
        ORDER BY best_score DESC
        LIMIT ?;
    """

    conn = connect_db()
    rows = conn.execute(
        query, (SOCIAL_MIN_SENTIMENT, cutoff, SOCIAL_MAX_TICKERS)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ----------------------------
# Stocktwits fetch
# ----------------------------

def fetch_stocktwits_messages(ticker: str) -> list[dict]:
    """
    Fetch recent Stocktwits messages for one ticker. Returns the raw 'messages'
    list (each has id, body, created_at, user, entities, ...). Empty list on a
    soft failure (rate-limited / no symbol) so the run continues.
    """
    url = STOCKTWITS_URL.format(ticker=ticker.upper())
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    use_curl = os.getenv("USE_CURL_IMPERSONATE", "false").lower() == "true"
    try:
        if use_curl:
            from curl_cffi import requests as curl_requests
            resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=20)
        else:
            resp = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        print(f"  [{ticker}] request error: {e}")
        return []

    if resp.status_code == 429:
        print(f"  [{ticker}] rate-limited (429); skipping this cycle.")
        return []
    if resp.status_code == 404:
        print(f"  [{ticker}] no Stocktwits symbol stream (404).")
        return []
    if resp.status_code != 200:
        print(f"  [{ticker}] HTTP {resp.status_code}; skipping.")
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    return data.get("messages", []) or []


def _parse_post_time(created_at: str) -> str | None:
    if not created_at:
        return None
    try:
        dt = date_parser.parse(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NY_TZ)
        return dt.astimezone(NY_TZ).isoformat()
    except Exception:
        return None


def _basic_sentiment(message: dict) -> tuple[str | None, float | None]:
    """Stocktwits lets users tag a post Bullish/Bearish — store it as a bonus."""
    entities = message.get("entities") or {}
    sentiment = entities.get("sentiment") or {}
    basic = sentiment.get("basic") if isinstance(sentiment, dict) else None
    if basic == "Bullish":
        return "bullish", 1.0
    if basic == "Bearish":
        return "bearish", -1.0
    return None, None


def insert_social_mentions(ticker: str, messages: list[dict]) -> int:
    if not messages:
        return 0

    conn = connect_db()
    cur = conn.cursor()
    now_iso = ny_iso(ny_now())
    inserted = 0

    for msg in messages:
        body = (msg.get("body") or "").strip()
        post_time = _parse_post_time(msg.get("created_at", ""))
        if not body or not post_time:
            continue

        author = ((msg.get("user") or {}).get("username")) or None
        msg_id = msg.get("id")
        url = f"https://stocktwits.com/message/{msg_id}" if msg_id else None
        label, score = _basic_sentiment(msg)
        message_hash = make_hash("Stocktwits", ticker, body, post_time)

        cur.execute(
            """
            INSERT OR IGNORE INTO social_mentions (
                source, ticker, message_text, author, post_time, collect_time,
                url, sentiment_label, sentiment_score, message_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            ("Stocktwits", ticker.upper(), body, author, post_time, now_iso,
             url, label, score, message_hash),
        )
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    conn.close()
    return inserted


# ----------------------------
# Main
# ----------------------------

def run_social_collection(tickers: list[str] | None = None) -> None:
    import news_sources
    if not news_sources.is_enabled("Stocktwits"):
        print("Stocktwits collector is disabled; skipping.")
        return
    if tickers is None:
        tickers = get_candidate_tickers()

    if not tickers:
        print("No candidate tickers (no growing ticker with recent positive news). Nothing to collect.")
        return

    print(f"Collecting Stocktwits mentions for {len(tickers)} ticker(s): {', '.join(tickers)}")
    total_new = 0

    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(SOCIAL_PAGE_DELAY)
        messages = fetch_stocktwits_messages(ticker)
        new = insert_social_mentions(ticker, messages)
        total_new += new
        print(f"  {ticker}: {len(messages)} messages fetched, {new} new inserted")

    print(f"Done. Inserted {total_new} new social mentions.")


if __name__ == "__main__":
    # Test specific tickers from the CLI, e.g.:
    #   python src/collect_social_stocktwits.py AAPL TSLA
    # Otherwise use the pipeline's candidate selection.
    cli_tickers = [t.upper() for t in sys.argv[1:]] or None
    run_social_collection(cli_tickers)
