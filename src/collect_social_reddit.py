"""
Social-mention collector — Reddit via public search RSS (no API key).

Reddit closed new free API keys (the "Responsible Builder Policy"), so instead of
PRAW we use Reddit's public search RSS feed, which needs no credentials:

    https://www.reddit.com/search.rss?q=<ticker>&sort=new&t=week

Matching posts are inserted into `social_mentions` (source="Reddit"), so
calculate_density.py and the Gemini crowd-mood summarizer pick them up
automatically. Same candidate-only approach as the Stocktwits collector, so only
a few tickers are searched per run.

Reddit rate-limits this RSS hard (429), so we pace requests and back off. Reddit
posts carry no Bullish/Bearish tag — they add to DENSITY and feed the mood
summary, but not the bull/bear ratio. Public-RSS use is subject to Reddit's ToS;
keep it light (the candidate-only design does).

Config (.env):
    REDDIT_SEARCH_URL    default https://www.reddit.com/search.rss
    REDDIT_QUERY_FORMAT  default "{ticker}" (try "${ticker}" or "{ticker} stock")
    REDDIT_TIME_FILTER   hour|day|week|month (default "week")
    REDDIT_PAGE_DELAY    seconds between ticker searches (default 4.0)
    REDDIT_MAX_RETRIES   429/5xx retries (default 3)
    REDDIT_BACKOFF       base backoff seconds (default 8)
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from collect_social_stocktwits import get_candidate_tickers  # reuse candidate logic


from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

load_dotenv(BASE_DIR / ".env")

# Restrict to finance subreddits — global Reddit search returns spammy
# "stock message board" subs and irrelevant matches; these subs are on-topic.
SUBREDDITS = os.getenv("REDDIT_SUBREDDITS", "wallstreetbets+stocks+pennystocks+StockMarket+smallstreetbets")
SEARCH_URL = os.getenv("REDDIT_SEARCH_URL", f"https://www.reddit.com/r/{SUBREDDITS}/search.rss")
RESTRICT_SR = os.getenv("REDDIT_RESTRICT_SR", "on")
QUERY_FORMAT = os.getenv("REDDIT_QUERY_FORMAT", "{ticker}")
TIME_FILTER = os.getenv("REDDIT_TIME_FILTER", "week")
PAGE_DELAY = float(os.getenv("REDDIT_PAGE_DELAY", "8.0"))     # gentle spacing between tickers
MAX_RETRIES = int(os.getenv("REDDIT_MAX_RETRIES", "2"))
BACKOFF = float(os.getenv("REDDIT_BACKOFF", "8"))
# Reddit is the lowest-priority source and rate-limits hard, so fetch it rarely
# (cadence guard) and only for the top few candidates per run.
MIN_INTERVAL_SEC = int(os.getenv("REDDIT_MIN_INTERVAL_SEC", "1800"))  # at most every 30 min
MAX_TICKERS = int(os.getenv("REDDIT_MAX_TICKERS", "3"))
USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
MAX_TEXT_CHARS = 400


def make_hash(source: str, ticker: str, message_text: str, post_time: str) -> str:
    raw = f"{source}|{ticker}|{message_text}|{post_time}".lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clean(html: str) -> str:
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _post_time(entry) -> str | None:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if not st:
        return None
    try:
        return datetime(*st[:6], tzinfo=timezone.utc).astimezone(NY_TZ).isoformat()
    except Exception:
        return None


def fetch_reddit_posts(ticker: str) -> list[dict]:
    """Fetch Reddit search-RSS results for a ticker, with 429/5xx backoff."""
    query = QUERY_FORMAT.format(ticker=ticker)
    params = {"q": query, "restrict_sr": RESTRICT_SR, "sort": "new", "t": TIME_FILTER}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml,*/*"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=headers, timeout=20)
        except Exception as e:
            print(f"  [{ticker}] request error: {e}")
            return []
        if resp.status_code == 200:
            feed = feedparser.parse(resp.text)
            posts = []
            for e in feed.entries:
                posts.append({
                    "title": e.get("title", ""),
                    "body": _clean(e.get("summary") or (e.get("content", [{}])[0].get("value") if e.get("content") else "")),
                    "author": e.get("author"),
                    "post_time": _post_time(e),
                    "url": e.get("link"),
                })
            return posts
        if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
            wait = BACKOFF * attempt
            print(f"  [{ticker}] Reddit {resp.status_code} rate-limited; backing off {wait:.0f}s...")
            time.sleep(wait)
            continue
        print(f"  [{ticker}] Reddit HTTP {resp.status_code}; skipping.")
        return []
    return []


def insert_mentions(ticker: str, posts: list[dict]) -> int:
    if not posts:
        return 0
    conn = connect_db()
    cur = conn.cursor()
    now_iso = ny_now_iso()
    inserted = 0
    for p in posts:
        body = (p["title"] + " " + (p["body"] or "")).strip()[:MAX_TEXT_CHARS]
        post_time = p["post_time"]
        if not body or not post_time:
            continue
        message_hash = make_hash("Reddit", ticker, body, post_time)
        cur.execute(
            """
            INSERT OR IGNORE INTO social_mentions (
                source, ticker, message_text, author, post_time, collect_time,
                url, sentiment_label, sentiment_score, message_hash
            )
            VALUES ('Reddit', ?, ?, ?, ?, ?, ?, NULL, NULL, ?);
            """,
            (ticker.upper(), body, p.get("author"), post_time, now_iso, p.get("url"), message_hash),
        )
        if cur.rowcount > 0:
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


def _last_run() -> datetime | None:
    conn = connect_db()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS collector_state (name TEXT PRIMARY KEY, last_run TEXT);")
        row = conn.execute("SELECT last_run FROM collector_state WHERE name='reddit';").fetchone()
    finally:
        conn.close()
    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            return None
    return None


def _mark_run() -> None:
    conn = connect_db()
    conn.execute("CREATE TABLE IF NOT EXISTS collector_state (name TEXT PRIMARY KEY, last_run TEXT);")
    conn.execute("INSERT OR REPLACE INTO collector_state (name, last_run) VALUES ('reddit', ?);", (ny_now_iso(),))
    conn.commit()
    conn.close()


def reddit_due() -> bool:
    last = _last_run()
    if last is None:
        return True
    return (datetime.now(NY_TZ) - last).total_seconds() >= MIN_INTERVAL_SEC


def run_reddit_collection(tickers: list[str] | None = None) -> None:
    import news_sources
    if not news_sources.is_enabled("Reddit"):
        print("Reddit collector is disabled; skipping.")
        return
    pipeline_mode = tickers is None
    if pipeline_mode:
        # Cadence guard: Reddit is low-priority and rate-limits hard, so only
        # actually fetch it every MIN_INTERVAL_SEC — most cycles do nothing.
        if not reddit_due():
            print(f"Reddit: skipping (fetched within last {MIN_INTERVAL_SEC}s).")
            return
        tickers = get_candidate_tickers()[:MAX_TICKERS]

    if not tickers:
        print("No candidate tickers for Reddit collection.")
        return

    if pipeline_mode:
        _mark_run()  # mark now so a 429/failure still spaces the next attempt

    print(f"Collecting Reddit mentions for {len(tickers)} ticker(s): {', '.join(tickers)}")
    total = 0
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(PAGE_DELAY)
        posts = fetch_reddit_posts(ticker)
        new = insert_mentions(ticker, posts)
        total += new
        print(f"  {ticker}: {len(posts)} posts fetched, {new} new inserted")
    print(f"Done. Inserted {total} new Reddit mentions.")


if __name__ == "__main__":
    cli = [t.upper() for t in sys.argv[1:]] or None
    run_reddit_collection(cli)
