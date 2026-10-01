"""
collect_ticker_news.py — per-ticker news for the active universe.

For each ticker in active_ticker_universe, fetch its Yahoo Finance news feed
(Yahoo aggregates ALL sources for a symbol — Business Wire, ACCESSWIRE, PR wires,
analysts, ...), keep only headlines that actually mention the ticker or its
company, and ingest them PRE-ATTACHED to the ticker:

    raw_news -> clean_news -> news_ticker_links (+ ticker_extraction_status)

Why: the broad RSS wires only carry a fraction of small-cap news, and even when
they do, the company-name matcher can miss shortened names. Fetching per-ticker
fixes both — the news is captured regardless of wire AND arrives already tied to
the ticker (no matching step to miss). Deduped against existing news by content
hash so it never double-counts a wire article. Cadence-guarded so it doesn't hit
Yahoo every cycle.

Runs after load_live_active_universe (needs the current universe) and before
preprocess/extract (its rows are already clean + linked, so those stages skip
them and sentiment_stage picks them up).
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
from curl_cffi import requests as cr

import preprocess_news as pp  # reuse clean_text + make_content_hash for consistency

from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

YAHOO_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
IMPERSONATE = "chrome"

# Config (.env overridable)
MIN_INTERVAL_SEC = int(os.getenv("TICKER_NEWS_MIN_INTERVAL_SEC", "600"))   # 10 min cadence guard
MAX_AGE_DAYS = int(os.getenv("TICKER_NEWS_MAX_AGE_DAYS", "3"))             # ignore stale headlines
MAX_PER_TICKER = int(os.getenv("TICKER_NEWS_MAX_PER_TICKER", "8"))
PER_TICKER_DELAY = float(os.getenv("TICKER_NEWS_DELAY", "0.4"))            # be polite to Yahoo
FETCH_TIMEOUT = int(os.getenv("TICKER_NEWS_TIMEOUT", "12"))

_COMMON = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "plc", "group", "holding", "holdings", "sa", "ag", "nv", "adr", "the", "and",
    "class", "common", "stock", "l", "p",
}


def ny_now() -> datetime:
    return datetime.now(NY_TZ)


DEDUP_WINDOW_DAYS = int(os.getenv("TICKER_NEWS_DEDUP_DAYS", "7"))


def _title_key(title: str) -> str:
    """Normalized headline key for cross-source dedup — lowercase, alnum only.
    The same article from a wire and from Yahoo has an identical title but a
    different summary (so a content-hash on title+text won't match); dedup on the
    title instead so we don't ingest it twice."""
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _company_tokens(company: str) -> list[str]:
    words = re.sub(r"[^a-z0-9 ]", " ", (company or "").lower()).split()
    return [w for w in words if w not in _COMMON and len(w) >= 2]


def _on_topic(title: str, ticker: str, company: str) -> bool:
    """Keep a Yahoo headline only if it actually concerns this ticker — the feed
    occasionally slips in unrelated market news."""
    t = re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())
    if re.search(rf"\b{re.escape(ticker.lower())}\b", t):
        return True
    tokens = _company_tokens(company)
    if not tokens:
        return False
    # require the first (up to two) distinctive company words to all appear
    return all(tok in t for tok in tokens[:2])


def _due(conn: sqlite3.Connection, force: bool) -> bool:
    """Cadence guard: skip if we fetched within MIN_INTERVAL_SEC. `force` bypasses."""
    if force:
        return True
    row = conn.execute("SELECT MAX(collect_time) FROM raw_news WHERE source='Yahoo';").fetchone()
    if not row or not row[0]:
        return True
    try:
        last = datetime.fromisoformat(row[0])
        return (ny_now() - last).total_seconds() >= MIN_INTERVAL_SEC
    except Exception:
        return True


def load_universe(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [
        (r[0], r[1] or "")
        for r in conn.execute(
            "SELECT ticker, company FROM active_ticker_universe WHERE ticker IS NOT NULL;"
        ).fetchall()
    ]


def _parse_time(entry) -> str | None:
    pp_time = getattr(entry, "published_parsed", None)
    if not pp_time:
        return None
    dt = datetime(*pp_time[:6], tzinfo=timezone.utc).astimezone(NY_TZ)
    return dt.isoformat()


def _ingest_headline(conn, seen, ticker, company, title, summary, url, publish_time) -> str:
    """Insert one headline pre-attached to `ticker`, deduped by title against
    `seen` (title_key -> clean_news_id). Returns 'new'/'linked'/'dup'."""
    cur = conn.cursor()
    title_clean = pp.clean_text(title)
    summary_clean = pp.clean_text(summary)
    now = ny_now().isoformat()

    clean_id = seen.get(_title_key(title_clean))

    if clean_id is not None:
        # same headline already ingested (from a wire or an earlier fetch) —
        # link this ticker to it instead of creating a duplicate row.
        result = "linked"
    else:
        content_hash = pp.make_content_hash(title_clean, summary_clean)
        cur.execute(
            """INSERT INTO raw_news (source, title, publish_time, url, summary,
                   partial_text, collect_time, source_type, record_hash)
               VALUES ('Yahoo', ?, ?, ?, ?, ?, ?, 'rss', ?);""",
            (title, publish_time, url, summary, summary, now, content_hash),
        )
        raw_id = cur.lastrowid
        cur.execute(
            """INSERT INTO clean_news (raw_news_id, source, source_type, title_clean,
                   summary_clean, text_clean, publish_time, collect_time, url,
                   content_hash, is_duplicate, is_ready_for_analysis,
                   preprocessing_notes, created_at)
               VALUES (?, 'Yahoo', 'rss', ?, ?, ?, ?, ?, ?, ?, 0, 1, 'yahoo_per_ticker', ?);""",
            (raw_id, title_clean, summary_clean, summary_clean, publish_time, now, url,
             content_hash, now),
        )
        clean_id = cur.lastrowid
        # mark processed so extract_tickers skips it (we link it directly below)
        cur.execute(
            """INSERT OR REPLACE INTO ticker_extraction_status
                   (clean_news_id, processed_at, match_count, status, notes)
               VALUES (?, ?, 1, 'matched', 'yahoo_per_ticker');""",
            (clean_id, now),
        )
        seen[_title_key(title_clean)] = clean_id  # dedup later headlines this run
        result = "new"

    # pre-attach the link (idempotent)
    before = cur.execute(
        "SELECT COUNT(*) FROM news_ticker_links WHERE clean_news_id=? AND ticker=?;",
        (clean_id, ticker),
    ).fetchone()[0]
    cur.execute(
        """INSERT OR IGNORE INTO news_ticker_links
               (clean_news_id, ticker, company, confidence_score, match_method,
                is_primary, is_ambiguous, extraction_notes, created_at)
           VALUES (?, ?, ?, 0.95, 'yahoo_per_ticker', 1, 0, 'per-ticker Yahoo feed', ?);""",
        (clean_id, ticker, company, now),
    )
    if result == "linked" and before:
        result = "dup"  # already had this news linked to this ticker
    return result


def run_collection(force: bool = False) -> None:
    import news_sources
    if not news_sources.is_enabled("Yahoo per-ticker"):
        print("Yahoo per-ticker collector is disabled; skipping.")
        return
    conn = connect_db()
    if not _due(conn, force):
        print("Ticker-news: within cadence window; skipping this cycle.")
        conn.close()
        return

    universe = load_universe(conn)
    cutoff = ny_now() - timedelta(days=MAX_AGE_DAYS)
    print(f"Fetching Yahoo per-ticker news for {len(universe)} universe tickers...")

    # Title-based dedup map so we never re-ingest a headline a wire already
    # carried (identical title, different summary -> different content_hash).
    dedup_from = (ny_now() - timedelta(days=DEDUP_WINDOW_DAYS)).isoformat()
    seen: dict[str, int] = {}
    for cid, tclean in conn.execute(
        "SELECT id, title_clean FROM clean_news WHERE publish_time >= ?;", (dedup_from,)
    ).fetchall():
        k = _title_key(tclean or "")
        if k and k not in seen:
            seen[k] = cid

    tot_new = tot_linked = tot_seen = 0
    for ticker, company in universe:
        try:
            r = cr.get(YAHOO_URL.format(t=ticker), impersonate=IMPERSONATE, timeout=FETCH_TIMEOUT)
            feed = feedparser.parse(r.text)
        except Exception as e:
            print(f"  {ticker}: fetch error ({str(e)[:50]})")
            continue

        kept = 0
        for entry in feed.entries:
            if kept >= MAX_PER_TICKER:
                break
            title = entry.get("title", "")
            if not title or not _on_topic(title, ticker, company):
                continue
            publish_time = _parse_time(entry)
            if not publish_time:
                continue
            try:
                if datetime.fromisoformat(publish_time) < cutoff:
                    continue  # too old
            except Exception:
                continue
            tot_seen += 1
            res = _ingest_headline(
                conn, seen, ticker, company, title,
                entry.get("summary", ""), entry.get("link", ""), publish_time,
            )
            if res == "new":
                tot_new += 1
                kept += 1
            elif res == "linked":
                tot_linked += 1
                kept += 1
        conn.commit()
        if PER_TICKER_DELAY:
            time.sleep(PER_TICKER_DELAY)

    conn.close()
    print(f"Ticker-news done: {tot_new} new headline(s) ingested, "
          f"{tot_linked} existing linked, {tot_seen} on-topic seen.")


if __name__ == "__main__":
    import sys
    run_collection(force="--force" in sys.argv)
