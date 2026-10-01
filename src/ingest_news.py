from __future__ import annotations

import hashlib
import html
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Iterable, Optional

import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# ----------------------------
# Time zone
# ----------------------------
NY_TZ = ZoneInfo("America/New_York")

# ----------------------------
# Project paths
# ----------------------------

from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = BASE_DIR / "financial_news.db"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# RSS sources
# ----------------------------
# You can change these later.
#
# SEC:
# This is a common EDGAR company filings RSS-style endpoint.
# For Week 2, this is enough to test SEC-style RSS ingestion.
#
# PR Newswire:
# PR Newswire has RSS feeds by topic. If one URL fails, replace it
# with another PR Newswire RSS URL from their RSS page.

RSS_SOURCES = [
    {
        "source": "SEC",
        "source_type": "rss",
        "feed_url": "https://www.sec.gov/news/pressreleases.rss",
    },
    {
        "source": "PR Newswire",
        "source_type": "rss",
        "feed_url": "https://www.prnewswire.com/rss/news-releases-list.rss",
    },
    # GlobeNewswire carries the press releases small/micro-cap companies actually
    # issue (content-rich headlines), which fits the nano-cap screen far better
    # than the large-cap-skewed feeds above.
    {
        "source": "GlobeNewswire",
        "source_type": "rss",
        "feed_url": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
    },
    {
        "source": "GlobeNewswire US",
        "source_type": "rss",
        "feed_url": "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/US",
    },
    # FDA press releases — official, real-time drug/device approvals & decisions.
    # An FDA approval/rejection is the top catalyst for biotech/pharma nano-caps.
    {
        "source": "FDA",
        "source_type": "rss",
        "feed_url": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    },
]

# Sent as the HTTP User-Agent when fetching feeds (SEC and some wires expect one).
FEED_USER_AGENT = "financialNewsProject/1.0 (research; contact: student@example.edu)"
# Per-feed HTTP timeout. feedparser.parse(url) fetches with NO timeout and will
# hang forever on a slow server (e.g. GlobeNewswire), stalling the whole pipeline
# — so we fetch with requests + a bounded timeout and skip a feed that's too slow.
FEED_TIMEOUT = int(os.getenv("FEED_TIMEOUT_SEC", "12"))


# ----------------------------
# Data schema
# ----------------------------

@dataclass
class NewsRecord:
    source: str
    title: str
    publish_time: Optional[str]
    url: str
    summary: str
    partial_text: str  # Keeping your requested schema spelling.
    collect_time: str
    source_type: str


# ----------------------------
# Helper functions
# ----------------------------


def clean_html_text(text: Optional[str]) -> str:
    """
    Convert HTML-ish RSS summaries into plain text.
    """
    if not text:
        return ""

    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned


def normalize_time_to_ny(time_value):
    """
    Convert RSS time formats to New York time.
    If parsing fails, return None.
    """
    if not time_value:
        return None

    try:
        dt = date_parser.parse(time_value)

        # If the source gives no timezone, assume New York time.
        # This is reasonable for your project because your professor wants NY time.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NY_TZ)

        return dt.astimezone(NY_TZ).isoformat()

    except Exception:
        return None


def make_record_hash(*values: str) -> str:
    """
    Used for duplicate detection.
    """
    joined = "||".join(v.strip().lower() for v in values if v)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ----------------------------
# RSS ingestion
# ----------------------------

def collect_rss_feed(source: str, source_type: str, feed_url: str) -> list[NewsRecord]:
    """
    Collect records from one RSS/Atom feed and return records
    using your requested schema.
    """
    try:
        resp = requests.get(feed_url, headers={"User-Agent": FEED_USER_AGENT}, timeout=FEED_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"[Warning] {source} fetch failed/timed out in {FEED_TIMEOUT}s ({type(e).__name__}); "
              f"skipping this cycle.")
        return []
    records: list[NewsRecord] = []

    if feed.bozo:
        print(f"[Warning] Feed parse issue for {source}: {feed.bozo_exception}")

    for entry in feed.entries:
        title = clean_html_text(entry.get("title", ""))
        url = entry.get("link", "")

        raw_summary = (
            entry.get("summary")
            or entry.get("description")
            or entry.get("content", [{}])[0].get("value", "")
            if entry.get("content")
            else ""
        )

        summary = clean_html_text(raw_summary)

        # "partial_text" means we only store the title + RSS summary for now,
        # not necessarily the full article body.
        partial_text = clean_html_text(f"{title}. {summary}")

        publish_time_raw = (
            entry.get("published")
            or entry.get("updated")
            or entry.get("created")
        )

        publish_time = normalize_time_to_ny(publish_time_raw)

        record = NewsRecord(
            source=source,
            title=title,
            publish_time=publish_time,
            url=url,
            summary=summary,
            partial_text=partial_text,
            collect_time=ny_now_iso(),
            source_type=source_type,
        )

        # Skip completely empty/broken records
        if record.title or record.url:
            records.append(record)

    return records


def collect_all_rss_sources() -> list[NewsRecord]:
    all_records: list[NewsRecord] = []

    # Live list comes from the DB (dashboard-editable); RSS_SOURCES above is the
    # one-time seed. Falls back to the seed if the registry is somehow empty.
    import news_sources
    sources = news_sources.get_enabled_rss() or RSS_SOURCES

    warn_after = int(os.getenv("EMPTY_FEED_WARN_CYCLES", "3"))
    for src in sources:
        print(f"Collecting RSS from {src['source']}...")
        records = collect_rss_feed(
            source=src["source"],
            source_type=src["source_type"],
            feed_url=src["feed_url"],
        )
        print(f"  Collected {len(records)} records from {src['source']}")
        all_records.extend(records)

        # Empty-feed guard: a source that returns 0 items for several cycles in a
        # row is likely dead/misconfigured (bad URL, expired token, single-company
        # feed with no recent news). It fetches 200-OK so nothing else flags it.
        empty_streak = news_sources.record_ingest_result(src["source"], len(records))
        if empty_streak >= warn_after:
            print(f"  [Warning] '{src['source']}' has returned 0 items for {empty_streak} "
                  f"cycles in a row — the feed may be dead/misconfigured; check its URL "
                  f"in the Sources tab.")

    return all_records


def _title_key(title: str) -> str:
    """Normalized headline key for cross-source dedup (lowercase, alnum only)."""
    import re
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dedupe_obvious_duplicates(records: list[NewsRecord]) -> list[NewsRecord]:
    """
    Collapse the SAME headline arriving from multiple wires in one collection
    (e.g. a press release syndicated to PR Newswire and GlobeNewswire). Keeps the
    first occurrence; identical-content items from later cycles are still caught
    downstream by the content_hash in preprocess_news.
    """
    seen: set[str] = set()
    unique: list[NewsRecord] = []
    removed = 0
    for r in records:
        key = _title_key(r.title)
        if key and key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(r)
    if removed:
        print(f"  Removed {removed} obvious duplicate headline(s) across sources.")
    return unique


# ----------------------------
# Finviz CSV ingestion
# ----------------------------

def load_finviz_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load Finviz CSV export.

    This is not stored in the news table because it is not news.
    It is better used as a ticker reference table for Week 4.
    """
    if not csv_path.exists():
        print(f"[Warning] Finviz CSV not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Add collect_time
    df["collect_time"] = ny_now_iso()
    df["source"] = "Finviz"
    df["source_type"] = "csv"

    return df


# ----------------------------
# SQLite storage
# ----------------------------


def init_db() -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT,
            publish_time TEXT,
            url TEXT,
            summary TEXT,
            partial_text TEXT,
            collect_time TEXT NOT NULL,
            source_type TEXT NOT NULL,
            record_hash TEXT UNIQUE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_reference (
            ticker TEXT PRIMARY KEY,
            company TEXT,
            sector TEXT,
            industry TEXT,
            country TEXT,
            market_cap TEXT,
            price REAL,
            volume INTEGER,
            collect_time TEXT,
            source TEXT,
            source_type TEXT
        );
        """
    )

    conn.commit()
    conn.close()


def insert_news_records(records: Iterable[NewsRecord]) -> tuple[int, int]:
    """
    Insert news records into SQLite.
    Returns:
        inserted_count, skipped_count
    """
    conn = connect_db()
    cur = conn.cursor()

    inserted = 0
    skipped = 0

    for record in records:
        record_hash = make_record_hash(record.source, record.title, record.url)

        try:
            cur.execute(
                """
                INSERT INTO raw_news (
                    source,
                    title,
                    publish_time,
                    url,
                    summary,
                    partial_text,
                    collect_time,
                    source_type,
                    record_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.source,
                    record.title,
                    record.publish_time,
                    record.url,
                    record.summary,
                    record.partial_text,
                    record.collect_time,
                    record.source_type,
                    record_hash,
                ),
            )
            inserted += 1

        except sqlite3.IntegrityError:
            # Duplicate record_hash
            skipped += 1

    conn.commit()
    conn.close()

    return inserted, skipped


def insert_finviz_reference(df: pd.DataFrame) -> int:
    """
    Insert Finviz CSV rows into ticker_reference.

    Finviz CSV column names may differ depending on your export settings.
    This function tries to handle common columns.
    """
    if df.empty:
        return 0

    conn = connect_db()
    cur = conn.cursor()

    inserted = 0

    for _, row in df.iterrows():
        ticker = row.get("ticker") or row.get("symbol")
        company = row.get("company")

        # Skip rows without ticker
        if pd.isna(ticker) or not str(ticker).strip():
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO ticker_reference (
                ticker,
                company,
                sector,
                industry,
                country,
                market_cap,
                price,
                volume,
                collect_time,
                source,
                source_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(ticker).strip(),
                None if pd.isna(company) else str(company).strip(),
                None if pd.isna(row.get("sector")) else str(row.get("sector")).strip(),
                None if pd.isna(row.get("industry")) else str(row.get("industry")).strip(),
                None if pd.isna(row.get("country")) else str(row.get("country")).strip(),
                None if pd.isna(row.get("market_cap")) else str(row.get("market_cap")).strip(),
                safe_float(row.get("price")),
                safe_int(row.get("volume")),
                row.get("collect_time"),
                "Finviz",
                "csv",
            ),
        )

        inserted += 1

    conn.commit()
    conn.close()

    return inserted


def safe_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None

    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except ValueError:
        return None


def safe_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None

    try:
        return int(float(str(value).replace(",", "")))
    except ValueError:
        return None


# ----------------------------
# CSV backup export
# ----------------------------

def export_news_to_csv(records: list[NewsRecord], output_path: Path) -> None:
    if not records:
        return

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(output_path, index=False)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("Initializing database...")
    init_db()

    print("\nCollecting RSS news records...")
    news_records = collect_all_rss_sources()
    news_records = dedupe_obvious_duplicates(news_records)

    backup_path = RAW_DIR / "news_records.csv"
    export_news_to_csv(news_records, backup_path)
    print(f"Saved CSV backup to: {backup_path}")

    inserted, skipped = insert_news_records(news_records)

    print("\nRSS ingestion summary")
    print("---------------------")
    print(f"Collected records: {len(news_records)}")
    print(f"Inserted new records: {inserted}")
    print(f"Skipped duplicates: {skipped}")

    print("\nLoading Finviz CSV...")
    finviz_csv_path = RAW_DIR / "finviz_export.csv"
    finviz_df = load_finviz_csv(finviz_csv_path)
    finviz_inserted = insert_finviz_reference(finviz_df)

    print("\nFinviz ingestion summary")
    print("------------------------")
    print(f"CSV path: {finviz_csv_path}")
    print(f"Rows loaded into ticker_reference: {finviz_inserted}")

    print("\nDone.")


if __name__ == "__main__":
    main()