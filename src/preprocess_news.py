from __future__ import annotations

import hashlib
import html
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from zoneinfo import ZoneInfo


# ----------------------------
# Project paths and constants
# ----------------------------

from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"

NY_TZ = ZoneInfo("America/New_York")


# ----------------------------
# Database connection
# ----------------------------


def init_preprocessing_tables() -> None:
    """
    Create clean_news and ingestion_log if they do not exist.
    raw_news and ticker_reference should already exist from Week 2.
    """
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clean_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_news_id INTEGER UNIQUE,
            source TEXT,
            source_type TEXT,
            title_clean TEXT,
            summary_clean TEXT,
            text_clean TEXT,
            publish_time TEXT,
            collect_time TEXT,
            url TEXT,
            content_hash TEXT,
            is_duplicate INTEGER DEFAULT 0,
            is_ready_for_analysis INTEGER DEFAULT 1,
            preprocessing_notes TEXT,
            created_at TEXT,
            FOREIGN KEY (raw_news_id) REFERENCES raw_news(id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TEXT,
            stage TEXT,
            records_read INTEGER,
            records_inserted INTEGER,
            records_skipped INTEGER,
            status TEXT,
            notes TEXT
        );
        """
    )

    conn.commit()
    conn.close()


# ----------------------------
# Cleaning helpers
# ----------------------------


def clean_text(text: Optional[str]) -> str:
    """
    Clean RSS/news text:
    - handle None
    - decode HTML entities
    - remove HTML tags
    - remove repeated whitespace
    - remove simple boilerplate phrases
    """
    if text is None:
        return ""

    text = str(text)

    # Decode HTML entities like &amp;, &nbsp;
    text = html.unescape(text)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

    # Remove common RSS/news boilerplate fragments
    boilerplate_patterns = [
        r"click here to read more",
        r"read more",
        r"continue reading",
        r"view original content",
        r"source version on .*",
        r"for further information.*",
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_source(source: Optional[str]) -> str:
    """
    Standardize source names.
    Useful now, and more useful later when more sources are added.
    """
    if not source:
        return "Unknown"

    key = source.strip().lower()

    source_map = {
        "sec": "SEC",
        "u.s. sec": "SEC",
        "u.s. securities and exchange commission": "SEC",
        "securities and exchange commission": "SEC",
        "prnewswire": "PR Newswire",
        "pr newswire": "PR Newswire",
        "prn": "PR Newswire",
        "pr newswire association": "PR Newswire",
        "globenewswire": "GlobeNewswire",
        "globenewswire us": "GlobeNewswire",
        "globe newswire": "GlobeNewswire",
        "finviz": "Finviz",
    }

    return source_map.get(key, source.strip())


def normalize_source_type(source_type: Optional[str]) -> str:
    if not source_type:
        return "unknown"

    return source_type.strip().lower()


def normalize_time_to_ny(time_value: Optional[str]) -> Optional[str]:
    """
    Validate and normalize timestamp to America/New_York.

    Since your Week 2 code already stores NY time, this mainly validates it.
    But if a timestamp is UTC or another timezone, this still converts it to NY.
    """
    if not time_value:
        return None

    try:
        dt = date_parser.parse(str(time_value))

        # If timezone is missing, assume New York time.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NY_TZ)

        return dt.astimezone(NY_TZ).isoformat()

    except Exception:
        return None


def make_content_hash(title: str, text: str, source: str = "") -> str:
    """
    Create a content hash for duplicate detection.

    We intentionally do NOT include URL because the same article may appear
    under different URLs. The title + cleaned text is more useful.
    """
    combined = f"{title} {text}"

    combined = combined.lower()
    combined = re.sub(r"[^\w\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()

    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def check_ready_for_analysis(
    title_clean: str,
    text_clean: str,
    publish_time: Optional[str],
    url: str,
) -> tuple[int, list[str]]:
    """
    Decide whether a record is usable for ticker extraction and sentiment later.
    """
    notes = []

    if not title_clean:
        notes.append("missing_title")

    if not publish_time:
        notes.append("missing_publish_time")

    if not url:
        notes.append("missing_url")

    if not text_clean:
        notes.append("missing_text")

    # Minimum requirement for future analysis:
    # title + timestamp are the most important.
    is_ready = 1

    if not title_clean:
        is_ready = 0

    if not publish_time:
        is_ready = 0

    return is_ready, notes


# ----------------------------
# Preprocessing pipeline
# ----------------------------

def fetch_unprocessed_raw_news() -> list[sqlite3.Row]:
    """
    Read raw_news records that have not yet been inserted into clean_news.
    """
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM raw_news
        WHERE id NOT IN (
            SELECT raw_news_id
            FROM clean_news
            WHERE raw_news_id IS NOT NULL
        )
        ORDER BY id ASC;
        """
    )

    rows = cur.fetchall()
    conn.close()

    return rows


def content_hash_exists(content_hash: str) -> bool:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*)
        FROM clean_news
        WHERE content_hash = ?;
        """,
        (content_hash,),
    )

    count = cur.fetchone()[0]
    conn.close()

    return count > 0


def insert_clean_record(
    raw_news_id: int,
    source: str,
    source_type: str,
    title_clean: str,
    summary_clean: str,
    text_clean: str,
    publish_time: Optional[str],
    collect_time: Optional[str],
    url: str,
    content_hash: str,
    is_duplicate: int,
    is_ready_for_analysis: int,
    preprocessing_notes: str,
) -> bool:
    conn = connect_db()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO clean_news (
                raw_news_id,
                source,
                source_type,
                title_clean,
                summary_clean,
                text_clean,
                publish_time,
                collect_time,
                url,
                content_hash,
                is_duplicate,
                is_ready_for_analysis,
                preprocessing_notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                raw_news_id,
                source,
                source_type,
                title_clean,
                summary_clean,
                text_clean,
                publish_time,
                collect_time,
                url,
                content_hash,
                is_duplicate,
                is_ready_for_analysis,
                preprocessing_notes,
                ny_now_iso(),
            ),
        )

        conn.commit()
        inserted = True

    except sqlite3.IntegrityError:
        inserted = False

    conn.close()
    return inserted


def log_preprocessing_run(
    records_read: int,
    records_inserted: int,
    records_skipped: int,
    status: str,
    notes: str,
) -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO ingestion_log (
            run_time,
            stage,
            records_read,
            records_inserted,
            records_skipped,
            status,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            ny_now_iso(),
            "preprocessing",
            records_read,
            records_inserted,
            records_skipped,
            status,
            notes,
        ),
    )

    conn.commit()
    conn.close()


def preprocess_raw_news() -> None:
    init_preprocessing_tables()

    raw_rows = fetch_unprocessed_raw_news()

    records_read = len(raw_rows)
    records_inserted = 0
    records_skipped = 0
    duplicate_count = 0
    not_ready_count = 0

    print(f"Raw records to preprocess: {records_read}")

    for row in raw_rows:
        raw_news_id = row["id"]

        source = normalize_source(row["source"])
        source_type = normalize_source_type(row["source_type"])

        title_clean = clean_text(row["title"])
        summary_clean = clean_text(row["summary"])

        text_clean = clean_text(row["partial_text"])

        if not text_clean:
            text_clean = clean_text(f"{title_clean}. {summary_clean}")

        publish_time = normalize_time_to_ny(row["publish_time"])
        collect_time = normalize_time_to_ny(row["collect_time"])

        url = row["url"] or ""

        content_hash = make_content_hash(title_clean, text_clean, source)

        is_duplicate = 1 if content_hash_exists(content_hash) else 0
        if is_duplicate:
            duplicate_count += 1

        is_ready, notes = check_ready_for_analysis(
            title_clean=title_clean,
            text_clean=text_clean,
            publish_time=publish_time,
            url=url,
        )

        if is_duplicate:
            notes.append("duplicate_content_hash")

        if not is_ready:
            not_ready_count += 1

        preprocessing_notes = ",".join(notes)

        inserted = insert_clean_record(
            raw_news_id=raw_news_id,
            source=source,
            source_type=source_type,
            title_clean=title_clean,
            summary_clean=summary_clean,
            text_clean=text_clean,
            publish_time=publish_time,
            collect_time=collect_time,
            url=url,
            content_hash=content_hash,
            is_duplicate=is_duplicate,
            is_ready_for_analysis=is_ready,
            preprocessing_notes=preprocessing_notes,
        )

        if inserted:
            records_inserted += 1
        else:
            records_skipped += 1

    notes = (
        f"duplicates_marked={duplicate_count}; "
        f"not_ready={not_ready_count}"
    )

    log_preprocessing_run(
        records_read=records_read,
        records_inserted=records_inserted,
        records_skipped=records_skipped,
        status="success",
        notes=notes,
    )

    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Records read: {records_read}")
    print(f"Records inserted into clean_news: {records_inserted}")
    print(f"Records skipped: {records_skipped}")
    print(f"Duplicates marked: {duplicate_count}")
    print(f"Not ready for analysis: {not_ready_count}")
    print("Done.")


if __name__ == "__main__":
    preprocess_raw_news()