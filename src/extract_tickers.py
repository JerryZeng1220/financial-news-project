from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo


# ----------------------------
# Project paths and constants
# ----------------------------

from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

# Re-scan window: recent news that failed to link (its ticker wasn't in the
# growing universe yet) is re-matched against the CURRENT universe each cycle,
# so a ticker that joins the universe *after* its catalyst news still gets the
# news linked -> scored -> alertable. Nano-caps usually pop (join the universe)
# because of the news, so without this the best signals are permanently missed.
RESCAN_HOURS = int(os.getenv("EXTRACT_RESCAN_HOURS", "48"))


# Some tickers are normal English words.
# We should not match these unless they appear in strong ticker formats,
# such as $AI, NASDAQ:AI, NYSE:AI, or (AI).
AMBIGUOUS_TICKERS = {
    "AI", "CAT", "NOW", "ON", "ARE", "IT", "FOR", "ALL", "BY",
    "CAN", "A", "I", "F", "T", "U", "OUT", "LOVE", "GOOD",
    "REAL", "BIG", "RUN", "PLAY", "LIFE", "OPEN", "EVER",
    "NEW", "ONE", "TRUE", "VERY", "LOW", "HIGH", "FAST"
}

# Bare-token fallback: a plain word that IS a current-universe ticker (e.g. the
# stylized brand "AIxC" -> ticker AIXC) links, but only for symbols this long, to
# keep short common-word tickers from false-matching.
UNIVERSE_BARE_MATCH_MIN_LEN = 4


# Words that often appear in official company names but do not help matching.
COMMON_COMPANY_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "plc", "holdings", "holding", "group",
    "common", "stock", "ordinary", "shares", "class", "a", "b",
    "adr", "ads", "sa", "nv", "ag", "lp", "llc", "limited",
    "the"
}


# Optional aliases for companies that are commonly mentioned by brand name.
# This is small for now. You can expand later.
COMPANY_ALIASES = {
    "google": "GOOGL",
    "youtube": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "instagram": "META",
    "meta": "META",
    "amazon": "AMZN",
    "aws": "AMZN",
    "apple": "AAPL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "moderna": "MRNA",
    "pfizer": "PFE",
    "netflix": "NFLX",
    "disney": "DIS",
    "walt disney": "DIS",
}


# ----------------------------
# General helpers
# ----------------------------


def normalize_text(text: Optional[str]) -> str:
    """
    Lowercase and remove punctuation for safer matching.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_company_name(company: Optional[str]) -> str:
    """
    Convert company names into searchable names.

    Example:
        "Apple Inc." -> "apple"
        "NVIDIA Corporation" -> "nvidia"
        "Moderna, Inc." -> "moderna"
    """
    text = normalize_text(company)
    words = text.split()

    filtered_words = [
        word for word in words
        if word not in COMMON_COMPANY_SUFFIXES
    ]

    return " ".join(filtered_words).strip()


# ----------------------------
# Database setup
# ----------------------------

def init_ticker_extraction_table() -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS news_ticker_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_news_id INTEGER,
            ticker TEXT,
            company TEXT,
            sector TEXT,
            industry TEXT,
            confidence_score REAL,
            match_method TEXT,
            is_primary INTEGER DEFAULT 1,
            is_ambiguous INTEGER DEFAULT 0,
            extraction_notes TEXT,
            created_at TEXT,
            UNIQUE(clean_news_id, ticker),
            FOREIGN KEY (clean_news_id) REFERENCES clean_news(id),
            FOREIGN KEY (ticker) REFERENCES ticker_reference(ticker)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_extraction_status (
            clean_news_id INTEGER PRIMARY KEY,
            processed_at TEXT,
            match_count INTEGER,
            status TEXT,
            notes TEXT,
            FOREIGN KEY (clean_news_id) REFERENCES clean_news(id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS extraction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TEXT,
            records_read INTEGER,
            records_with_match INTEGER,
            records_without_match INTEGER,
            ticker_links_inserted INTEGER,
            status TEXT,
            notes TEXT
        );
        """
    )

    conn.commit()
    conn.close()


# ----------------------------
# Data loading
# ----------------------------

def load_unprocessed_clean_news() -> pd.DataFrame:
    """
    Load clean news records that are ready for analysis, not duplicates, and
    either:
      (a) have never gone through ticker extraction, OR
      (b) are recent (within RESCAN_HOURS) and still have NO ticker link.

    Case (b) is the re-scan: the active universe changes over time, and a
    nano-cap typically joins the "growing" universe *after* its catalyst news
    was already processed (and skipped for no match). Re-matching recent
    unlinked news against the current universe each cycle lets those late
    joiners finally get linked. Links use INSERT OR IGNORE and status uses
    INSERT OR REPLACE, so reprocessing is idempotent.
    """
    conn = connect_db()

    rescan_cutoff = (datetime.now(NY_TZ) - timedelta(hours=RESCAN_HOURS)).isoformat()

    query = """
        SELECT *
        FROM clean_news
        WHERE is_ready_for_analysis = 1
          AND is_duplicate = 0
          AND (
              id NOT IN (SELECT clean_news_id FROM ticker_extraction_status)
              OR (
                  publish_time >= ?
                  AND id NOT IN (SELECT clean_news_id FROM news_ticker_links)
              )
          )
        ORDER BY id ASC;
    """

    df = pd.read_sql_query(query, conn, params=(rescan_cutoff,))
    conn.close()

    return df


def active_universe_exists() -> bool:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'active_ticker_universe';
        """
    )

    table_exists = cur.fetchone() is not None

    if not table_exists:
        conn.close()
        return False

    cur.execute("SELECT COUNT(*) FROM active_ticker_universe;")
    count = cur.fetchone()[0]

    conn.close()

    return count > 0


def load_ticker_reference() -> pd.DataFrame:
    """
    Prefer active_ticker_universe if it exists.
    Fall back to full ticker_reference if active universe is missing.
    """
    conn = connect_db()

    if active_universe_exists():
        print("Using active_ticker_universe for ticker extraction.")

        query = """
            SELECT ticker, company, sector, industry
            FROM active_ticker_universe
            WHERE ticker IS NOT NULL
              AND ticker != ''
              AND company IS NOT NULL
              AND company != '';
        """
    else:
        print("Using full ticker_reference for ticker extraction.")

        query = """
            SELECT ticker, company, sector, industry
            FROM ticker_reference
            WHERE ticker IS NOT NULL
              AND ticker != ''
              AND company IS NOT NULL
              AND company != '';
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["company"] = df["company"].astype(str).str.strip()

    return df


def prepare_reference_table(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized company names and remove weak rows.
    """
    ticker_df = ticker_df.copy()

    ticker_df["normalized_company"] = ticker_df["company"].apply(normalize_company_name)

    # Remove companies whose normalized name becomes too short.
    # This helps avoid matching generic words.
    ticker_df = ticker_df[
        ticker_df["normalized_company"].str.len() >= 3
    ]

    # Remove duplicate tickers if somehow present.
    ticker_df = ticker_df.drop_duplicates(subset=["ticker"])

    return ticker_df


# ----------------------------
# Matching functions
# ----------------------------

def find_explicit_ticker_matches(text: str, ticker_set: set[str]) -> list[dict]:
    """
    Find strong ticker patterns:
    - $TSLA
    - NASDAQ:AAPL
    - NYSE:IBM
    - (AAPL)
    """
    matches: list[dict] = []

    if not text:
        return matches

    # $TSLA
    dollar_matches = re.findall(r"\$([A-Z]{1,6})\b", text)
    for ticker in dollar_matches:
        ticker = ticker.upper()
        if ticker in ticker_set:
            matches.append({
                "ticker": ticker,
                "confidence_score": 0.98,
                "match_method": "dollar_ticker_match",
                "is_ambiguous": 1 if ticker in AMBIGUOUS_TICKERS else 0,
                "notes": "Matched ticker with dollar sign"
            })

    # NASDAQ:AAPL, NYSE:IBM, AMEX:XYZ
    exchange_matches = re.findall(
        r"\b(?:NASDAQ|NYSE|AMEX|NYSEAMERICAN|OTC):\s*([A-Z]{1,6})\b",
        text
    )
    for ticker in exchange_matches:
        ticker = ticker.upper()
        if ticker in ticker_set:
            matches.append({
                "ticker": ticker,
                "confidence_score": 0.99,
                "match_method": "exchange_ticker_match",
                "is_ambiguous": 1 if ticker in AMBIGUOUS_TICKERS else 0,
                "notes": "Matched ticker with exchange prefix"
            })

    # (AAPL)
    paren_matches = re.findall(r"\(([A-Z]{1,6})\)", text)
    for ticker in paren_matches:
        ticker = ticker.upper()
        if ticker in ticker_set:
            matches.append({
                "ticker": ticker,
                "confidence_score": 0.95 if ticker not in AMBIGUOUS_TICKERS else 0.85,
                "match_method": "parentheses_ticker_match",
                "is_ambiguous": 1 if ticker in AMBIGUOUS_TICKERS else 0,
                "notes": "Matched ticker inside parentheses"
            })

    return matches


def find_alias_matches(
    normalized_news_text: str,
    alias_map: dict[str, str],
    ticker_set: set[str],
) -> list[dict]:
    """
    Match common brand/company aliases manually.

    Example:
        "google" -> GOOGL
        "facebook" -> META
    """
    matches: list[dict] = []

    for alias, ticker in alias_map.items():
        ticker = ticker.upper()

        if ticker not in ticker_set:
            continue

        pattern = r"\b" + re.escape(normalize_text(alias)) + r"\b"

        if re.search(pattern, normalized_news_text):
            matches.append({
                "ticker": ticker,
                "confidence_score": 0.92,
                "match_method": "alias_match",
                "is_ambiguous": 0,
                "notes": f"Matched alias: {alias}"
            })

    return matches


def find_company_name_matches(
    normalized_news_text: str,
    ticker_df: pd.DataFrame,
) -> list[dict]:
    """
    Match normalized company names against normalized news text.

    This first version loops through all companies. With 11k rows and modest
    news volume, it is okay for the prototype. Later we can optimize.
    """
    matches: list[dict] = []

    if not normalized_news_text:
        return matches

    for _, row in ticker_df.iterrows():
        normalized_company = row["normalized_company"]

        if not normalized_company:
            continue

        # Avoid matching company names that are too generic.
        if len(normalized_company) < 3:
            continue

        pattern = r"\b" + re.escape(normalized_company) + r"\b"

        if re.search(pattern, normalized_news_text):
            matches.append({
                "ticker": row["ticker"],
                "confidence_score": 0.90,
                "match_method": "company_name_match",
                "is_ambiguous": 0,
                "notes": f"Matched company name: {row['company']}"
            })

    return matches


def find_universe_ticker_matches(
    normalized_news_text: str,
    ticker_set: set[str],
) -> list[dict]:
    """
    Bare-token fallback: match a plain word that IS a current-universe ticker
    (e.g. the stylized brand "AIxC" -> normalized "aixc" -> ticker AIXC). Only safe
    because ticker_set here is the small daily active universe, so this is called
    only when the reference is universe-scoped. Guarded: skip ambiguous/common
    tickers and short symbols to avoid false hits.
    """
    matches: list[dict] = []
    if not normalized_news_text:
        return matches

    tokens = set(normalized_news_text.split())
    for ticker in ticker_set:
        if len(ticker) < UNIVERSE_BARE_MATCH_MIN_LEN or ticker in AMBIGUOUS_TICKERS:
            continue
        if ticker.lower() in tokens:
            matches.append({
                "ticker": ticker,
                "confidence_score": 0.80,
                "match_method": "universe_bare_match",
                "is_ambiguous": 0,
                "notes": "Bare token matched a current-universe ticker",
            })

    return matches


def deduplicate_matches(matches: list[dict]) -> list[dict]:
    """
    If the same ticker is matched by multiple methods,
    keep the match with the highest confidence.
    """
    best_by_ticker: dict[str, dict] = {}

    for match in matches:
        ticker = match["ticker"]

        if ticker not in best_by_ticker:
            best_by_ticker[ticker] = match
        else:
            if match["confidence_score"] > best_by_ticker[ticker]["confidence_score"]:
                best_by_ticker[ticker] = match

    return list(best_by_ticker.values())


def mark_primary_match(matches: list[dict]) -> list[dict]:
    """
    Mark the highest-confidence ticker as primary.
    If multiple tickers are tied, the first remains primary.
    """
    if not matches:
        return matches

    matches = sorted(matches, key=lambda m: m["confidence_score"], reverse=True)

    for i, match in enumerate(matches):
        match["is_primary"] = 1 if i == 0 else 0

    return matches


def extract_tickers_from_news(
    title: str,
    text: str,
    ticker_df: pd.DataFrame,
    ticker_set: set[str],
    universe_scoped: bool = False,
) -> list[dict]:
    """
    Extract ticker matches from one news item.
    """
    original_text = f"{title or ''} {text or ''}"
    normalized_news_text = normalize_text(original_text)

    matches: list[dict] = []

    # 1. Strong explicit ticker patterns
    matches.extend(find_explicit_ticker_matches(original_text, ticker_set))

    # 2. Common alias matches
    matches.extend(find_alias_matches(
        normalized_news_text=normalized_news_text,
        alias_map=COMPANY_ALIASES,
        ticker_set=ticker_set,
    ))

    # 3. Company name matches
    matches.extend(find_company_name_matches(
        normalized_news_text=normalized_news_text,
        ticker_df=ticker_df,
    ))

    # 4. Bare-token match against the CURRENT universe (fallback for brand
    #    abbreviations that equal the ticker, e.g. "AIxC" -> AIXC). Only when the
    #    reference is the small active universe — never against the full 11k list.
    if universe_scoped:
        matches.extend(find_universe_ticker_matches(
            normalized_news_text=normalized_news_text,
            ticker_set=ticker_set,
        ))

    # 5. Deduplicate and mark primary
    matches = deduplicate_matches(matches)
    matches = mark_primary_match(matches)

    return matches


# ----------------------------
# Insert results
# ----------------------------

def insert_ticker_links(
    clean_news_id: int,
    matches: list[dict],
    ticker_df: pd.DataFrame,
) -> int:
    if not matches:
        return 0

    conn = connect_db()
    cur = conn.cursor()

    inserted = 0

    ticker_lookup = ticker_df.set_index("ticker").to_dict(orient="index")

    for match in matches:
        ticker = match["ticker"]
        ref = ticker_lookup.get(ticker, {})

        company = ref.get("company")
        sector = ref.get("sector")
        industry = ref.get("industry")

        try:
            cur.execute(
                """
                INSERT OR IGNORE INTO news_ticker_links (
                    clean_news_id,
                    ticker,
                    company,
                    sector,
                    industry,
                    confidence_score,
                    match_method,
                    is_primary,
                    is_ambiguous,
                    extraction_notes,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    clean_news_id,
                    ticker,
                    company,
                    sector,
                    industry,
                    match["confidence_score"],
                    match["match_method"],
                    match.get("is_primary", 1),
                    match["is_ambiguous"],
                    match["notes"],
                    ny_now_iso(),
                ),
            )

            if cur.rowcount > 0:
                inserted += 1

        except Exception as e:
            print(f"Insert error for clean_news_id={clean_news_id}, ticker={ticker}: {e}")

    conn.commit()
    conn.close()

    return inserted


def log_extraction_run(
    records_read: int,
    records_with_match: int,
    records_without_match: int,
    ticker_links_inserted: int,
    status: str,
    notes: str,
) -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO extraction_log (
            run_time,
            records_read,
            records_with_match,
            records_without_match,
            ticker_links_inserted,
            status,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            ny_now_iso(),
            records_read,
            records_with_match,
            records_without_match,
            ticker_links_inserted,
            status,
            notes,
        ),
    )

    conn.commit()
    conn.close()


def insert_extraction_status(
    clean_news_id: int,
    match_count: int,
    status: str,
    notes: str = "",
) -> None:
    """
    Record that this clean_news row has gone through ticker extraction,
    even if no ticker was found.
    """
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO ticker_extraction_status (
            clean_news_id,
            processed_at,
            match_count,
            status,
            notes
        )
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            clean_news_id,
            ny_now_iso(),
            match_count,
            status,
            notes,
        ),
    )

    conn.commit()
    conn.close()
# ----------------------------
# Main pipeline
# ----------------------------

def run_ticker_extraction() -> None:
    print("Initializing ticker extraction table...")
    init_ticker_extraction_table()

    print("Loading clean news...")
    clean_news_df = load_unprocessed_clean_news()

    print("Loading ticker reference...")
    ticker_df = load_ticker_reference()
    ticker_df = prepare_reference_table(ticker_df)

    ticker_set = set(ticker_df["ticker"].tolist())
    # Bare-token matching is only safe against the small daily universe, not the
    # full 11k reference — gate it on which one we're actually using.
    universe_scoped = active_universe_exists()

    print(f"Clean news records to process: {len(clean_news_df)}")
    print(f"Ticker reference rows: {len(ticker_df)} (universe-scoped: {universe_scoped})")

    total_links_inserted = 0
    records_with_match = 0
    records_without_match = 0

    for _, row in clean_news_df.iterrows():
        clean_news_id = int(row["id"])
        title = row.get("title_clean", "")
        text = row.get("text_clean", "")

        matches = extract_tickers_from_news(
            title=title,
            text=text,
            ticker_df=ticker_df,
            ticker_set=ticker_set,
            universe_scoped=universe_scoped,
        )

        if matches:
            records_with_match += 1
            status = "matched"
            notes = f"matched_tickers={','.join([m['ticker'] for m in matches])}"
        else:
            records_without_match += 1
            status = "no_match"
            notes = "no ticker matched by current rules"

        inserted = insert_ticker_links(
            clean_news_id=clean_news_id,
            matches=matches,
            ticker_df=ticker_df,
        )

        total_links_inserted += inserted

        insert_extraction_status(
            clean_news_id=clean_news_id,
            match_count=len(matches),
            status=status,
            notes=notes,
        )

    log_extraction_run(
        records_read=len(clean_news_df),
        records_with_match=records_with_match,
        records_without_match=records_without_match,
        ticker_links_inserted=total_links_inserted,
        status="success",
        notes="first rule-based ticker extraction run",
    )

    print("\nTicker extraction summary")
    print("-------------------------")
    print(f"Clean news processed: {len(clean_news_df)}")
    print(f"Records with at least one ticker: {records_with_match}")
    print(f"Records without ticker match: {records_without_match}")
    print(f"Ticker links inserted: {total_links_inserted}")
    print("Done.")


if __name__ == "__main__":
    run_ticker_extraction()