from __future__ import annotations

import io
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


from common import connect_db, ny_now_iso
import app_settings

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
FINVIZ_CSV_PATH = BASE_DIR / "data" / "raw" / "finviz_export.csv"
NY_TZ = ZoneInfo("America/New_York")

# ----------------------------
# Finviz live screener config
# ----------------------------
# Base URL for the FREE Finviz screener (HTML pages, 20 rows per page).
FINVIZ_SCREENER_BASE = "https://finviz.com/screener.ashx"

# Default filters = Change up (over 0%) + Relative volume over 2 + Current volume over 300K.
#   ta_change_u    -> Change: Up (positive)
#   sh_relvol_o2   -> Relative Volume: Over 2
#   sh_curvol_o300 -> Current Volume: Over 300K
# Override via the FINVIZ_FILTERS env var. The safest way to get the exact codes
# is to build the filters in the Finviz UI and copy the f=... part of the URL.
DEFAULT_FINVIZ_FILTERS = "ta_change_u,sh_relvol_o2,sh_curvol_o300"

load_dotenv(BASE_DIR / ".env")


def parse_percent(value) -> float | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().replace("%", "").replace(",", "")

    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).replace(",", "").replace("$", "").strip()

    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).replace(",", "").strip()

    if text == "":
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


def init_tables() -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS active_ticker_universe (
            ticker TEXT PRIMARY KEY,
            company TEXT,
            sector TEXT,
            industry TEXT,
            country TEXT,
            market_cap TEXT,
            price REAL,
            volume INTEGER,
            change_percent REAL,
            source TEXT,
            filter_name TEXT,
            collected_at TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS active_ticker_universe_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            company TEXT,
            price REAL,
            volume INTEGER,
            change_percent REAL,
            source TEXT,
            filter_name TEXT,
            collected_at TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS active_universe_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TEXT,
            mode TEXT,
            source TEXT,
            active_count INTEGER,
            added_count INTEGER,
            removed_count INTEGER,
            status TEXT,
            notes TEXT
        );
        """
    )

    conn.commit()
    conn.close()


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    normalized_cols = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    for name in possible_names:
        key = name.lower().strip().replace(" ", "_")
        if key in normalized_cols:
            return normalized_cols[key]

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_from_csv_snapshot() -> pd.DataFrame:
    if not FINVIZ_CSV_PATH.exists():
        raise FileNotFoundError(f"Finviz CSV not found: {FINVIZ_CSV_PATH}")

    modified_time = datetime.fromtimestamp(FINVIZ_CSV_PATH.stat().st_mtime, tz=NY_TZ)
    print(f"Using local CSV snapshot: {FINVIZ_CSV_PATH}")
    print(f"CSV last modified: {modified_time.isoformat()}")

    df = pd.read_csv(FINVIZ_CSV_PATH)
    return normalize_columns(df)


# Market-cap band for the screener — dashboard-editable (app_settings 'market_cap').
# Swaps the Finviz `cap_` filter in the export URL so Nano/Micro/Small/Mid/Large
# can be flipped live, no restart. Finviz codes: cap_nano/micro/small/mid/large.
_VALID_CAPS = {"nano", "micro", "small", "mid", "large"}


def _market_cap() -> str:
    v = str(app_settings.get("market_cap", "nano")).strip().lower()
    return v if v in _VALID_CAPS else "nano"


def _apply_market_cap(url: str) -> str:
    """Replace the Finviz cap_ filter in the URL with the selected band."""
    cap = _market_cap()
    return re.sub(r"cap_[a-z]+", f"cap_{cap}", url) if "cap_" in url else url


def load_from_authorized_url() -> pd.DataFrame:
    """
    Loads live/premium screener data from an authorized URL.

    Use this only with permission and valid account access.
    Do not use this to bypass paywalls, CAPTCHAs, or access controls.
    """
    url = os.getenv("FINVIZ_ACTIVE_UNIVERSE_URL", "").strip()
    cookie = os.getenv("FINVIZ_COOKIE", "").strip()
    use_curl_impersonate = os.getenv("USE_CURL_IMPERSONATE", "false").lower() == "true"

    if not url:
        raise ValueError("FINVIZ_ACTIVE_UNIVERSE_URL is empty in .env")
    url = _apply_market_cap(url)  # dashboard-selected market-cap band

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/json,text/plain,*/*",
    }

    if cookie:
        headers["Cookie"] = cookie

    print(f"Fetching authorized screener URL: {url}")
    print(f"Using curl impersonate: {use_curl_impersonate}")

    max_retries = int(os.getenv("FINVIZ_MAX_RETRIES", "4"))
    backoff_base = float(os.getenv("FINVIZ_BACKOFF", "10"))

    def _get():
        if use_curl_impersonate:
            from curl_cffi import requests as curl_requests
            return curl_requests.get(url, headers=headers, impersonate="chrome", timeout=30)
        return requests.get(url, headers=headers, timeout=30)

    # The Elite export endpoint throttles rapid repeated requests with HTTP 429.
    # Retry with growing backoff so a temporary throttle doesn't crash the pipeline.
    response = None
    for attempt in range(1, max_retries + 1):
        response = _get()
        if response.status_code == 200:
            break
        if response.status_code in (429, 503) and attempt < max_retries:
            wait = backoff_base * attempt
            print(
                f"  [Finviz {response.status_code} rate-limited] backing off "
                f"{wait:.0f}s (attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)
            continue
        break

    if response is not None and response.status_code in (429, 503):
        raise RuntimeError(
            f"Finviz Elite export returned HTTP {response.status_code} (rate limited) after "
            f"{max_retries} attempts. The export endpoint throttles rapid repeated requests — "
            f"wait a minute and try again, or raise FINVIZ_BACKOFF."
        )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    text = response.text

    if "application/json" in content_type or text.strip().startswith("{") or text.strip().startswith("["):
        data = response.json()
        df = pd.DataFrame(data)
    else:
        # Expired/invalid cookie: Finviz serves its login HTML with a 200 instead of
        # the CSV export, which pandas would choke on with a cryptic tokenizing error.
        # Detect it and fail with an actionable message.
        low = text.lstrip()[:200].lower()
        if "text/html" in content_type or low.startswith("<!doctype") or low.startswith("<html"):
            raise RuntimeError(
                "Finviz returned an HTML page, not the CSV export — FINVIZ_COOKIE has "
                "almost certainly expired. Refresh it: log into elite.finviz.com, copy the "
                "'Cookie' request header (DevTools → Network → any elite.finviz.com request), "
                "and update FINVIZ_COOKIE in .env."
            )
        df = pd.read_csv(io.StringIO(text))

    return normalize_columns(df)


# ----------------------------------------------------------------------
# NEW: live Finviz free-screener scraper (HTML, paginated)
# ----------------------------------------------------------------------

def fetch_finviz_html(url: str) -> str:
    """
    Fetch one Finviz screener HTML page, retrying with backoff on rate limits.

    When USE_CURL_IMPERSONATE=true, use curl_cffi so the request's TLS/HTTP2
    fingerprint looks like a real Chrome browser (helps avoid anti-bot blocks).
    Otherwise fall back to plain requests.

    Finviz throttles aggressive scraping with HTTP 429 (and sometimes 503).
    On those we back off (FINVIZ_BACKOFF seconds, growing each attempt) and
    retry up to FINVIZ_MAX_RETRIES times before giving up with a clear message.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    cookie = os.getenv("FINVIZ_COOKIE", "").strip()
    if cookie:
        headers["Cookie"] = cookie

    use_curl_impersonate = os.getenv("USE_CURL_IMPERSONATE", "false").lower() == "true"
    max_retries = int(os.getenv("FINVIZ_MAX_RETRIES", "4"))
    backoff_base = float(os.getenv("FINVIZ_BACKOFF", "6"))

    def _get():
        if use_curl_impersonate:
            from curl_cffi import requests as curl_requests

            return curl_requests.get(url, headers=headers, impersonate="chrome", timeout=30)
        return requests.get(url, headers=headers, timeout=30)

    response = None
    for attempt in range(1, max_retries + 1):
        response = _get()

        if response.status_code == 200:
            return response.text

        if response.status_code in (429, 503) and attempt < max_retries:
            wait = backoff_base * attempt
            print(
                f"  [Finviz {response.status_code} rate-limited] "
                f"backing off {wait:.0f}s (attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait)
            continue

        # Any other status, or retries exhausted -> raise.
        break

    if response is not None and response.status_code in (429, 503):
        raise RuntimeError(
            f"Finviz kept returning HTTP {response.status_code} (rate limited) after "
            f"{max_retries} attempts. Wait a few minutes, and/or raise FINVIZ_PAGE_DELAY "
            f"and FINVIZ_BACKOFF, then try again."
        )

    response.raise_for_status()
    return response.text


def parse_finviz_total(html: str) -> int | None:
    """
    Finviz shows the result count near the top, e.g. "1 / 234 Total".
    Returns the total number of matching tickers, or None if not found.
    Only used for logging; pagination stops when no new tickers appear.
    """
    match = re.search(r"([\d,]+)\s*Total", html)
    if not match:
        return None

    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_finviz_screener_table(html: str) -> pd.DataFrame | None:
    """
    Parse one Finviz "Overview" (v=111) screener page into a DataFrame.

    Strategy (no lxml / pandas.read_html needed, just BeautifulSoup):
      - Locate the results table (class "screener_table").
      - Read its HEADER row to get the real column names ("No.", "Ticker",
        "Company", "Sector", "Industry", "Country", "Market Cap", "P/E",
        "Price", "Change", "Volume"), then map each data row's cells onto those
        names by position. Mapping by the table's own header (instead of a
        hard-coded offset) is what keeps the Ticker/Company/Change columns
        correctly aligned even though the first column is the row number.
      - Downstream, build_active_universe() picks the columns it needs
        ("ticker", "change", "price", "volume", ...) by name via find_column.

    Returns None if the results table or its header cannot be found
    (e.g. blocked page or a layout change).
    """
    soup = BeautifulSoup(html, "html.parser")

    results_table = soup.find(
        "table",
        class_=lambda c: bool(c) and "screener_table" in c,
    )
    if results_table is None:
        return None

    rows = results_table.find_all("tr")
    if not rows:
        return None

    # The header row is the one containing a "Ticker" cell.
    header_labels: list[str] | None = None
    header_row_index = 0
    for i, row in enumerate(rows):
        labels = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
        if any(label.strip().lower() == "ticker" for label in labels):
            header_labels = labels
            header_row_index = i
            break

    if not header_labels:
        return None

    # Normalize header names the same way normalize_columns() does, so that
    # find_column() downstream can locate ticker/company/change/price/volume.
    column_names = [
        label.strip().lower().replace(" ", "_") for label in header_labels
    ]

    records: list[dict] = []

    for row in rows[header_row_index + 1:]:
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        if len(cells) != len(column_names):
            # Spacer / malformed row; skip rather than misalign columns.
            continue

        record = dict(zip(column_names, cells))

        ticker = record.get("ticker", "")
        # A real ticker is a short alphanumeric symbol; skip junk/spacer rows.
        if not re.fullmatch(r"[A-Za-z0-9.\-]{1,8}", ticker):
            continue

        records.append(record)

    if not records:
        return None

    return pd.DataFrame(records)


def load_from_finviz_screener() -> pd.DataFrame:
    """
    Live-scrape the FREE Finviz screener, page by page, and return all matching
    tickers as a normalized DataFrame ready for build_active_universe().

    Controlled by env vars:
      FINVIZ_FILTERS     filter codes for f= (default DEFAULT_FINVIZ_FILTERS)
      FINVIZ_VIEW        screener view id (default "111" = Overview)
      FINVIZ_PAGE_DELAY  seconds to sleep between page requests (default 1.5)
      FINVIZ_MAX_PAGES   safety cap on number of pages (default 100 = 2000 rows)
      USE_CURL_IMPERSONATE  "true" to route through curl_cffi browser impersonation

    NOTE: Automated scraping of the free Finviz site is subject to Finviz's
    Terms of Service. Keep volume low (one run per pipeline, a delay between
    pages). For heavy/commercial use, a Finviz Elite subscription + the official
    export endpoint (use load_from_authorized_url) is the clean path.
    """
    filters = os.getenv("FINVIZ_FILTERS", DEFAULT_FINVIZ_FILTERS).strip()
    view = os.getenv("FINVIZ_VIEW", "111").strip()
    page_delay = float(os.getenv("FINVIZ_PAGE_DELAY", "1.5"))
    max_pages = int(os.getenv("FINVIZ_MAX_PAGES", "100"))
    rows_per_page = 20

    use_curl_impersonate = os.getenv("USE_CURL_IMPERSONATE", "false").lower() == "true"
    print("Live Finviz screener scrape")
    print(f"  filters: {filters}")
    print(f"  view: v={view}")
    print(f"  curl impersonate: {use_curl_impersonate}")

    frames: list[pd.DataFrame] = []
    seen_tickers: set[str] = set()

    for page_index in range(max_pages):
        start_row = page_index * rows_per_page + 1
        url = f"{FINVIZ_SCREENER_BASE}?v={view}&f={filters}&r={start_row}"

        if page_index == 0:
            print(f"Fetching: {url}")
        else:
            time.sleep(page_delay)
            print(f"  page r={start_row} ...")

        html = fetch_finviz_html(url)

        if page_index == 0:
            total = parse_finviz_total(html)
            if total is not None:
                print(f"Finviz reports {total} matching tickers.")

        table = parse_finviz_screener_table(html)

        if table is None:
            if page_index == 0:
                raise ValueError(
                    "Could not locate the Finviz screener results table. "
                    "The page HTML may have changed, or the request was blocked. "
                    "Try USE_CURL_IMPERSONATE=true, or verify the filter URL in a browser."
                )
            # No more rows on later pages -> we're done.
            break

        page_tickers = set(table["ticker"].astype(str).str.upper())
        new_tickers = page_tickers - seen_tickers

        if not new_tickers:
            # Finviz clamps an out-of-range r back to the last valid page;
            # seeing only already-collected tickers means we've reached the end.
            break

        seen_tickers |= new_tickers
        frames.append(table)

    if not frames:
        raise ValueError("Finviz screener returned no rows.")

    df = pd.concat(frames, ignore_index=True)
    df = normalize_columns(df)
    df = df.drop_duplicates(subset=["ticker"])

    print(f"Scraped {len(df)} unique tickers across {len(frames)} page(s).")

    return df


def build_active_universe(df: pd.DataFrame) -> pd.DataFrame:
    ticker_col = find_column(df, ["ticker", "symbol"])
    company_col = find_column(df, ["company", "company_name", "name"])
    change_col = find_column(df, ["change", "change_%", "change_percent", "chg"])
    price_col = find_column(df, ["price", "last", "last_price"])
    volume_col = find_column(df, ["volume", "vol"])
    sector_col = find_column(df, ["sector"])
    industry_col = find_column(df, ["industry"])
    country_col = find_column(df, ["country"])
    market_cap_col = find_column(df, ["market_cap", "marketcap", "market_capitalization"])

    if ticker_col is None:
        raise ValueError(f"Could not find ticker/symbol column. Columns found: {list(df.columns)}")

    if company_col is None:
        print("[Warning] Could not find company column. Using ticker as company.")
        df["company_fallback"] = df[ticker_col]
        company_col = "company_fallback"

    if change_col is None:
        print("[Warning] Could not find Change column. Using all rows as active universe.")
        df["change_percent_parsed"] = None
        active_df = df.copy()
    else:
        df["change_percent_parsed"] = df[change_col].apply(parse_percent)
        # Finviz already filters change > 0 server-side via ta_change_u, but we
        # keep this as a harmless safety net for non-screener data sources.
        active_df = df[df["change_percent_parsed"] > 0].copy()

    output = pd.DataFrame()
    output["ticker"] = active_df[ticker_col].astype(str).str.strip().str.upper()
    output["company"] = active_df[company_col].astype(str).str.strip()
    output["sector"] = active_df[sector_col].astype(str).str.strip() if sector_col else None
    output["industry"] = active_df[industry_col].astype(str).str.strip() if industry_col else None
    output["country"] = active_df[country_col].astype(str).str.strip() if country_col else None
    output["market_cap"] = active_df[market_cap_col].astype(str).str.strip() if market_cap_col else None
    output["price"] = active_df[price_col].apply(safe_float) if price_col else None
    output["volume"] = active_df[volume_col].apply(safe_int) if volume_col else None
    output["change_percent"] = active_df["change_percent_parsed"]
    output["source"] = "Finviz"
    output["filter_name"] = "positive_change"
    output["collected_at"] = ny_now_iso()

    output = output.dropna(subset=["ticker"])
    output = output[output["ticker"] != ""]
    output = output.drop_duplicates(subset=["ticker"])

    return output


def get_existing_active_tickers() -> set[str]:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT ticker FROM active_ticker_universe;")
    tickers = {row[0] for row in cur.fetchall()}

    conn.close()
    return tickers


def save_active_universe(active_df: pd.DataFrame) -> tuple[int, int]:
    new_tickers = set(active_df["ticker"].tolist())
    old_tickers = get_existing_active_tickers()

    added = new_tickers - old_tickers
    removed = old_tickers - new_tickers

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM active_ticker_universe;")

    for _, row in active_df.iterrows():
        values = (
            row["ticker"],
            row["company"],
            row["sector"],
            row["industry"],
            row["country"],
            row["market_cap"],
            row["price"],
            row["volume"],
            row["change_percent"],
            row["source"],
            row["filter_name"],
            row["collected_at"],
        )

        cur.execute(
            """
            INSERT OR REPLACE INTO active_ticker_universe (
                ticker,
                company,
                sector,
                industry,
                country,
                market_cap,
                price,
                volume,
                change_percent,
                source,
                filter_name,
                collected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            values,
        )

        cur.execute(
            """
            INSERT INTO active_ticker_universe_history (
                ticker,
                company,
                price,
                volume,
                change_percent,
                source,
                filter_name,
                collected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row["ticker"],
                row["company"],
                row["price"],
                row["volume"],
                row["change_percent"],
                row["source"],
                row["filter_name"],
                row["collected_at"],
            ),
        )

    conn.commit()
    conn.close()

    return len(added), len(removed)


def log_run(
    mode: str,
    source: str,
    active_count: int,
    added_count: int,
    removed_count: int,
    status: str,
    notes: str,
) -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO active_universe_runs (
            run_time,
            mode,
            source,
            active_count,
            added_count,
            removed_count,
            status,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            ny_now_iso(),
            mode,
            source,
            active_count,
            added_count,
            removed_count,
            status,
            notes,
        ),
    )

    conn.commit()
    conn.close()


def universe_is_fresh() -> bool:
    """
    True if the active universe was refreshed within UNIVERSE_MIN_REFRESH_SEC.
    Lets the main app loop run fast (1-2 min) while the Finviz fetch only actually
    happens every few minutes — so we never trip the Elite export rate limit.
    """
    threshold = int(os.getenv("UNIVERSE_MIN_REFRESH_SEC", "240"))
    if threshold <= 0:
        return False
    conn = connect_db()
    row = conn.execute("SELECT MAX(collected_at) FROM active_ticker_universe;").fetchone()
    conn.close()
    if not row or not row[0]:
        return False  # empty -> must fetch
    try:
        age = (datetime.now(NY_TZ) - datetime.fromisoformat(row[0])).total_seconds()
    except Exception:
        return False
    return age < threshold


def main() -> None:
    init_tables()

    mode = os.getenv("ACTIVE_UNIVERSE_MODE", "csv_snapshot").strip()

    # Rate-limit guard for the network modes: skip the fetch if we refreshed recently.
    if mode in ("authorized_url", "finviz_screener") and universe_is_fresh():
        print(f"Active universe refreshed within UNIVERSE_MIN_REFRESH_SEC "
              f"({os.getenv('UNIVERSE_MIN_REFRESH_SEC', '240')}s); skipping Finviz fetch this cycle.")
        return

    try:
        if mode == "authorized_url":
            raw_df = load_from_authorized_url()
            source = "authorized_url"
        elif mode == "finviz_screener":
            raw_df = load_from_finviz_screener()
            source = "finviz_screener"
        elif mode == "csv_snapshot":
            raw_df = load_from_csv_snapshot()
            source = "csv_snapshot"
        else:
            raise ValueError(f"Unknown ACTIVE_UNIVERSE_MODE: {mode}")

        print(f"Raw screener rows: {len(raw_df)}")

        active_df = build_active_universe(raw_df)

        print(f"Positive-change active tickers: {len(active_df)}")

        added_count, removed_count = save_active_universe(active_df)

        print(f"Added tickers since previous run: {added_count}")
        print(f"Removed tickers since previous run: {removed_count}")

        log_run(
            mode=mode,
            source=source,
            active_count=len(active_df),
            added_count=added_count,
            removed_count=removed_count,
            status="success",
            notes="active universe refreshed",
        )

        print("Saved active_ticker_universe.")
        print("Done.")

    except Exception as e:
        log_run(
            mode=mode,
            source="unknown",
            active_count=0,
            added_count=0,
            removed_count=0,
            status="failed",
            notes=str(e),
        )
        # Don't hard-stop the whole pipeline on a universe-refresh failure (e.g. an
        # expired Finviz cookie or a network blip). save_active_universe only clears
        # the table AFTER a successful load, so the previous universe is intact —
        # keep it so news/alerts/tracking on existing tickers still run this cycle.
        conn = connect_db()
        n_prev = conn.execute("SELECT COUNT(*) FROM active_ticker_universe;").fetchone()[0]
        conn.close()
        print(f"[Warning] active-universe refresh failed: {type(e).__name__}: {e}")
        if n_prev:
            print(f"[Warning] keeping the last-known universe ({n_prev} tickers) and continuing. "
                  f"If this persists, refresh FINVIZ_COOKIE in .env.")
        else:
            print("[Warning] no previous universe to fall back to — this cycle has no tickers. "
                  "Refresh FINVIZ_COOKIE and re-run.")
        return  # exit 0 so run_pipeline continues to the remaining stages


if __name__ == "__main__":
    main()
