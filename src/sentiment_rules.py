from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


# ----------------------------
# Project paths and constants
# ----------------------------

from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

# Rows scored by the rule-based fallback while the LLM was down/over-quota are
# flagged pending_llm_rescore=1 and re-scored by the LLM on a later cycle. Only
# recent news is worth re-scoring (older can't alert anyway).
RESCORE_WINDOW_HOURS = int(os.getenv("SENTIMENT_RESCORE_HOURS", "72"))


# ----------------------------
# Database setup
# ----------------------------

def init_sentiment_tables() -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_news_id INTEGER,
            ticker TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            confidence_score REAL,
            materiality TEXT,
            method TEXT,
            reason TEXT,
            model_version TEXT,
            pending_llm_rescore INTEGER DEFAULT 0,
            created_at TEXT,
            UNIQUE(clean_news_id, ticker),
            FOREIGN KEY (clean_news_id) REFERENCES clean_news(id),
            FOREIGN KEY (ticker) REFERENCES ticker_reference(ticker)
        );
        """
    )
    # Migration: add materiality to a pre-existing news_sentiment table.
    existing = [r[1] for r in cur.execute("PRAGMA table_info(news_sentiment);").fetchall()]
    if "materiality" not in existing:
        cur.execute("ALTER TABLE news_sentiment ADD COLUMN materiality TEXT;")

    # Migration: add the LLM-rescore flag, and one-time backfill — flag recent
    # rule-based rows (they likely fell back while the LLM was over quota) so the
    # LLM re-scores them and recovers any stuck-neutral catalysts.
    if "pending_llm_rescore" not in existing:
        cur.execute("ALTER TABLE news_sentiment ADD COLUMN pending_llm_rescore INTEGER DEFAULT 0;")
        cutoff = (datetime.now(NY_TZ) - timedelta(hours=RESCORE_WINDOW_HOURS)).isoformat()
        cur.execute(
            """
            UPDATE news_sentiment
            SET pending_llm_rescore = 1
            WHERE method = 'rule_based'
              AND clean_news_id IN (
                  SELECT id FROM clean_news WHERE publish_time >= ?
              );
            """,
            (cutoff,),
        )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TEXT,
            records_read INTEGER,
            records_scored INTEGER,
            status TEXT,
            notes TEXT
        );
        """
    )

    conn.commit()
    conn.close()


# ----------------------------
# Load ticker-linked news
# ----------------------------

def load_unscored_ticker_news() -> pd.DataFrame:
    """
    Load ticker-linked news records that do not yet have sentiment scores.
    """
    conn = connect_db()

    query = """
        SELECT
            ntl.clean_news_id,
            ntl.ticker,
            ntl.company,
            ntl.confidence_score AS ticker_confidence,
            ntl.match_method,
            cn.source,
            cn.title_clean,
            cn.summary_clean,
            cn.text_clean,
            cn.publish_time,
            cn.url
        FROM news_ticker_links ntl
        JOIN clean_news cn
            ON ntl.clean_news_id = cn.id
        LEFT JOIN news_sentiment ns
            ON ntl.clean_news_id = ns.clean_news_id
            AND ntl.ticker = ns.ticker
        WHERE ns.id IS NULL
        ORDER BY cn.publish_time DESC;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# ----------------------------
# Sentiment rules
# ----------------------------

POSITIVE_PATTERNS = [
    # FDA / biotech
    (r"\bfda approves\b", 0.90, "FDA approval is generally financially positive."),
    (r"\bfda approval\b", 0.85, "FDA approval is generally financially positive."),
    (r"\bclearance from the fda\b", 0.75, "FDA clearance can be positive for a company."),
    (r"\breceives fda clearance\b", 0.75, "FDA clearance can be positive for a company."),
    (r"\bphase 3.*positive\b", 0.80, "Positive Phase 3 trial result can be favorable."),
    (r"\bpositive topline results\b", 0.80, "Positive clinical results can be favorable."),

    # Earnings / guidance
    (r"\bbeats expectations\b", 0.75, "Beating expectations is generally positive."),
    (r"\bbeats estimates\b", 0.75, "Beating estimates is generally positive."),
    (r"\brecord revenue\b", 0.65, "Record revenue is generally positive."),
    (r"\brevenue growth\b", 0.55, "Revenue growth is generally positive."),
    (r"\braises guidance\b", 0.80, "Raising guidance is generally positive."),
    (r"\braises outlook\b", 0.75, "Raising outlook is generally positive."),
    (r"\bincreases guidance\b", 0.75, "Increasing guidance is generally positive."),

    # Capital return / shareholder friendly
    (r"\bshare repurchase\b", 0.65, "Share repurchase can be positive for shareholders."),
    (r"\bstock buyback\b", 0.65, "Stock buyback can be positive for shareholders."),
    (r"\bbuyback\b", 0.60, "Buyback can be positive for shareholders."),
    (r"\bdividend increase\b", 0.60, "Dividend increase is generally positive."),
    (r"\bincreases dividend\b", 0.60, "Dividend increase is generally positive."),

    # Analyst / deal
    (r"\bupgrade[sd]?\b", 0.55, "Analyst upgrade is generally positive."),
    (r"\bprice target raised\b", 0.55, "Raised price target is generally positive."),
    (r"\bacquisition agreement\b", 0.60, "Acquisition-related news can be positive depending on context."),
    (r"\bstrategic partnership\b", 0.45, "Strategic partnership can be mildly positive."),
    (r"\bmajor contract\b", 0.60, "Major contract can be positive."),

    # Additional Rules
    (r"\bgeneral availability\b", 0.55, "Product availability or launch announcements are generally positive business-development news."),
    (r"\bproduct launch\b", 0.55, "Product availability or launch announcements are generally positive business-development news."),

]


NEGATIVE_PATTERNS = [
    # FDA / biotech
    (r"\bfda rejects\b", -0.90, "FDA rejection is generally negative."),
    (r"\bcomplete response letter\b", -0.85, "FDA complete response letter is generally negative."),
    (r"\bclinical hold\b", -0.85, "Clinical hold is generally negative."),
    (r"\btrial failed\b", -0.90, "Failed trial is generally negative."),
    (r"\bmissed primary endpoint\b", -0.85, "Missing primary endpoint is generally negative."),
    (r"\bnegative topline results\b", -0.80, "Negative trial results are generally negative."),

    # Earnings / guidance
    (r"\bmisses expectations\b", -0.75, "Missing expectations is generally negative."),
    (r"\bmisses estimates\b", -0.75, "Missing estimates is generally negative."),
    (r"\bcuts guidance\b", -0.85, "Cutting guidance is generally negative."),
    (r"\blowers guidance\b", -0.85, "Lowering guidance is generally negative."),
    (r"\breduces outlook\b", -0.75, "Reducing outlook is generally negative."),
    (r"\blower[s]? outlook\b", -0.75, "Lower outlook is generally negative."),

    # Financing / dilution
    (r"\bpublic offering\b", -0.70, "Public offering may dilute existing shareholders."),
    (r"\bregistered direct offering\b", -0.75, "Registered direct offering may dilute existing shareholders."),
    (r"\bprivate placement\b", -0.60, "Private placement can be dilutive."),
    (r"\bshare offering\b", -0.70, "Share offering may dilute existing shareholders."),
    (r"\bat-the-market offering\b", -0.65, "ATM offering may dilute existing shareholders."),
    (r"\bconvertible notes\b", -0.45, "Convertible notes can be dilutive depending on terms."),

    # Legal / risk
    (r"\blawsuit\b", -0.65, "Lawsuit may create legal or financial risk."),
    (r"\bclass action\b", -0.70, "Class action may create legal or financial risk."),
    (r"\binvestigation\b", -0.60, "Investigation may create legal or regulatory risk."),
    (r"\bsec investigation\b", -0.75, "SEC investigation is generally negative."),
    (r"\bdoj investigation\b", -0.75, "DOJ investigation is generally negative."),
    (r"\bsubpoena\b", -0.65, "Subpoena may indicate legal or regulatory risk."),
    (r"\bsettlement\b", -0.35, "Settlement may indicate legal or financial cost."),

    # Distress
    (r"\bbankruptcy\b", -0.95, "Bankruptcy is strongly negative."),
    (r"\bgoing concern\b", -0.90, "Going concern warning is strongly negative."),
    (r"\bdelisting\b", -0.85, "Delisting risk is generally negative."),
    (r"\bresignation\b", -0.35, "Unexpected resignation can be negative depending on context."),
    (r"\blayoffs\b", -0.45, "Layoffs can signal business weakness, though context matters."),

    # Analyst
    (r"\bdowngrade[sd]?\b", -0.55, "Analyst downgrade is generally negative."),
    (r"\bprice target lowered\b", -0.55, "Lowered price target is generally negative."),
]


NEUTRAL_PATTERNS = [
    (r"\bto present at\b", "Conference or presentation announcements are usually neutral."),
    (r"\binvestor conference\b", "Investor conference announcements are usually neutral."),
    (r"\bearnings release date\b", "Earnings date announcement is usually neutral."),
    (r"\bto report .* results on\b", "Scheduling earnings results is usually neutral."),
    (r"\bwebcast\b", "Webcast announcement is usually neutral."),
    (r"\bappoints\b", "Appointment news is often neutral without more context."),
    (r"\bnames\b", "Personnel naming announcement is often neutral without more context."),
    (r"\bto present at\b", "Conference or presentation announcements are usually neutral.")
]

# Special Cases
MAJOR_RETAILERS = [
    "walmart", "target", "costco", "walgreens", "cvs",
    "kroger", "amazon", "best buy", "home depot", "lowe's"
]

RETAIL_EXPANSION_PATTERNS = [
    r"\bnow available at\b",
    r"\bavailable at\b",
    r"\bavailable in\b",
    r"\blaunches at\b",
    r"\brolls out to\b",
    r"\bstores nationwide\b",
]

def is_retail_expansion(title_lower):
    return (
        any(re.search(pattern, title_lower) for pattern in RETAIL_EXPANSION_PATTERNS)
        and any(retailer in title_lower for retailer in MAJOR_RETAILERS)
    )

# End of special case


def normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_sentiment_rule_based(title: str, text: str) -> dict:
    """
    Return:
        sentiment_label: positive/negative/neutral/mixed
        sentiment_score: -1 to +1
        confidence_score: 0 to 1
        reason: short explanation
    """
    combined_text = normalize_for_matching(f"{title or ''}. {text or ''}")

    positive_hits = []
    negative_hits = []
    neutral_hits = []

    title_lower = str(title).lower()
    if is_retail_expansion(title_lower):
        return {
            "sentiment_label": "positive",
            "sentiment_score": 0.55,
            "confidence_score": 0.75,
            "reason": "Product availability through a major retailer suggests expanded distribution and potential revenue growth.",
        }


    for pattern, score, reason in POSITIVE_PATTERNS:
        if re.search(pattern, combined_text):
            positive_hits.append((pattern, score, reason))

    for pattern, score, reason in NEGATIVE_PATTERNS:
        if re.search(pattern, combined_text):
            negative_hits.append((pattern, score, reason))

    for pattern, reason in NEUTRAL_PATTERNS:
        if re.search(pattern, combined_text):
            neutral_hits.append((pattern, reason))

    # If both positive and negative signals exist, mark mixed.
    if positive_hits and negative_hits:
        strongest_positive = max(positive_hits, key=lambda x: abs(x[1]))
        strongest_negative = max(negative_hits, key=lambda x: abs(x[1]))

        # Mixed score leans toward stronger side but stays moderate.
        raw_score = strongest_positive[1] + strongest_negative[1]
        score = max(min(raw_score / 2, 0.35), -0.35)

        reason = (
            f"Mixed signals: {strongest_positive[2]} "
            f"However, {strongest_negative[2]}"
        )

        return {
            "sentiment_label": "mixed",
            "sentiment_score": score,
            "confidence_score": 0.75,
            "reason": reason,
        }

    if positive_hits:
        strongest = max(positive_hits, key=lambda x: abs(x[1]))
        return {
            "sentiment_label": "positive",
            "sentiment_score": strongest[1],
            "confidence_score": 0.85,
            "reason": strongest[2],
        }

    if negative_hits:
        strongest = max(negative_hits, key=lambda x: abs(x[1]))
        return {
            "sentiment_label": "negative",
            "sentiment_score": strongest[1],
            "confidence_score": 0.85,
            "reason": strongest[2],
        }

    if neutral_hits:
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "confidence_score": 0.75,
            "reason": neutral_hits[0][1],
        }

    # Default fallback
    return {
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "confidence_score": 0.50,
        "reason": "No strong financial sentiment pattern matched.",
    }


# ----------------------------
# Insert sentiment results
# ----------------------------

def insert_sentiment_result(
    clean_news_id: int,
    ticker: str,
    sentiment_label: str,
    sentiment_score: float,
    confidence_score: float,
    method: str,
    reason: str,
    model_version: str = "rule_based_v1",
    materiality: str | None = None,
    pending_llm_rescore: int = 0,
) -> bool:
    conn = connect_db()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO news_sentiment (
                clean_news_id,
                ticker,
                sentiment_label,
                sentiment_score,
                confidence_score,
                materiality,
                method,
                reason,
                model_version,
                pending_llm_rescore,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                clean_news_id,
                ticker,
                sentiment_label,
                sentiment_score,
                confidence_score,
                materiality,
                method,
                reason,
                model_version,
                pending_llm_rescore,
                ny_now_iso(),
            ),
        )

        inserted = cur.rowcount > 0
        conn.commit()

    finally:
        conn.close()

    return inserted


def load_pending_llm_rescore(window_hours: int = RESCORE_WINDOW_HOURS) -> pd.DataFrame:
    """Recent rows flagged for an LLM re-score (they fell back to rule-based while
    the LLM was unavailable). Limited to recent news so we don't re-score stale
    items that can no longer alert."""
    conn = connect_db()
    cutoff = (datetime.now(NY_TZ) - timedelta(hours=window_hours)).isoformat()
    query = """
        SELECT ns.clean_news_id, ns.ticker, cn.title_clean, cn.text_clean
        FROM news_sentiment ns
        JOIN clean_news cn ON ns.clean_news_id = cn.id
        WHERE ns.pending_llm_rescore = 1
          AND cn.publish_time >= ?
        ORDER BY cn.publish_time DESC;
    """
    df = pd.read_sql_query(query, conn, params=(cutoff,))
    conn.close()
    return df


def update_sentiment_llm(
    clean_news_id: int,
    ticker: str,
    sentiment_label: str,
    sentiment_score: float,
    confidence_score: float,
    reason: str,
    materiality: str | None,
    model_version: str,
) -> None:
    """Overwrite a fallback row with a fresh LLM result and clear the flag."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE news_sentiment
        SET sentiment_label = ?, sentiment_score = ?, confidence_score = ?,
            reason = ?, materiality = ?, method = 'llm', model_version = ?,
            pending_llm_rescore = 0, created_at = ?
        WHERE clean_news_id = ? AND ticker = ?;
        """,
        (sentiment_label, sentiment_score, confidence_score, reason, materiality,
         model_version, ny_now_iso(), clean_news_id, ticker),
    )
    conn.commit()
    conn.close()


def log_sentiment_run(
    records_read: int,
    records_scored: int,
    status: str,
    notes: str,
) -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO sentiment_log (
            run_time,
            records_read,
            records_scored,
            status,
            notes
        )
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            ny_now_iso(),
            records_read,
            records_scored,
            status,
            notes,
        ),
    )

    conn.commit()
    conn.close()


# ----------------------------
# Main pipeline
# ----------------------------

def run_sentiment_scoring() -> None:
    print("Initializing sentiment tables...")
    init_sentiment_tables()

    print("Loading unscored ticker-linked news...")
    df = load_unscored_ticker_news()

    print(f"Ticker-linked news records to score: {len(df)}")

    records_scored = 0

    for _, row in df.iterrows():
        clean_news_id = int(row["clean_news_id"])
        ticker = str(row["ticker"]).strip().upper()

        title = row.get("title_clean", "")
        text = row.get("text_clean", "")

        result = classify_sentiment_rule_based(title=title, text=text)

        inserted = insert_sentiment_result(
            clean_news_id=clean_news_id,
            ticker=ticker,
            sentiment_label=result["sentiment_label"],
            sentiment_score=result["sentiment_score"],
            confidence_score=result["confidence_score"],
            method="rule_based",
            reason=result["reason"],
            model_version="rule_based_v1",
        )

        if inserted:
            records_scored += 1

    log_sentiment_run(
        records_read=len(df),
        records_scored=records_scored,
        status="success",
        notes="rule_based_v1 sentiment scoring",
    )

    print("\nSentiment scoring summary")
    print("-------------------------")
    print(f"Records read: {len(df)}")
    print(f"Records scored: {records_scored}")
    print("Done.")


if __name__ == "__main__":
    run_sentiment_scoring()