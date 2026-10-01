from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")


def init_combined_ranking_table() -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS combined_ticker_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            calculated_at TEXT,
            combined_score REAL,
            news_sentiment_score REAL,
            news_sentiment_label TEXT,
            density_score REAL,
            density_level TEXT,
            source_credibility REAL,
            latest_headline TEXT,
            latest_source TEXT,
            latest_news_time TEXT,
            alert_candidate INTEGER DEFAULT 0,
            explanation TEXT
        );
        """
    )

    conn.commit()
    conn.close()


def load_latest_news_sentiment() -> pd.DataFrame:
    conn = connect_db()

    query = """
        SELECT
            ns.ticker,
            ns.sentiment_label,
            ns.sentiment_score,
            ns.confidence_score,
            cn.title_clean,
            cn.source,
            cn.publish_time
        FROM news_sentiment ns
        JOIN clean_news cn
            ON ns.clean_news_id = cn.id
        JOIN (
            SELECT
                ns2.ticker,
                MAX(cn2.publish_time) AS latest_time
            FROM news_sentiment ns2
            JOIN clean_news cn2
                ON ns2.clean_news_id = cn2.id
            GROUP BY ns2.ticker
        ) latest
            ON ns.ticker = latest.ticker
            AND cn.publish_time = latest.latest_time;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def load_latest_density() -> pd.DataFrame:
    conn = connect_db()

    query = """
        SELECT tds.*
        FROM ticker_density_snapshots tds
        JOIN (
            SELECT ticker, MAX(calculated_at) AS latest_time
            FROM ticker_density_snapshots
            GROUP BY ticker
        ) latest
            ON tds.ticker = latest.ticker
            AND tds.calculated_at = latest.latest_time;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def load_source_credibility() -> dict[str, float]:
    conn = connect_db()

    query = """
        SELECT source_name, credibility_score
        FROM source_registry;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return dict(zip(df["source_name"], df["credibility_score"]))


def calculate_combined_score(
    sentiment_score: float,
    density_score: float,
    source_credibility: float,
) -> float:
    """
    First combined formula:
    positive news + discussion density + source credibility.
    """
    positive_news_score = max(sentiment_score, 0.0)

    score = (
        0.45 * positive_news_score
        + 0.40 * density_score
        + 0.15 * source_credibility
    )

    return round(float(score), 4)


def save_combined_ranking(row: dict) -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO combined_ticker_rankings (
            ticker,
            calculated_at,
            combined_score,
            news_sentiment_score,
            news_sentiment_label,
            density_score,
            density_level,
            source_credibility,
            latest_headline,
            latest_source,
            latest_news_time,
            alert_candidate,
            explanation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row["ticker"],
            row["calculated_at"],
            row["combined_score"],
            row["news_sentiment_score"],
            row["news_sentiment_label"],
            row["density_score"],
            row["density_level"],
            row["source_credibility"],
            row["latest_headline"],
            row["latest_source"],
            row["latest_news_time"],
            row["alert_candidate"],
            row["explanation"],
        ),
    )

    conn.commit()
    conn.close()


def run_combined_ranking() -> None:
    print("Initializing combined ranking table...")
    init_combined_ranking_table()

    print("Loading latest news sentiment...")
    news_df = load_latest_news_sentiment()

    print("Loading latest density...")
    density_df = load_latest_density()

    if news_df.empty:
        print("No news sentiment records found.")
        return

    source_scores = load_source_credibility()

    # Merge on ticker. If no density exists, fill with 0.
    merged = news_df.merge(
        density_df[["ticker", "density_score", "density_level"]],
        on="ticker",
        how="left",
    )

    merged["density_score"] = merged["density_score"].fillna(0.0)
    merged["density_level"] = merged["density_level"].fillna("none")

    calculated_at = ny_now_iso()
    saved = 0

    for _, row in merged.iterrows():
        ticker = row["ticker"]
        sentiment_score = float(row["sentiment_score"])
        density_score = float(row["density_score"])
        source = row["source"]
        credibility = float(source_scores.get(source, 0.50))

        combined_score = calculate_combined_score(
            sentiment_score=sentiment_score,
            density_score=density_score,
            source_credibility=credibility,
        )

        alert_candidate = 1 if (
            sentiment_score >= 0.50
            and density_score >= 0.70
        ) else 0

        if alert_candidate:
            explanation = (
                f"{ticker} has positive news sentiment "
                f"({sentiment_score:.2f}) and elevated discussion density "
                f"({density_score:.2f})."
            )
        else:
            explanation = (
                f"{ticker} combined score based on news sentiment, "
                f"discussion density, and source credibility."
            )

        save_combined_ranking(
            {
                "ticker": ticker,
                "calculated_at": calculated_at,
                "combined_score": combined_score,
                "news_sentiment_score": sentiment_score,
                "news_sentiment_label": row["sentiment_label"],
                "density_score": density_score,
                "density_level": row["density_level"],
                "source_credibility": credibility,
                "latest_headline": row["title_clean"],
                "latest_source": source,
                "latest_news_time": row["publish_time"],
                "alert_candidate": alert_candidate,
                "explanation": explanation,
            }
        )

        saved += 1

    print(f"Saved combined rankings: {saved}")

    print("\nTop combined rankings:")
    print(
        merged[["ticker", "sentiment_label", "sentiment_score", "density_score", "density_level"]]
        .sort_values(by=["density_score", "sentiment_score"], ascending=False)
        .head(20)
    )

    print("Done.")


if __name__ == "__main__":
    run_combined_ranking()