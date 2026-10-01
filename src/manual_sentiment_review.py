from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")


def init_review_table() -> None:
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_manual_review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_news_id INTEGER,
            ticker TEXT,
            title TEXT,
            system_sentiment_label TEXT,
            system_sentiment_score REAL,
            human_sentiment_label TEXT,
            is_correct INTEGER,
            reviewer_notes TEXT,
            reviewed_at TEXT,
            UNIQUE(clean_news_id, ticker),
            FOREIGN KEY (clean_news_id) REFERENCES clean_news(id),
            FOREIGN KEY (ticker) REFERENCES ticker_reference(ticker)
        );
        """
    )

    conn.commit()
    conn.close()


def load_sentiment_for_review(limit: int = 100) -> pd.DataFrame:
    """
    Load sentiment-scored ticker-news records.
    Left join manual review so previous reviews are visible.
    """
    conn = connect_db()

    query = """
        SELECT
            ns.clean_news_id,
            ns.ticker,
            cn.title_clean AS title,
            ns.sentiment_label AS system_sentiment_label,
            ns.sentiment_score AS system_sentiment_score,
            ns.confidence_score AS system_confidence,
            ns.reason AS system_reason,
            smr.human_sentiment_label,
            smr.is_correct,
            smr.reviewer_notes
        FROM news_sentiment ns
        JOIN clean_news cn
            ON ns.clean_news_id = cn.id
        LEFT JOIN sentiment_manual_review smr
            ON ns.clean_news_id = smr.clean_news_id
            AND ns.ticker = smr.ticker
        ORDER BY ns.created_at DESC
        LIMIT ?;
    """

    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()

    if df.empty:
        return df

    # Convert is_correct from SQLite integer to readable labels
    def correct_to_label(value):
        if pd.isna(value):
            return "Unreviewed"
        if int(value) == 1:
            return "Correct"
        return "Incorrect"

    df["review_status"] = df["is_correct"].apply(correct_to_label)

    # Fill editable fields
    df["human_sentiment_label"] = df["human_sentiment_label"].fillna("Unreviewed")
    df["reviewer_notes"] = df["reviewer_notes"].fillna("")

    return df


def save_review_row(row: pd.Series) -> None:
    conn = connect_db()
    cur = conn.cursor()

    review_status = row.get("review_status", "Unreviewed")
    human_label = row.get("human_sentiment_label", "Unreviewed")

    if review_status == "Correct":
        is_correct = 1
    elif review_status == "Incorrect":
        is_correct = 0
    else:
        is_correct = None

    # Do not save fully unreviewed rows
    if review_status == "Unreviewed" and human_label == "Unreviewed":
        conn.close()
        return

    cur.execute(
        """
        INSERT OR REPLACE INTO sentiment_manual_review (
            clean_news_id,
            ticker,
            title,
            system_sentiment_label,
            system_sentiment_score,
            human_sentiment_label,
            is_correct,
            reviewer_notes,
            reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            int(row["clean_news_id"]),
            str(row["ticker"]),
            str(row["title"]),
            str(row["system_sentiment_label"]),
            float(row["system_sentiment_score"]),
            str(human_label),
            is_correct,
            str(row.get("reviewer_notes", "")),
            ny_now_iso(),
        ),
    )

    conn.commit()
    conn.close()


def save_reviews(edited_df: pd.DataFrame) -> int:
    saved_count = 0

    for _, row in edited_df.iterrows():
        before = row.get("review_status", "Unreviewed")
        human = row.get("human_sentiment_label", "Unreviewed")

        if before != "Unreviewed" or human != "Unreviewed":
            save_review_row(row)
            saved_count += 1

    return saved_count


def load_review_summary() -> pd.DataFrame:
    conn = connect_db()

    query = """
        SELECT
            COUNT(*) AS reviewed_count,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct_count,
            SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) AS incorrect_count
        FROM sentiment_manual_review;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def main() -> None:
    st.set_page_config(
        page_title="Manual Sentiment Review",
        layout="wide"
    )

    init_review_table()

    st.title("Manual Sentiment Review")
    st.caption("Review system-generated sentiment labels for ticker-linked news.")

    with st.sidebar:
        st.header("Settings")
        limit = st.slider("Records to load", min_value=10, max_value=500, value=100, step=10)

        st.markdown("### Label guide")
        st.markdown(
            """
            - **Positive**: likely favorable for the company  
            - **Negative**: likely unfavorable for the company  
            - **Neutral**: no clear financial direction  
            - **Mixed**: both positive and negative signals  
            """
        )

    df = load_sentiment_for_review(limit=limit)

    if df.empty:
        st.warning("No sentiment records found. Run the pipeline (`run_pipeline.py`) first.")
        return

    summary = load_review_summary()
    st.subheader("Review Summary")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Review Table")

    editable_cols = [
        "clean_news_id",
        "ticker",
        "title",
        "system_sentiment_label",
        "system_sentiment_score",
        "system_confidence",
        "system_reason",
        "human_sentiment_label",
        "review_status",
        "reviewer_notes",
    ]

    review_df = df[editable_cols].copy()

    edited_df = st.data_editor(
        review_df,
        width="stretch",
        hide_index=True,
        column_config={
            "clean_news_id": st.column_config.NumberColumn(
                "News ID",
                disabled=True,
            ),
            "ticker": st.column_config.TextColumn(
                "Ticker",
                disabled=True,
            ),
            "title": st.column_config.TextColumn(
                "Title",
                disabled=True,
                width="large",
            ),
            "system_sentiment_label": st.column_config.TextColumn(
                "System Sentiment",
                disabled=True,
            ),
            "system_sentiment_score": st.column_config.NumberColumn(
                "System Score",
                disabled=True,
                format="%.2f",
            ),
            "system_confidence": st.column_config.NumberColumn(
                "Confidence",
                disabled=True,
                format="%.2f",
            ),
            "system_reason": st.column_config.TextColumn(
                "System Reason",
                disabled=True,
                width="large",
            ),
            "human_sentiment_label": st.column_config.SelectboxColumn(
                "Your Sentiment",
                options=["Unreviewed", "positive", "negative", "neutral", "mixed"],
                required=True,
            ),
            "review_status": st.column_config.SelectboxColumn(
                "System Correct?",
                options=["Unreviewed", "Correct", "Incorrect"],
                required=True,
            ),
            "reviewer_notes": st.column_config.TextColumn(
                "Notes",
                width="medium",
            ),
        },
        disabled=[
            "clean_news_id",
            "ticker",
            "title",
            "system_sentiment_label",
            "system_sentiment_score",
            "system_confidence",
            "system_reason",
        ],
    )

    if st.button("Save Reviews", type="primary"):
        saved_count = save_reviews(edited_df)
        st.success(f"Saved {saved_count} reviewed rows.")
        st.rerun()

    st.markdown("---")
    st.subheader("How to use this")
    st.markdown(
        """
        1. Read the title and ticker.
        2. Compare the system sentiment with your own judgment.
        3. Fill **Your Sentiment**.
        4. Mark whether the system was **Correct** or **Incorrect**.
        5. Add notes if needed.
        6. Click **Save Reviews**.
        """
    )


if __name__ == "__main__":
    main()