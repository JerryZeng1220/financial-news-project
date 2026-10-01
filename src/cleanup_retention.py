"""
Retention / cleanup step — keeps the DB lean without breaking dedup or losing
the alert history we want for evaluation.

What it prunes (by age, all configurable):
  - social_mentions          fast-growing, only used in 2-6h windows  -> 48h
  - ticker_density_snapshots accumulate per run, only latest needed    -> 7d
  - combined_ticker_rankings same; dashboard reads the latest batch    -> 7d
  - ticker_crowd_mood        stale moods                               -> 7d
  - active_ticker_universe_history  short history for momentum (Ph.2)  -> 30d
  - *_log / active_universe_runs    run logs                          -> 30d

What it deliberately KEEPS:
  - raw_news / clean_news / news_ticker_links / news_sentiment / *_status:
      these hold the DEDUP fingerprints + "already processed/scored" state.
      Deleting them would make old RSS items look new and get RE-SENT to Gemini.
      News volume is low, so we leave them intact (pruning them safely needs a
      persistent seen-hash store first).
  - notified_alerts: kept long-term (1y) — it's your record of what fired, the
      memory you need to later evaluate whether the picks were good.

Config (.env):
    RET_SOCIAL_HOURS=48  RET_SNAPSHOT_DAYS=7  RET_RANKING_DAYS=7
    RET_MOOD_DAYS=7  RET_UNIVERSE_HISTORY_DAYS=30  RET_LOG_DAYS=30
    RET_NOTIFIED_DAYS=365  CLEANUP_VACUUM=false
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

load_dotenv(BASE_DIR / ".env")


def _hours(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def ny_now() -> datetime:
    return datetime.now(NY_TZ)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _prune(conn: sqlite3.Connection, table: str, time_col: str, cutoff_iso: str) -> int:
    if not table_exists(conn, table):
        return 0
    cur = conn.execute(
        f"DELETE FROM {table} WHERE {time_col} IS NOT NULL AND {time_col} < ?",
        (cutoff_iso,),
    )
    return cur.rowcount


def run_cleanup() -> None:
    now = ny_now()

    # (table, time column, cutoff datetime)
    plan = [
        ("social_mentions", "post_time", now - timedelta(hours=_hours("RET_SOCIAL_HOURS", 48))),
        ("ticker_density_snapshots", "calculated_at", now - timedelta(days=_hours("RET_SNAPSHOT_DAYS", 7))),
        ("combined_ticker_rankings", "calculated_at", now - timedelta(days=_hours("RET_RANKING_DAYS", 7))),
        ("ticker_crowd_mood", "calculated_at", now - timedelta(days=_hours("RET_MOOD_DAYS", 7))),
        ("active_ticker_universe_history", "collected_at", now - timedelta(days=_hours("RET_UNIVERSE_HISTORY_DAYS", 30))),
        ("active_universe_runs", "run_time", now - timedelta(days=_hours("RET_LOG_DAYS", 30))),
        ("ingestion_log", "run_time", now - timedelta(days=_hours("RET_LOG_DAYS", 30))),
        ("extraction_log", "run_time", now - timedelta(days=_hours("RET_LOG_DAYS", 30))),
        ("sentiment_log", "run_time", now - timedelta(days=_hours("RET_LOG_DAYS", 30))),
        # Kept long-term as the evaluation memory:
        ("notified_alerts", "notified_at", now - timedelta(days=_hours("RET_NOTIFIED_DAYS", 365))),
    ]

    conn = connect_db()
    total = 0
    try:
        for table, col, cutoff in plan:
            n = _prune(conn, table, col, cutoff.isoformat())
            if n:
                print(f"  pruned {n:>5} rows from {table} (older than {cutoff.date()})")
            total += n
        conn.commit()

        if os.getenv("CLEANUP_VACUUM", "false").lower() == "true" and total:
            print("  VACUUM (reclaiming disk)...")
            conn.execute("VACUUM;")
    finally:
        conn.close()

    print(f"Cleanup done. Pruned {total} rows. (news/sentiment kept for dedup; alerts kept for evaluation)")


if __name__ == "__main__":
    run_cleanup()
