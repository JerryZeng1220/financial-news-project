"""
Shared paths, DB connection, and time helpers used across the app.

These were previously copy-pasted into ~18 modules (identical `connect_db`,
`ny_now_iso`, and the BASE_DIR / DB_PATH / NY_TZ constants). Importing them from
here keeps the DB path, WAL mode, and timezone consistent in one place.

    from common import connect_db, ny_now_iso, DB_PATH, NY_TZ, BASE_DIR
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")


def connect_db() -> sqlite3.Connection:
    """Open the app DB in WAL mode (shared by every module)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def ny_now_iso() -> str:
    """Current time in America/New_York as an ISO-8601 string."""
    return datetime.now(NY_TZ).isoformat()
