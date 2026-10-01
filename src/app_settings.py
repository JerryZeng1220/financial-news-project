"""
Tiny shared settings store (a key/value table in the DB) so the dashboard can
change runtime knobs that the background worker reads.

The dashboard writes a setting (e.g. the density rolling-window length); the
worker's next pipeline cycle reads it. Precedence for a value is:
    DB setting  ->  environment variable  ->  code default.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT);")
    return conn


def get(key: str, default: str | None = None) -> str | None:
    conn = _conn()
    try:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?;", (key,)).fetchone()
        return row[0] if row and row[0] is not None else default
    finally:
        conn.close()


def get_int(key: str, default: int | None = None) -> int | None:
    v = get(key)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def set(key: str, value) -> None:
    conn = _conn()
    try:
        conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?);", (key, str(value)))
        conn.commit()
    finally:
        conn.close()
