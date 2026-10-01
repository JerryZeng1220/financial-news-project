"""
Discussion-density from social mentions — rolling window, scored RELATIVE to the
ticker's own baseline (like relative volume, but for chatter).

Each cycle (~every minute) we count a ticker's social mentions in the last
DENSITY_WINDOW_MIN minutes (the rolling window, tunable 10 min .. 16 h), and
compare that rate to its baseline rate over the preceding DENSITY_BASELINE_MIN
minutes. So a "high" density means chatter is running well above normal for THAT
ticker — not just that it has a lot of posts.

Why relative: a fixed count threshold can't work across window lengths (a 16h
window accumulates far more than a 10min one) and can't tell "busy stock" from
"suddenly busy stock". Relative-to-baseline fixes both.

Config (.env):
    DENSITY_WINDOW_MIN     rolling window minutes, clamped 10..960 (default 30)
    DENSITY_BASELINE_MIN   baseline period minutes (default 480 = 8h)
    DENSITY_MIN_MENTIONS   min posts in the window to score above 'low' (default 3)
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")


import app_settings


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _resolve(setting_key: str, env_key: str, default: int, lo: int, hi: int) -> int:
    """DB setting (dashboard) overrides env var, which overrides the default."""
    v = app_settings.get_int(setting_key, None)
    if v is None:
        v = int(os.getenv(env_key, str(default)))
    return _clamp(v, lo, hi)


WINDOW_MIN = _resolve("density_window_min", "DENSITY_WINDOW_MIN", 30, 10, 960)
BASELINE_MIN = _resolve("density_baseline_min", "DENSITY_BASELINE_MIN", 480, WINDOW_MIN + 10, 5760)
MIN_MENTIONS = int(os.getenv("DENSITY_MIN_MENTIONS", "3"))


def ny_now() -> datetime:
    return datetime.now(NY_TZ)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def init_density_columns(conn: sqlite3.Connection) -> None:
    """Add the rolling-window columns to the existing snapshots table if missing."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ticker_density_snapshots);").fetchall()]
    for name, decl in (("window_min", "INTEGER"), ("window_count", "INTEGER"),
                       ("relative_density", "REAL")):
        if name not in cols:
            conn.execute(f"ALTER TABLE ticker_density_snapshots ADD COLUMN {name} {decl};")
    conn.commit()


def load_recent_social_mentions(minutes: int) -> pd.DataFrame:
    conn = connect_db()
    cutoff = (ny_now() - timedelta(minutes=minutes)).isoformat()
    df = pd.read_sql_query(
        """
        SELECT ticker, post_time FROM social_mentions
        WHERE post_time >= ? AND ticker IS NOT NULL AND ticker != '';
        """,
        conn, params=(cutoff,),
    )
    conn.close()
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def density_from_window(cur_count: int, cur_rate: float, base_rate: float | None) -> tuple[float, str, float | None]:
    """
    Score the current window relative to baseline.
    Returns (density_score 0..1, level, relative-multiple or None).
    """
    if cur_count < MIN_MENTIONS:
        return 0.10, "low", None
    if base_rate is None or base_rate <= 0:
        # Enough chatter but no prior baseline -> a fresh spike from a quiet base.
        return 0.85, "elevated", None
    rel = cur_rate / base_rate
    if rel >= 3.0:
        return 1.00, "high", rel
    if rel >= 1.8:
        return 0.70, "elevated", rel
    if rel >= 1.0:
        return 0.40, "normal", rel
    return 0.10, "low", rel


def save_density_snapshot(conn, ticker, calculated_at, window_count, relative, score, level, notes) -> None:
    conn.execute(
        """
        INSERT INTO ticker_density_snapshots (
            ticker, calculated_at, density_score, density_level, notes,
            window_min, window_count, relative_density
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (ticker, calculated_at, score, level, notes, WINDOW_MIN, window_count, relative),
    )


def calculate_density() -> None:
    print(f"Computing density (rolling {WINDOW_MIN}m window vs {BASELINE_MIN}m baseline)...")
    df = load_recent_social_mentions(BASELINE_MIN + WINDOW_MIN)
    if df.empty:
        print("No social mentions found.")
        return

    df["post_dt"] = df["post_time"].apply(parse_time)
    now = ny_now()
    calc_at = now.isoformat()
    win_start = now - timedelta(minutes=WINDOW_MIN)
    base_start = now - timedelta(minutes=BASELINE_MIN)
    base_span = BASELINE_MIN - WINDOW_MIN  # minutes in the baseline period

    conn = connect_db()
    init_density_columns(conn)

    for ticker in sorted(df["ticker"].dropna().unique().tolist()):
        tdf = df[df["ticker"] == ticker]
        cur = int((tdf["post_dt"] >= win_start).sum())
        cur_rate = cur / WINDOW_MIN
        base = int(((tdf["post_dt"] >= base_start) & (tdf["post_dt"] < win_start)).sum())
        base_rate = (base / base_span) if base_span > 0 else None

        score, level, rel = density_from_window(cur, cur_rate, base_rate)
        rel_txt = f"{rel:.1f}x" if rel is not None else ("spike" if level != "low" else "-")
        notes = f"rolling_{WINDOW_MIN}m={cur} vs baseline_rate={base_rate or 0:.3f}/min ({rel_txt})"
        save_density_snapshot(conn, ticker, calc_at, cur, rel, score, level, notes)
        print(f"  {ticker}: {cur} in {WINDOW_MIN}m -> {level} ({score:.2f}) [{rel_txt}]")

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    calculate_density()
