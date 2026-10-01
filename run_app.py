"""
Background runner: turns the one-shot pipeline into a living app.

Each cycle it (1) runs the full data pipeline (run_pipeline.py) and then
(2) checks for new alerts and sends instant macOS notifications
(src/notify_alerts.py). It loops forever until you press Ctrl+C.

Open it, minimize it, and it keeps watching. Configuration via env vars:

  APP_INTERVAL_SECONDS   seconds between cycles (default 300 = 5 min)
  ACTIVE_UNIVERSE_MODE   csv_snapshot | finviz_screener (live) | authorized_url
  USE_CURL_IMPERSONATE   "true" for the live Finviz scraper
  ALERT_MIN_SCORE        minimum sentiment score to alert (default 0.5)
  APP_NEWS_WINDOW_MIN    only alert on news newer than this many minutes (default 1440)
  APP_PIPELINE_TIMEOUT_SEC  watchdog: force-kill a pipeline that runs longer than
                            this (default 120 = 2 min)
  APP_PIPELINE_RETRY_SEC    wait this long after a kill/failure before a fresh
                            pipeline (default 60 = 1 min)

Watchdog: each cycle runs the pipeline under a hard timeout. If it finishes, the
next cycle starts normally; if it stalls past APP_PIPELINE_TIMEOUT_SEC it's
force-killed (whole process group, no orphaned stage), then the app waits
APP_PIPELINE_RETRY_SEC and starts a fresh pipeline — so one hung stage can never
freeze the app.

The loop can run fast (1-2 min): the Finviz universe fetch self-throttles to
UNIVERSE_MIN_REFRESH_SEC (default 240s) regardless of loop speed, so you get
fast news/alert checks without tripping Finviz's rate limit. Gemini is gated by
*new* news (batched) + a mood cooldown, so faster cycles don't increase its usage.
"""
from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent
NY_TZ = ZoneInfo("America/New_York")
DB_PATH = BASE_DIR / "financial_news.db"

sys.path.insert(0, str(BASE_DIR / "src"))
import app_settings  # noqa: E402 — dashboard-editable trading-session window

INTERVAL = int(os.getenv("APP_INTERVAL_SECONDS", "90"))
# Off-hours the loop still runs the data pipeline (so no news/sentiment is missed)
# but does NOT trade; poll a bit slower then to be gentle on the news servers.
OFFHOURS_INTERVAL = int(os.getenv("APP_OFFHOURS_INTERVAL_SECONDS", "300"))
# Trading is ALWAYS confined to the dashboard-configured session window (read fresh
# each cycle from app_settings — see is_market_hours). There is no separate static
# on/off flag anymore: the old APP_MARKET_HOURS_ONLY env was read once at startup,
# so a process that outlived a config change kept trading pre-market. To run around
# the clock, widen the session window (dashboard → Trading session) instead.

# Watchdog: force-kill the pipeline if a cycle stalls past this many seconds,
# then wait before starting a fresh one (so a hung stage can't freeze the app).
PIPELINE_TIMEOUT = int(os.getenv("APP_PIPELINE_TIMEOUT_SEC", "120"))  # 2 min
RETRY_WAIT = int(os.getenv("APP_PIPELINE_RETRY_SEC", "60"))           # 1 min

PIPELINE = BASE_DIR / "run_pipeline.py"
NOTIFIER = BASE_DIR / "src" / "notify_alerts.py"
TRACKER = BASE_DIR / "src" / "track_positions.py"
END_OF_DAY = BASE_DIR / "src" / "end_of_day.py"


def _session_window() -> tuple[int, int]:
    """Trading window as [start, end) minutes-from-midnight ET — dashboard-editable
    (app_settings), read fresh each cycle so changes take effect live. Default
    9:30 AM–8:00 PM (regular + after-hours)."""
    def _mins(key: str, default: int) -> int:
        try:
            return int(app_settings.get(key, default))
        except (TypeError, ValueError):
            return default
    return _mins("session_start_min", 570), _mins("session_end_min", 1200)


def is_market_hours(now: datetime) -> bool:
    """True inside the dashboard-configured trading window (Mon-Fri, ET)."""
    if now.weekday() >= 5:  # Sat/Sun
        return False
    start, end = _session_window()
    return start <= (now.hour * 60 + now.minute) < end


def run_pipeline_watchdog(timeout: int) -> str:
    """Run the full pipeline under a hard timeout (watchdog).

    Returns 'ok' (finished, exit 0), 'failed' (finished non-zero), or 'timeout'
    (ran longer than `timeout`s → force-killed). On timeout the WHOLE process
    group is killed (start_new_session), so a stuck stage isn't left orphaned.
    """
    proc = subprocess.Popen([sys.executable, str(PIPELINE)], start_new_session=True)
    try:
        return "ok" if proc.wait(timeout=timeout) == 0 else "failed"
    except subprocess.TimeoutExpired:
        print(f"  [watchdog] pipeline exceeded {timeout}s — force-ending it.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return "timeout"


def collect_data() -> str:
    """Run the full data pipeline (ingest news → score sentiment → rank) under the
    watchdog. Runs EVERY cycle, in or out of the trading window, so overnight
    news/sentiment is never missed — RSS feeds roll over, so we can't just wait
    until the open to fetch it."""
    return run_pipeline_watchdog(PIPELINE_TIMEOUT)


def run_trading() -> None:
    """Fire alerts + track positions. Called ONLY inside the trading window: off
    hours the prices are unreliable (thin/absent quotes), so we still collect the
    data but don't act on it. The data-trust / corroboration guards handle any
    remaining unreliable-price cases during the session itself."""
    subprocess.run([sys.executable, str(NOTIFIER)])   # records alerts + notifications
    subprocess.run([sys.executable, str(TRACKER)])    # opens/tracks positions


def run_end_of_day() -> None:
    """Flatten all open positions + emit the daily summary (no overnight holds)."""
    subprocess.run([sys.executable, str(END_OF_DAY)])


def _has_open_positions() -> bool:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        return conn.execute(
            "SELECT 1 FROM positions WHERE status='open' LIMIT 1;"
        ).fetchone() is not None
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def record_app_start() -> str:
    """Log this worker launch so you can see when the app started each day
    (e.g. to confirm it was live at the market open)."""
    started = datetime.now(NY_TZ).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS app_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT);"
    )
    conn.execute("INSERT INTO app_sessions (started_at) VALUES (?);", (started,))
    conn.commit()
    conn.close()
    return started


def main() -> None:
    started = record_app_start()
    print("=" * 50)
    print("Financial News background app")
    print(f"  started at: {started[:19].replace('T', ' ')}")
    print(f"  interval: {INTERVAL}s")
    _ss, _se = _session_window()
    print(f"  trading window: {_ss//60:02d}:{_ss%60:02d}-{_se//60:02d}:{_se%60:02d} ET "
          f"(always enforced, editable in the dashboard)")
    print(f"  data collection: 24/7 (news + sentiment) — trading only inside the window")
    print(f"  universe mode: {os.getenv('ACTIVE_UNIVERSE_MODE', 'csv_snapshot')}")
    print("  press Ctrl+C to stop.")
    print("=" * 50)

    # Track session membership so we can flatten + summarise exactly once when the
    # trading day ends. If we START outside the session with positions still open,
    # they're stale overnight holds (a missed EOD) — flatten them now.
    was_in_session = is_market_hours(datetime.now(NY_TZ))
    if not was_in_session and _has_open_positions():
        print("  [startup] open positions found outside the session — running end-of-day.")
        run_end_of_day()

    try:
        while True:
            now = datetime.now(NY_TZ)
            in_session = is_market_hours(now)
            print(f"\n===== cycle @ {now.isoformat()}  [{'TRADING' if in_session else 'data-only'}] =====")

            # 1) ALWAYS collect news + score sentiment (never miss data), under watchdog.
            try:
                status = collect_data()
            except Exception as e:  # keep the loop alive on a bad cycle
                print(f"  [cycle] error: {e}")
                status = "failed"
            if status != "ok":
                print(f"  [cycle] pipeline {status}; waiting {RETRY_WAIT}s before retry.")
                time.sleep(RETRY_WAIT)
                continue

            # 2) TRADE only inside the trading window (reliable prices). Off hours we
            #    keep collecting but never suggest a trade.
            if in_session:
                was_in_session = True
                run_trading()
                time.sleep(INTERVAL)
            else:
                if was_in_session:
                    # Session just ended — flatten every position (no overnight holds)
                    # and emit the daily "trade every signal" summary, once.
                    print(f"  [{now.isoformat()}] session ended — end-of-day flatten + summary.")
                    run_end_of_day()
                    was_in_session = False
                else:
                    print("  off-hours: collected news + sentiment, not trading.")
                time.sleep(OFFHOURS_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped. Goodbye.")


if __name__ == "__main__":
    main()
