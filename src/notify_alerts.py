from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import sqlite3

import news_filter
import app_settings


# ----------------------------
# Paths / constants
# ----------------------------

from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

# Load .env so SMTP creds (and any ALERT_* overrides) are available even when run
# as a subprocess by run_app.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Defaults (overridable via env or CLI flags).
DEFAULT_MIN_SCORE = float(os.getenv("ALERT_MIN_SCORE", "0.5"))
DEFAULT_WINDOW_MIN = int(os.getenv("APP_NEWS_WINDOW_MIN", "1440"))  # 24h; lower for true real-time
DENSITY_HOT = float(os.getenv("ALERT_DENSITY_HOT", "0.7"))

# Structured-news selectivity: only alert on news at/above this materiality (the
# Gemini catalyst tag). Default "moderate" filters out routine PR; set "major"
# to alert only on FDA/earnings/M&A-class catalysts, or "routine" to allow all.
_MATERIALITY_RANK = {"routine": 0, "moderate": 1, "major": 2}
ALERT_MIN_MATERIALITY = os.getenv("ALERT_MIN_MATERIALITY", "moderate").strip().lower()


def _allowed_materiality() -> list[str]:
    min_rank = _MATERIALITY_RANK.get(ALERT_MIN_MATERIALITY, 1)
    return [m for m, r in _MATERIALITY_RANK.items() if r >= min_rank]

# Audible cue when a strong ticker appears. ENTRY alerts (📈) use a bright "Glass"
# ding; EXIT alerts (🛑, trailing stop) use a lower "Basso" tone so you can tell
# them apart by ear without looking.
ALERT_SOUND_ENABLED = os.getenv("ALERT_SOUND_ENABLED", "true").strip().lower() == "true"
ALERT_SOUND_FILE = os.getenv("ALERT_SOUND_FILE", "/System/Library/Sounds/Glass.aiff")
ALERT_SOUND_NAME = os.getenv("ALERT_SOUND_NAME", "Glass")  # macOS notification sound (entry)
ALERT_EXIT_SOUND_FILE = os.getenv("ALERT_EXIT_SOUND_FILE", "/System/Library/Sounds/Basso.aiff")
ALERT_EXIT_SOUND_NAME = os.getenv("ALERT_EXIT_SOUND_NAME", "Basso")  # macOS notification sound (exit)

# Email alerts (optional). SMTP creds are secrets -> .env; the recipient + the
# on/off toggles are dashboard-editable (app_settings). Sound and email are two
# independent delivery methods; either can be on or off.
ALERT_SMTP_HOST = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
ALERT_SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", "587"))
ALERT_SMTP_USER = os.getenv("ALERT_SMTP_USER", "").strip()
ALERT_SMTP_PASSWORD = os.getenv("ALERT_SMTP_PASSWORD", "").strip()


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _sound_enabled() -> bool:
    """Sound toggle — dashboard setting overrides the env default."""
    v = app_settings.get("alert_sound_enabled", None)
    return ALERT_SOUND_ENABLED if v is None else _truthy(v)


def _email_enabled() -> bool:
    return _truthy(app_settings.get("alert_email_enabled", "false"))


def _email_recipient() -> str:
    """Dashboard-entered recipient; falls back to the SMTP account (send to self)."""
    return (str(app_settings.get("alert_email", "") or "").strip()) or ALERT_SMTP_USER


KEYWORD_BYPASS_SCORE_DEFAULT = float(os.getenv("KEYWORD_BYPASS_SCORE", "0.8"))


def _keyword_bypass() -> tuple[bool, float]:
    """(enabled, min_score) for the high-sentiment keyword bypass — dashboard-editable.
    When on, a headline scoring >= min_score alerts even if it matches no keyword
    (catches narrative-catalyst movers whose headlines skip the whitelist words)."""
    enabled = _truthy(app_settings.get("keyword_bypass_enabled", "false"))
    try:
        score = float(app_settings.get("keyword_bypass_score", KEYWORD_BYPASS_SCORE_DEFAULT))
    except (TypeError, ValueError):
        score = KEYWORD_BYPASS_SCORE_DEFAULT
    return enabled, score


def send_email(subject: str, body: str, to_addr: str) -> bool:
    """Send one email via SMTP (Gmail by default). Returns True on success."""
    if not (ALERT_SMTP_USER and ALERT_SMTP_PASSWORD and to_addr):
        print("  [email] not configured (need ALERT_SMTP_USER/PASSWORD in .env + a recipient); skipping.")
        return False
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = ALERT_SMTP_USER
    msg["To"] = to_addr
    msg.set_content(body)
    try:
        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT, timeout=20) as s:
            s.starttls()
            s.login(ALERT_SMTP_USER, ALERT_SMTP_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"  [email] send failed: {e}")
        return False


def _send_email_digest(rows) -> None:
    """One email per cycle summarising the new alerts (no-op if email is off)."""
    if not _email_enabled() or not rows:
        return
    to = _email_recipient()
    tickers = [str(r["ticker"]) for r in rows]
    subject = f"🔔 {len(rows)} stock alert{'s' if len(rows) > 1 else ''}: " + ", ".join(tickers[:4])
    parts = ["New alerts from your screener:\n"]
    for r in rows:
        mat = r.get("materiality") if hasattr(r, "get") else None
        parts.append(f"• {r['ticker']}  {_fmt_change(r['change_percent'])}  "
                     f"(score {float(r['sentiment_score']):.2f}{', ' + str(mat) if mat else ''})")
        parts.append(f"    {r['title_clean']}")
        parts.append(f"    why: {r['reason']}")
        if r.get("source"):
            parts.append(f"    source: {r['source']}")
        parts.append("")
    parts.append("Open the dashboard for live prices and the Top Pick.")
    if send_email(subject, "\n".join(parts), to):
        print(f"  [email] sent to {to} ({len(rows)} alert(s)).")


def ny_now() -> datetime:
    return datetime.now(NY_TZ)


def ny_now_iso() -> str:
    return ny_now().isoformat()


# ----------------------------
# Dedup table
# ----------------------------

def init_notified_alerts_table() -> None:
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notified_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_news_id INTEGER,
            ticker TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            change_percent REAL,
            density_score REAL,
            price_at_alert REAL,
            headline TEXT,
            notified_at TEXT,
            UNIQUE(clean_news_id, ticker)
        );
        """
    )
    # Migration: add price_at_alert to a pre-existing table that lacks it.
    cols = [r[1] for r in cur.execute("PRAGMA table_info(notified_alerts);").fetchall()]
    if "price_at_alert" not in cols:
        cur.execute("ALTER TABLE notified_alerts ADD COLUMN price_at_alert REAL;")
    conn.commit()
    conn.close()


# ----------------------------
# Find new alerts
# ----------------------------

ALERT_MAJOR_NEUTRAL_KEY = "alert_major_neutral"


def _alert_major_neutral() -> bool:
    """Option A (dashboard toggle, default OFF = Option B): also alert on
    MAJOR-materiality news that scored neutral-or-positive (score >= 0), not only
    the usual score >= min_score. Lets direction-ambiguous catalysts like mergers
    / buyouts through, while still excluding clearly-negative major news."""
    return str(app_settings.get(ALERT_MAJOR_NEUTRAL_KEY, "false")).strip().lower() in ("1", "true", "yes")


def find_new_alerts(min_score: float, window_min: int) -> pd.DataFrame:
    """
    An alert = a positive-enough news item on a ticker that Finviz currently
    flags as growing (it's in active_ticker_universe), that is recent, and that
    we have not already notified.

    Density (from the latest ticker_density_snapshots) is attached when present
    but is NOT required, so alerts can fire before a real social collector exists.
    """
    cutoff = (ny_now() - timedelta(minutes=window_min)).isoformat()

    # Selectivity: only alert on news at/above a materiality bar (default excludes
    # 'routine' PR). NULL materiality (old rows) is treated as passing.
    allowed = _allowed_materiality()
    placeholders = ",".join("?" for _ in allowed)
    major_neutral = 1 if _alert_major_neutral() else 0

    query = f"""
        WITH latest_density AS (
            SELECT tds.ticker, tds.density_score, tds.density_level
            FROM ticker_density_snapshots tds
            JOIN (
                SELECT ticker, MAX(calculated_at) AS mx
                FROM ticker_density_snapshots
                GROUP BY ticker
            ) l ON tds.ticker = l.ticker AND tds.calculated_at = l.mx
        )
        SELECT
            ns.clean_news_id,
            ns.ticker,
            ns.sentiment_label,
            ns.sentiment_score,
            ns.materiality,
            ns.reason,
            cn.title_clean,
            cn.publish_time,
            cn.collect_time,
            cn.source,
            atu.change_percent,
            atu.price,
            ld.density_score
        FROM news_sentiment ns
        JOIN clean_news cn
            ON ns.clean_news_id = cn.id
        JOIN active_ticker_universe atu
            ON ns.ticker = atu.ticker
        LEFT JOIN latest_density ld
            ON ld.ticker = ns.ticker
        LEFT JOIN notified_alerts na
            ON na.clean_news_id = ns.clean_news_id
           AND na.ticker = ns.ticker
        WHERE (
                ns.sentiment_score >= ?
                OR (? = 1 AND ns.materiality = 'major' AND ns.sentiment_score >= 0)
              )
          AND (ns.materiality IS NULL OR ns.materiality IN ({placeholders}))
          AND na.id IS NULL
          AND cn.publish_time IS NOT NULL
          AND cn.publish_time >= ?
        ORDER BY ns.sentiment_score DESC, atu.change_percent DESC;
    """

    conn = connect_db()
    df = pd.read_sql_query(query, conn, params=(min_score, major_neutral, *allowed, cutoff))
    conn.close()

    # User keyword whitelist (edited from the dashboard). When non-empty, only
    # alert on headlines containing one of the keywords; empty list = no filter.
    # High-sentiment bypass (optional toggle): a strongly-positive headline alerts
    # even without a keyword — catches narrative-catalyst movers whose headlines
    # skip the whitelist vocabulary (e.g. AIxC "pivot to robotics", +86%).
    if not df.empty:
        keywords = news_filter.get_keywords()
        if keywords:
            bypass_on, bypass_score = _keyword_bypass()

            def _keep(row) -> bool:
                if news_filter.matches(row["title_clean"] or "", keywords):
                    return True
                if bypass_on and row["sentiment_score"] is not None:
                    try:
                        return float(row["sentiment_score"]) >= bypass_score
                    except (TypeError, ValueError):
                        return False
                return False

            df = df[df.apply(_keep, axis=1)].reset_index(drop=True)
    return df


def record_notified(row: pd.Series) -> None:
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO notified_alerts (
            clean_news_id, ticker, sentiment_label, sentiment_score,
            change_percent, density_score, price_at_alert, headline, notified_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            int(row["clean_news_id"]),
            str(row["ticker"]),
            str(row["sentiment_label"]),
            float(row["sentiment_score"]),
            None if pd.isna(row["change_percent"]) else float(row["change_percent"]),
            None if pd.isna(row.get("density_score")) else float(row["density_score"]),
            None if pd.isna(row.get("price")) else float(row["price"]),
            str(row["title_clean"]),
            ny_now_iso(),
        ),
    )
    conn.commit()
    conn.close()


# ----------------------------
# macOS notification
# ----------------------------

def _osa_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def send_macos_notification(title: str, message: str, subtitle: str | None = None,
                            sound_name: str | None = None) -> None:
    """
    Fire an instant macOS notification via osascript (no extra dependency).
    sound_name overrides the notification sound (defaults to the entry sound).
    """
    name = sound_name if sound_name is not None else ALERT_SOUND_NAME
    script = f'display notification "{_osa_escape(message)}" with title "{_osa_escape(title)}"'
    if subtitle:
        script += f' subtitle "{_osa_escape(subtitle)}"'
    if _sound_enabled() and name:
        script += f' sound name "{_osa_escape(name)}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False)
    except FileNotFoundError:
        # Not on macOS / osascript unavailable -> degrade to console.
        print(f"[notify] {title} | {subtitle or ''} | {message}")


def play_alert_sound(sound_file: str | None = None) -> None:
    """Play an audible cue (defaults to the entry sound; pass a file for exits)."""
    if not _sound_enabled():
        return
    try:
        subprocess.run(["afplay", sound_file or ALERT_SOUND_FILE], check=False)
    except FileNotFoundError:
        # Not on macOS / afplay unavailable -> the notification sound still plays.
        pass


def _fmt_change(value) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"


# ----------------------------
# Main entry
# ----------------------------

def run_notifications(min_score: float, window_min: int, send: bool = True) -> int:
    init_notified_alerts_table()
    alerts = find_new_alerts(min_score, window_min)

    if alerts.empty:
        print("No new alerts.")
        return 0

    print(f"Found {len(alerts)} new alert(s).")

    sent_rows = []
    for _, row in alerts.iterrows():
        change_str = _fmt_change(row["change_percent"])
        density = row.get("density_score")
        hot = (density is not None) and (not pd.isna(density)) and (float(density) >= DENSITY_HOT)
        prefix = "🔥 " if hot else ""

        title = f"📈 {row['ticker']}  {change_str}".strip()
        message = f"{prefix}{row['reason']} — {row['title_clean']}"
        subtitle = str(row["source"]) if row.get("source") else None

        print(
            f"  ALERT {row['ticker']} score={row['sentiment_score']:.2f} "
            f"chg={change_str} {'[hot density]' if hot else ''}"
        )

        if send:
            send_macos_notification(title, message, subtitle=subtitle)
            record_notified(row)
            sent_rows.append(row)

    if send:
        # One ding + one email digest per cycle that produced new alerts.
        play_alert_sound()
        _send_email_digest(sent_rows)
    else:
        print("(dry run: nothing sent or recorded)")

    return len(alerts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send instant alerts for growing tickers with positive news.")
    parser.add_argument("--test", action="store_true", help="send one sample notification and exit")
    parser.add_argument("--test-email", action="store_true", help="send one test email and exit")
    parser.add_argument("--no-send", action="store_true", help="evaluate alerts but do not send or record (dry run)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--window-min", type=int, default=DEFAULT_WINDOW_MIN)
    args = parser.parse_args()

    if args.test_email:
        to = _email_recipient()
        ok = send_email("🔔 Test alert from your stock screener",
                        "This is a test. If you got this, email alerts are working.\n\n"
                        "Enable them from the dashboard sidebar (📧 Email alert).", to)
        print(f"Test email {'sent to ' + to if ok else 'FAILED — check ALERT_SMTP_USER/PASSWORD in .env'}.")
        return

    if args.test:
        send_macos_notification(
            "📈 TEST — TSLA  +5.20%",
            "FDA approval is generally positive. — Sample headline for the alert test.",
            subtitle="PR Newswire",
        )
        play_alert_sound()
        print("Sent test notification (with sound).")
        return

    run_notifications(args.min_score, args.window_min, send=not args.no_send)


if __name__ == "__main__":
    main()
