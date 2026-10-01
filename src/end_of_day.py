"""
End-of-day: flatten all open positions (no overnight holds) + a daily recap.

Two jobs, run when the trading session ends (and defensively at worker startup if
positions lingered from a prior session):

  1. FLATTEN  — close every open position at its current price (exit_reason
     'End of day'). Nothing is ever held overnight.
  2. SUMMARISE — "if you traded every signal today, how did the day go?" — the
     realized win rate / average / compounded return over today's trades. Saved to
     `daily_summaries`, printed, and delivered by desktop notification + optional
     email.

Invoked by run_app at the session-end transition. Safe to run manually too:
    python src/end_of_day.py
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import live_price
import notify_alerts as notifier
import check_accuracy
from common import connect_db, ny_now_iso

NY_TZ = ZoneInfo("America/New_York")


def _init_summary_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_summaries (
            day TEXT PRIMARY KEY,
            created_at TEXT,
            n_trades INTEGER,
            win_rate REAL,
            avg_return REAL,
            compounded REAL,
            best TEXT,
            worst TEXT,
            text TEXT
        );
        """
    )


def flatten_open_positions(conn) -> int:
    """Close every open position at its current price. Returns the count closed."""
    now = ny_now_iso()
    rows = conn.execute(
        "SELECT id, ticker, entry_price, last_price FROM positions WHERE status='open';"
    ).fetchall()
    if not rows:
        return 0
    live = live_price.fetch_live_prices([r[1] for r in rows])
    for pid, ticker, entry, last in rows:
        px = live.get(ticker)
        if px is None:                       # no fresh quote -> last tracked, else entry
            px = last if last is not None else entry
        conn.execute(
            """
            UPDATE positions
            SET status='stopped', exit_price=?, exit_time=?, last_price=?, last_time=?,
                exit_reason='End of day'
            WHERE id=?;
            """,
            (px, now, px, now, pid),
        )
    conn.commit()
    return len(rows)


def _fmt_summary(day: str, s: dict, n_flat: int) -> str:
    if s["closed"] == 0:
        base = f"📅 Daily summary — {day}\nNo completed trades today."
    else:
        b, w = s["best"], s["worst"]
        base = (
            f"📅 Daily summary — {day}\n"
            f"If you had traded every signal:\n"
            f"  Trades: {s['closed']}   Win rate: {s['win_rate']:.0f}%\n"
            f"  Avg / trade: {s['avg_return']:+.1f}%   Compounded: {s['compounded_return']:+.1f}%\n"
            f"  Median: {s['median_return']:+.1f}%   "
            f"Avg win {(('%+.1f%%' % s['avg_win']) if s['avg_win'] is not None else '—')} / "
            f"avg loss {(('%+.1f%%' % s['avg_loss']) if s['avg_loss'] is not None else '—')}\n"
            f"  Best: {b['ticker']} {b['ret_pct']:+.0f}%   Worst: {w['ticker']} {w['ret_pct']:+.0f}%"
        )
    return base + f"\n({n_flat} position(s) force-closed at end of day — no overnight holds.)"


def run_end_of_day() -> str:
    """Flatten, summarise today's 'trade every signal' result, save + deliver it."""
    conn = connect_db()
    _init_summary_table(conn)
    n_flat = flatten_open_positions(conn)
    conn.close()

    day = ny_now_iso()[:10]
    # Today's realized "buy every alert, sell when told (or at EOD)" stats — every
    # trade entered today is now closed, so this is the complete day.
    now = datetime.now(NY_TZ)
    minutes_today = now.hour * 60 + now.minute + 1
    s = check_accuracy.compute_strategy_stats(minutes_today)
    text = _fmt_summary(day, s, n_flat)
    print(text)

    best = f"{s['best']['ticker']} {s['best']['ret_pct']:+.0f}%" if s.get("best") else None
    worst = f"{s['worst']['ticker']} {s['worst']['ret_pct']:+.0f}%" if s.get("worst") else None
    conn = connect_db()
    _init_summary_table(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO daily_summaries
            (day, created_at, n_trades, win_rate, avg_return, compounded, best, worst, text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (day, ny_now_iso(), s["closed"], s.get("win_rate"), s.get("avg_return"),
         s.get("compounded_return"), best, worst, text),
    )
    conn.commit()
    conn.close()

    # Deliver (desktop notification always; email if enabled in the dashboard).
    if n_flat > 0 or s["closed"] > 0:
        subtitle = (f"Win {s['win_rate']:.0f}% · {s['compounded_return']:+.1f}% compounded"
                    if s["closed"] else f"{n_flat} flattened")
        notifier.send_macos_notification(f"📅 EOD {day}: {s['closed']} trades", text, subtitle=subtitle)
        try:
            if notifier._email_enabled():
                notifier.send_email(f"📅 Daily summary {day}", text, notifier._email_recipient())
        except Exception as e:
            print(f"  [eod] email failed: {e}")
    return text


if __name__ == "__main__":
    run_end_of_day()
