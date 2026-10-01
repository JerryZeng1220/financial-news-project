"""
check_accuracy.py — how good were the app's alerts?

For every ENTRY alert fired in the look-back window (default 1 day, max 7 days),
it compares the price at alert to the current live price (Yahoo) and reports:

  - Win rate     — share of alerts that are UP since the alert (to now)
  - Avg / median return since alert
  - Best / worst call
  - "Pop" hit-rate — share that reached >= HIT_PCT at any point (using the peak
    price from the trailing-stop tracker) — i.e. did the signal actually move up

Honest caveats (printed too):
  - Hypothetical PAPER returns on the signals: entry = screener price at the
    alert, "now" = live Yahoo quote. No fees/slippage — these are signals, not
    fills.
  - Return is measured "to now", so a 6-day-old alert is a 6-day return while a
    10-minute-old alert is a 10-minute return — not apples-to-apples across ages.
    The peak-based hit-rate ("did it pop") is the more age-robust measure.
  - Small samples (a few alerts/day) — read as a sanity check, not a backtest.

Usage:
    python check_accuracy.py                 # last 1 day (default)
    python check_accuracy.py --window 6h
    python check_accuracy.py --window 90m
    python check_accuracy.py --window 7d --hit 10
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import live_price

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

MAX_MINUTES = 7 * 24 * 60          # 7-day cap
DEFAULT_MINUTES = 24 * 60          # 1 day
_UNIT_MIN = {"m": 1, "h": 60, "d": 1440}


def parse_window(s: str) -> int:
    """'30m' / '6h' / '2d' -> minutes, capped at 7 days."""
    m = re.fullmatch(r"\s*(\d+)\s*([mhd])\s*", (s or "").lower())
    if not m:
        raise argparse.ArgumentTypeError("window must look like 30m, 6h, or 2d")
    minutes = int(m.group(1)) * _UNIT_MIN[m.group(2)]
    if minutes <= 0:
        raise argparse.ArgumentTypeError("window must be positive")
    return min(minutes, MAX_MINUTES)


def _fmt_window(minutes: int) -> str:
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def compute_accuracy(minutes: int = DEFAULT_MINUTES, hit_pct: float = 5.0) -> dict:
    """Return an accuracy report dict for alerts fired in the last `minutes`."""
    minutes = min(max(1, int(minutes)), MAX_MINUTES)
    now_dt = datetime.now(NY_TZ)
    cutoff = (now_dt - timedelta(minutes=minutes)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        rows = conn.execute(
            """
            SELECT notified_at, ticker, price_at_alert, headline
            FROM notified_alerts
            WHERE notified_at >= ?
            ORDER BY notified_at DESC;
            """,
            (cutoff,),
        ).fetchall()
        # Max favourable move since entry, from the trailing-stop tracker.
        peaks = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT ticker, MAX(peak_price) FROM positions WHERE peak_price IS NOT NULL GROUP BY ticker;"
            ).fetchall()
        }
    finally:
        conn.close()

    tickers = sorted({r[1] for r in rows})
    live = live_price.fetch_live_prices(tickers) if tickers else {}

    scored: list[dict] = []
    excluded = 0
    for notified_at, ticker, entry, headline in rows:
        now_px = live.get(ticker)
        try:
            entry = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            entry = None
        if not entry or now_px is None:
            excluded += 1
            continue
        ret = (now_px - entry) / entry * 100.0
        peak = peaks.get(ticker)
        mfe = (peak - entry) / entry * 100.0 if peak else None  # max favourable excursion
        scored.append({
            "notified_at": notified_at, "ticker": ticker, "entry": entry,
            "now": now_px, "ret_pct": ret, "peak_pct": mfe, "headline": headline,
        })

    rets = [a["ret_pct"] for a in scored]
    # "Pop": reached >= hit_pct at any point (peak if we have it, else current).
    def _reached(a):
        best = a["peak_pct"] if a["peak_pct"] is not None else a["ret_pct"]
        return best >= hit_pct

    n = len(scored)
    report = {
        "window_minutes": minutes,
        "window_label": _fmt_window(minutes),
        "from": cutoff[:19].replace("T", " "),
        "to": now_dt.isoformat()[:19].replace("T", " "),
        "hit_pct": hit_pct,
        "n_alerts": len(rows),
        "n_scored": n,
        "n_excluded": excluded,
        "n_distinct_tickers": len({a["ticker"] for a in scored}),
        "win_rate": (sum(1 for r in rets if r > 0) / n * 100.0) if n else None,
        "avg_return": (statistics.fmean(rets)) if n else None,
        "median_return": (statistics.median(rets)) if n else None,
        "hit_rate": (sum(1 for a in scored if _reached(a)) / n * 100.0) if n else None,
        "best": max(scored, key=lambda a: a["ret_pct"]) if n else None,
        "worst": min(scored, key=lambda a: a["ret_pct"]) if n else None,
        "alerts": scored,
    }
    return report


def compute_strategy_stats(minutes: int = DEFAULT_MINUTES) -> dict:
    """Backtest-style stats under the assumption: BUY at every alert, SELL when
    the tracker says (trailing stop or profit target). Uses the positions table
    (one round-trip per ticker that alerted), for trades ENTERED in the window.

    'closed' = completed round-trips (realized return exit vs entry). 'open' =
    still-running trades, marked to the last tracked price (unrealized).
    """
    minutes = min(max(1, int(minutes)), MAX_MINUTES)
    now_dt = datetime.now(NY_TZ)
    cutoff = (now_dt - timedelta(minutes=minutes)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        # exit_reason may be absent on DBs written before it was added.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(positions);").fetchall()}
        reason_col = "p.exit_reason" if "exit_reason" in cols else "NULL AS exit_reason"
        # news_published = when the entry catalyst was published (via the alert's
        # clean_news row) — surfaced in the dashboard's per-row info tooltip.
        rows = conn.execute(
            f"""SELECT p.ticker, p.entry_price, p.exit_price, p.last_price, p.status, p.exit_time,
                      p.entry_time, p.entry_headline, {reason_col},
                      (SELECT cn.publish_time FROM notified_alerts na
                         JOIN clean_news cn ON na.clean_news_id = cn.id
                        WHERE na.ticker = p.ticker AND na.headline = p.entry_headline
                        ORDER BY na.notified_at LIMIT 1) AS news_published,
                      (SELECT cn.collect_time FROM notified_alerts na
                         JOIN clean_news cn ON na.clean_news_id = cn.id
                        WHERE na.ticker = p.ticker AND na.headline = p.entry_headline
                        ORDER BY na.notified_at LIMIT 1) AS news_detected
               FROM positions p WHERE p.entry_time >= ? AND p.entry_price > 0;""",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    closed, open_ = [], []
    for ticker, entry, exitp, lastp, status, xtime, etime, headline, xreason, news_pub, news_det in rows:
        entry = float(entry)
        if status == "stopped" and exitp is not None:
            r = (float(exitp) - entry) / entry * 100.0
            closed.append({"ticker": ticker, "entry": entry, "exit": float(exitp),
                           "ret_pct": r, "exit_time": xtime, "entry_time": etime,
                           "entry_reason": headline, "exit_reason": xreason,
                           "news_published": news_pub, "news_detected": news_det})
        elif status == "open":
            mark = float(lastp) if lastp is not None else None
            open_.append({"ticker": ticker, "entry": entry, "now": mark,
                          "ret_pct": ((mark - entry) / entry * 100.0) if mark else None,
                          "entry_time": etime, "entry_reason": headline,
                          "news_published": news_pub, "news_detected": news_det})

    rets = [t["ret_pct"] for t in closed]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100.0)
    open_rets = [t["ret_pct"] for t in open_ if t["ret_pct"] is not None]
    n = len(rets)

    return {
        "window_minutes": minutes,
        "window_label": _fmt_window(minutes),
        "from": cutoff[:19].replace("T", " "),
        "to": now_dt.isoformat()[:19].replace("T", " "),
        "closed": n,
        "win_rate": (len(wins) / n * 100.0) if n else None,
        "avg_return": statistics.fmean(rets) if n else None,
        "median_return": statistics.median(rets) if n else None,
        "compounded_return": (comp - 1) * 100.0 if n else None,
        "avg_win": statistics.fmean(wins) if wins else None,
        "avg_loss": statistics.fmean(losses) if losses else None,
        "best": max(closed, key=lambda t: t["ret_pct"]) if n else None,
        "worst": min(closed, key=lambda t: t["ret_pct"]) if n else None,
        "open": len(open_),
        "open_unrealized_avg": statistics.fmean(open_rets) if open_rets else None,
        "trades": sorted(closed, key=lambda t: t["exit_time"] or "", reverse=True),
    }


def print_report(r: dict) -> None:
    print(f"\nAlert accuracy — last {r['window_label']}  ({r['from']} .. {r['to']})")
    print("-" * 64)
    if r["n_alerts"] == 0:
        print("No alerts fired in this window.")
        return
    print(f"Alerts fired:        {r['n_alerts']}  "
          f"(scored {r['n_scored']}, excluded {r['n_excluded']} — missing entry/live price)")
    if r["n_scored"] == 0:
        print("Nothing scorable (no entry or live prices available).")
        return
    print(f"Distinct tickers:    {r['n_distinct_tickers']}")
    print(f"Win rate (up now):   {r['win_rate']:.0f}%")
    print(f"Avg return:          {r['avg_return']:+.1f}%")
    print(f"Median return:       {r['median_return']:+.1f}%")
    print(f"Reached +{r['hit_pct']:.0f}% (peak): {r['hit_rate']:.0f}%")
    b, w = r["best"], r["worst"]
    print(f"Best:  {b['ticker']:6} {b['ret_pct']:+.1f}%")
    print(f"Worst: {w['ticker']:6} {w['ret_pct']:+.1f}%")
    print("\n  when              ticker   entry     now    ret%   peak%")
    for a in r["alerts"]:
        pk = f"{a['peak_pct']:+.1f}" if a["peak_pct"] is not None else "  -  "
        print(f"  {a['notified_at'][:16].replace('T',' ')}  {a['ticker']:6} "
              f"{a['entry']:7.2f} {a['now']:7.2f}  {a['ret_pct']:+6.1f}  {pk:>6}")
    print("\nNote: hypothetical paper returns (screener price at alert -> live Yahoo now), "
          "no fees/slippage. Return is 'to now', so it spans each alert's own age; the "
          "peak-based hit-rate is the more age-robust 'did it pop' measure. Small samples.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the app's alert accuracy over a window.")
    ap.add_argument("--window", type=parse_window, default=DEFAULT_MINUTES,
                    help="look-back like 30m / 6h / 2d (default 1d, max 7d)")
    ap.add_argument("--hit", type=float, default=5.0,
                    help="'pop' threshold %% for the hit-rate (default 5)")
    args = ap.parse_args()
    print_report(compute_accuracy(args.window, args.hit))


if __name__ == "__main__":
    main()
