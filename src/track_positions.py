"""
Trailing stop-loss tracker — the exit side of the system.

When the app alerts on a ticker (your entry), this opens a tracked position and
follows it. Each cycle it updates the position's running PEAK price and computes
a trailing stop:

    stop_price = peak_price * (1 - TRAIL_STOP_PCT)      # default 30%

If the current price falls to/below that stop, the position is marked 'stopped'
and an EXIT alert (sound + notification) fires: "down 30% from the peak."

It turns the app from entry-alerts-only into an entry + exit assistant. These are
SIGNALS you act on — nothing is actually traded.

Price source (in priority order): a live per-ticker Yahoo quote (live_price),
then the current active_ticker_universe (this cycle), then the latest
active_ticker_universe_history price. The live Yahoo quote means the stop keeps
tracking even after a ticker drops off the Finviz screener (previously the stop
froze in that case).

Config (.env):
    TRAIL_STOP_PCT   trailing stop fraction (default 0.30 = -30% from peak)

Runs after the notifier (so it sees this cycle's new alerts) — wired into run_app.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import notify_alerts as notifier  # reuse send_macos_notification + play_alert_sound
import live_price                  # live per-ticker Yahoo quotes (off-screener tracking)
import app_settings                # dashboard-editable profit-target setting


from common import connect_db, ny_now_iso

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

load_dotenv(BASE_DIR / ".env")

TRAIL_STOP_PCT = float(os.getenv("TRAIL_STOP_PCT", "0.30"))

# When opening a position, the entry price (Finviz screener) must agree with the
# live tracking price (Yahoo) within this fraction. Illiquid pre-market spikers
# can have the two sources disagree 5x; opening then records a fake instant
# stop-out. If they disagree beyond this, skip until the data is consistent.
POSITION_PRICE_TOLERANCE = float(os.getenv("POSITION_PRICE_TOLERANCE", "0.25"))

# Stop corroboration: before firing a normal (trailing) stop, require the screener
# (Finviz) price not clearly CONTRADICT it. If only the Yahoo tracking price has
# dropped below the stop while Finviz still shows the price comfortably above, the
# Yahoo read is suspect (spread / bad prints on a thin sub-$1 name) — hold this
# cycle instead of booking a bad exit (e.g. VVOS: Yahoo $0.295 vs Finviz $0.33 —
# Finviz says the stop wasn't really hit). A real drop is corroborated by Finviz
# within a refresh and still exits (e.g. ALP, whose Finviz was also falling).
# The 25% guard above is too loose for a ~10% thin-name gap; this is the finer net.
# Applies to the trailing stop (and catastrophe check) — the raw catastrophe stays fast.
STOP_CORROBORATION_MARGIN = float(os.getenv("STOP_CORROBORATION_MARGIN", "0.03"))

# Grace period: for the first GRACE seconds after entry, a position can't be
# stopped out on a *drop* (the peak still tracks, and a profit target can still
# fire). Just enough to block the "exit on the same tick as entry" whipsaw — one
# cycle (~90s). The SMA (noise filter) now does most of what the old 3-min grace
# did, so it's kept short. Catastrophe still bypasses it.
POSITION_GRACE_SEC = int(os.getenv("POSITION_GRACE_SEC", "90"))

# Optional profit-target exit (off by default; toggled from the dashboard). When
# on, a position also exits the moment it's up PROFIT_TARGET_PCT% from entry —
# a "take profits" signal, independent of the trailing stop.
PROFIT_TARGET_DEFAULT = float(os.getenv("PROFIT_TARGET_PCT", "50"))

# Exit SMA (noise filter). When on, the normal trailing-stop decision compares a
# short simple moving average of the 1-min price (not the raw tick) to the stop,
# so a single bad/flickery quote can't trigger an exit. The PEAK is tracked on
# the same smoothed series (so the stop and peak stay consistent); the profit
# TARGET stays on the raw price (grab real spikes). Default 3 minutes.
EXIT_SMA_DEFAULT_MIN = int(os.getenv("EXIT_SMA_MINUTES", "10"))

# Catastrophe stop (lag insurance for the SMA). A WIDE stop measured on the RAW
# price that bypasses the SMA: if the raw price falls this far below the peak,
# exit immediately — a real crash blows through it before the SMA catches up.
# Set much wider than the normal trail so noise never reaches it. Still gated on
# a trusted quote (so a lone Yahoo glitch can't fake it). Default 20%.
CATASTROPHE_STOP_DEFAULT = float(os.getenv("CATASTROPHE_STOP_PCT", "20"))

# Breakeven stop (+ price-based re-entry). After a per-position warmup of
# BREAKEVEN_WARMUP cycles ("room to settle"), if the smoothed price drops below
# the ENTRY price the position exits — never hold a loser. After such an exit,
# when the price RECLAIMS entry the ticker is re-entered at the current price, up
# to REENTRY_CAP times per ticker per day. Off unless enabled in the dashboard.
BREAKEVEN_WARMUP_DEFAULT = int(os.getenv("BREAKEVEN_WARMUP_CYCLES", "5"))
REENTRY_CAP_DEFAULT = int(os.getenv("REENTRY_CAP_PER_DAY", "2"))


def _profit_target() -> tuple[bool, float]:
    """(enabled, pct) for the optional profit-target exit — dashboard-editable."""
    enabled = str(app_settings.get("profit_target_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
    try:
        pct = float(app_settings.get("profit_target_pct", PROFIT_TARGET_DEFAULT))
    except (TypeError, ValueError):
        pct = PROFIT_TARGET_DEFAULT
    return enabled, pct


def _exit_mode() -> str:
    """Trailing-stop mode (dashboard-toggleable):
      'price'  — stop = peak × (1 − TRAIL_STOP_PCT)   [30% below the peak PRICE]
      'profit' — exit if the gain falls below (1 − TRAIL_STOP_PCT) of the PEAK
                 gain, i.e. give back 30% of profits. When there's no gain yet
                 (peak = entry) this exits on any drop below entry — cutting
                 losers fast.
    """
    m = str(app_settings.get("exit_mode", os.getenv("EXIT_MODE", "price"))).strip().lower()
    return "profit" if m == "profit" else "price"


def _grace_sec() -> int:
    """Grace-period seconds — dashboard setting overrides the env default."""
    try:
        return int(app_settings.get("position_grace_sec", POSITION_GRACE_SEC))
    except (TypeError, ValueError):
        return POSITION_GRACE_SEC


def _exit_sma() -> tuple[bool, int]:
    """(enabled, minutes) for the exit SMA noise filter — dashboard-editable."""
    enabled = str(app_settings.get("exit_sma_enabled", "true")).strip().lower() in ("1", "true", "yes", "on")
    try:
        mins = int(float(app_settings.get("exit_sma_minutes", EXIT_SMA_DEFAULT_MIN)))
    except (TypeError, ValueError):
        mins = EXIT_SMA_DEFAULT_MIN
    return enabled, max(1, mins)


def _catastrophe_stop() -> tuple[bool, float]:
    """(enabled, fraction) for the raw-price catastrophe backstop — dashboard-editable."""
    enabled = str(app_settings.get("catastrophe_stop_enabled", "true")).strip().lower() in ("1", "true", "yes", "on")
    try:
        pct = float(app_settings.get("catastrophe_stop_pct", CATASTROPHE_STOP_DEFAULT))
    except (TypeError, ValueError):
        pct = CATASTROPHE_STOP_DEFAULT
    return enabled, max(0.0, pct) / 100.0


def _breakeven_stop() -> tuple[bool, int]:
    """(enabled, warmup_cycles) for the breakeven stop — dashboard-editable."""
    enabled = str(app_settings.get("breakeven_stop_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
    try:
        warmup = int(float(app_settings.get("breakeven_warmup_cycles", BREAKEVEN_WARMUP_DEFAULT)))
    except (TypeError, ValueError):
        warmup = BREAKEVEN_WARMUP_DEFAULT
    return enabled, max(0, warmup)


def _reentry_cap() -> int:
    """Max price-based re-entries per ticker per day — dashboard-editable."""
    try:
        return max(0, int(float(app_settings.get("reentry_cap", REENTRY_CAP_DEFAULT))))
    except (TypeError, ValueError):
        return REENTRY_CAP_DEFAULT


def init_positions_table() -> None:
    conn = connect_db()
    # Trade-log schema: autoincrement id, ticker is a plain (indexed) column — so a
    # ticker can hold MANY trades over time (re-entry on different days). Fresh DBs
    # get this directly; older one-row-per-ticker tables are migrated below.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            entry_price REAL,
            entry_time TEXT,
            peak_price REAL,
            peak_time TEXT,
            last_price REAL,
            last_time TEXT,
            trail_pct REAL,
            stop_price REAL,
            status TEXT,            -- 'open' | 'stopped'
            exit_price REAL,
            exit_time TEXT,
            entry_headline TEXT,
            exit_reason TEXT        -- why it exited: 'Profit trail' | 'Breakeven stop' | ...
        );
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(positions);").fetchall()}
    # Migrate the old one-row-per-ticker table (ticker PRIMARY KEY, no `id`). The
    # CREATE above is a no-op when the table already exists, so do it here: preserve
    # every existing trade, just move to the id-keyed multi-row shape.
    if "id" not in cols:
        if "exit_reason" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN exit_reason TEXT;")
        conn.executescript(
            """
            ALTER TABLE positions RENAME TO positions_old;
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, entry_price REAL, entry_time TEXT, peak_price REAL,
                peak_time TEXT, last_price REAL, last_time TEXT, trail_pct REAL,
                stop_price REAL, status TEXT, exit_price REAL, exit_time TEXT,
                entry_headline TEXT, exit_reason TEXT, cycles INTEGER DEFAULT 0
            );
            INSERT INTO positions
                (ticker, entry_price, entry_time, peak_price, peak_time, last_price,
                 last_time, trail_pct, stop_price, status, exit_price, exit_time,
                 entry_headline, exit_reason)
            SELECT ticker, entry_price, entry_time, peak_price, peak_time, last_price,
                 last_time, trail_pct, stop_price, status, exit_price, exit_time,
                 entry_headline, exit_reason
            FROM positions_old;
            DROP TABLE positions_old;
            CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            """
        )
    # Add the per-position cycle counter (breakeven-stop warmup) to id-schema
    # tables that predate it.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(positions);").fetchall()}
    if "cycles" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN cycles INTEGER DEFAULT 0;")
    conn.commit()
    conn.close()


def _exit_reason_label(reason: str, exit_mode: str) -> str:
    """Human label for why a position exited (stored + shown in the dashboard)."""
    if reason == "target":
        return "Profit target"
    if reason == "catastrophe":
        return "Catastrophe stop"
    if reason == "breakeven":
        return "Breakeven stop"
    return "Profit trail" if exit_mode == "profit" else "Trailing stop"


def _momentum_gate() -> bool:
    """Entry momentum confirmation — dashboard toggle. When on, only open a position
    if the price is at/above its short SMA (i.e. actually rising, not fading from a
    higher open). Filters the 'up on the day but rolling over' top-tick entries."""
    return str(app_settings.get("momentum_gate_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")


def open_new_positions(conn: sqlite3.Connection) -> int:
    """Open a position for each newly-alerted ticker eligible for entry.

    Re-entry rule: a ticker can be traded again on a LATER day, but not while it
    still has an open position and not twice in the same day. So we exclude any
    ticker that currently has an open position, or one entered OR exited today —
    a trade closed today can re-enter tomorrow, not again today.
    """
    now = ny_now_iso()
    today = now[:10]
    rows = conn.execute(
        """
        SELECT ticker, MIN(notified_at) AS first_at, price_at_alert, headline
        FROM notified_alerts
        WHERE price_at_alert IS NOT NULL
          AND substr(notified_at, 1, 10) = ?
          AND ticker NOT IN (
              SELECT ticker FROM positions
              WHERE status = 'open'
                 OR substr(entry_time, 1, 10) = ?
                 OR substr(exit_time, 1, 10) = ?
          )
        GROUP BY ticker;
        """,
        (today, today, today),
    ).fetchall()

    opened = 0
    candidates = [(t, fa, float(e), h) for (t, fa, e, h) in rows if e is not None]
    _, sma_min = _exit_sma()
    gate_on = _momentum_gate()
    # Live quote + short SMA per candidate: raw for the entry-reconciliation check,
    # SMA for the momentum gate.
    quotes = live_price.fetch_live_and_sma([t for t, _, _, _ in candidates], sma_min) if candidates else {}
    for ticker, first_at, entry, headline in candidates:
        q = quotes.get(ticker) or {}
        live_px = q.get("raw")
        sma = q.get("sma")
        if live_px is not None and entry > 0 and abs(live_px - entry) / entry >= POSITION_PRICE_TOLERANCE:
            # Screener entry and live price disagree too much — untrusted pricing
            # (e.g. an illiquid pre-market spiker). Skip so we don't record a
            # bogus trade that "stops out" instantly; retry once they reconcile.
            print(f"  skip {ticker}: entry ${entry:.2f} (screener) vs live ${live_px:.2f} "
                  f"disagree >{int(POSITION_PRICE_TOLERANCE * 100)}% — untrusted pricing.")
            continue
        # Momentum gate: only enter if the price is at/above its short SMA — i.e. the
        # move is still rising, not fading from a higher open. A catalyst that's up on
        # the day but rolling over is skipped; it re-qualifies next cycle if it turns up.
        if gate_on and live_px is not None and sma is not None and live_px < sma:
            print(f"  skip {ticker}: ${live_px:.2f} below its SMA ${sma:.2f} — no upward "
                  f"momentum yet (waiting for the price to confirm).")
            continue
        stop = entry * (1 - TRAIL_STOP_PCT)
        conn.execute(
            """
            INSERT INTO positions (
                ticker, entry_price, entry_time, peak_price, peak_time,
                last_price, last_time, trail_pct, stop_price, status, entry_headline
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?);
            """,
            (ticker, entry, first_at, entry, first_at, entry, now,
             TRAIL_STOP_PCT, stop, headline),
        )
        opened += 1
        print(f"  opened position {ticker} @ ${entry:.2f} (stop ${stop:.2f})")
    conn.commit()
    return opened


def _current_prices(conn: sqlite3.Connection) -> dict[str, float]:
    return {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT ticker, price FROM active_ticker_universe WHERE price IS NOT NULL;"
        ).fetchall()
    }


def _history_price(conn: sqlite3.Connection, ticker: str) -> float | None:
    row = conn.execute(
        """
        SELECT price FROM active_ticker_universe_history
        WHERE ticker = ? AND price IS NOT NULL
        ORDER BY collected_at DESC LIMIT 1;
        """,
        (ticker,),
    ).fetchone()
    return row[0] if row else None


def update_positions(conn: sqlite3.Connection) -> list[dict]:
    """Update peak/last for open positions; return the list that just stopped out."""
    prices = _current_prices(conn)
    now_dt = datetime.now(NY_TZ)
    now = now_dt.isoformat()
    exits: list[dict] = []

    # No overnight holds: if a position is somehow still open from a PRIOR day (a
    # missed end-of-day, e.g. after a crash), close it now at its last price before
    # tracking the rest. The normal end-of-day flatten (end_of_day.py) handles the
    # common case; this is the belt-and-suspenders enforcer.
    today = now_dt.strftime("%Y-%m-%d")
    conn.execute(
        """
        UPDATE positions
        SET status='stopped', exit_price=last_price, exit_time=?, exit_reason='End of day (late)'
        WHERE status='open' AND last_price IS NOT NULL AND substr(entry_time, 1, 10) < ?;
        """,
        (now, today),
    )

    open_rows = conn.execute(
        "SELECT id, ticker, entry_price, peak_price, peak_time, entry_time, cycles FROM positions WHERE status='open';"
    ).fetchall()

    pt_enabled, pt_pct = _profit_target()
    exit_mode = _exit_mode()
    grace_sec = _grace_sec()
    sma_enabled, sma_min = _exit_sma()
    cat_enabled, cat_frac = _catastrophe_stop()
    be_enabled, be_warmup = _breakeven_stop()

    # Live Yahoo quotes (raw + N-min SMA) for the open positions so the stop tracks
    # even after a ticker leaves the screener; screener/history are fallbacks.
    open_tickers = [r[1] for r in open_rows]
    quotes = live_price.fetch_live_and_sma(open_tickers, sma_min) if open_tickers else {}

    for pid, ticker, entry, peak, peak_time, entry_time, cycles in open_rows:
        cycles = cycles or 0
        q = quotes.get(ticker) or {}
        yahoo_px = q.get("raw")
        yahoo_sma = q.get("sma")
        screener_px = prices.get(ticker)
        price = yahoo_px if yahoo_px is not None else screener_px
        if price is None:
            price = _history_price(conn, ticker)
        if price is None:
            continue  # no fresh price anywhere this cycle — leave the position as-is

        # Data-trust check: if the live (Yahoo) and screener (Finviz) prices
        # disagree wildly, the quote is unreliable (illiquid pre-market spiker).
        # Track the peak with the HIGHER of the two, but DON'T stop out on it this
        # cycle — prevents fake stop-outs from one bad read (e.g. AUUD/FCUV, where
        # Yahoo misread a rising nano-cap ~23% low and triggered a bogus exit).
        trusted = True
        if yahoo_px is not None and screener_px is not None:
            hi = max(yahoo_px, screener_px)
            # '>=' so an exactly-at-threshold gap counts as untrusted (DRMA hit
            # exactly 25%: Finviz 1.44 vs Yahoo's stale 1.08 = 0.2500, and a
            # strict '>' let it through → fake catastrophe exit).
            if hi > 0 and abs(yahoo_px - screener_px) / hi >= POSITION_PRICE_TOLERANCE:
                trusted = False
                price = hi

        # `price` is the RAW trusted quote (used for the profit target, the
        # catastrophe backstop and display). `eval_price` is what the NORMAL
        # trailing stop compares — the SMA when enabled and available, else raw.
        # The peak tracks the same series as the normal stop, so the two stay
        # consistent (a smoothed stop referencing a raw peak would false-exit on
        # the smoothing lag right after a spike).
        eval_price = yahoo_sma if (sma_enabled and yahoo_sma is not None) else price

        if peak is None or eval_price > peak:
            peak, peak_time = eval_price, now

        # Grace period: within GRACE seconds of entry, don't stop out on a drop.
        in_grace = False
        if entry_time:
            try:
                in_grace = (now_dt - datetime.fromisoformat(entry_time)).total_seconds() < grace_sec
            except (ValueError, TypeError):
                in_grace = False

        # Normal trailing stop (compared against the smoothed eval_price) — two modes:
        if exit_mode == "profit":
            # give back TRAIL_STOP_PCT of the PEAK gain; stop rides at that level.
            stop = entry + (peak - entry) * (1 - TRAIL_STOP_PCT)
            stop_hit = eval_price < stop
            # Lane split: when the breakeven stop is on, IT owns the at/below-entry
            # cut (with its warmup + re-entry). So the profit-trail only trails a
            # REAL gain — it fires only when its level is above entry. With no gain
            # (peak == entry) the stop is exactly `entry`; hand that region to the
            # breakeven so its warmup actually applies. (If breakeven is off, the
            # profit-trail still cuts at entry, preserving downside protection.)
            if be_enabled and stop <= entry:
                stop_hit = False
        else:
            stop = peak * (1 - TRAIL_STOP_PCT)   # 30% below the peak price
            stop_hit = eval_price <= stop

        # Corroboration guard: suppress a normal (trailing) stop that the screener flatly
        # contradicts (Finviz still shows the price comfortably above the stop while
        # only the Yahoo read dipped below). A real drop is confirmed by Finviz
        # within a refresh and still exits next cycle.
        stop_corroborated = True
        if screener_px is not None and screener_px > stop * (1 + STOP_CORROBORATION_MARGIN):
            stop_corroborated = False
            if stop_hit and not in_grace:
                print(f"  hold {ticker}: tracking px hit stop ${stop:.4f} but screener "
                      f"${screener_px:.4f} disagrees — waiting for corroboration.")

        # Catastrophe backstop: a WIDE stop on the RAW price that bypasses the SMA
        # (and the grace period) so a genuine fast crash exits immediately instead
        # of waiting for the moving average to catch up. Still trusted-gated.
        cat_stop = peak * (1 - cat_frac) if cat_enabled else None
        cat_hit = cat_stop is not None and trusted and price <= cat_stop
        # ...but don't fire it when the screener flatly contradicts the crash. A
        # source mismatch (e.g. DRMA: Finviz 1.44 pre-market vs Yahoo's stale 1.08
        # close) makes the raw price look 25% down while Finviz shows it well above
        # the crash line. A REAL crash has Finviz falling too and still exits within
        # a refresh, so this only suppresses fakes.
        if cat_hit and screener_px is not None and screener_px > cat_stop * (1 + STOP_CORROBORATION_MARGIN):
            print(f"  hold {ticker}: raw ${price:.4f} <= catastrophe ${cat_stop:.4f} but screener "
                  f"${screener_px:.4f} disagrees — likely a source mismatch, not exiting.")
            cat_hit = False

        target = entry * (1 + pt_pct / 100.0) if (pt_enabled and entry) else None

        # Breakeven stop: after the per-position warmup, never hold below entry.
        # Uses the smoothed price + trust + screener corroboration (a raw check would
        # fire on noise and Yahoo-mismatch fakes). During warmup it's inactive and
        # the normal rules (grace/trailing-stop/catastrophe) govern.
        be_hit = (be_enabled and entry and cycles >= be_warmup and trusted
                  and eval_price < entry
                  and (screener_px is None or screener_px <= entry * (1 + STOP_CORROBORATION_MARGIN)))

        # Exit priority: profit target (up +pt_pct%, on the raw price) → catastrophe
        # (raw crash, bypasses grace) → breakeven (below entry, post-warmup) → normal
        # trailing stop (smoothed, only after grace AND trusted AND corroborated).
        reason = None
        if target is not None and price >= target:
            reason = "target"
        elif cat_hit:
            reason = "catastrophe"
        elif be_hit:
            reason = "breakeven"
        elif stop_hit and not in_grace and trusted and stop_corroborated:
            reason = "stop"

        if reason:
            reason_label = _exit_reason_label(reason, exit_mode)
            stored_stop = {"catastrophe": cat_stop, "breakeven": entry}.get(reason, stop)
            conn.execute(
                """
                UPDATE positions
                SET peak_price=?, peak_time=?, last_price=?, last_time=?, stop_price=?,
                    status='stopped', exit_price=?, exit_time=?, exit_reason=?
                WHERE id=?;
                """,
                (peak, peak_time, price, now, stored_stop, price, now, reason_label, pid),
            )
            exits.append({"ticker": ticker, "entry": entry, "peak": peak,
                          "price": price, "stop": stored_stop, "reason": reason,
                          "target": target, "mode": exit_mode, "cat_frac": cat_frac})
            tag = {"target": "TARGET", "catastrophe": "CATASTROPHE",
                   "breakeven": "BREAKEVEN"}.get(reason, "STOP")
            print(f"  {tag} {ticker}: ${price:.2f} (entry ${entry:.2f}, peak ${peak:.2f})")
        else:
            conn.execute(
                """
                UPDATE positions
                SET peak_price=?, peak_time=?, last_price=?, last_time=?, stop_price=?,
                    cycles=cycles+1
                WHERE id=?;
                """,
                (peak, peak_time, price, now, stop, pid),
            )

    conn.commit()
    return exits


def reenter_positions(conn: sqlite3.Connection) -> int:
    """Price-based re-entry: after a breakeven exit, re-open a ticker when the
    smoothed price reclaims its entry — up to the daily cap per ticker. Each
    re-entry is a fresh position (new row) with its own warmup."""
    be_enabled, _ = _breakeven_stop()
    if not be_enabled:
        return 0
    cap = _reentry_cap()
    if cap <= 0:
        return 0
    now = ny_now_iso()
    today = now[:10]
    _, sma_min = _exit_sma()
    prices = _current_prices(conn)

    # Most-recent position TODAY per ticker + how many trades it's had today.
    rows = conn.execute(
        """
        SELECT p.ticker, p.entry_price, p.status, p.exit_reason,
               (SELECT COUNT(*) FROM positions x
                  WHERE x.ticker = p.ticker AND substr(x.entry_time, 1, 10) = ?) AS n_today
        FROM positions p
        WHERE substr(p.entry_time, 1, 10) = ?
          AND p.id = (SELECT MAX(y.id) FROM positions y
                        WHERE y.ticker = p.ticker AND substr(y.entry_time, 1, 10) = ?);
        """,
        (today, today, today),
    ).fetchall()

    # Candidate = last trade today exited on the breakeven stop, ticker not open,
    # and under the daily cap (original + cap re-entries).
    candidates = [(t, float(e)) for t, e, status, reason, n in rows
                  if status == "stopped" and reason == "Breakeven stop" and e and n < 1 + cap]
    if not candidates:
        return 0

    quotes = live_price.fetch_live_and_sma([t for t, _ in candidates], sma_min)
    reentered = 0
    for ticker, entry in candidates:
        q = quotes.get(ticker) or {}
        raw, sma = q.get("raw"), q.get("sma")
        eval_px = sma if sma is not None else raw
        screener_px = prices.get(ticker)
        if eval_px is None or eval_px < entry:
            continue                                    # hasn't reclaimed entry yet
        # Corroboration: the screener should also be at/above entry (a real reclaim,
        # not a lone Yahoo blip).
        if screener_px is not None and screener_px < entry * (1 - STOP_CORROBORATION_MARGIN):
            continue
        newpx = raw if raw is not None else eval_px
        stop = newpx * (1 - TRAIL_STOP_PCT)
        conn.execute(
            """
            INSERT INTO positions (
                ticker, entry_price, entry_time, peak_price, peak_time,
                last_price, last_time, trail_pct, stop_price, status, entry_headline
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?);
            """,
            (ticker, newpx, now, newpx, now, newpx, now, TRAIL_STOP_PCT, stop,
             "Re-entry (price reclaimed entry)"),
        )
        reentered += 1
        print(f"  RE-ENTRY {ticker} @ ${newpx:.2f} (reclaimed entry ${entry:.2f})")
        notifier.send_macos_notification(
            f"📈 RE-ENTRY {ticker}  ${newpx:.2f}",
            f"Price reclaimed entry ${entry:.2f} after a breakeven exit — re-entered.",
            subtitle="Re-entry", sound_name=notifier.ALERT_SOUND_NAME)
    conn.commit()
    return reentered


def notify_exits(exits: list[dict]) -> None:
    if not exits:
        return
    trail_pct = int(round(TRAIL_STOP_PCT * 100))
    for e in exits:
        pl = (e["price"] - e["entry"]) / e["entry"] * 100 if e["entry"] else 0.0
        if e.get("reason") == "target":
            title = f"🎯 TARGET {e['ticker']}  {pl:+.0f}%"
            message = (
                f"Profit target hit: +{pl:.0f}% from entry ${e['entry']:.2f} "
                f"(now ${e['price']:.2f}). Consider taking profits."
            )
            subtitle = "Profit target"
        elif e.get("reason") == "breakeven":
            title = f"🛑 BREAKEVEN {e['ticker']}  {pl:+.1f}%"
            message = (
                f"Dropped below entry ${e['entry']:.2f} (now ${e['price']:.2f}) — "
                f"exited to avoid holding a loser. P/L {pl:+.1f}%. "
                f"Will re-enter if it reclaims entry."
            )
            subtitle = "Breakeven stop"
        elif e.get("reason") == "catastrophe":
            cat_pct = int(round(e.get("cat_frac", 0.20) * 100))
            title = f"🚨 CATASTROPHE {e['ticker']}  {pl:+.1f}%"
            message = (
                f"Raw price crashed {cat_pct}%+ below the peak ${e['peak']:.2f} — "
                f"hard exit (bypassing the SMA). Now ${e['price']:.2f} · "
                f"entry ${e['entry']:.2f} · P/L {pl:+.1f}%."
            )
            subtitle = "Catastrophe stop"
        elif e.get("mode") == "profit":
            title = f"🛑 EXIT {e['ticker']}  {pl:+.1f}%"
            message = (
                f"Profit trail hit — gave back {trail_pct}% of the peak gain "
                f"(peak ${e['peak']:.2f}). Now ${e['price']:.2f} · entry ${e['entry']:.2f} · "
                f"P/L {pl:+.1f}%."
            )
            subtitle = "Profit trail"
        else:
            title = f"🛑 EXIT {e['ticker']}"
            message = (
                f"Trailing stop hit (-{trail_pct}% from peak ${e['peak']:.2f}). "
                f"Now ${e['price']:.2f} · entry ${e['entry']:.2f} · P/L {pl:+.1f}%."
            )
            subtitle = "Trailing stop"
        notifier.send_macos_notification(title, message, subtitle=subtitle,
                                         sound_name=notifier.ALERT_EXIT_SOUND_NAME)
    notifier.play_alert_sound(notifier.ALERT_EXIT_SOUND_FILE)


def run_position_tracking() -> None:
    init_positions_table()
    conn = connect_db()
    try:
        opened = open_new_positions(conn)
        exits = update_positions(conn)
        reentered = reenter_positions(conn)   # price reclaimed entry after a breakeven exit
        n_open = conn.execute("SELECT COUNT(*) FROM positions WHERE status='open';").fetchone()[0]
    finally:
        conn.close()

    notify_exits(exits)
    print(f"Positions: {opened} opened, {reentered} re-entered, {len(exits)} stopped out, "
          f"{n_open} still open (trailing stop {int(TRAIL_STOP_PCT*100)}%).")


if __name__ == "__main__":
    run_position_tracking()
