"""
Decision dashboard — "the app" you open to see why a ticker looks strong.

Reads what the pipeline already produces (combined_ticker_rankings + the supporting
tables) and renders one decision card per current strong ticker:
price/change, the news headline + the sentiment reason, discussion density, the
Stocktwits bullish/bearish crowd ratio, the combined score, and links out to the
Stocktwits and Finviz pages for the chart and comments.

Run it with:
    streamlit run src/dashboard.py

It auto-refreshes on an interval (sidebar), so after the background worker plays
its alert sound you just glance here for the detail.
"""
from __future__ import annotations

import html
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import app_settings
import news_filter
import live_price
import check_accuracy
import news_sources


from common import connect_db

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "financial_news.db"
NY_TZ = ZoneInfo("America/New_York")

SOCIAL_LOOKBACK_HOURS = 6


def table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


# ----------------------------
# Data assembly
# ----------------------------

def load_decision_data() -> pd.DataFrame:
    conn = connect_db()
    try:
        if not table_exists(conn, "combined_ticker_rankings"):
            return pd.DataFrame()

        rank = pd.read_sql_query(
            """
            SELECT ticker, combined_score, news_sentiment_label, news_sentiment_score,
                   density_score, density_level, alert_candidate,
                   latest_headline, latest_source, latest_news_time, explanation
            FROM combined_ticker_rankings
            WHERE calculated_at = (SELECT MAX(calculated_at) FROM combined_ticker_rankings);
            """,
            conn,
        )
        if rank.empty:
            return rank

        univ = pd.read_sql_query(
            "SELECT ticker, price, change_percent, volume, market_cap FROM active_ticker_universe;",
            conn,
        )

        # The actual Gemini/rule reason for each ticker's most recent news.
        reason = pd.read_sql_query(
            """
            SELECT ns.ticker, ns.reason, ns.method, ns.materiality
            FROM news_sentiment ns
            JOIN clean_news cn ON ns.clean_news_id = cn.id
            JOIN (
                SELECT ns2.ticker, MAX(cn2.publish_time) AS mx
                FROM news_sentiment ns2
                JOIN clean_news cn2 ON ns2.clean_news_id = cn2.id
                GROUP BY ns2.ticker
            ) latest ON ns.ticker = latest.ticker AND cn.publish_time = latest.mx;
            """,
            conn,
        )

        # Stocktwits bullish/bearish crowd ratio (recent).
        cutoff = (datetime.now(NY_TZ) - timedelta(hours=SOCIAL_LOOKBACK_HOURS)).isoformat()
        social = pd.read_sql_query(
            """
            SELECT ticker,
                   SUM(CASE WHEN sentiment_label='bullish' THEN 1 ELSE 0 END) AS bull,
                   SUM(CASE WHEN sentiment_label='bearish' THEN 1 ELSE 0 END) AS bear,
                   COUNT(*) AS total
            FROM social_mentions
            WHERE post_time >= ?
            GROUP BY ticker;
            """,
            conn,
            params=(cutoff,),
        )

        # Gemini one-line crowd-mood summary (when available).
        if table_exists(conn, "ticker_crowd_mood"):
            mood = pd.read_sql_query(
                "SELECT ticker, mood, summary AS mood_summary FROM ticker_crowd_mood;", conn
            )
        else:
            mood = pd.DataFrame(columns=["ticker", "mood", "mood_summary"])

        # Most recent alert per ticker (for the "🔔 alerted" badge).
        if table_exists(conn, "notified_alerts"):
            alerted = pd.read_sql_query(
                """
                SELECT ticker, MAX(notified_at) AS alerted_at
                FROM notified_alerts GROUP BY ticker;
                """, conn
            )
        else:
            alerted = pd.DataFrame(columns=["ticker", "alerted_at"])
    finally:
        conn.close()

    df = rank.merge(univ, on="ticker", how="left")
    df = df.merge(reason, on="ticker", how="left")
    df = df.merge(social, on="ticker", how="left")
    df = df.merge(mood, on="ticker", how="left")
    df = df.merge(alerted, on="ticker", how="left")
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return df


def load_alert_log(limit: int = 50) -> pd.DataFrame:
    """Recent alerts with their entry price and the latest known price (outcome)."""
    conn = connect_db()
    try:
        if not table_exists(conn, "notified_alerts"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            """
            WITH latest_price AS (
                SELECT h.ticker, h.price AS latest_price
                FROM active_ticker_universe_history h
                JOIN (
                    SELECT ticker, MAX(collected_at) AS mx
                    FROM active_ticker_universe_history GROUP BY ticker
                ) m ON h.ticker = m.ticker AND h.collected_at = m.mx
            )
            SELECT na.notified_at, na.ticker, na.price_at_alert,
                   lp.latest_price, na.change_percent, na.density_score, na.headline
            FROM notified_alerts na
            LEFT JOIN latest_price lp ON na.ticker = lp.ticker
            ORDER BY na.notified_at DESC
            LIMIT ?;
            """,
            conn, params=(limit,),
        )
    finally:
        conn.close()

    if not df.empty:
        df["price_at_alert"] = pd.to_numeric(df["price_at_alert"], errors="coerce")
        df["latest_price"] = pd.to_numeric(df["latest_price"], errors="coerce")
        df["pct_since_alert"] = (
            (df["latest_price"] - df["price_at_alert"]) / df["price_at_alert"] * 100
        ).round(1)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def live_prices_cached(tickers: tuple) -> dict:
    """Live Yahoo prices for the tickers, cached ~60s so the dashboard's
    auto-refresh doesn't re-hit Yahoo every render."""
    return live_price.fetch_live_prices(list(tickers))


@st.cache_data(ttl=60, show_spinner=False)
def accuracy_cached(minutes: int, hit: float) -> dict:
    """Alert-accuracy report for the window, cached ~60s (it fetches live prices)."""
    return check_accuracy.compute_accuracy(minutes, hit)


@st.cache_data(ttl=60, show_spinner=False)
def strategy_cached(minutes: int) -> dict:
    """Buy-every-alert / sell-when-told backtest stats for the window (cached)."""
    return check_accuracy.compute_strategy_stats(minutes)


@st.cache_data(ttl=60, show_spinner=False)
def would_fire_cached() -> pd.DataFrame:
    """Qualifying catalyst news that WOULD fire an alert but hasn't been traded —
    typically because it was detected off-hours (the trader only runs in-session).
    Reuses the alerter's exact criteria (score/materiality/universe/fresh/keyword)."""
    import notify_alerts
    try:
        df = notify_alerts.find_new_alerts(notify_alerts.DEFAULT_MIN_SCORE,
                                           notify_alerts.DEFAULT_WINDOW_MIN)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return df.drop_duplicates("ticker").reset_index(drop=True)  # one row per ticker


def load_positions() -> pd.DataFrame:
    """Open + stopped positions with current P/L and distance from the trailing stop."""
    conn = connect_db()
    try:
        if not table_exists(conn, "positions"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            """
            SELECT ticker, status, entry_price, peak_price, last_price, stop_price,
                   trail_pct, exit_time
            FROM positions
            ORDER BY (status='open') DESC, peak_price DESC;
            """, conn
        )
    finally:
        conn.close()

    if not df.empty:
        for c in ["entry_price", "peak_price", "last_price", "stop_price"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["pl_pct"] = ((df["last_price"] - df["entry_price"]) / df["entry_price"] * 100).round(1)
        df["from_peak_pct"] = ((df["last_price"] - df["peak_price"]) / df["peak_price"] * 100).round(1)
    return df


EXIT_BANNER_HOURS = int(os.getenv("EXIT_BANNER_HOURS", "4"))


def load_recent_exits(hours: int = EXIT_BANNER_HOURS) -> pd.DataFrame:
    """Positions that stopped out in the last `hours` — the tickers to sell now."""
    conn = connect_db()
    try:
        if not table_exists(conn, "positions"):
            return pd.DataFrame()
        cutoff = (datetime.now(NY_TZ) - timedelta(hours=hours)).isoformat()
        df = pd.read_sql_query(
            """
            SELECT ticker, entry_price, exit_price, exit_time
            FROM positions
            WHERE status='stopped' AND exit_time >= ? AND exit_price IS NOT NULL
            ORDER BY exit_time DESC;
            """, conn, params=(cutoff,),
        )
    finally:
        conn.close()
    if not df.empty:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
        df["ret_pct"] = ((df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100).round(1)
    return df


def _market_status() -> str:
    """Human label for the current US market session, so a frozen 'Current $'
    reads as 'market closed' rather than a stuck feed. Regular session is
    Mon–Fri 9:30–16:00 ET (holidays not accounted for)."""
    now = datetime.now(NY_TZ)
    mins = now.hour * 60 + now.minute
    if now.weekday() >= 5:
        return "🔴 Weekend — prices are last Friday's close"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "🟢 Market open — prices live (refresh ~60s)"
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "🟡 Pre-market — regular prices frozen at last close"
    if 16 * 60 <= mins < 20 * 60:
        return "🟡 After-hours — regular prices frozen at today's close"
    return "🔴 Market closed — prices are the last close"


def load_open_positions() -> pd.DataFrame:
    """Currently-open positions with entry price, a live current price, and the
    unrealized return."""
    conn = connect_db()
    try:
        if not table_exists(conn, "positions"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            """SELECT p.ticker, p.entry_price, p.entry_time, p.entry_headline,
                  (SELECT cn.publish_time FROM notified_alerts na
                     JOIN clean_news cn ON na.clean_news_id = cn.id
                    WHERE na.ticker = p.ticker AND na.headline = p.entry_headline
                    ORDER BY na.notified_at LIMIT 1) AS news_published,
                  (SELECT cn.collect_time FROM notified_alerts na
                     JOIN clean_news cn ON na.clean_news_id = cn.id
                    WHERE na.ticker = p.ticker AND na.headline = p.entry_headline
                    ORDER BY na.notified_at LIMIT 1) AS news_detected
               FROM positions p WHERE p.status='open' ORDER BY p.entry_time DESC;""",
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    live = live_prices_cached(tuple(sorted(df["ticker"].dropna().unique())))
    df["current"] = df["ticker"].map(live)
    df["ret_pct"] = ((df["current"] - df["entry_price"]) / df["entry_price"] * 100).round(1)
    return df


def last_updated() -> str | None:
    conn = connect_db()
    try:
        if not table_exists(conn, "combined_ticker_rankings"):
            return None
        row = conn.execute("SELECT MAX(calculated_at) FROM combined_ticker_rankings").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def app_started_today() -> str | None:
    """First time the background worker launched today (to confirm it was live at the open)."""
    conn = connect_db()
    try:
        if not table_exists(conn, "app_sessions"):
            return None
        today = datetime.now(NY_TZ).date().isoformat()
        row = conn.execute(
            "SELECT MIN(started_at) FROM app_sessions WHERE started_at >= ?;", (today,)
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


# ----------------------------
# Rendering
# ----------------------------

def _fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):+.2f}%"


_MOOD_EMOJI = {"bullish": "🐂", "bearish": "🐻", "mixed": "⚖️", "quiet": "💤"}


def crowd_label(row) -> str:
    # Prefer the Gemini one-line summary when we have it.
    summary = row.get("mood_summary")
    if summary and pd.notna(summary):
        emoji = _MOOD_EMOJI.get(str(row.get("mood") or "").lower(), "💬")
        return f"{emoji} {summary}"

    # Fallback: bullish/bearish ratio from stored tags.
    total = row.get("total")
    if total is None or pd.isna(total) or int(total) == 0:
        return "💬 no recent chatter"
    bull = int(row.get("bull") or 0)
    bear = int(row.get("bear") or 0)
    tagged = bull + bear
    if tagged == 0:
        return f"💬 {int(total)} posts (untagged mood)"
    pct_bull = round(100 * bull / tagged)
    mood = "🐂 bullish" if pct_bull >= 60 else ("🐻 bearish" if pct_bull <= 40 else "⚖️ mixed")
    return f"{mood} — {pct_bull}% bull ({bull}/{tagged} tagged, {int(total)} posts)"


def render_card(row) -> None:
    ticker = row["ticker"]
    with st.container(border=True):
        top = st.columns([2, 1, 1, 1])
        with top[0]:
            flag = "🔥 " if int(row.get("alert_candidate") or 0) == 1 else ""
            st.subheader(f"{flag}{ticker}")
            mc = row.get("market_cap")
            st.caption(f"Market cap: {mc if pd.notna(mc) else '—'}M" if mc not in (None, "") else "")
            alerted_at = row.get("alerted_at")
            if alerted_at and pd.notna(alerted_at):
                when = str(alerted_at)[:16].replace("T", " ")
                st.caption(f"🔔 already alerted {when}")
        with top[1]:
            price = row.get("price")
            st.metric("Price", f"${float(price):.2f}" if pd.notna(price) else "—",
                      _fmt_pct(row.get("change_percent")))
        with top[2]:
            st.metric("Combined", f"{float(row['combined_score']):.2f}")
        with top[3]:
            st.metric("Density", row.get("density_level") or "none")

        # News + reason
        label = (row.get("news_sentiment_label") or "neutral").lower()
        emoji = {"positive": "🟢", "negative": "🔴", "mixed": "🟡", "neutral": "⚪"}.get(label, "⚪")
        headline = row.get("latest_headline") or "(no recent news)"
        src = row.get("latest_source") or ""
        mat = row.get("materiality")
        mat_txt = f" · {mat}" if mat and pd.notna(mat) else ""
        st.markdown(f"**{emoji} News ({label}{mat_txt})** · {src}")
        st.write(headline)
        reason = row.get("reason")
        if reason and pd.notna(reason):
            method = row.get("method") or ""
            st.caption(f"Why: {reason}  _( {method} )_")

        # Crowd + links
        st.markdown(f"**Crowd:** {crowd_label(row)}")
        links = st.columns(2)
        with links[0]:
            st.link_button("💬 Stocktwits", f"https://stocktwits.com/symbol/{ticker}", width="stretch")
        with links[1]:
            st.link_button("📈 Finviz chart", f"https://finviz.com/quote.ashx?t={ticker}", width="stretch")


_SENTI_EMOJI = {"positive": "🟢", "negative": "🔴", "mixed": "🟡", "neutral": "⚪"}


def _screener_frame(view: pd.DataFrame) -> pd.DataFrame:
    """Build the compact screener table (separate from rendering so it's testable)."""
    sent = view.get("news_sentiment_label")
    sent = (sent.fillna("neutral").astype(str).str.lower() if sent is not None
            else pd.Series(["neutral"] * len(view), index=view.index))
    if "mood" in view.columns:
        crowd = view["mood"].fillna("").astype(str).str.lower().map(lambda m: _MOOD_EMOJI.get(m, ""))
    else:
        crowd = pd.Series([""] * len(view), index=view.index)

    return pd.DataFrame({
        "🔔": view["alerted_at"].apply(lambda x: "🔔" if pd.notna(x) else ""),
        "Ticker": view["ticker"],
        "Price": pd.to_numeric(view.get("price"), errors="coerce"),
        "Chg %": pd.to_numeric(view.get("change_percent"), errors="coerce"),
        "Score": pd.to_numeric(view["combined_score"], errors="coerce"),
        "News": sent.map(lambda s: f"{_SENTI_EMOJI.get(s, '⚪')} {s}"),
        "Density": view.get("density_level").fillna("none") if view.get("density_level") is not None else "none",
        "Crowd": crowd,
    })


def render_screener_table(view: pd.DataFrame) -> None:
    """Compact screener-style overview of the strong tickers — a quick scan above the cards."""
    st.dataframe(
        _screener_frame(view),
        width="stretch", hide_index=True,
        column_config={
            "🔔": st.column_config.TextColumn(width="small"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Chg %": st.column_config.NumberColumn(format="%.1f%%"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def _suppress_stale_news(df: pd.DataFrame, cutoff_iso: str) -> tuple[pd.DataFrame, int]:
    """Blank the news fields for tickers whose newest news is older than the cutoff.

    Design A: keep the ticker card (its growth may still be valid) but hide the
    stale headline/reason so it reads as "(no recent news)". Display-only —
    nothing is deleted; the old story stays in the DB and shows in the archive.
    """
    if df.empty or "latest_news_time" not in df.columns:
        return df, 0

    def _stale(v):
        return bool(v) and pd.notna(v) and str(v) < cutoff_iso

    mask = df["latest_news_time"].map(_stale)
    if not mask.any():
        return df, 0
    df = df.copy()
    for col in ("latest_headline", "latest_source", "reason", "materiality",
                "news_sentiment_label", "news_sentiment_score"):
        if col in df.columns:
            df.loc[mask, col] = None
    return df, int(mask.sum())


def load_archived_news(cutoff_iso: str, limit: int = 300) -> pd.DataFrame:
    """News older than the freshness cutoff — hidden from the board but retained."""
    conn = connect_db()
    try:
        return pd.read_sql_query(
            """
            SELECT cn.publish_time AS when_utc,
                   COALESCE(ntl.ticker, '—') AS ticker,
                   cn.title_clean AS title,
                   cn.source AS source
            FROM clean_news cn
            LEFT JOIN news_ticker_links ntl ON ntl.clean_news_id = cn.id
            WHERE cn.publish_time IS NOT NULL AND cn.publish_time < ?
            ORDER BY cn.publish_time DESC
            LIMIT ?;
            """,
            conn, params=(cutoff_iso, limit),
        )
    finally:
        conn.close()


def load_all_news(days: int = 3, limit: int = 500, search: str = "") -> pd.DataFrame:
    """Every ingested headline from the last `days` days — all sources, whether or
    not the news is linked to a ticker or that ticker is in the growing universe.
    Newest first. Optional case-insensitive search (headline/source/ticker) is
    applied in SQL so it scans the whole window, not just the capped result set."""
    conn = connect_db()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).isoformat()
        params: list = [cutoff]
        where_search = ""
        s = (search or "").strip()
        if s:
            like = f"%{s}%"
            where_search = """
              AND (cn.title_clean LIKE ? COLLATE NOCASE
                   OR cn.source LIKE ? COLLATE NOCASE
                   OR cn.id IN (SELECT clean_news_id FROM news_ticker_links
                                WHERE ticker LIKE ? COLLATE NOCASE))
            """
            params += [like, like, like]
        params.append(limit)
        df = pd.read_sql_query(
            f"""
            SELECT cn.publish_time AS when_t,
                   COALESCE(GROUP_CONCAT(DISTINCT ntl.ticker), '') AS tickers,
                   cn.source AS source,
                   cn.title_clean AS title,
                   cn.url AS url
            FROM clean_news cn
            LEFT JOIN news_ticker_links ntl ON ntl.clean_news_id = cn.id
            WHERE cn.publish_time IS NOT NULL AND cn.publish_time >= ?
              AND cn.is_duplicate = 0 {where_search}
            GROUP BY cn.id
            ORDER BY cn.publish_time DESC
            LIMIT ?;
            """,
            conn, params=params,
        )
    finally:
        conn.close()
    return df


def render_recommendation(df: pd.DataFrame, updated) -> None:
    """The program's strongest current trade signal, computed live when this tab
    is viewed: the highest-combined-score alert candidate + a live price. Honest
    when nothing clears the bar. Not financial advice."""
    st.caption("🎯 The program's strongest current signal — a growing ticker with fresh, "
               "positive, material news (and crowd interest). **Not financial advice; you decide.**")

    if updated:
        stamp = str(updated)[:19].replace("T", " ")
        age_note = ""
        try:
            secs = (datetime.now(NY_TZ) - datetime.fromisoformat(str(updated))).total_seconds()
            if secs > 1800:
                age_note = (f"  ⚠️ signals are ~{int(secs // 60)} min old "
                            "(worker may be off — market closed / after hours)")
        except Exception:
            pass
        st.caption(f"Signals as of {stamp}{age_note} · price below is live.")

    recs = df[(df["alert_candidate"] == 1) & df["combined_score"].notna()].copy()
    recs = recs.sort_values("combined_score", ascending=False)
    if recs.empty:
        st.info("🚦 **No strong signal right now.** Nothing currently clears the bar "
                "(a growing ticker with fresh positive, material news). Sit tight — "
                "this updates the moment one does.")
        return

    live = live_prices_cached(tuple(recs["ticker"].head(8)))
    top = recs.iloc[0]
    tk = str(top["ticker"])
    px = live.get(tk)

    st.markdown(f"## 🎯 {tk}")
    c = st.columns(4)
    c[0].metric("Live price", f"${px:.2f}" if px else "—", _fmt_pct(top.get("change_percent")))
    c[1].metric("Combined score", f"{float(top['combined_score']):.2f}")
    mat = top.get("materiality")
    news = (top.get("news_sentiment_label") or "—").title()
    c[2].metric("News", news + (f" · {mat}" if mat and pd.notna(mat) else ""))
    c[3].metric("Density", top.get("density_level") or "none")

    if top.get("alerted_at") and pd.notna(top.get("alerted_at")):
        st.caption(f"🔔 already alerted {str(top['alerted_at'])[:16].replace('T', ' ')}")

    hl = top.get("latest_headline")
    if hl and pd.notna(hl):
        st.markdown(f"**Why:** {hl}")
        reason = top.get("reason")
        if reason and pd.notna(reason):
            st.caption(reason)
    st.markdown(f"**Crowd:** {crowd_label(top)}")

    links = st.columns(2)
    links[0].link_button("💬 Stocktwits", f"https://stocktwits.com/symbol/{tk}", width="stretch")
    links[1].link_button("📈 Finviz chart", f"https://finviz.com/quote.ashx?t={tk}", width="stretch")

    if len(recs) > 1:
        st.markdown("**Other candidates**")
        others = recs.iloc[1:6].copy()
        others["now"] = others["ticker"].map(live)
        others = others.rename(columns={
            "ticker": "Ticker", "now": "Now $", "combined_score": "Score",
            "news_sentiment_label": "News", "density_level": "Density",
        })
        st.dataframe(
            others[["Ticker", "Now $", "Score", "News", "Density"]],
            width="stretch", hide_index=True,
            column_config={
                "Now $": st.column_config.NumberColumn(format="$%.2f"),
                "Score": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def render_sources() -> None:
    """Manage the sources the app collects from — enable/disable any, add RSS
    feeds, remove ones you added. Changes apply on the worker's next cycle."""
    st.caption("Everything the app collects news from. Toggle any source on/off, add an RSS feed, "
               "or remove one. **Changes apply on the worker's next cycle.**")
    df = news_sources.get_all()

    st.markdown("#### 🗞️ RSS news feeds")
    rss = df[df["source_type"] == "rss"]
    if rss.empty:
        st.caption("No RSS feeds — add one below.")
    for _, row in rss.iterrows():
        name = row["name"]
        c = st.columns([3, 6, 1.4, 1])
        c[0].markdown(f"**{name}**")
        c[1].caption(str(row["feed_url"]))
        on = c[2].toggle("on", value=bool(row["enabled"]), key=f"src_tog_{name}",
                         label_visibility="collapsed")
        if on != bool(row["enabled"]):
            news_sources.set_enabled(name, on)
        if c[3].button("🗑️", key=f"src_del_{name}", help=f"Remove {name}"):
            news_sources.delete_source(name)
            st.rerun()

    with st.form("add_source_form", clear_on_submit=True):
        st.markdown("**➕ Add an RSS feed**")
        fc = st.columns([2, 5, 1.4])
        new_name = fc[0].text_input("Name", key="src_new_name", placeholder="e.g. Business Wire")
        new_url = fc[1].text_input("Feed URL", key="src_new_url", placeholder="https://…/rss")
        if fc[2].form_submit_button("Add", width="stretch"):
            ok, msg = news_sources.add_source(new_name, new_url)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    st.divider()
    st.markdown("#### 🔌 Collectors")
    st.caption("Built-in collectors (not URL feeds). Toggle to include/exclude their data.")
    coll = df[df["source_type"] == "collector"]
    for _, row in coll.iterrows():
        name = row["name"]
        c = st.columns([3, 6, 1.4])
        c[0].markdown(f"**{name}**")
        c[1].caption(news_sources.COLLECTOR_NOTES.get(name, "collector"))
        on = c[2].toggle("on", value=bool(row["enabled"]), key=f"src_tog_{name}",
                         label_visibility="collapsed")
        if on != bool(row["enabled"]):
            news_sources.set_enabled(name, on)


def load_daily_summaries(limit: int = 10) -> pd.DataFrame:
    """Recent end-of-day summaries — one row per trading day (from end_of_day.py)."""
    conn = connect_db()
    try:
        if not table_exists(conn, "daily_summaries"):
            return pd.DataFrame()
        return pd.read_sql_query(
            "SELECT day, n_trades, win_rate, avg_return, compounded, best, worst "
            "FROM daily_summaries ORDER BY day DESC LIMIT ?;",
            conn, params=(limit,),
        )
    finally:
        conn.close()


_OUTCOMES_CSS = """
<style>
.to-tbl{width:100%;border-collapse:collapse;font-size:0.9rem;margin:0.25rem 0 0.5rem;}
.to-tbl th{text-align:left;padding:6px 10px;border-bottom:2px solid rgba(128,128,128,.35);font-weight:600;white-space:nowrap;}
.to-tbl td{padding:6px 10px;border-bottom:1px solid rgba(128,128,128,.18);white-space:nowrap;}
.to-info{position:relative;cursor:help;user-select:none;}
.to-info .to-tip{visibility:hidden;opacity:0;transition:opacity .1s;position:absolute;right:0;top:150%;z-index:9999;
  width:340px;max-width:70vw;padding:10px 12px;border-radius:8px;background:#20232b;color:#e8e8e8;
  border:1px solid rgba(255,255,255,.18);box-shadow:0 6px 22px rgba(0,0,0,.5);font-size:.82rem;
  line-height:1.45;white-space:normal;text-align:left;font-weight:400;}
.to-info:hover .to-tip{visibility:visible;opacity:1;}
.to-pos{color:#16a34a;font-weight:600;} .to-neg{color:#dc2626;font-weight:600;}
</style>
"""


def _ret_html(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    cls = "to-pos" if v > 0 else ("to-neg" if v < 0 else "")
    return f'<span class="{cls}">{v:+.1f}%</span>'


def _fmt_ts(v) -> str:
    return html.escape(str(v)[:16].replace("T", " ")) if v else "—"


def _info_html(reason_in, news_published, news_detected=None) -> str:
    """Regular (single-entry) info button — symbol ⓘ. Reason-in headline, when the
    news was published, and when the app detected it. Styled popup only."""
    ri = html.escape(str(reason_in) if reason_in else "—")
    tip = (f'<span class="to-tip"><b>Reason in</b><br>{ri}<br><br>'
           f'<b>News published:</b> {_fmt_ts(news_published)}<br>'
           f'<b>News detected:</b> {_fmt_ts(news_detected)}</span>')
    return f'<span class="to-info">ⓘ{tip}</span>'


def _info_html_reentry(first: dict) -> str:
    """Re-entry info button — symbol ℹ️ — for a ticker that entered 2+ times the same
    day. Shows the FIRST entry's context: its reason-in news, the news publish and
    detect times, and the first entry/exit times."""
    ri = html.escape(str(first.get("entry_reason")) if first.get("entry_reason") else "—")
    tip = (f'<span class="to-tip"><b>First entry — reason in</b><br>{ri}<br><br>'
           f'<b>News published:</b> {_fmt_ts(first.get("news_published"))}<br>'
           f'<b>News detected:</b> {_fmt_ts(first.get("news_detected"))}<br>'
           f'<b>First entry:</b> {_fmt_ts(first.get("entry_time"))}<br>'
           f'<b>First exit:</b> {_fmt_ts(first.get("exit_time"))}</span>')
    return f'<span class="to-info">ℹ️{tip}</span>'


def _fmt_time(v) -> str:
    return html.escape(str(v)[:16].replace("T", " ")) if v else "—"


def _outcomes_table(headers: list[str], rows_cells: list[list[str]]) -> str:
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                   for cells in rows_cells)
    return (_OUTCOMES_CSS
            + f'<table class="to-tbl"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>')


def render_accuracy() -> None:
    """Alert-accuracy view with an adjustable look-back (minutes/hours/days, max
    7 days). Did the app's alerts actually go up? Paper returns, small samples."""
    # --- End-of-day daily summaries: "if you traded every signal, how did the day go?" ---
    days = load_daily_summaries()
    if not days.empty:
        st.markdown("#### 📅 Daily summary — traded every signal, flat overnight")
        latest = days.iloc[0]
        cA = st.columns(4)
        cA[0].metric(f"{latest['day']} · trades", int(latest["n_trades"]))
        cA[1].metric("Win rate", f"{latest['win_rate']:.0f}%" if pd.notna(latest["win_rate"]) else "—")
        cA[2].metric("Avg / trade", f"{latest['avg_return']:+.1f}%" if pd.notna(latest["avg_return"]) else "—")
        cA[3].metric("Compounded", f"{latest['compounded']:+.1f}%" if pd.notna(latest["compounded"]) else "—")
        if len(days) > 1:
            hist = days.rename(columns={
                "day": "Day", "n_trades": "Trades", "win_rate": "Win %",
                "avg_return": "Avg %", "compounded": "Compounded %", "best": "Best", "worst": "Worst"})
            st.dataframe(
                hist[["Day", "Trades", "Win %", "Avg %", "Compounded %", "Best", "Worst"]],
                width="stretch", hide_index=True,
                column_config={
                    "Win %": st.column_config.NumberColumn(format="%.0f%%"),
                    "Avg %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Compounded %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
        st.caption("All positions are force-closed at session end — nothing is held overnight.")
        st.divider()

    st.caption("Did the app's alerts actually go up? **Paper** returns: price at alert → live price now. "
               "Not financial advice; small samples — read as a sanity check.")

    c = st.columns([1, 2])
    unit = c[0].selectbox("Unit", ["Minutes", "Hours", "Days"], index=2, key="acc_unit")
    if unit == "Minutes":
        minutes = c[1].slider("Look back (minutes)", 5, 7 * 24 * 60, 60, step=5, key="acc_min")
    elif unit == "Hours":
        minutes = c[1].slider("Look back (hours)", 1, 168, 24, step=1, key="acc_hr") * 60
    else:
        minutes = c[1].slider("Look back (days)", 1, 7, 1, step=1, key="acc_day") * 1440
    hit = float(st.slider("‘Pop’ threshold — count as a hit if it reached +X% (peak)",
                          1, 50, 5, step=1, key="acc_hit"))

    r = accuracy_cached(int(minutes), hit)
    if r["n_alerts"] == 0:
        st.info(f"No alerts fired in the last {r['window_label']}.")
        return
    if r["n_scored"] == 0:
        st.warning(f"{r['n_alerts']} alert(s) fired but none scorable (missing entry/live prices).")
        return

    m = st.columns(4)
    m[0].metric("Alerts", r["n_alerts"], help=f"{r['n_distinct_tickers']} distinct tickers"
                + (f", {r['n_excluded']} excluded" if r["n_excluded"] else ""))
    m[1].metric("Win rate (up now)", f"{r['win_rate']:.0f}%")
    m[2].metric("Avg return", f"{r['avg_return']:+.1f}%")
    m[3].metric(f"Reached +{int(r['hit_pct'])}% (peak)", f"{r['hit_rate']:.0f}%")
    m2 = st.columns(3)
    m2[0].metric("Median return", f"{r['median_return']:+.1f}%")
    m2[1].metric("Best call", f"{r['best']['ticker']}", f"{r['best']['ret_pct']:+.1f}%")
    m2[2].metric("Worst call", f"{r['worst']['ticker']}", f"{r['worst']['ret_pct']:+.1f}%")

    dfa = pd.DataFrame(r["alerts"])
    dfa["when"] = dfa["notified_at"].str[:16].str.replace("T", " ", regex=False)
    dfa = dfa.rename(columns={
        "ticker": "Ticker", "entry": "Entry $", "now": "Now $",
        "ret_pct": "Δ% since", "peak_pct": "Peak %", "headline": "Headline",
    })
    st.dataframe(
        dfa[["when", "Ticker", "Entry $", "Now $", "Δ% since", "Peak %", "Headline"]],
        width="stretch", hide_index=True,
        column_config={
            "Entry $": st.column_config.NumberColumn(format="$%.2f"),
            "Now $": st.column_config.NumberColumn(format="$%.2f"),
            "Δ% since": st.column_config.NumberColumn(format="%.1f%%"),
            "Peak %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(f"{r['from']} .. {r['to']} · returns are 'to now' (span each alert's own age); "
               "the peak-based hit-rate is the age-robust 'did it pop' measure. Paper returns, no fees.")

    # --- Theoretical Outcomes: buy every alert, sell when the exit signal fires ---
    st.divider()
    st.markdown("#### 📊 Theoretical Outcomes")
    st.caption("Assumes you **buy at every alert** and **sell when the exit signal fires** "
               "(trailing stop or profit target) — completed round-trips from the position tracker. "
               "Paper, equal-weight, no fees/slippage.")
    st.caption("Info button: **ⓘ** = single entry that day · **ℹ️** = ticker **re-entered the same "
               "day** — its ℹ️ tooltip shows the *first* entry's news, publish/detect times, and first exit.")
    s = strategy_cached(int(minutes))
    if s["closed"] == 0:
        st.info(f"No completed round-trip trades entered in the last {s['window_label']}"
                + (f" — {s['open']} still open." if s["open"] else "."))
    else:
        m = st.columns(4)
        m[0].metric("Closed trades", s["closed"], help=f"{s['open']} still open")
        m[1].metric("Win rate", f"{s['win_rate']:.0f}%")
        m[2].metric("Avg return / trade", f"{s['avg_return']:+.1f}%")
        m[3].metric("Compounded", f"{s['compounded_return']:+.1f}%",
                    help="If you rolled the full bankroll trade-by-trade (sequential).")
        m2 = st.columns(3)
        m2[0].metric("Median / trade", f"{s['median_return']:+.1f}%")
        m2[1].metric("Avg win", f"{s['avg_win']:+.1f}%" if s["avg_win"] is not None else "—")
        m2[2].metric("Avg loss", f"{s['avg_loss']:+.1f}%" if s["avg_loss"] is not None else "—")
        st.caption(
            f"Best: **{s['best']['ticker']} {s['best']['ret_pct']:+.0f}%** · "
            f"Worst: **{s['worst']['ticker']} {s['worst']['ret_pct']:+.0f}%**"
            + (f" · plus **{s['open']} open** trade(s), avg unrealized "
               f"**{s['open_unrealized_avg']:+.1f}%**" if s["open"] and s["open_unrealized_avg"] is not None else "")
        )
        # Flag same-day re-entries: group trades by (ticker, entry-date) and find the
        # first entry of each. A ticker with 2+ entries the same day gets the ℹ️ symbol
        # and its tooltip points back to that first entry; single entries get ⓘ.
        by_day = defaultdict(list)
        for t in s["trades"]:
            by_day[(t.get("ticker"), str(t.get("entry_time") or "")[:10])].append(t)
        first_of = {k: min(g, key=lambda x: str(x.get("entry_time") or "")) for k, g in by_day.items()}

        rows_cells = []
        for t in s["trades"]:
            key = (t.get("ticker"), str(t.get("entry_time") or "")[:10])
            if len(by_day[key]) >= 2:          # re-entered the same day
                info = _info_html_reentry(first_of[key])
            else:
                info = _info_html(t.get("entry_reason"), t.get("news_published"), t.get("news_detected"))
            rows_cells.append([
                html.escape(str(t.get("ticker", ""))),
                _fmt_time(t.get("entry_time")),
                _fmt_time(t.get("exit_time")),
                f'${t["entry"]:.2f}',
                f'${t["exit"]:.2f}',
                _ret_html(t.get("ret_pct")),
                html.escape(str(t.get("exit_reason") or "—")),
                info,
            ])
        st.markdown(
            _outcomes_table(
                ["Ticker", "Entered", "Exited", "Entry $", "Exit $", "Return %", "Reason out", "Info"],
                rows_cells),
            unsafe_allow_html=True)

    # --- Open positions (live, unrealized) — the still-running theoretical trades ---
    st.markdown("##### 📌 Open positions (live)")
    open_df = load_open_positions()
    if open_df.empty:
        st.caption("No open positions right now.")
    else:
        rows_cells = []
        for _, r in open_df.iterrows():
            cur = r.get("current")
            rows_cells.append([
                html.escape(str(r.get("ticker", ""))),
                _fmt_time(r.get("entry_time")),
                f'${float(r["entry_price"]):.2f}' if pd.notna(r.get("entry_price")) else "—",
                f'${float(cur):.2f}' if pd.notna(cur) else "—",
                _ret_html(r.get("ret_pct")),
                "Still open",
                _info_html(r.get("entry_headline"), r.get("news_published"), r.get("news_detected")),
            ])
        st.markdown(
            _outcomes_table(
                ["Ticker", "Entered", "Entry $", "Current $", "Return %", "Reason out", "ℹ️"],
                rows_cells),
            unsafe_allow_html=True)
        avg = open_df["ret_pct"].dropna().mean()
        head = (f"{len(open_df)} open · avg unrealized **{avg:+.1f}%**"
                if pd.notna(avg) else f"{len(open_df)} open")
        st.caption(f"{head} · {_market_status()}")

    # --- Would-have-fired: qualifying catalysts not (yet) traded — typically detected
    #     off-hours, when the trader doesn't run. Shows what you'd have entered. ---
    st.markdown("##### 🕒 Would-fire signals (detected, not traded)")
    wf = would_fire_cached()
    if wf.empty:
        st.caption("Nothing pending — every qualifying catalyst has been acted on (or none right now).")
    else:
        rows_cells = []
        for _, r in wf.iterrows():
            rows_cells.append([
                html.escape(str(r.get("ticker", ""))),
                _fmt_time(r.get("collect_time") or r.get("publish_time")),
                _info_html(r.get("title_clean"), r.get("publish_time"), r.get("collect_time")),
            ])
        st.markdown(_outcomes_table(["Ticker", "Would fire at", "ℹ️"], rows_cells), unsafe_allow_html=True)
        st.caption("Met all alert criteria but weren't traded — usually detected **outside market "
                   "hours**. They fire when the market opens (if still fresh and up).")


@st.fragment(run_every=30)
def render_et_clock() -> None:
    """Live Eastern-time clock for the sidebar. Every timestamp in the app is ET, so
    this is the reference — handy when you're viewing from another timezone."""
    et = datetime.now(NY_TZ)
    h12 = et.hour % 12 or 12
    ampm = "AM" if et.hour < 12 else "PM"
    st.caption(f"🕐 **{h12}:{et.minute:02d} {ampm} ET** · {et.strftime('%a %b %d')} — "
               f"all times shown are Eastern")


def main() -> None:
    st.set_page_config(page_title="Ticker Radar", page_icon="📡", layout="wide")
    st.title("📡 Ticker Radar")
    st.caption("Growing tickers with positive news and crowd buzz — your decision board.")

    with st.sidebar:
        st.header("Settings")
        render_et_clock()
        # Market-cap band for the screener universe — swaps the Finviz cap_ filter live.
        _CAP_LABELS = {"nano": "Nano", "micro": "Micro", "small": "Small", "mid": "Mid", "large": "Large"}
        _cur_cap = str(app_settings.get("market_cap", "nano")).strip().lower()
        if _cur_cap not in _CAP_LABELS:
            _cur_cap = "nano"
        _cap_pick = st.selectbox("📊 Market cap (screener universe)", list(_CAP_LABELS.values()),
                                 index=list(_CAP_LABELS).index(_cur_cap), key="market_cap_sel")
        _new_cap = next(k for k, v in _CAP_LABELS.items() if v == _cap_pick)
        if _new_cap != _cur_cap:
            app_settings.set("market_cap", _new_cap)
            st.caption("Takes effect on the next universe refresh (~4 min).")

        auto = st.checkbox("Auto-refresh", value=True)
        interval = st.slider("Refresh every (seconds)", 15, 300, 60, step=15)
        st.divider()
        min_score = st.slider("Min combined score", 0.0, 1.0, 0.0, step=0.05)
        alerts_only = st.checkbox("Alert candidates only 🔥", value=False)
        top_n = st.slider("Show top N", 5, 50, 20, step=5)
        fresh_days = st.slider(
            "News freshness (days)", 1, 30, 3,
            help="Hide news older than this from the board (kept in the DB; view it in Archived news below).",
        )

        st.divider()
        st.caption("📡 Density rolling window (applies on the next worker cycle)")
        cur_win = app_settings.get_int("density_window_min", None) or int(os.getenv("DENSITY_WINDOW_MIN", "30"))
        win = st.slider("Window (min)", 10, 960, int(min(max(cur_win, 10), 960)), step=5)
        if win != cur_win:
            app_settings.set("density_window_min", win)
        cur_base = app_settings.get_int("density_baseline_min", None) or int(os.getenv("DENSITY_BASELINE_MIN", "480"))
        base = st.slider("Baseline (min)", 30, 2880, int(min(max(cur_base, 30), 2880)), step=30)
        if base != cur_base:
            app_settings.set("density_baseline_min", base)

        st.divider()
        st.caption("🔤 News keyword filter (one per line; empty = show all). Gates alerts.")
        kw_text = st.text_area(
            "Keywords",
            value="\n".join(news_filter.get_keywords()),
            height=150,
            label_visibility="collapsed",
        )
        if st.button("💾 Save keywords", width="stretch"):
            news_filter.set_keywords(kw_text)
            n = len([k for k in kw_text.splitlines() if k.strip()])
            st.success(f"Saved {n} keyword(s)." if n else "Cleared — filter off (showing all).")
        # High-sentiment bypass: let a strongly-positive headline alert even with no
        # keyword match (catches narrative-catalyst movers like AIxC / INBS).
        _kb_en = str(app_settings.get("keyword_bypass_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
        _kb = st.toggle("⚡ High-sentiment bypass — alert on strong headlines even without a keyword",
                        value=_kb_en, key="kw_bypass")
        if _kb != _kb_en:
            app_settings.set("keyword_bypass_enabled", "true" if _kb else "false")
        if _kb:
            try:
                _kb_sc = float(app_settings.get("keyword_bypass_score", "0.8"))
            except (TypeError, ValueError):
                _kb_sc = 0.8
            _kb_new = st.slider("Bypass when sentiment ≥", 0.5, 1.0, _kb_sc, step=0.05, key="kw_bypass_score")
            if abs(_kb_new - _kb_sc) > 1e-9:
                app_settings.set("keyword_bypass_score", str(_kb_new))
        board_kw_filter = st.checkbox("Filter cards by keywords too", value=False)

        st.divider()
        st.caption("🚀 Entry momentum confirmation")
        _mg_en = str(app_settings.get("momentum_gate_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
        _mg = st.toggle("Only enter if price ≥ its SMA (rising, not fading from a higher open)",
                        value=_mg_en, key="mom_gate")
        if _mg != _mg_en:
            app_settings.set("momentum_gate_enabled", "true" if _mg else "false")

        st.divider()
        st.caption("🚨 Alert rule")
        _cur_major = str(app_settings.get("alert_major_neutral", "false")).strip().lower() in ("1", "true", "yes")
        _opt_a = st.toggle(
            "Alert on MAJOR news even if neutral (Option A)", value=_cur_major,
            help="ON: direction-ambiguous major catalysts (mergers, buyouts) alert even at neutral "
                 "sentiment — negative major news is still excluded. OFF (default): positive-only "
                 "(sentiment ≥ 0.5). Applies on the worker's next cycle.",
        )
        if _opt_a != _cur_major:
            app_settings.set("alert_major_neutral", "true" if _opt_a else "false")
        st.caption("**A** = major news alerts even if neutral · **B** = positive-only")

        st.divider()
        st.caption("🕐 Trading session — when the app screens & trades (ET)")
        _SESSIONS = {
            "Regular only (9:30 AM–4 PM)": (570, 960),
            "Regular + after-hours (9:30 AM–8 PM)": (570, 1200),
            "Full extended (4 AM–8 PM)": (240, 1200),
            "24/7 (all hours)": (0, 1440),
        }
        try:
            _cs = int(app_settings.get("session_start_min", "570"))
            _ce = int(app_settings.get("session_end_min", "1200"))
        except (TypeError, ValueError):
            _cs, _ce = 570, 1200
        _labels = list(_SESSIONS)
        _cur = next((k for k, v in _SESSIONS.items() if v == (_cs, _ce)), _labels[1])
        _pick = st.selectbox("Session window", _labels, index=_labels.index(_cur),
                             key="session_window", label_visibility="collapsed")
        if _SESSIONS[_pick] != (_cs, _ce):
            app_settings.set("session_start_min", str(_SESSIONS[_pick][0]))
            app_settings.set("session_end_min", str(_SESSIONS[_pick][1]))
        st.caption("Overnight (8 PM–4 AM) stays off — no real trading, unreliable data. "
                   "Takes effect within a cycle; no restart needed.")

        st.divider()
        st.caption("🎚️ Exit stop mode")
        _em = "profit" if str(app_settings.get("exit_mode", "price")).strip().lower() == "profit" else "price"
        _choice = st.radio(
            "Trailing stop basis", ["Price-based", "Profit-based"],
            index=(1 if _em == "profit" else 0), key="exit_mode_radio", label_visibility="collapsed",
            captions=["Exit 30% below the peak **price**",
                      "Give back 30% of the peak **gain** — cuts losers fast (exits on any drop below entry before a gain)"],
        )
        _new_em = "profit" if _choice == "Profit-based" else "price"
        if _new_em != _em:
            app_settings.set("exit_mode", _new_em)
        try:
            _cur_grace = int(app_settings.get("position_grace_sec", "90"))
        except (TypeError, ValueError):
            _cur_grace = 90
        _grace_sec_ui = st.slider("Grace period (sec) — no stop-out right after entry", 0, 300,
                                  _cur_grace, step=30, key="grace_sec")
        if _grace_sec_ui != _cur_grace:
            app_settings.set("position_grace_sec", str(_grace_sec_ui))

        # SMA noise filter (applies to both exit modes).
        _sma_en = str(app_settings.get("exit_sma_enabled", "true")).strip().lower() in ("1", "true", "yes", "on")
        _sma = st.toggle("🌊 Smooth exits with an SMA — ignore single-tick noise",
                         value=_sma_en, key="sma_en")
        if _sma != _sma_en:
            app_settings.set("exit_sma_enabled", "true" if _sma else "false")
        if _sma:
            try:
                _sma_cur = int(float(app_settings.get("exit_sma_minutes", "10")))
            except (TypeError, ValueError):
                _sma_cur = 10
            _sma_new = st.slider("SMA window (min) — bigger = smoother but slower to react", 2, 30,
                                 _sma_cur, step=1, key="sma_min")
            if _sma_new != _sma_cur:
                app_settings.set("exit_sma_minutes", str(_sma_new))

        # Catastrophe backstop (raw price; covers the SMA's lag on a real crash).
        _cat_en = str(app_settings.get("catastrophe_stop_enabled", "true")).strip().lower() in ("1", "true", "yes", "on")
        _cat = st.toggle("🚨 Catastrophe stop — hard exit on a real crash (raw price)",
                         value=_cat_en, key="cat_en")
        if _cat != _cat_en:
            app_settings.set("catastrophe_stop_enabled", "true" if _cat else "false")
        if _cat:
            try:
                _cat_cur = int(float(app_settings.get("catastrophe_stop_pct", "20")))
            except (TypeError, ValueError):
                _cat_cur = 20
            _cat_new = st.slider("Exit immediately if raw price falls X% below peak", 10, 40,
                                 _cat_cur, step=1, key="cat_pct")
            if _cat_new != _cat_cur:
                app_settings.set("catastrophe_stop_pct", str(_cat_new))

        # Breakeven stop + price-based re-entry (applies to both exit modes).
        _be_en = str(app_settings.get("breakeven_stop_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
        _be = st.toggle("🛡️ Breakeven stop — after a warmup, exit if price drops below entry",
                        value=_be_en, key="be_en")
        if _be != _be_en:
            app_settings.set("breakeven_stop_enabled", "true" if _be else "false")
        if _be:
            try:
                _bw = int(float(app_settings.get("breakeven_warmup_cycles", "5")))
            except (TypeError, ValueError):
                _bw = 5
            _bw_new = st.slider("Warmup — cycles of room before breakeven arms", 0, 20, _bw, step=1, key="be_warmup")
            if _bw_new != _bw:
                app_settings.set("breakeven_warmup_cycles", str(_bw_new))
            try:
                _rc = int(float(app_settings.get("reentry_cap", "2")))
            except (TypeError, ValueError):
                _rc = 2
            _rc_new = st.slider("Max re-entries/ticker/day (when price reclaims entry)", 0, 5, _rc, step=1, key="reentry_cap")
            if _rc_new != _rc:
                app_settings.set("reentry_cap", str(_rc_new))

        st.divider()
        st.caption("🎯 Profit-target exit (optional) — complements the 30% trailing stop")
        _pt_en = str(app_settings.get("profit_target_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
        _pt = st.toggle("Signal a sell when up +X% from entry", value=_pt_en, key="pt_en")
        if _pt != _pt_en:
            app_settings.set("profit_target_enabled", "true" if _pt else "false")
        if _pt:
            try:
                cur_pct = int(float(app_settings.get("profit_target_pct", "50")))
            except (TypeError, ValueError):
                cur_pct = 50
            new_pct = st.slider("Target % gain from entry", 10, 300, cur_pct, step=5, key="pt_pct")
            if new_pct != cur_pct:
                app_settings.set("profit_target_pct", str(new_pct))

        st.divider()
        st.caption("🔔 Alert delivery — sound & email are independent")
        _snd = str(app_settings.get("alert_sound_enabled", "true")).strip().lower() in ("1", "true", "yes", "on")
        _s = st.toggle("🔊 Sound alert", value=_snd, key="al_sound")
        if _s != _snd:
            app_settings.set("alert_sound_enabled", "true" if _s else "false")
        _eml = str(app_settings.get("alert_email_enabled", "false")).strip().lower() in ("1", "true", "yes", "on")
        _e = st.toggle("📧 Email alert", value=_eml, key="al_email")
        if _e != _eml:
            app_settings.set("alert_email_enabled", "true" if _e else "false")
        _addr = str(app_settings.get("alert_email", "") or "")
        _na = st.text_input("Email address", value=_addr, key="al_email_addr", placeholder="you@gmail.com")
        if _na.strip() != _addr.strip():
            app_settings.set("alert_email", _na.strip())
        if _e:
            st.caption("Needs a Gmail **App Password** in `.env` (`ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD`). "
                       "Test with `python src/notify_alerts.py --test-email`.")

        st.divider()
        if st.button("🔄 Refresh now", width="stretch"):
            st.rerun()

    run_every = interval if auto else None

    @st.fragment(run_every=run_every)
    def render_board():
        # 🛑 Exit signals first — recently-stopped positions (the tickers to sell).
        # Shown at the very top because a stopped ticker often leaves the screener,
        # so it won't appear among the ranked cards below.
        exits = load_recent_exits()
        if not exits.empty:
            items = "  ·  ".join(
                f"**{r['ticker']}** → sell ~${r['exit_price']:.2f} ({r['ret_pct']:+.1f}%, {str(r['exit_time'])[11:16]})"
                for _, r in exits.iterrows()
            )
            st.error(f"🛑 **EXIT SIGNAL — sell these** (trailing stop / target fired in the last "
                     f"{EXIT_BANNER_HOURS}h):  {items}")

        updated = last_updated()
        df = load_decision_data()

        if df.empty:
            st.info("No rankings yet. Run the pipeline (`run_pipeline.py`) or start the background app.")
            return

        # Freshness gate: news older than the cutoff is hidden from the board
        # (professor's "3 days" rule) but retained in the DB for the archive view.
        fresh_cutoff = (datetime.now(NY_TZ) - timedelta(days=fresh_days)).isoformat()
        df, n_stale = _suppress_stale_news(df, fresh_cutoff)

        tab_pick, tab_board, tab_acc, tab_src = st.tabs(
            ["🎯 Top Pick", "📋 Board", "📊 Accuracy", "🗞️ Sources"])

        with tab_pick:
            render_recommendation(df, updated)

        with tab_acc:
            render_accuracy()

        with tab_src:
            render_sources()

        with tab_board:
            view = df[df["combined_score"] >= min_score]
            if alerts_only:
                view = view[view["alert_candidate"] == 1]
            if board_kw_filter:
                kws = news_filter.get_keywords()
                if kws:
                    view = view[view["latest_headline"].fillna("").map(lambda t: news_filter.matches(t, kws))]
            view = view.head(top_n)

            n_alerts = int((df["alert_candidate"] == 1).sum())
            started = app_started_today()
            cols = st.columns(4)
            cols[0].metric("Tickers ranked", len(df))
            cols[1].metric("🔥 Alert candidates", n_alerts)
            cols[2].metric("Last updated", (updated or "—")[:19].replace("T", " "))
            cols[3].metric("App started today", started[11:16] if started else "— (not run)")
            st.divider()

            if view.empty:
                st.warning("No tickers match the current filters.")
            else:
                st.subheader("📋 Screener")
                render_screener_table(view)
                st.subheader("Details")
                for _, row in view.iterrows():
                    render_card(row)

            # ---- Alerts log: review what fired and how it did ----
            st.divider()
            log = load_alert_log()
            with st.expander(f"🔔 Alerts log ({len(log)})", expanded=False):
                if log.empty:
                    st.caption("No alerts recorded yet.")
                else:
                    # Live per-ticker price (Yahoo) so "since alert" stays accurate
                    # even after a ticker drops off the Finviz screener.
                    live = live_prices_cached(tuple(sorted(log["ticker"].dropna().unique())))
                    show = log.copy()
                    show["when"] = show["notified_at"].str[:16].str.replace("T", " ", regex=False)
                    show["now"] = show["ticker"].map(live)
                    show["chg_pct"] = (show["now"] - show["price_at_alert"]) / show["price_at_alert"] * 100
                    show["chg_usd"] = show["now"] - show["price_at_alert"]
                    show = show.rename(columns={
                        "ticker": "Ticker", "price_at_alert": "Entry $",
                        "now": "Now $", "chg_pct": "Δ% since", "chg_usd": "Δ$ since",
                        "headline": "Headline",
                    })
                    st.dataframe(
                        show[["when", "Ticker", "Entry $", "Now $", "Δ% since", "Δ$ since", "Headline"]],
                        width="stretch", hide_index=True,
                        column_config={
                            "Entry $": st.column_config.NumberColumn(format="$%.2f"),
                            "Now $": st.column_config.NumberColumn(format="$%.2f"),
                            "Δ% since": st.column_config.NumberColumn(format="%.1f%%"),
                            "Δ$ since": st.column_config.NumberColumn(format="$%.2f"),
                        },
                    )
                    st.caption("**Entry $** = price when the alert fired. **Now $** is a live Yahoo quote "
                               "(regular / pre / post-market), refreshed ~60s — so Δ-since-alert stays accurate "
                               "even after the ticker leaves the screener. Blank 'Now' = Yahoo had no quote.")

            # ---- All news feed: every ingested headline, any ticker or none ----
            with st.expander("📰 All news feed (last 3 days — every source, growing or not)", expanded=False):
                feed_search = st.text_input(
                    "Filter (ticker, keyword, or source)", key="feed_search",
                    placeholder="e.g. FDA, acquisition, AAPL, GlobeNewswire",
                )
                feed = load_all_news(days=3, limit=500, search=feed_search)
                st.caption(f"{len(feed)} headline(s)"
                           + (f" matching “{feed_search}”" if feed_search else " in the last 3 days")
                           + " · capped at 500")
                if feed.empty:
                    st.info("No news in the window.")
                else:
                    show = feed.copy()
                    show["when_t"] = show["when_t"].str[:16].str.replace("T", " ", regex=False)
                    show = show.rename(columns={
                        "when_t": "Published", "tickers": "Ticker(s)",
                        "source": "Source", "title": "Headline", "url": "Link",
                    })
                    st.dataframe(
                        show[["Published", "Ticker(s)", "Source", "Headline", "Link"]],
                        width="stretch", hide_index=True,
                        column_config={"Link": st.column_config.LinkColumn("Link", display_text="open ↗")},
                    )

            # ---- Archived news: older than the freshness cutoff; kept, not deleted ----
            with st.expander(
                f"📁 Archived news — older than {fresh_days} day(s) "
                f"({n_stale} ticker(s) had news hidden from the board)",
                expanded=False,
            ):
                arch = load_archived_news(fresh_cutoff)
                if arch.empty:
                    st.caption("No archived news yet.")
                else:
                    show = arch.copy()
                    show["when_utc"] = show["when_utc"].str[:16].str.replace("T", " ", regex=False)
                    show = show.rename(columns={
                        "when_utc": "Published", "ticker": "Ticker",
                        "title": "Headline", "source": "Source",
                    })
                    st.caption(f"{len(show)} record(s) retained in the database "
                               f"(newest-first, capped at 300). Nothing is deleted.")
                    st.dataframe(
                        show[["Published", "Ticker", "Source", "Headline"]],
                        width="stretch", hide_index=True,
                    )

            # ---- Positions & trailing stops ----
            positions = load_positions()
            n_open = int((positions["status"] == "open").sum()) if len(positions) else 0
            with st.expander(f"🎯 Positions & trailing stops ({n_open} open)", expanded=(n_open > 0)):
                if positions.empty:
                    st.caption("No positions yet — one opens automatically when a ticker alerts, then the "
                               "trailing stop follows its peak.")
                else:
                    show = positions.rename(columns={
                        "ticker": "Ticker", "status": "Status", "entry_price": "Entry $",
                        "peak_price": "Peak $", "last_price": "Now $", "stop_price": "Stop $",
                        "pl_pct": "P/L %", "from_peak_pct": "From peak %",
                    })
                    st.dataframe(
                        show[["Ticker", "Status", "Entry $", "Peak $", "Now $", "Stop $", "P/L %", "From peak %"]],
                        width="stretch", hide_index=True,
                        column_config={
                            c: st.column_config.NumberColumn(format="%.2f")
                            for c in ["Entry $", "Peak $", "Now $", "Stop $"]
                        } | {
                            "P/L %": st.column_config.NumberColumn(format="%.1f%%"),
                            "From peak %": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                    )
                    trail = int((positions["trail_pct"].dropna().iloc[0] * 100)) if positions["trail_pct"].notna().any() else 30
                    st.caption(f"Stop = peak × (1 − {trail}%). 'From peak %' near −{trail}% means the trailing "
                               f"stop is about to trigger; 'stopped' rows already exited.")

    render_board()


if __name__ == "__main__":
    main()
