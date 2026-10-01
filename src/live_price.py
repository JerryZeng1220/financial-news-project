"""
Fresh per-ticker prices from Yahoo Finance (free, no key), independent of the
Finviz screener.

Why this exists: alert "price since entry" used to rely on the Finviz screener
price, which only updates while a ticker is ON the screener — so for a nano-cap
that pops and drops off, the number froze. Yahoo gives a live per-ticker quote
(regular, plus pre/post-market when in session) so the difference-since-alert
stays accurate.

Fetched with curl_cffi (browser impersonation) to avoid bot-blocking. Best
effort: a ticker that fails or is unknown is simply omitted from the result.
"""
from __future__ import annotations

from curl_cffi import requests as cr

_IMPERSONATE = "chrome"
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?interval=1m&range=1d"


def _fetch_chart(ticker: str) -> dict | None:
    """Raw Yahoo chart result for one ticker (meta + 1-min bars), or None."""
    t = (ticker or "").strip().upper()
    if not t:
        return None
    try:
        r = cr.get(_CHART_URL.format(t=t), impersonate=_IMPERSONATE, timeout=10)
        return r.json()["chart"]["result"][0]
    except Exception:
        return None


def _price_from_meta(meta: dict) -> float | None:
    """Latest tradeable price from a chart's meta: post/pre-market when in that
    session, otherwise the regular-session price."""
    for key in ("postMarketPrice", "preMarketPrice", "regularMarketPrice"):
        val = meta.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _sma_from_result(result: dict, minutes: int) -> float | None:
    """Simple moving average of the last `minutes` 1-min closes, or None if there
    aren't enough bars to be worth smoothing (fall back to the raw price then)."""
    try:
        closes = result["indicators"]["quote"][0].get("close") or []
    except (KeyError, IndexError, TypeError):
        return None
    vals = [float(c) for c in closes if c is not None]
    if len(vals) < 2:
        return None  # nothing to smooth — caller falls back to the raw price
    window = vals[-max(1, int(minutes)):]
    return sum(window) / len(window)


def fetch_live_price(ticker: str) -> float | None:
    """Latest tradeable price for one ticker, or None if unavailable."""
    res = _fetch_chart(ticker)
    return _price_from_meta(res.get("meta") or {}) if res else None


def fetch_live_and_sma(tickers, sma_minutes: int) -> dict[str, dict]:
    """{TICKER: {'raw': float|None, 'sma': float|None}} in a single chart call
    per ticker. 'raw' is the latest quote; 'sma' is the N-minute simple moving
    average of the 1-min closes (None when there aren't enough bars)."""
    out: dict[str, dict] = {}
    seen = set()
    for raw in tickers:
        t = (raw or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        res = _fetch_chart(t)
        if res is None:
            out[t] = {"raw": None, "sma": None}
            continue
        out[t] = {"raw": _price_from_meta(res.get("meta") or {}),
                  "sma": _sma_from_result(res, sma_minutes)}
    return out


def fetch_live_prices(tickers) -> dict[str, float]:
    """{TICKER: price} for the tickers we could fetch (others omitted)."""
    out: dict[str, float] = {}
    seen = set()
    for raw in tickers:
        t = (raw or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        px = fetch_live_price(t)
        if px is not None:
            out[t] = px
    return out
