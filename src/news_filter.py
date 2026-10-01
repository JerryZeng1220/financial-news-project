"""
User-editable news keyword filter.

A whitelist of catalyst keywords (edited from the dashboard). When non-empty,
only news whose text contains one of the keywords passes — used to gate alerts
and, optionally, the dashboard cards. Empty list = filter OFF (everything passes).

Matching is case-insensitive and word-boundary based, with automatic plural/verb
forms, so:
  - "Earning" matches "earnings"; "Beat" matches "beats"/"beating"
  - but "SEC" does NOT match "second" (word-boundary). Watch the auto -ing/-es
    suffixes, though: use "Missed"/"Misses" (not "Miss", which would hit "missing")
    and avoid short stems like "SPAC" (which would hit "spaces"/"spacing").

Keywords are stored in app_settings under 'news_keywords' (one per line). Until
you edit them, the DEFAULT_KEYWORDS below are active.
"""
from __future__ import annotations

import re

import app_settings

SETTING_KEY = "news_keywords"

DEFAULT_KEYWORDS = [
    "Acquisition", "Buyback", "Contract", "Listing", "Delisting", "Compliance",
    "IPO", "Merger", "Partnership", "Earning", "Beat", "Missed", "Misses",
    "Investigation", "Probe", "FDA", "Approval", "SEC", "Bankruptcy",
]


def get_keywords() -> list[str]:
    """Current keyword list: the saved setting, or DEFAULT_KEYWORDS until edited."""
    raw = app_settings.get(SETTING_KEY, None)
    if raw is None:
        return list(DEFAULT_KEYWORDS)          # not set yet -> defaults active
    # Saved (even ""), so honor exactly what the user has: "" -> [] -> filter off.
    return [k.strip() for k in raw.replace(",", "\n").splitlines() if k.strip()]


def set_keywords(text: str) -> None:
    app_settings.set(SETTING_KEY, text)


def _compile(keywords: list[str]) -> re.Pattern | None:
    if not keywords:
        return None
    # \b(?:kw1|kw2|...)(?:es|s|ed|ing)?\b  — whole-word with common suffixes
    # ("es" before "s" so "misses" matches as miss+es, not miss+<fail>).
    body = "|".join(re.escape(k) for k in keywords)
    return re.compile(r"\b(?:" + body + r")(?:es|s|ed|ing)?\b", re.IGNORECASE)


def matches(text: str, keywords: list[str] | None = None) -> bool:
    """True if the filter is off (no keywords) or the text contains one."""
    if keywords is None:
        keywords = get_keywords()
    if not keywords:
        return True
    rx = _compile(keywords)
    return bool(rx.search(text or ""))
