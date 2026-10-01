"""
LLM-based financial-news sentiment classifier (provider-agnostic, BATCHED).

Purpose
-------
A more accurate replacement for the rule-based `classify_sentiment_rule_based()`
in sentiment_rules.py. Each classification has the same shape:

    {
        "sentiment_label": "positive" | "negative" | "neutral" | "mixed",
        "sentiment_score": float,    # -1.0 .. +1.0
        "confidence_score": float,   #  0.0 .. 1.0
        "reason": str,
    }

so the rest of the pipeline (ticker links, combined ranking, alerts) is
unchanged.

Why batched
-----------
The Gemini free tier rate-limits requests (e.g. ~5 requests/minute for
gemini-2.5-flash). Sending one headline per request burns through that fast.
Instead we send MANY headlines in a SINGLE request and get all the labels back
at once — typically one API call per pipeline cycle. Each input item carries a
`clean_news_id`, and the model echoes it back, so results map reliably to rows
(we never rely on list order). Large sets are split into chunks of
GEMINI_BATCH_SIZE to stay within token limits.

Provider-agnostic by design
----------------------------
The actual model call is isolated in `_score_batch_with_gemini()`. To switch
providers later (Ollama for fully-offline, or the Claude API for production),
add another `_score_batch_with_<backend>()` and a branch in
`classify_sentiments_llm()` — nothing else in the project changes.

Configuration (.env / environment)
----------------------------------
    LLM_SENTIMENT_BACKEND   "gemini" (default)
    GEMINI_API_KEY          your free Google AI Studio key
    GEMINI_MODEL            model id (default "gemini-2.5-flash")
    GEMINI_BATCH_SIZE       max items per request (default 25)
    GEMINI_MAX_RETRIES      retries on 429/503 (default 5)

Failures raise `LLMUnavailable` so the caller can fall back to the rule-based
scorer instead of crashing the background loop.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
BACKEND = os.getenv("LLM_SENTIMENT_BACKEND", "gemini").strip().lower()
BATCH_SIZE = int(os.getenv("GEMINI_BATCH_SIZE", "25"))
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
# Hard per-request timeout (ms) so a genuinely hung call fails fast instead of
# stalling the whole pipeline cycle.
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

# gemini-2.5-* models "think" before answering, which adds several seconds of
# latency. Sentiment labeling doesn't need it, so we disable it by default
# (thinking_budget=0). Set GEMINI_THINKING_BUDGET=-1 to OMIT the setting entirely
# (e.g. if you switch to a model like gemini-2.0-flash that doesn't accept it).
THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

# Stored in news_sentiment.model_version so LLM rows are distinguishable from
# the "rule_based_v1" rows.
LLM_MODEL_VERSION = f"{BACKEND}:{DEFAULT_GEMINI_MODEL}"

# Keep each summary short so a batch prompt stays well within token limits.
MAX_TEXT_CHARS = 500


class LLMUnavailable(RuntimeError):
    """Raised when the LLM backend can't be reached or returns nothing usable."""


# ----------------------------
# Structured output schema
# ----------------------------

class SentimentItem(BaseModel):
    """One classification, tagged with the id of the news row it belongs to."""
    id: int
    sentiment_label: Literal["positive", "negative", "neutral", "mixed"]
    sentiment_score: float        # -1.0 (very bearish) .. +1.0 (very bullish)
    confidence_score: float       # 0.0 .. 1.0
    materiality: Literal["major", "moderate", "routine"]  # catalyst strength
    reason: str                   # one short sentence


SYSTEM_INSTRUCTION = (
    "You are a financial news sentiment classifier. You are given a JSON array "
    "of news items, each with an id, a stock ticker, a headline, and a summary. "
    "For EACH item, judge the likely SHORT-TERM impact on that company's share "
    "price and return one classification per item.\n"
    "- Echo back the same id you were given for each item.\n"
    "- sentiment_label: 'positive' if likely to push the price up, 'negative' "
    "if down, 'neutral' if no clear directional impact, 'mixed' if there are "
    "both meaningful positive and negative signals.\n"
    "- sentiment_score: -1.0 (very bearish) to +1.0 (very bullish); 0.0 for "
    "neutral.\n"
    "- confidence_score: 0.0 to 1.0.\n"
    "- materiality: how MARKET-MOVING the event is, independent of direction:\n"
    "    'major'    = FDA approval/rejection/CRL, earnings beat/miss, guidance "
    "change, M&A/buyout/merger, major contract or deal, pivotal clinical results, "
    "bankruptcy/going-concern, uplisting/delisting.\n"
    "    'moderate' = partnerships, analyst upgrade/downgrade, a notable product "
    "launch, secondary/registered offering, smaller contracts.\n"
    "    'routine'  = product availability, conference/webcast/presentation, "
    "scheduling or date announcements, personnel appointments, marketing fluff.\n"
    "- reason: ONE short sentence.\n"
    "Judge from the perspective of an investor in the given ticker. Return exactly "
    "one object per input id."
)


# ----------------------------
# Helpers
# ----------------------------

def _clamp(value: float, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(low, min(high, v))


def _normalize(item: SentimentItem) -> dict:
    return {
        "sentiment_label": item.sentiment_label,
        "sentiment_score": _clamp(item.sentiment_score, -1.0, 1.0),
        "confidence_score": _clamp(item.confidence_score, 0.0, 1.0),
        "materiality": item.materiality,
        "reason": (item.reason or "").strip() or "No explanation provided.",
    }


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _retry_delay_seconds(error_text: str, attempt: int) -> float:
    """Use the server-suggested retryDelay if present, else exponential backoff."""
    m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", error_text)
    if m:
        return float(m.group(1)) + 1.0
    return min(2.0 ** attempt, 30.0)


def _is_retryable(error_text: str) -> bool:
    return any(tok in error_text for tok in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def _is_fatal_quota(error_text: str) -> bool:
    """A 429 that won't clear soon: a per-DAY request quota or depleted prepaid
    credits/billing. Retrying 5x with backoff just stalls the pipeline ~2 min,
    so we fail fast to the rule-based fallback (rows get flagged for LLM
    re-score once billing/quota is healthy). A per-MINUTE 429 is NOT fatal —
    it clears in ~a minute, so that path still retries."""
    t = error_text.lower()
    if "resource_exhausted" not in t and "429" not in t:
        return False
    return any(tok in t for tok in (
        "perday", "per day", "requestsperday",
        "prepayment credits", "credits are depleted", "billing",
    ))


# ----------------------------
# Gemini backend
# ----------------------------

_gemini_client = None  # lazily initialised, reused across calls


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMUnavailable("GEMINI_API_KEY is not set in the environment/.env")

    try:
        from google import genai  # google-genai package
    except ImportError as e:
        raise LLMUnavailable(
            "google-genai is not installed. Run: pip install google-genai"
        ) from e

    _gemini_client = genai.Client(
        api_key=api_key,
        http_options={"timeout": GEMINI_TIMEOUT_MS},
    )
    return _gemini_client


def _score_batch_with_gemini(batch: list[dict]) -> dict[int, dict]:
    """
    Classify one batch (<= BATCH_SIZE items) in a single request.
    Returns {clean_news_id: result_dict}. Ids missing from the model's response
    are simply absent (the caller can fall back for those).
    """
    client = _get_gemini_client()

    payload = [
        {
            "id": int(item["id"]),
            "ticker": item.get("ticker", ""),
            "headline": (item.get("title") or "").strip(),
            "summary": (item.get("text") or "").strip()[:MAX_TEXT_CHARS],
        }
        for item in batch
    ]
    prompt = "Classify these news items:\n" + json.dumps(payload, ensure_ascii=False)

    config = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "response_mime_type": "application/json",
        "response_schema": list[SentimentItem],
        "temperature": 0.0,
    }
    if THINKING_BUDGET >= 0:
        # 0 disables the model's internal reasoning step (faster); >0 caps it.
        config["thinking_config"] = {"thinking_budget": THINKING_BUDGET}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            parsed: Optional[list[SentimentItem]] = getattr(response, "parsed", None)
            if not parsed:
                raise LLMUnavailable("Gemini returned no parseable structured output.")

            requested_ids = {int(item["id"]) for item in batch}
            results: dict[int, dict] = {}
            for entry in parsed:
                if entry.id in requested_ids:
                    results[entry.id] = _normalize(entry)
            return results

        except Exception as e:
            last_error = e
            text = str(e)
            if _is_fatal_quota(text):
                # Daily quota / depleted credits — won't clear for hours. Fail
                # fast (no 5x backoff stall); these rows fall back to rule-based
                # and are flagged for LLM re-score on a later, healthy cycle.
                raise LLMUnavailable(f"Gemini quota/credits exhausted (fail-fast): {e}") from e
            if _is_retryable(text) and attempt < MAX_RETRIES:
                wait = _retry_delay_seconds(text, attempt)
                print(f"  [Gemini {('429/503')} retry {attempt}/{MAX_RETRIES}] waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise LLMUnavailable(f"Gemini request failed: {e}") from e

    raise LLMUnavailable(f"Gemini request failed after {MAX_RETRIES} retries: {last_error}")


# ----------------------------
# Public entry points
# ----------------------------

def classify_sentiments_llm(items: list[dict]) -> dict[int, dict]:
    """
    Batch-classify many news items with the configured LLM backend.

    items: list of {"id": int, "ticker": str, "title": str, "text": str}
    returns: {id: {sentiment_label, sentiment_score, confidence_score, reason}}
             (an id is absent if the model didn't return a result for it)

    Raises LLMUnavailable if the backend is unreachable so the caller can fall
    back to the rule-based scorer.
    """
    if not items:
        return {}

    if BACKEND != "gemini":
        # Future: ollama / claude backends.
        raise LLMUnavailable(f"Unknown LLM_SENTIMENT_BACKEND: {BACKEND!r}")

    results: dict[int, dict] = {}
    chunks = list(_chunks(items, BATCH_SIZE))
    for idx, batch in enumerate(chunks):
        if idx > 0:
            time.sleep(1.0)  # gentle pacing between chunk requests
        results.update(_score_batch_with_gemini(batch))
    return results


def classify_sentiment_llm(title: str, text: str, ticker: str = "") -> dict:
    """
    Single-item convenience wrapper (used for smoke tests / one-offs).
    Internally still goes through the batch path with one item.
    """
    out = classify_sentiments_llm([{"id": 0, "ticker": ticker, "title": title, "text": text}])
    if 0 not in out:
        raise LLMUnavailable("Gemini returned no result for the item.")
    return out[0]


# ----------------------------
# Crowd-mood summarization (Stocktwits chatter)
# ----------------------------

class CrowdMood(BaseModel):
    id: int
    mood: Literal["bullish", "bearish", "mixed", "quiet"]
    summary: str  # one short sentence


CROWD_SYSTEM_INSTRUCTION = (
    "You summarize retail-investor chatter about stocks from Stocktwits. You receive a "
    "JSON array of items, each with an id, a ticker, and a list of recent messages. For "
    "EACH item, judge the overall crowd mood and write ONE short sentence on what people "
    "are saying (the catalyst or vibe, plus any caution like 'looks like a pump').\n"
    "- mood: 'bullish', 'bearish', 'mixed', or 'quiet' (quiet = little real signal).\n"
    "- summary: one concise plain-language sentence.\n"
    "- Echo back the same id. Return exactly one object per input id."
)


def _gemini_structured(prompt: str, system_instruction: str, response_schema):
    """Shared Gemini call: thinking off + retry/backoff. Returns response.parsed."""
    client = _get_gemini_client()
    config = {
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "temperature": 0.0,
    }
    if THINKING_BUDGET >= 0:
        config["thinking_config"] = {"thinking_budget": THINKING_BUDGET}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL, contents=prompt, config=config,
            )
            parsed = getattr(response, "parsed", None)
            if not parsed:
                raise LLMUnavailable("Gemini returned no parseable structured output.")
            return parsed
        except Exception as e:
            last_error = e
            text = str(e)
            if _is_retryable(text) and attempt < MAX_RETRIES:
                wait = _retry_delay_seconds(text, attempt)
                print(f"  [Gemini retry {attempt}/{MAX_RETRIES}] waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise LLMUnavailable(f"Gemini request failed: {e}") from e
    raise LLMUnavailable(f"Gemini request failed after {MAX_RETRIES} retries: {last_error}")


def summarize_crowd_moods(items: list[dict]) -> dict[int, dict]:
    """
    Batch-summarize Stocktwits chatter for a few candidate tickers in ONE request.

    items: list of {"id": int, "ticker": str, "comments": [str, ...]}
    returns: {id: {"mood": str, "summary": str}}  (absent id = no result)

    Raises LLMUnavailable so the caller can fall back to the bullish/bearish ratio.
    """
    if not items:
        return {}
    if BACKEND != "gemini":
        raise LLMUnavailable(f"Unknown LLM_SENTIMENT_BACKEND: {BACKEND!r}")

    payload = [
        {
            "id": int(it["id"]),
            "ticker": it.get("ticker", ""),
            "messages": [str(c).strip()[:200] for c in (it.get("comments") or [])][:30],
        }
        for it in items
    ]
    prompt = "Summarize the crowd mood for each item:\n" + json.dumps(payload, ensure_ascii=False)

    parsed = _gemini_structured(prompt, CROWD_SYSTEM_INSTRUCTION, list[CrowdMood])
    requested = {int(it["id"]) for it in items}
    out: dict[int, dict] = {}
    for entry in parsed:
        if entry.id in requested:
            out[entry.id] = {"mood": entry.mood, "summary": (entry.summary or "").strip()}
    return out


if __name__ == "__main__":
    # Smoke test: classifies several headlines in ONE request.
    samples = [
        {"id": 1, "ticker": "NVDA", "title": "NVIDIA beats earnings estimates and raises full-year guidance", "text": ""},
        {"id": 2, "ticker": "ABCD", "title": "ABCD Corp announces $200M public offering of common stock", "text": ""},
        {"id": 3, "ticker": "XYZ", "title": "XYZ to present at the Goldman Sachs technology conference", "text": ""},
        {"id": 4, "ticker": "BIO", "title": "BIO Therapeutics receives FDA approval for lead drug", "text": ""},
        {"id": 5, "ticker": "LAW", "title": "Investors with losses urged to lead securities class action against LAW Inc", "text": ""},
    ]
    try:
        results = classify_sentiments_llm(samples)
        for s in samples:
            r = results.get(s["id"])
            if r:
                print(f"{s['ticker']:6} {r['sentiment_label']:8} "
                      f"score={r['sentiment_score']:+.2f} conf={r['confidence_score']:.2f} | {r['reason']}")
            else:
                print(f"{s['ticker']:6} (no result returned)")
        print(f"\nClassified {len(results)}/{len(samples)} items in 1 request.")
    except LLMUnavailable as e:
        print(f"LLM unavailable -> {e}")
