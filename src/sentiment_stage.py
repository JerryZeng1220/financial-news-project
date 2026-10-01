"""
Sentiment scoring stage — LLM-first with rule-based fallback.

Reuses everything from sentiment_rules (DB helpers, the rule-based classifier,
the patterns) and only changes HOW each record is scored:

  1. Load all unscored ticker-linked news (rules.load_unscored_ticker_news).
  2. Classify them all in ONE batched Gemini request (llm.classify_sentiments_llm).
  3. For any record the LLM didn't return — or if Gemini is unavailable
     (no key, offline, quota) — fall back to the rule-based classifier.
  4. Insert with method/model_version reflecting which path scored it.

Toggle with USE_LLM_SENTIMENT=false to force pure rule-based (original behaviour).

Each news item is scored per (clean_news_id, ticker) pair, because sentiment is
ticker-relative. We give each pair a batch-local index as the LLM `id`, then map
the results back to the correct (clean_news_id, ticker).
"""
from __future__ import annotations

import os

import sentiment_rules as rules         # DB helpers + rule-based classifier + patterns
import llm_sentiment as llm             # batched Gemini classifier


USE_LLM = os.getenv("USE_LLM_SENTIMENT", "true").strip().lower() == "true"


def _rule_based_safe(title: str, text: str) -> dict:
    """
    Call the existing rule-based classifier, tolerating its special-case branch
    that returns a (label, reason) tuple instead of a dict. This keeps the
    background loop from crashing on that path.
    """
    result = rules.classify_sentiment_rule_based(title=title, text=text)
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        label, reason = result
        score = 0.5 if label == "positive" else (-0.5 if label == "negative" else 0.0)
        return {
            "sentiment_label": label,
            "sentiment_score": score,
            "confidence_score": 0.70,
            "reason": reason,
        }
    return {
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "confidence_score": 0.50,
        "reason": "Fallback: unrecognized rule-based result.",
    }


def rescore_pending_llm() -> None:
    """Re-run the LLM on rows that fell back to rule-based while it was
    down/over-quota (pending_llm_rescore=1), now that it may be healthy. Updates
    each row in place and clears the flag; rows the LLM still can't score stay
    flagged for the next cycle. No-op if USE_LLM is off."""
    if not USE_LLM:
        return

    df = rules.load_pending_llm_rescore()
    total = len(df)
    if total == 0:
        return

    print(f"Re-scoring {total} pending rule-based-fallback row(s) via LLM...")
    records = []
    for idx, row in enumerate(df.itertuples(index=False)):
        records.append({
            "idx": idx,
            "clean_news_id": int(row.clean_news_id),
            "ticker": str(row.ticker).strip().upper(),
            "title": row.title_clean or "",
            "text": row.text_clean or "",
        })

    items = [
        {"id": r["idx"], "ticker": r["ticker"], "title": r["title"], "text": r["text"]}
        for r in records
    ]
    try:
        llm_results = llm.classify_sentiments_llm(items)
    except llm.LLMUnavailable as e:
        print(f"  LLM still unavailable ({e}); leaving rows flagged for next cycle.")
        return

    updated = 0
    for r in records:
        res = llm_results.get(r["idx"])
        if not res:
            continue  # LLM didn't return this one; stays flagged, retried later
        rules.update_sentiment_llm(
            clean_news_id=r["clean_news_id"],
            ticker=r["ticker"],
            sentiment_label=res["sentiment_label"],
            sentiment_score=res["sentiment_score"],
            confidence_score=res["confidence_score"],
            reason=res["reason"],
            materiality=res.get("materiality", "moderate"),
            model_version=llm.LLM_MODEL_VERSION,
        )
        updated += 1
    print(f"  Re-scored {updated}/{total} rows via LLM.")


def run_sentiment_scoring() -> None:
    print("Initializing sentiment tables...")
    rules.init_sentiment_tables()

    # First recover any rows that fell back to rule-based while the LLM was
    # unavailable (e.g. over quota) — re-score them now that it may be healthy.
    rescore_pending_llm()

    print("Loading unscored ticker-linked news...")
    df = rules.load_unscored_ticker_news()
    total = len(df)
    print(f"Ticker-linked news records to score: {total}")

    if total == 0:
        rules.log_sentiment_run(0, 0, "success", "no unscored records")
        print("Nothing to score. Done.")
        return

    # Build batch-local index -> record mapping. One entry per (news, ticker) pair.
    records = []  # list of dicts: idx, clean_news_id, ticker, title, text
    for idx, row in enumerate(df.itertuples(index=False)):
        records.append({
            "idx": idx,
            "clean_news_id": int(row.clean_news_id),
            "ticker": str(row.ticker).strip().upper(),
            "title": row.title_clean or "",
            "text": row.text_clean or "",
        })

    # 1) Try the batched LLM path (unless disabled).
    llm_results: dict[int, dict] = {}
    if USE_LLM:
        items = [
            {"id": r["idx"], "ticker": r["ticker"], "title": r["title"], "text": r["text"]}
            for r in records
        ]
        try:
            print(f"Scoring {len(items)} records via LLM ({llm.LLM_MODEL_VERSION})...")
            llm_results = llm.classify_sentiments_llm(items)
            print(f"  LLM returned {len(llm_results)}/{len(items)} results.")
        except llm.LLMUnavailable as e:
            print(f"  LLM unavailable ({e}); falling back to rule-based for all.")
    else:
        print("USE_LLM_SENTIMENT=false; using rule-based only.")

    # 2) Insert each record, using the LLM result when present, else rule-based.
    scored_llm = 0
    scored_rule = 0

    for r in records:
        if r["idx"] in llm_results:
            result = llm_results[r["idx"]]
            method = "llm"
            model_version = llm.LLM_MODEL_VERSION
            pending = 0
            scored_llm += 1
        else:
            result = _rule_based_safe(r["title"], r["text"])
            method = "rule_based"
            model_version = "rule_based_v1"
            # We wanted the LLM but it was down/over-quota (or skipped this item);
            # flag it so a later healthy cycle re-scores it. If USE_LLM is off by
            # choice, rule-based is the intended final answer (don't re-score).
            pending = 1 if USE_LLM else 0
            scored_rule += 1

        rules.insert_sentiment_result(
            clean_news_id=r["clean_news_id"],
            ticker=r["ticker"],
            sentiment_label=result["sentiment_label"],
            sentiment_score=result["sentiment_score"],
            confidence_score=result["confidence_score"],
            method=method,
            reason=result["reason"],
            model_version=model_version,
            # LLM tags materiality; rule-based fallback can't judge it, so default
            # to "moderate" (passes a moderate+ alert filter but isn't "major").
            materiality=result.get("materiality", "moderate"),
            pending_llm_rescore=pending,
        )

    rules.log_sentiment_run(
        records_read=total,
        records_scored=scored_llm + scored_rule,
        status="success",
        notes=f"llm={scored_llm}, rule_based_fallback={scored_rule}, model={llm.LLM_MODEL_VERSION}",
    )

    print("\nSentiment scoring summary")
    print("-------------------------")
    print(f"Records read: {total}")
    print(f"Scored by LLM: {scored_llm}")
    print(f"Scored by rule-based fallback: {scored_rule}")
    print("Done.")


if __name__ == "__main__":
    run_sentiment_scoring()
