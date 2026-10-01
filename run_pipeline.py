import subprocess
import sys

scripts = [
    # Source and schema setup
    "src/init_db.py",

    # News ingestion
    "src/ingest_news.py",
    "src/load_live_active_universe.py",

    # Per-ticker Yahoo news for the current universe (captures news the broad
    # wires miss, pre-attached to the ticker). Runs after the universe loads.
    "src/collect_ticker_news.py",

    "src/preprocess_news.py",
    "src/extract_tickers.py",
    # LLM-first sentiment (Gemini) with rule-based fallback.
    # Falls back to pure rule-based if GEMINI_API_KEY is missing or USE_LLM_SENTIMENT=false.
    "src/sentiment_stage.py",

    # Real social-mention collectors (run AFTER sentiment so they only gather
    # discussion for growing tickers with positive recent news). Both feed
    # social_mentions -> density + crowd-mood. Reddit no-ops if not configured.
    "src/collect_social_stocktwits.py",
    "src/collect_social_reddit.py",

    # Social density + crowd-mood summary (Gemini, candidates only) + combined ranking
    "src/calculate_density.py",
    "src/score_crowd_mood.py",
    "src/combined_ranking.py",

    # Retention: prune fast-growing / short-lived data each run.
    # Keeps dedup fingerprints (news) and alert history (for evaluation) intact.
    "src/cleanup_retention.py",
]

for script in scripts:
    print(f"\n==============================")
    print(f"Running {script}...")
    print(f"==============================")

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Stopping pipeline.")
        sys.exit(result.returncode)

print("\nPipeline completed successfully.")