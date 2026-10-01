# Financial News Project — Nano-Cap Catalyst Screener

A Python application that combines stock screening, financial news analysis, social discussion, alerts, and simulated position tracking in a Streamlit dashboard.

The project helps users investigate potential news catalysts for small publicly traded companies. It generates signals and records simulated positions; **it does not connect to a broker or execute trades**. Scores are heuristics, not validated predictions of future returns.

## Features

- Load a stock universe from a local CSV, a Finviz screener, or an authorized export URL.
- Collect financial news from configured RSS sources and per-ticker Yahoo feeds.
- Clean articles, detect duplicate content, and match news to stock tickers.
- Analyze ticker-specific sentiment and catalyst materiality using Gemini, with a rule-based fallback.
- Collect Stocktwits and Reddit discussion when enabled and available.
- Calculate discussion density relative to a ticker's observed baseline.
- Rank tickers using sentiment, discussion density, and source credibility.
- Generate desktop notifications and optional email alerts.
- Track simulated positions with configurable exit conditions and end-of-day closure.
- Display rankings, news, alerts, positions, and evaluation metrics in a dashboard.

## Requirements

- Python 3.10 or newer; use a version compatible with the packages in `requirements.txt`.
- Internet access for news, social sources, and live quotes.
- A stock-universe input: a CSV export or an accessible Finviz source.
- Optional Gemini API key for model-based sentiment and crowd summaries.
- Finviz Elite credentials for the Elite authorized-export configuration.
- Optional SMTP credentials for email alerts.

macOS is required for AppleScript desktop notifications, alert sounds, and the supplied LaunchAgent integration. Those features are platform-specific. The core pipeline and Streamlit interface use Python, but a fresh installation should be tested on the intended operating system.

**The current project does not include a complete offline demo mode.** CSV input removes the need to fetch the stock universe, but the remaining collection and quote stages still use external services. A reviewer needs the documented inputs and working network access. Do not assume that a successful process exit means every external source returned data.

## Installation

Download or clone this repository, then open a terminal in its root directory—the folder containing `run_pipeline.py`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation is unavailable, invoke the environment's Python executable directly: `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.

Edit `.env` before running. Replace example credentials with your own, or select the CSV/rule-based configuration below. Never publish `.env`.

## Configure the stock universe

Choose one of these modes in `.env`.

### Option A: Local CSV snapshot

```dotenv
ACTIVE_UNIVERSE_MODE=csv_snapshot
USE_LLM_SENTIMENT=false
```

Create `data/raw/` if necessary and place a stock CSV at:

```text
data/raw/finviz_export.csv
```

Recommended columns:

```csv
Ticker,Company,Price,Change,Volume,Sector,Industry,Country,Market Cap
```

`Ticker` is required. Include actual company names for useful company-name matching, and prices for price-based tracking. If `Change` is present, only rows with a parsed positive percentage enter the active universe. Use percentage values such as `5.20%`; price should be numeric and volume should be a numeric count.

Supply your own shareable CSV; do not assume this input is bundled with the repository. A CSV snapshot is historical input and does not update itself. Network-based news collection still runs in this configuration, and current news may not match the snapshot's tickers.

### Option B: Finviz Elite authorized export

```dotenv
ACTIVE_UNIVERSE_MODE=authorized_url
USE_CURL_IMPERSONATE=true
FINVIZ_ACTIVE_UNIVERSE_URL=YOUR_AUTHORIZED_EXPORT_URL
FINVIZ_COOKIE=YOUR_COOKIE
```

Use an export URL and cookie from your own authorized Finviz session. Cookies may expire. Do not publish either credentials or URLs containing private access tokens.

### Option C: Finviz HTML screener

```dotenv
ACTIVE_UNIVERSE_MODE=finviz_screener
USE_CURL_IMPERSONATE=true
```

The code also supports fetching Finviz screener HTML. This depends on page structure and source access; blocking or format changes can prevent collection. It is not a guaranteed substitute for an authorized export.

## Configure sentiment and optional delivery

To enable Gemini sentiment:

```dotenv
USE_LLM_SENTIMENT=true
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-2.5-flash
```

The model name shown is the code's default. Your account must have access to the configured model. Quotas and billing depend on your provider account; this project does not guarantee a particular free quota or cost.

If Gemini is unavailable, new sentiment records can fall back to rules and be marked for later model rescoring. Setting `USE_LLM_SENTIMENT=false` explicitly selects rule-based news scoring. Crowd-mood summarization is a separate Gemini stage and can skip when the model is unavailable.

Optional email configuration:

```dotenv
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=YOUR_EMAIL_ADDRESS
ALERT_SMTP_PASSWORD=YOUR_SMTP_OR_APP_PASSWORD
```

Enable email delivery and set the recipient in the dashboard. Use credentials appropriate for your SMTP provider. Social collectors can be enabled or disabled through the Sources controls; unavailable or blocked sources may produce no discussion data.

## Run the application

Run every command below from the repository root, with the virtual environment active.

### 1. Process one data cycle

```bash
python run_pipeline.py
```

This runs schema initialization, collection, cleaning, matching, sentiment, social processing, ranking, and retention in order. SQLite tables are initialized by the relevant stages. The database is stored at `financial_news.db` in the project root.

Expected console output includes stage names, processed-record counts, ranking output when records are available, and a completion message. On repeat runs, already-processed data may produce zero new records. A source failure or empty universe can leave no rankings even if the pipeline continues; inspect warnings and counts.

### 2. Open the dashboard

```bash
python run_dashboard.py
```

Open http://localhost:8501 if a browser tab does not appear. The dashboard displays stored results; opening it alone does not start the collection worker. The launcher attempts to close its dashboard browser tab when stopped on macOS.

Alternatively:

```bash
python -m streamlit run src/dashboard.py
```

### 3. Start continuous collection and tracking

In a second terminal, activate the same environment, change to the repository root, and run:

```bash
python run_app.py
```

The worker repeats the pipeline and runs notifications and position tracking within the dashboard-configured weekday trading window. Its default window is 09:30–20:00 America/New_York. The schedule uses weekdays and does not implement an exchange-holiday calendar.

Press Ctrl+C to stop either process. Avoid starting multiple workers against the same database. The worker may generate configured notifications and update simulated positions.

Worker timing options can be exported in the launching terminal. For example, on macOS/Linux:

```bash
export APP_INTERVAL_SECONDS=90
export APP_PIPELINE_TIMEOUT_SEC=120
python run_app.py
```

The worker reads these options at startup; setting them in the shell ensures they are available regardless of individual modules' `.env` loading behavior.

## Dashboard guide

| Area | Purpose |
| --- | --- |
| Top Pick | Display a leading candidate according to the ranking logic |
| Board | Review ranked tickers, news, alerts, and tracked positions |
| Accuracy | Inspect alert outcomes and simulated-position statistics |
| Sources | Configure collection sources |
| Sidebar | Adjust filters, session times, delivery, and exit settings |

The combined ranking and notification eligibility use different conditions in the current implementation. A high-ranked ticker or dashboard candidate is not guaranteed to produce a notification.

## Project structure

| File or group | Responsibility |
| --- | --- |
| `run_pipeline.py` | Run one ordered data-processing cycle |
| `run_app.py` | Continuous worker, pipeline watchdog, session handling |
| `run_dashboard.py` | Launch the Streamlit interface |
| `src/common.py` | Shared database path, connection, and time helpers |
| `src/init_db.py` | Initialize source and social tables |
| `src/ingest_news.py` | Collect configured RSS news |
| `src/load_live_active_universe.py` | Load and normalize the stock universe |
| `src/collect_ticker_news.py` | Collect per-ticker Yahoo news |
| `src/preprocess_news.py` | Clean text, normalize dates, and detect duplicates |
| `src/extract_tickers.py` | Match articles to companies and tickers |
| `src/sentiment_stage.py` | Coordinate model sentiment and rule fallback |
| `src/llm_sentiment.py` | Batched Gemini classification and crowd summaries |
| `src/sentiment_rules.py` | Rule classifier and sentiment persistence |
| `src/collect_social_stocktwits.py`, `src/collect_social_reddit.py` | Collect social discussion |
| `src/calculate_density.py`, `src/score_crowd_mood.py` | Discussion density and crowd summaries |
| `src/combined_ranking.py` | Calculate and save combined scores |
| `src/notify_alerts.py` | Select, deduplicate, record, and deliver alerts |
| `src/live_price.py`, `src/track_positions.py` | Quote retrieval and simulated entry/exit tracking |
| `src/end_of_day.py` | Close simulated positions and generate a daily recap |
| `src/check_accuracy.py` | Calculate evaluation statistics |
| `src/dashboard.py` | Render the interface |
| `src/app_settings.py`, `src/news_filter.py`, `src/news_sources.py` | Persist settings and source/filter controls |
| `scripts/` | Backup, restore, and optional macOS service utilities |
| `archive/` | Legacy utilities, if included; not part of the current pipeline |

The standard pipeline order is:

```text
init_db → ingest_news → load_live_active_universe → collect_ticker_news
→ preprocess_news → extract_tickers → sentiment_stage
→ collect_social_stocktwits → collect_social_reddit
→ calculate_density → score_crowd_mood → combined_ranking → cleanup_retention
```

## Scoring and stored data

The current combined score is:

```text
0.45 × max(news_sentiment, 0)
+ 0.40 × discussion_density
+ 0.15 × source_credibility
```

These weights are selected heuristics. Sentiment and materiality depend on model/rule outputs. Density depends on collected observations, which may not represent all discussion.

Important tables include `raw_news`, `clean_news`, `news_ticker_links`, `news_sentiment`, `active_ticker_universe`, `social_mentions`, `ticker_density_snapshots`, `combined_ticker_rankings`, `notified_alerts`, and `positions`. Additional tables store history, settings, logs, mood summaries, and daily recaps as stages run.

Retention prunes configured histories and snapshots. Raw/clean news and sentiment are intentionally retained for deduplication, so the database can continue growing.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Missing Python module | Activate the environment and reinstall `requirements.txt` using that environment's Python |
| Missing CSV or empty universe | Check `data/raw/finviz_export.csv`, mode selection, column names, and positive Change values |
| Finviz refresh warning | Verify source access, export URL, and cookie; an existing universe may be retained after failure |
| No new records | Existing records may already be processed; inspect collection counts and freshness |
| No alerts | Check the active universe, news age, sentiment, materiality, keywords, deduplication, and session window |
| No social data | Check source toggles, candidate availability, and HTTP/rate-limit warnings |
| Gemini unavailable | Verify credentials/model access; inspect fallback messages and provider quota |
| Dashboard cannot open | Check terminal errors and whether port 8501 is already in use |
| Missing desktop sound/notification | These delivery features require macOS and applicable permissions |
| Worker repeatedly times out | Inspect the slow stage; review network delays and the watchdog timeout |

## Known limitations

- Source availability, delayed prices, and differing quote timestamps can affect signals and simulated results.
- The current worker gates tracking and end-of-day handling behind pipeline success; repeated collection failures can delay those actions.
- Delayed simulated entries can retain an earlier alert's price/time. Entry accounting requires further correction.
- Price trust checks are not applied consistently to all exits and peak updates.
- Alert peak statistics can use a historical peak from another position in the same ticker.
- The displayed compounded-return calculation multiplies trade returns without a capital-allocation model; it is not a realized portfolio return, especially for overlapping positions.
- Transaction costs, bid/ask spreads, slippage, liquidity constraints, and actual execution are not modeled.
- There is no bundled fully offline, deterministic end-to-end demo.

Treat evaluation output as exploratory until these issues are corrected and validated.

## Fresh-install verification for reviewers

1. Clone/download the repository into a new folder.
2. Create a new environment and install the declared dependencies.
3. Configure your own source access, or supply the CSV input and select rule-based sentiment.
4. Run `python run_pipeline.py` and check stage warnings and record counts.
5. Run `python run_dashboard.py` and inspect the resulting stored output.
6. If testing the worker, run it separately and verify settings and session behavior.

These instructions describe the current implementation. A successful clean-install execution on another machine should be verified before claiming reproducibility.

## Public-repository hygiene

Commit source code, `requirements.txt`, documentation, and a placeholder-only `.env.example`. Exclude actual credentials, virtual environments, caches, logs, database files and their WAL/SHM companions, and backups. Share only data you are permitted to redistribute.

The optional macOS service scripts contain machine-specific paths. Update them for your installation before use; they are not required for the manual setup above.
