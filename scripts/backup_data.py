#!/usr/bin/env python3
"""
Create a full, consistent, restorable snapshot of financial_news.db.

Uses SQLite's online backup API, so it's safe to run even while the app is
writing (it captures a transactionally-consistent copy, WAL included). The
snapshot is a single standalone .db file containing EVERYTHING — alert logs
(notified_alerts), positions, accuracy history, sources (source_registry /
news_sources) and settings (app_settings). Restore with restore_data.py.

    python scripts/backup_data.py            # -> backups/financial_news_<timestamp>.db
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB = BASE / "financial_news.db"
BK = BASE / "backups"


def main() -> None:
    if not DB.exists():
        sys.exit(f"no database at {DB}")
    BK.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BK / f"financial_news_{ts}.db"

    src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = sqlite3.connect(str(dst))
    try:
        with out:
            src.backup(out)            # atomic, consistent full copy
    finally:
        src.close()
        out.close()

    # sanity: confirm the copy opens and has the key tables
    chk = sqlite3.connect(str(dst))
    tabs = {r[0] for r in chk.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    npos = chk.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    nal = chk.execute("SELECT COUNT(*) FROM notified_alerts").fetchone()[0]
    chk.close()
    ok = {"positions", "notified_alerts", "app_settings", "source_registry"} <= tabs
    print(f"backup -> {dst}")
    print(f"  size: {dst.stat().st_size/1e6:.0f} MB · tables: {len(tabs)} · "
          f"positions: {npos} · alerts: {nal} · key tables present: {ok}")


if __name__ == "__main__":
    main()
