#!/usr/bin/env python3
"""
Restore financial_news.db from a backup made by backup_data.py.

    python scripts/restore_data.py                 # restore the LATEST backup
    python scripts/restore_data.py backups/financial_news_20260811_231500.db

IMPORTANT: stop the app first (so nothing is writing) — e.g.
    launchctl unload ~/Library/LaunchAgents/com.example.financialnews.marketsession.plist
The current database is safety-copied to backups/pre_restore_<timestamp>.db before
being overwritten, so a restore is itself reversible.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB = BASE / "financial_news.db"
BK = BASE / "backups"


def main() -> None:
    backups = sorted(BK.glob("financial_news_*.db"))
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        if not src.is_absolute():
            src = BASE / src
    elif backups:
        src = backups[-1]
    else:
        sys.exit("no backups found in backups/")
    if not src.exists():
        sys.exit(f"backup not found: {src}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Safety-copy the current DB (so this restore is reversible), then clear the
    # live WAL/SHM so they can't overlay the restored file.
    if DB.exists():
        safety = BK / f"pre_restore_{ts}.db"
        shutil.copy2(DB, safety)
        print(f"  saved current db -> {safety.name}")
    for ext in ("-wal", "-shm"):
        p = Path(str(DB) + ext)
        if p.exists():
            p.unlink()

    shutil.copy2(src, DB)
    print(f"restored {src.name} -> financial_news.db")
    print("  (make sure the app was stopped; start it again when ready)")


if __name__ == "__main__":
    main()
