#!/bin/bash
# Market-hours session runner for the financial-news app.
#
# Launched by the LaunchAgent at 07:00 on weekdays (and whenever the agent is
# (re)loaded). It keeps the Mac awake (no idle sleep) and runs run_app.py until
# 20:00, then lets go so the Mac sleeps normally overnight and on weekends.
#
# Design notes:
#   - Only runs Mon-Fri, and only before 20:00 (guards below) so a login on a
#     weekend / late night doesn't spin it up.
#   - `caffeinate -i` prevents *idle* sleep only; closing the lid still sleeps
#     (by design — run it lid-open or on a hard surface, plugged in).
#   - Restarts run_app if it ever exits early, but never past 20:00.
#   - Won't start a duplicate if a worker is already running.

set -u
PROJECT="/Users/junwen/PycharmProjects/financialNewsProject"
PY="/Users/junwen/Documents/financialNewsProject/bin/python3"
cd "$PROJECT" || exit 1
mkdir -p logs
LOG="$PROJECT/logs/run_app.log"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# --- Only run during weekday US-market hours ---
DOW=$(date +%u)                         # 1=Mon .. 7=Sun
[ "$DOW" -ge 6 ] && { log "weekend; not starting"; exit 0; }
END=$(date -v20H -v0M -v0S +%s)         # today 20:00 local
[ "$(date +%s)" -ge "$END" ] && { log "after 20:00; not starting"; exit 0; }

# --- Don't start a duplicate worker ---
# Match only an actual python process executing run_app.py (e.g. "python3
# run_app.py" or ".../Python run_app.py") — NOT arbitrary shell commands that
# merely mention the filename.
if pgrep -f "[Pp]ython[^ ]* run_app.py" >/dev/null 2>&1; then
    log "worker already running; not starting a duplicate"
    exit 0
fi

log "session start — holding Mac awake and running until 20:00"

# Keep the Mac awake (prevent idle sleep) for the whole session, in background.
caffeinate -i -t "$(( END - $(date +%s) ))" &
CAF=$!
trap 'kill "$CAF" 2>/dev/null' EXIT     # release the awake-hold whenever we exit

# Run run_app until 20:00, restarting it if it exits early.
while [ "$(date +%s)" -lt "$END" ]; do
    "$PY" run_app.py >> "$LOG" 2>&1 &
    APP=$!
    ( sleep "$(( END - $(date +%s) ))"; kill "$APP" 2>/dev/null ) &   # stop at 20:00
    WD=$!
    wait "$APP" 2>/dev/null
    kill "$WD" 2>/dev/null
    if [ "$(date +%s)" -lt "$END" ]; then
        log "run_app exited early; restarting in 5s"
        sleep 5
    fi
done
log "session end (20:00) — Mac free to sleep"
