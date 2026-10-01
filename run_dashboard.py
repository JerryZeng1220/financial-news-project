"""
Dashboard launcher that auto-closes the browser tab on Ctrl+C.

Run this INSTEAD of `streamlit run src/dashboard.py`:
    /Users/junwen/Documents/financialNewsProject/bin/python3 run_dashboard.py

It starts Streamlit (which opens the dashboard tab), and when you press Ctrl+C it
stops the server AND closes the browser tab pointing at the dashboard.

macOS only (uses osascript). Supports Chrome, Brave, Edge, Arc, and Safari. The
first time, macOS may ask to allow the terminal to control your browser
(System Settings → Privacy & Security → Automation) — approve it once and the
auto-close works from then on.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP = BASE_DIR / "src" / "dashboard.py"
PORT = os.getenv("DASHBOARD_PORT", "8501")

# Chrome-family and Safari all expose windows -> tabs -> (URL, close), so one
# template works for each; we only target browsers that are actually running.
BROWSERS = ["Google Chrome", "Brave Browser", "Microsoft Edge", "Arc", "Safari"]


def _running(app_name: str) -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", app_name], capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False


def close_browser_tab() -> None:
    frag1 = f"localhost:{PORT}"
    frag2 = f"127.0.0.1:{PORT}"
    for app in BROWSERS:
        if not _running(app):
            continue
        script = f'''
        tell application "{app}"
          repeat with w in windows
            repeat with t in (tabs of w)
              try
                set u to URL of t
                if (u contains "{frag1}") or (u contains "{frag2}") then close t
              end try
            end repeat
          end repeat
        end tell
        '''
        try:
            subprocess.run(["osascript", "-e", script], check=False, timeout=10,
                           capture_output=True)
        except Exception:
            pass


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", PORT]
    print(f"Starting dashboard on http://localhost:{PORT}  (Ctrl+C to stop + close the tab)")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        close_browser_tab()
        print("\nDashboard stopped and browser tab closed.")


if __name__ == "__main__":
    main()
