#!/usr/bin/env bash
# Start the main run detached: keeps going if the terminal closes, and blocks suspend/idle sleep
# while it runs (systemd-inhibit). Run from the repo root with the venv active:
#
#     bash scripts/overnight.sh            # all three runs (see run_main.sh)
#     tail -f logs/main_*.log              # watch progress (Ctrl+C only stops watching)
#     bash scripts/stop_run.sh             # stop it (e.g. to play games); rerun overnight.sh to resume
set -u
cd "$(dirname "$0")/.."
if pgrep -f "scripts/run_main.sh" >/dev/null; then echo "already running (bash scripts/stop_run.sh to stop)"; exit 1; fi
if command -v systemd-inhibit >/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who=position-fragility --why="overnight experiment")
else
  INHIBIT=()
fi
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then set +u; source .venv/bin/activate; set -u; fi
nohup "${INHIBIT[@]}" bash scripts/run_main.sh "$@" >/dev/null 2>&1 &
sleep 3
echo "started (pid $!). Sleep is blocked while it runs. Watch: tail -f logs/main_*.log"
echo "Stop any time: bash scripts/stop_run.sh   |   resume: bash scripts/overnight.sh"
