#!/bin/bash
# PBT2 watchdog. Run in tmux before leaving a run unattended:
#   tmux new -s watchdog 'bash scripts/watchdog_pbt.sh'

set -euo pipefail

ROOT="${CHESS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${TRAIN_CONFIG:-configs/pbt2_small.yaml}"
LOG="${TRAIN_MONITOR_LOG:-$ROOT/runs/pbt2_small/monitor.log}"
INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-3600}"

cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
mkdir -p "$(dirname "$LOG")"

while true; do
  {
    echo "========================================"
    echo "Watchdog check: $(date)"
    echo "========================================"
  } | tee -a "$LOG"

  if ! pgrep -f "chess_anti_engine.run" >/dev/null 2>&1; then
    echo "WARNING: Training process not found. Restarting..." | tee -a "$LOG"
    PYTHONPATH=. python3 -m chess_anti_engine.run \
      --config "$CONFIG" --mode tune --resume >> "$LOG" 2>&1 &
    echo "Restarted with PID $!" | tee -a "$LOG"
    sleep 120
    continue
  fi

  CHESS_ROOT="$ROOT" TRAIN_MONITOR_LOG="$LOG" bash "$ROOT/scripts/monitor_pbt.sh"
  sleep "$INTERVAL_SECONDS"
done
