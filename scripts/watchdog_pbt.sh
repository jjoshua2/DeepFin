#!/bin/bash
# PBT2 overnight watchdog — run in tmux before bed:
#   tmux new -s watchdog 'bash scripts/watchdog_pbt.sh'
#
# Auto-restarts training if the process dies.
# Logs status every 60 minutes.

LOG="/home/josh/projects/chess/runs/pbt2_small/monitor.log"
CMD="python -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune --resume"
cd /home/josh/projects/chess

source .venv/bin/activate 2>/dev/null

while true; do
    echo "========================================"  | tee -a "$LOG"
    echo "Watchdog check: $(date)"                   | tee -a "$LOG"
    echo "========================================"  | tee -a "$LOG"

    if ! pgrep -f "chess_anti_engine.run" > /dev/null 2>&1; then
        echo "WARNING: Training process not found! Restarting..." | tee -a "$LOG"
        $CMD >> "$LOG" 2>&1 &
        echo "Restarted with PID $!" | tee -a "$LOG"
        sleep 120  # give it time to initialize before next check
        continue
    fi

    # Quick status summary
    bash /home/josh/projects/chess/scripts/monitor_pbt.sh

    sleep 3600
done
