#!/bin/bash
set -euo pipefail

ROOT="/home/josh/projects/chess"
LOG="$ROOT/runs/pbt2_small/hourly_audit.log"
SCRIPT="$ROOT/scripts/pbt_hourly_audit.py"

cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

while true; do
  {
    echo
    echo "########################################################################"
    python3 "$SCRIPT"
  } >> "$LOG" 2>&1
  sleep 3600
done

