#!/bin/bash
set -euo pipefail

ROOT="/home/josh/projects/chess"
LOG="$ROOT/runs/pbt2_small/poll_30m.log"
SCRIPT="$ROOT/scripts/pbt_30m_poll.py"
INTERVAL_SECONDS=1800

cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p "$(dirname "$LOG")"

while true; do
  {
    echo
    echo "################################################################################"
    python3 "$SCRIPT" --root "$ROOT"
  } >> "$LOG" 2>&1
  sleep "$INTERVAL_SECONDS"
done

