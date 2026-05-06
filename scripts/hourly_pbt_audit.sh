#!/bin/bash
set -euo pipefail

ROOT="${CHESS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG="${PBT_HOURLY_AUDIT_LOG:-$ROOT/runs/pbt2_small/hourly_audit.log}"
SCRIPT="$ROOT/scripts/pbt_hourly_audit.py"
INTERVAL_SECONDS="${PBT_HOURLY_AUDIT_INTERVAL_SECONDS:-3600}"

cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

while true; do
  {
    echo
    echo "########################################################################"
    python3 "$SCRIPT" --root "$ROOT"
  } >> "$LOG" 2>&1
  sleep "$INTERVAL_SECONDS"
done
