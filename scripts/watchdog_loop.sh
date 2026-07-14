#!/bin/bash
# Training-health watchdog loop (managed by scripts/train.sh; runs forever).
#
# Every $WATCHDOG_EVERY seconds runs scripts/train_watchdog.py (detect-and-
# report ONLY — it never starts, stops, or signals anything, and neither does
# this loop; recovery is always a human/agent decision). All checks append to
# $LOGF; non-OK states also append to $ALERTF *unless* the intentional-stop
# marker exists (touched by `train.sh stop`, removed by `train.sh start`), so
# deliberately stopping training doesn't page anyone.
#
# Optional: set WATCHDOG_NOTIFY_CMD to a command; it is invoked with the
# status line as one argument on every unsuppressed non-OK check (fail-soft,
# handled inside train_watchdog.py --notify-cmd).
set -u
cd /home/josh/projects/chess
MARKER=/tmp/chess_training.intentional_stop
LOGF=scratchpad/watchdog.log
ALERTF=scratchpad/watchdog_alerts.log
STATEF=/tmp/chess_watchdog_state.json
WATCHDOG_EVERY="${WATCHDOG_EVERY:-600}"

while true; do
    ARGS=()
    [ -n "${WATCHDOG_NOTIFY_CMD:-}" ] && [ ! -f "$MARKER" ] && \
        ARGS=(--notify-cmd "$WATCHDOG_NOTIFY_CMD")
    LINE=$(PYTHONPATH=. python3 scripts/train_watchdog.py --state "$STATEF" "${ARGS[@]}" 2>&1)
    RC=$?
    echo "$(date '+%m-%d %H:%M:%S') $LINE" >> "$LOGF"
    if [ "$RC" -ne 0 ] && [ ! -f "$MARKER" ]; then
        echo "$(date '+%m-%d %H:%M:%S') $LINE" >> "$ALERTF"
    fi
    sleep "$WATCHDOG_EVERY"
done
