#!/bin/bash
# Training-health watchdog loop (managed by scripts/train.sh; runs forever).
#
# Every $WATCHDOG_EVERY seconds runs scripts/train_watchdog.py (detect-and-
# report). All checks append to $LOGF; non-OK states also append to $ALERTF
# *unless* the intentional-stop marker exists (touched by `train.sh stop`,
# removed by `train.sh start`), so deliberately stopping training doesn't page.
#
# AUTO-RECOVERY (WATCHDOG_AUTO_RECOVER=1, default on): on a confirmed STALLED
# verdict (exit 3 = 90 min flat, PID ALIVE, no pause.txt, no intentional-stop
# marker) the loop runs scripts/recover_stall.sh — a GPU-independent force
# teardown + GPU-bridge wait + train.sh start. This is the recurring WSL2 dxg
# vmbus wedge (memory: wsl2-gpu-vmbus-wedge-signature): the trainer wedges alive
# and `train.sh stop` can't help because ray stop needs the dead GPU.
#
# ABANDONED-PAUSE RECOVERY (exit 5, added 2026-08-09 with scripts/pause_window.sh).
# A held pause.txt makes the stall path structurally unreachable: `decide()`
# returns PAUSED-HELD whenever the loop is flat AND a marker exists, and STALLED
# requires no marker — so a wrapper that dies holding one parks production until
# a human notices, and no amount of waiting turns it into an exit 3. The nightly
# pause window makes that failure a live possibility rather than a thought
# experiment, so the watchdog now separates the two cases and this loop clears
# ONLY the recoverable one. Deliberately NOT recover_stall.sh: nothing is
# wedged, so SIGKILLing a healthy stack to delete one file would be the more
# destructive fix. The marker is removed and the trial resumes at its next
# `pause_poll_seconds`.
#
# Guards that keep this from fighting the operator or flapping:
#   * ONLY STALLED (exit 3) triggers the force-recovery — STOPPED (exit 1, PID
#     gone) never does, so a manual `kill` / deliberate stop is left alone.
#   * Only a marker that NAMES ITS OWNER (`pid=`, written by pause_window.sh)
#     can be cleared, and the removal re-checks that here rather than trusting
#     the parsed verdict line — an operator's graceful_restart.py marker carries
#     no pid and must outlive any bound we could pick.
#   * The intentional-stop marker suppresses recovery entirely.
#   * Anti-flap: after a recovery, refuse to auto-recover again for
#     $RECOVER_COOLDOWN_S (default 2h) — a re-stall that soon means restart isn't
#     fixing it, so escalate to $ALERTF for a human instead of loop-restarting.
#   * WATCHDOG_AUTO_RECOVER=0 disables recovery (revert to detect-only).
#
# Optional: WATCHDOG_NOTIFY_CMD is invoked with the status line on every
# unsuppressed non-OK check (fail-soft, inside train_watchdog.py --notify-cmd).
set -u
# WATCHDOG_ROOT and the three /tmp paths below are TEST SEAMS, not operator
# knobs (the same contract ratchet_common.sh states for RATCHET_ROOT). Without
# them nothing can execute this loop's body, and the abandoned-pause branch
# would be pinned only by reading it — while a test that ran the loop unseamed
# would clobber the LIVE watchdog's flatness state in /tmp and could be
# suppressed by a real intentional-stop marker on the same machine.
cd "${WATCHDOG_ROOT:-/home/josh/projects/chess}" || exit 2
MARKER="${WATCHDOG_STOP_MARKER:-/tmp/chess_training.intentional_stop}"
LOGF=scratchpad/watchdog.log
ALERTF=scratchpad/watchdog_alerts.log
STATEF="${WATCHDOG_STATE:-/tmp/chess_watchdog_state.json}"
RECOVER_STAMP="${WATCHDOG_RECOVER_STAMP:-/tmp/chess_watchdog_last_recover}"
WATCHDOG_EVERY="${WATCHDOG_EVERY:-600}"
AUTO_RECOVER="${WATCHDOG_AUTO_RECOVER:-1}"
RECOVER_COOLDOWN_S="${RECOVER_COOLDOWN_S:-7200}"
EXIT_STALLED=3
EXIT_PAUSE_ABANDONED=5
# --once runs a single check and exits with it; the scheduled invocation from
# scripts/train.sh passes no arguments.
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

mkdir -p scratchpad

stamp(){ date '+%m-%d %H:%M:%S'; }

while true; do
    ARGS=()
    [ -n "${WATCHDOG_NOTIFY_CMD:-}" ] && [ ! -f "$MARKER" ] && \
        ARGS=(--notify-cmd "$WATCHDOG_NOTIFY_CMD")
    LINE=$(PYTHONPATH=. python3 scripts/train_watchdog.py --state "$STATEF" "${ARGS[@]}" 2>&1)
    RC=$?
    echo "$(stamp) $LINE" >> "$LOGF"
    if [ "$RC" -ne 0 ] && [ ! -f "$MARKER" ]; then
        echo "$(stamp) $LINE" >> "$ALERTF"
    fi

    # ── auto-recovery on confirmed STALLED (wedged-but-alive) ────────────
    if [ "$AUTO_RECOVER" = 1 ] && [ "$RC" = "$EXIT_STALLED" ] && [ ! -f "$MARKER" ]; then
        now=$(date +%s)
        last=0; [ -f "$RECOVER_STAMP" ] && last=$(cat "$RECOVER_STAMP" 2>/dev/null || echo 0)
        if [ $((now - last)) -lt "$RECOVER_COOLDOWN_S" ]; then
            echo "$(stamp) AUTO-RECOVER SUPPRESSED (re-stall within cooldown ${RECOVER_COOLDOWN_S}s of last recovery) — NEEDS HUMAN: $LINE" >> "$ALERTF"
        else
            echo "$(stamp) AUTO-RECOVER FIRING (STALLED, no marker): $LINE" | tee -a "$LOGF" >> "$ALERTF"
            echo "$now" > "$RECOVER_STAMP"
            # Run recovery to completion before the next poll (it restarts the stack).
            bash scripts/recover_stall.sh >> "$LOGF" 2>&1
        fi
    fi

    # ── clear an ABANDONED pause marker (wrapper died holding it) ────────
    # No cooldown: this removes one file and cannot restart anything, so the
    # anti-flap reasoning that guards recover_stall.sh does not apply. It is
    # also self-limiting — once the marker is gone the trial resumes, rows
    # grow, and the verdict is OK.
    if [ "$AUTO_RECOVER" = 1 ] && [ "$RC" = "$EXIT_PAUSE_ABANDONED" ] && [ ! -f "$MARKER" ]; then
        pm=$(printf '%s\n' "$LINE" | sed -n 's/.* pause_txt=\([^ ]*\).*/\1/p')
        # ⚑ RE-CHECK `pid=` HERE. The verdict already required a self-identifying
        # marker, but this is the line that DELETES, and it must not depend on a
        # sed of a human-readable status string having parsed the right path. An
        # operator's graceful_restart.py marker ("graceful restart in progress")
        # has no pid= and can never pass this.
        if [ -n "$pm" ] && [ -f "$pm" ] && grep -q 'pid=[0-9]' "$pm"; then
            echo "$(stamp) CLEARING ABANDONED PAUSE MARKER $pm — held: $(tr '\n' ' ' < "$pm"): $LINE" \
                | tee -a "$LOGF" >> "$ALERTF"
            rm -f "$pm"
        else
            echo "$(stamp) PAUSE-ABANDONED but the marker was not clearable (parsed path='$pm') — NEEDS HUMAN: $LINE" \
                | tee -a "$LOGF" >> "$ALERTF"
        fi
    fi

    [ "$ONCE" -eq 1 ] && exit "$RC"
    sleep "$WATCHDOG_EVERY"
done
