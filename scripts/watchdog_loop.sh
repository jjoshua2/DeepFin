#!/bin/bash
# Training-health watchdog loop (managed by scripts/train.sh; runs forever).
#
# Every $WATCHDOG_EVERY seconds runs scripts/train_watchdog.py (detect-and-
# report). EVERY check appends to $LOGF — that stays the complete record.
# $ALERTF gets a line only on a STATE TRANSITION into a non-OK state, and only
# while the intentional-stop marker is absent (touched by `train.sh stop`,
# removed by `train.sh start`), so deliberately stopping training doesn't page.
#
# TRANSITION-ONLY, added 2026-08-08. The outage that day appended 27 identical
# STOPPED lines across 4h32m and nobody read any of them. Repeating yourself
# every 10 minutes trains the reader to ignore the file; one line per episode
# means every line in $ALERTF is news. $WATCHDOG_NOTIFY_CMD (optional) is
# invoked on the same transitions, fail-soft.
#
# A dead PID now reports CRASHED (no marker) vs STOPPED (marker present), so
# the log distinguishes "it died" from "the operator stopped it" — the
# distinction whose absence made that outage invisible.
#
# AUTO-RECOVERY (WATCHDOG_AUTO_RECOVER=1, default on): on a confirmed STALLED
# verdict (exit 3 = 90 min flat, PID ALIVE, no pause.txt, no intentional-stop
# marker) the loop runs scripts/recover_stall.sh — a GPU-independent force
# teardown + GPU-bridge wait + train.sh start. This is the recurring WSL2 dxg
# vmbus wedge (memory: wsl2-gpu-vmbus-wedge-signature): the trainer wedges alive
# and `train.sh stop` can't help because ray stop needs the dead GPU.
#
# Guards that keep this from fighting the operator or flapping:
#   * ONLY STALLED (exit 3) triggers recovery — STOPPED (exit 1, PID gone) never
#     does, so a manual `kill` / deliberate stop is left alone (could be intended).
#   * The intentional-stop marker suppresses recovery entirely.
#   * Anti-flap: after a recovery, refuse to auto-recover again for
#     $RECOVER_COOLDOWN_S (default 2h) — a re-stall that soon means restart isn't
#     fixing it, so escalate to $ALERTF for a human instead of loop-restarting.
#   * WATCHDOG_AUTO_RECOVER=0 disables recovery (revert to detect-only).
#
# Optional: WATCHDOG_NOTIFY_CMD is invoked with the status line on each
# unsuppressed non-OK TRANSITION (fail-soft, invoked here — NOT via
# train_watchdog.py --notify-cmd, which fires on every check and would
# reintroduce exactly the per-poll spam this loop now suppresses).
set -u
# WATCHDOG_ROOT / WATCHDOG_MAX_ITERS and the four path overrides below are TEST
# SEAMS, not operator knobs (same convention as ratchet_common.sh). Without them
# the transition-only alerter — the whole point of this file — could only be
# pinned by reading it, and "reads correct" is exactly how the 27-identical-lines
# behaviour survived in the first place.
cd "${WATCHDOG_ROOT:-/home/josh/projects/chess}" || exit 2
MARKER="${WATCHDOG_MARKER:-/tmp/chess_training.intentional_stop}"
LOGF="${WATCHDOG_LOGF:-scratchpad/watchdog.log}"
ALERTF="${WATCHDOG_ALERTF:-scratchpad/watchdog_alerts.log}"
STATEF="${WATCHDOG_STATEF:-/tmp/chess_watchdog_state.json}"
RECOVER_STAMP=/tmp/chess_watchdog_last_recover
MAX_ITERS="${WATCHDOG_MAX_ITERS:-0}"   # 0 = run forever (production)
WATCHDOG_EVERY="${WATCHDOG_EVERY:-600}"
AUTO_RECOVER="${WATCHDOG_AUTO_RECOVER:-1}"
RECOVER_COOLDOWN_S="${RECOVER_COOLDOWN_S:-7200}"
EXIT_STALLED=3
# Last state we ALERTED on, so a persisting condition alerts ONCE per episode
# rather than once per poll. The 2026-08-08 outage wrote 27 identical STOPPED
# lines over 4h32m; a file that repeats itself every 10 minutes is a log, not
# an alert, and nobody read it. Transition-only makes each line mean "something
# changed" — which is the only thing worth waking someone for.
LAST_ALERT_F="${WATCHDOG_LAST_ALERT_F:-/tmp/chess_watchdog_last_alert}"

stamp(){ date '+%m-%d %H:%M:%S'; }
# "watchdog: CRASHED pid=... rows=..." -> "CRASHED". Field 2 of the line.
state_of(){ printf '%s' "$1" | awk '{print $2}'; }

ITER=0
while true; do
    ITER=$((ITER + 1))
    # --notify-cmd is deliberately NOT passed: notification is a TRANSITION
    # decision, and train_watchdog.py fires it on every non-OK check. One policy,
    # one place — the loop owns "is this worth telling someone about".
    LINE=$(PYTHONPATH=. python3 scripts/train_watchdog.py --state "$STATEF" 2>&1)
    RC=$?
    echo "$(stamp) $LINE" >> "$LOGF"

    STATE=$(state_of "$LINE")
    LAST_ALERT=""; [ -f "$LAST_ALERT_F" ] && LAST_ALERT=$(cat "$LAST_ALERT_F" 2>/dev/null || echo "")
    if [ "$RC" -ne 0 ] && [ ! -f "$MARKER" ]; then
        if [ "$STATE" != "$LAST_ALERT" ]; then
            echo "$(stamp) $LINE" >> "$ALERTF"
            printf '%s' "$STATE" > "$LAST_ALERT_F"
            if [ -n "${WATCHDOG_NOTIFY_CMD:-}" ]; then
                # Fail-soft: a broken notifier must never kill the watchdog.
                "$WATCHDOG_NOTIFY_CMD" "$LINE" >/dev/null 2>&1 || true
            fi
        fi
    else
        # Back to OK (or a deliberate stop): re-arm so the NEXT episode alerts.
        [ -n "$LAST_ALERT" ] && rm -f "$LAST_ALERT_F"
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

    if [ "$MAX_ITERS" -gt 0 ] && [ "$ITER" -ge "$MAX_ITERS" ]; then
        break
    fi
    sleep "$WATCHDOG_EVERY"
done
