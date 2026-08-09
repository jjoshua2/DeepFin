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
#
# THE DEDUPE MUST NEVER BE ABLE TO SILENCE THE ALERTER. De-duplicating is one
# `!=` away from muting, so two properties are load-bearing and both are tested:
#   * The dedupe key is armed ONLY after the append to $ALERTF succeeds, so an
#     undelivered alert is retried rather than deduped away.
#   * Output the watchdog was never supposed to emit (empty — it was SIGKILLed;
#     or a traceback) gets a SYNTHESIZED state, so it can neither collide with
#     the empty initial key nor build a multi-line one. A broken watchdog is the
#     loudest case, not the quietest.
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
RECOVER_STAMP="${WATCHDOG_RECOVER_STAMP:-/tmp/chess_watchdog_last_recover}"
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
# The AUTO-RECOVER-SUPPRESSED escalation dedupes on its own key: it is a
# different piece of news from the STALLED verdict that produced it, but a
# persisting stall inside the cooldown must not re-say it every 10 minutes
# either. Both keys clear together when the state goes back to OK.
LAST_ESCALATE_F="${WATCHDOG_LAST_ESCALATE_F:-/tmp/chess_watchdog_last_escalate}"

stamp(){ date '+%m-%d %H:%M:%S'; }

# "watchdog: CRASHED pid=... rows=..." -> "CRASHED". Field 2 of the FIRST line:
# on an unhandled traceback $LINE is multi-line and a bare '{print $2}' would
# build a multi-line "state" that can never compare equal to anything.
# An empty state means the watchdog itself produced nothing (SIGKILL, interpreter
# death before main()). That is the MOST alarming case, so it must not fall
# through to the empty-string default and silently match the "" initial value of
# the dedupe key -- that would suppress this poll AND, since nothing gets
# written, every poll after it. Synthesize a state instead.
#
# Field 2 is only meaningful when the line IS the tool's contract. Trusting it
# unconditionally keys a traceback on "(most" (from "Traceback (most recent call
# last):"), so two DIFFERENT watchdog crashes collapse into one episode and the
# key left in /tmp tells an incident responder nothing.
state_of(){
    local s=""
    case "$1" in watchdog:*) s=$(printf '%s' "$1" | awk 'NR==1{print $2}') ;; esac
    [ -n "$s" ] || s="UNPARSEABLE-rc$2"
    printf '%s' "$s"
}

# Append to $ALERTF iff $2 (the dedupe key) differs from what $1 holds, and arm
# the key ONLY once the append has actually landed. Arming on a failed append
# (disk full on the repo partition, $ALERTF's directory gone -- $ALERTF and the
# key file are on DIFFERENT filesystems) would dedupe away an episode that was
# never delivered: an alert accepted and then silently dropped, which is the
# exact defect class this file exists to fix. Returns 0 iff a line was written.
# The ONLY writer to $ALERTF, so every alert in this file is checked and
# flattened by construction. ONE LINE PER ALERT is the file's whole contract --
# "every line here is news" -- and a traceback would otherwise land as N lines
# and read as N alerts. $LOGF keeps the unflattened text. Returns non-zero if
# the line did not land, so no caller can record "delivered" for an alert that
# was not.
alert_write(){
    local msg
    msg=$(printf '%s' "$1" | tr '\n' ' ')
    echo "$(stamp) $msg" >> "$ALERTF"
}

alert_once(){
    local keyf="$1" key="$2" msg="$3" last=""
    [ -f "$keyf" ] && last=$(cat "$keyf" 2>/dev/null || echo "")
    [ "$key" = "$last" ] && return 1
    alert_write "$msg" || return 1
    printf '%s' "$key" > "$keyf"
    return 0
}

# Notifier dispatch, kept at PARITY with train_watchdog.py's maybe_notify /
# _looks_like_shell: a multi-word command or pipeline goes through a shell, a
# bare executable path is exec'd directly. A path CONTAINING a space is
# unsupported by both paths -- it matches the metachar class and then word-splits
# in the shell. Moving notification out of --notify-cmd must not quietly narrow what
# WATCHDOG_NOTIFY_CMD accepts -- `notify-send training` is the form the tool's
# own docstring advertises, and "$cmd" "$msg" would look for an executable
# literally named "notify-send training" and fail into /dev/null.
notify(){
    case "$1" in
        *[\ \|\;\&\>\<\`\$]*) sh -c "$1 \"\$0\"" "$2" ;;
        *)                    "$1" "$2" ;;
    esac
}

ITER=0
while true; do
    ITER=$((ITER + 1))
    # --notify-cmd is deliberately NOT passed: notification is a TRANSITION
    # decision, and train_watchdog.py fires it on every non-OK check. One policy,
    # one place — the loop owns "is this worth telling someone about".
    LINE=$(PYTHONPATH=. python3 scripts/train_watchdog.py --state "$STATEF" 2>&1)
    RC=$?
    echo "$(stamp) $LINE" >> "$LOGF"

    STATE=$(state_of "$LINE" "$RC")
    # A silent watchdog still has to say something, or the alert line is a bare
    # timestamp and the reader learns nothing about why.
    MSG="$LINE"
    [ -n "$MSG" ] || MSG="watchdog: $STATE (the watchdog itself produced no output; rc=$RC)"

    if [ "$RC" -ne 0 ] && [ ! -f "$MARKER" ]; then
        if alert_once "$LAST_ALERT_F" "$STATE" "$MSG"; then
            # A NEW primary episode makes the escalation news again. Clearing it
            # only on the OK branch is not enough: STALLED -> CRASHED -> STALLED
            # never touches OK, so the second stall would re-alert but its
            # higher-severity "auto-recovery cannot fix this" line would be
            # deduped away against the first stall's key.
            rm -f "$LAST_ESCALATE_F"
            if [ -n "${WATCHDOG_NOTIFY_CMD:-}" ]; then
                # Fail-soft: a broken notifier must never kill the watchdog.
                notify "$WATCHDOG_NOTIFY_CMD" "$MSG" >/dev/null 2>&1 || true
            fi
        fi
    else
        # Back to OK (or a deliberate stop): re-arm so the NEXT episode alerts.
        # Clearing LAST_ESCALATE_F here is REDUNDANT and deliberately kept: every
        # route to a new escalation now passes through a new primary alert, which
        # clears it above, so no test can distinguish this line's presence (its
        # mutation survives, by design). It is the belt to that braces -- if the
        # primary path is ever restructured, the invariant "the escalate key never
        # outlives its primary episode" still holds at the OK boundary.
        rm -f "$LAST_ALERT_F" "$LAST_ESCALATE_F"
    fi

    # ── auto-recovery on confirmed STALLED (wedged-but-alive) ────────────
    if [ "$AUTO_RECOVER" = 1 ] && [ "$RC" = "$EXIT_STALLED" ] && [ ! -f "$MARKER" ]; then
        now=$(date +%s)
        last=0; [ -f "$RECOVER_STAMP" ] && last=$(cat "$RECOVER_STAMP" 2>/dev/null || echo 0)
        if [ $((now - last)) -lt "$RECOVER_COOLDOWN_S" ]; then
            # Deduped like any other alert: a stall that persists through the 2h
            # cooldown is ONE piece of news, not one per poll. Un-gated, this
            # branch alone reproduced the 27-identical-lines pattern the rest of
            # this file exists to kill.
            alert_once "$LAST_ESCALATE_F" "SUPPRESSED-$STATE" \
                "AUTO-RECOVER SUPPRESSED (re-stall within cooldown ${RECOVER_COOLDOWN_S}s of last recovery) — NEEDS HUMAN: $MSG" || true
        else
            # NOT deduped: the cooldown stamp below already bounds this to once
            # per $RECOVER_COOLDOWN_S, so every FIRING line is by construction a
            # distinct event. It goes through alert_write only for the CHECKED
            # append -- the previous `tee -a … >> "$ALERTF"` could drop the alert
            # silently while the stamp below still armed, which is the same
            # accept-then-drop shape as the two blocking bugs above.
            echo "$(stamp) AUTO-RECOVER FIRING (STALLED, no marker): $LINE" >> "$LOGF"
            alert_write "AUTO-RECOVER FIRING (STALLED, no marker): $MSG" \
                || echo "$(stamp) WARN: FIRING alert could not be appended to $ALERTF" >> "$LOGF"
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
