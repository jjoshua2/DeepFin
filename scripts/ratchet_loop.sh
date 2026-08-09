#!/bin/bash
# Daily strength-ratchet driver (managed by scripts/train.sh).
#
# Runs scripts/daily_gate_ratchet.sh at most once per calendar day, and ONLY
# while the trainer is actually up. Not cron: cron is not running in this WSL2
# instance, and more importantly a scheduled ratchet fires whenever the clock
# says so — including while training is stopped for a match, a reboot, or
# gaming, where it would silently take the GPU the operator just freed. Tying it
# to the training lifecycle (same pattern as monitor_fen.sh / watchdog_loop.sh)
# means "no training, no ratchet" is structural rather than remembered.
#
# Cadence is per calendar DAY, not per N hours, so a restart cannot make it fire
# twice in one day and a stopped run simply skips that day — the CSV's date
# column stays the honest index of when a regression appeared.
set -u
# The repo, the run directory, the exit statuses and the checkpoint-iteration
# parse are all defined ONCE, in scripts/ratchet_common.sh, and shared with
# daily_gate_ratchet.sh — see the header there for the divergence that made
# that necessary. Sourced by the script's own location so it does not depend on
# the caller's cwd, and before anything else because it performs the `cd`.
. "$(dirname "${BASH_SOURCE[0]}")/ratchet_common.sh"

# --once runs a single poll and exits with that poll's status. The scheduled
# invocation from scripts/train.sh passes no arguments.
#
# NOTE: it exits 0 on every path where the poll deliberately DID NOTHING
# (trainer down, paused, day already stamped, no trial/checkpoint yet, below
# MIN_ITER), so as a status it conflates "skipped" with "succeeded". That is
# fine for what it is — a test seam and a manual one-shot — but do not build a
# scheduler on it without giving "did nothing" its own status.
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

PIDFILE="${TRAIN_PIDFILE:-/tmp/chess_training.pid}"
STATE=data/ratchet/last_run_date
# ATTEMPTED, not SUCCEEDED. $STATE means "today has a reading"; $GIVEUP_STATE
# means "today has no reading and asking again cannot produce one". They are
# separate files on purpose: collapsing them would either stamp a dead day as
# done (the silent hole the exit-1 path exists to prevent) or leave the day
# retrying every $POLL seconds forever. A 90-minute 16-concurrent arena
# retried back-to-back is ~21.6 GPU-hours/day spent by the observer on the
# training it is supposed to be observing (the attempts cap bounds the real
# spend at 4.5), and it is self-reinforcing: contention -> no complete pairs
# -> no row -> retry -> more contention.
GIVEUP_STATE=data/ratchet/last_giveup_date
LOG=scratchpad/ratchet_loop.log
POLL="${RATCHET_POLL:-600}"
# Skip the first N iterations after a restart: a freshly-restarted trial spends
# its early checkpoints re-warming, and ratcheting against those produces a
# baseline that says more about warmup than about the day's training.
MIN_ITER="${RATCHET_MIN_ITER:-5}"

mkdir -p data/ratchet scratchpad

trainer_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

paused() {
    local tune_dir="$WORK_DIR/tune"
    [ -f "$tune_dir/pause.txt" ] && return 0
    ls "$tune_dir"/train_trial_*/pause.txt >/dev/null 2>&1
}

log() { echo "[$(date -Is)] $*" >> "$LOG"; }

# What the day's outcome does to the loop's state. A standalone function so
# tests can EXECUTE it: the whole reason this file needed changing is that the
# "stamps only on exit 0" contract was documented in a comment, asserted
# nowhere, and had no way to express "stop, but not because it worked".
ratchet_outcome () {   # $1=rc  $2=today
    case "$1" in
        0)  echo "$2" > "$STATE"
            log "daily ratchet done — see data/ratchet/ratchet.csv" ;;
        "$RATCHET_EXIT_NO_RETRY")
            # NOT $STATE: the day gets no reading and must not read as one.
            echo "$2" > "$GIVEUP_STATE"
            log "daily ratchet GAVE UP for $2 — zero rows and not retryable." \
                "NO strength measurement for this day; see data/ratchet/attempts.csv" ;;
        *)  log "daily ratchet FAILED (rc=$1) — will retry next poll" ;;
    esac
}

# ONE poll: decide whether to run the ratchet, run it, and record what came
# back. A function so the whole body — including the `ratchet_outcome "$?"`
# call site, which is the wiring the state machine hangs off — is reachable by a
# test rather than only by reading.
poll_once () {
    trainer_running || return 0
    # A paused trial still holds its PID, so PIDFILE alone would let the ratchet
    # run during a deliberate pause.
    paused && return 0

    local today trial ck iter rc
    today=$(date +%F)
    [ "$(cat "$STATE" 2>/dev/null)" = "$today" ] && return 0
    [ "$(cat "$GIVEUP_STATE" 2>/dev/null)" = "$today" ] && return 0

    trial=$(ls -td "$WORK_DIR"/tune/train_trial_*/ 2>/dev/null | head -1)
    [ -n "$trial" ] || return 0
    ck=$(ls -td "$trial"checkpoint_* 2>/dev/null | head -1)
    [ -n "$ck" ] || return 0
    iter=$(ratchet_iter_from_checkpoint "$ck")
    [ "${iter:-0}" -ge "$MIN_ITER" ] 2>/dev/null || return 0

    # Re-check immediately before spending GPU: the gap between the poll above
    # and here is where a stop/pause lands.
    trainer_running || return 0
    paused && return 0

    log "starting daily ratchet (iter=$iter)"
    # `$STATE` is stamped only on exit 0, so a failed run retries on the next
    # poll instead of silently skipping the whole day — but a failure that
    # reproduces would then retry until midnight, which the ratchet reports as
    # exit $RATCHET_EXIT_NO_RETRY and this loop honours WITHOUT claiming the day
    # succeeded. Capture the status into a local FIRST: any command inserted
    # between the run and the `ratchet_outcome` call would otherwise clobber
    # `$?` and turn every outcome into a success.
    # ⚑ RUN THE WHOLE RATCHET INSIDE ONE PAUSE WINDOW, not beside training.
    # Measured 2026-08-09: beside training the lineage series truncated at
    # 106/200 with a CI spanning zero — an unreadable result — while iterations
    # stretched 245s -> 611s. The same series in a pause window got 200/200 in
    # ~21 min. Wrapping HERE rather than inside daily_gate_ratchet.sh pauses
    # ONCE for both series (matching the ~35 min 2x200 measurement) and keeps
    # the wrapper's chatter out of the per-arena logs the outcome parser reads.
    #
    # The `paused` guard above has already run, so this cannot collide with an
    # operator's own pause: we only ever set the marker after finding none.
    #
    # Every trappable failure path releases the marker, so a wrapper failure
    # degrades to the old contended behaviour rather than parking production.
    # The residual is SIGKILL, which no trap can catch — `train.sh start` clears
    # stale markers, and the trainer watchdog alerts on a stalled loop.
    # Set CAE_RATCHET_PAUSE_WINDOW=0 to go back to running beside training.
    if [ "${CAE_RATCHET_PAUSE_WINDOW:-1}" = "1" ]; then
        bash scripts/pause_window.sh -- bash scripts/daily_gate_ratchet.sh >> "$LOG" 2>&1
    else
        bash scripts/daily_gate_ratchet.sh >> "$LOG" 2>&1
    fi
    rc=$?
    ratchet_outcome "$rc" "$today"
    return "$rc"
}

while true; do
    sleep "$POLL"
    poll_once
    rc=$?
    [ "$ONCE" -eq 1 ] && exit "$rc"
done
