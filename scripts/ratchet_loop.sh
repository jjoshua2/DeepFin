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
# Consecutive pause-window failures, as "<date> <count>". A wrapper that cannot
# park (wedged trial, long iteration, a drain that refuses) is NOT a ratchet
# verdict, so `ratchet_outcome` neither stamps the day nor gives up on it -- it
# just retries, every $POLL, setting and holding the marker each time. That is
# invisible too: `loop_health.ratchet_gap_alerts()` keys on attempts.csv, which
# never gets a row. After $PAUSE_MAX_FAILS the loop stops asking for a window
# and takes the contended reading instead, because a noisy measurement beats a
# day of repeatedly parking production and measuring nothing.
PAUSE_FAIL_STATE=data/ratchet/pause_window_fails
PAUSE_MAX_FAILS="${CAE_RATCHET_PAUSE_MAX_FAILS:-2}"
# ⚑ MUST EQUAL pause_window.sh's WRAPPER_FAILED_RC, and must not collide with
# any status daily_gate_ratchet.sh can return. It was 3, which ratchet_common.sh
# already documents as "the arena's own no-pairs status" -- the same class of
# mistake as the exit-5/CRASHED collision, found the same way (by reading the
# file the comment was paraphrasing). A test derives the taken set from those
# files rather than restating it, and mutating this constant to 1 -- which would
# make every routine RETRY count as a pause-window failure and silently disable
# the window for the rest of the day after two polls -- now fails.
PAUSE_WINDOW_FAILED_RC=7
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

    trial=$(ratchet_newest_trial_dir)
    [ -n "$trial" ] || return 0
    ck=$(ls -td "$trial"checkpoint_* 2>/dev/null | head -1)
    [ -n "$ck" ] || return 0
    # ⚑ Ray creates `checkpoint_NNNNNN/` before writing into it, and
    # `daily_gate_ratchet.sh` treats that as a cheap RETRY -- its comment calls
    # it "a live race that every fresh restart passes through" and says these
    # paths cost NO GPU and are not counted against the day's attempts. Note it
    # here so the WINDOW can be skipped below; the ratchet still runs, because
    # the loop surfacing its retryable status is an existing contract
    # (`test_nothing_to_measure_yet_is_retryable_not_a_finished_day`). What we
    # refuse to do is pay a park plus a four-worker relaunch (model reload, AOT
    # compile, re-lease) for a callee that will exit immediately.
    local ck_ready=0
    [ -f "$ck/trainer.pt" ] && ck_ready=1
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
    # Measured 2026-08-09 (same snapshots, same seed, same argv but for --label;
    # runs/arena_results.jsonl): beside training the lineage series truncated at
    # 106/200 games in 2520.3s, reporting at 53% of its configured resolution.
    # The same series in a pause window got 200/200 in 1160.1s. Cost to the loop,
    # from result.json: 15.1 iterations lost beside training against 7.1 parked.
    # Wrapping HERE rather than inside daily_gate_ratchet.sh pauses ONCE for both
    # series and keeps the wrapper's chatter out of the per-arena logs the
    # outcome parser reads.
    # ⚑ This comment used to say "iterations stretched 245s -> 611s ... 200/200
    # in ~21 min". Those are the RETRACTED figures (pause_window.sh:19-24): the
    # pre-arena median was 247.8s and the peak 1628.0s, not 611.3s. The
    # retraction was written into two files and missed this third one -- which is
    # why the sweep is now `grep -rn '245s\|2\.5x\|611s' scripts/ docs/`.
    #
    # The `paused` guard above has already run, so this cannot collide with an
    # operator's own pause: we only ever set the marker after finding none.
    #
    # Every trappable failure path inside the wrapper releases the marker, so a
    # wrapper failure degrades to the old contended behaviour rather than
    # parking production. The residual is SIGKILL, which no trap can catch.
    # Three things cover it, and the third exists because of this PR:
    #   * `train.sh start` clears stale markers;
    #   * the marker records `pid=` and `started=`, so an operator can tell "the
    #     ratchet is running" from "a dead window parked production";
    #   * `train_watchdog.py` returns PAUSE-ABANDONED (exit 6) for a marker whose
    #     named owner is gone, and `watchdog_loop.sh` removes it.
    # ⚑ THAT THIRD ONE IS NEW, AND THE CLAIM IT REPLACES WAS FALSE. This comment
    # used to say "the trainer watchdog alerts on a stalled loop". It alerts, but
    # it could not RECOVER: `decide()` returns PAUSED-HELD whenever a marker is
    # present and STALLED requires `pause_txt is None`, so with the marker held
    # the exit-3 branch `watchdog_loop.sh` recovers on was unreachable by
    # construction, and so was `recover_stall.sh`'s `rm -f ... pause.txt`.
    # Set CAE_RATCHET_PAUSE_WINDOW=0 to go back to running beside training.
    #
    # ⚑ THE PRESENCE TEST IS NOT DEFENSIVE PADDING — WITHOUT IT THE SENTENCE
    # ABOVE IS FALSE. `bash <missing-script>` exits 127, which `ratchet_outcome`
    # reads as an ordinary ratchet failure: no row, no give-up stamp, and a
    # retry every $POLL until midnight. So the one thing that cannot find the
    # wrapper loses the whole day's strength measurement AND spends the day
    # asking again. CI caught it (three tests, rc=127) because the test sandbox
    # copies only the scripts it names. "Degrades to the old behaviour" has to
    # be a branch, not a hope.
    # `fail_day` is local too: `read` leaves its targets UNCHANGED when the file
    # is missing, so an undeclared one would carry the previous poll's date into
    # this poll's comparison. Harmless today only because `pause_fails` is
    # re-initialised to 0 on every call — i.e. the guard is doing nothing and the
    # safety comes from somewhere else, which is the shape worth not shipping.
    local pause_ok=0 pause_fails=0 fail_day=
    read -r fail_day pause_fails < "$PAUSE_FAIL_STATE" 2>/dev/null || true
    [ "${fail_day:-}" = "$today" ] || pause_fails=0
    if [ "${CAE_RATCHET_PAUSE_WINDOW:-1}" = "1" ]; then
        if [ "${pause_fails:-0}" -ge "$PAUSE_MAX_FAILS" ] 2>/dev/null; then
            log "pause window failed ${pause_fails}x today (cap $PAUSE_MAX_FAILS)" \
                "-- running BESIDE training for the rest of $today so the day still gets a reading"
        elif [ "$ck_ready" = "0" ]; then
            log "checkpoint has no trainer.pt yet -- running the ratchet WITHOUT a window;" \
                "it exits retryable at no GPU cost, and a park here buys nothing"
        elif [ -r scripts/pause_window.sh ]; then
            pause_ok=1
        else
            log "WARNING: scripts/pause_window.sh not readable — running the ratchet" \
                "BESIDE training; expect ~half the games and a CI that may span zero"
        fi
    fi
    if [ "$pause_ok" = "1" ]; then
        # ⚑ PASS THE WORK DIR. `ratchet_common.sh` exists because the loop and the
        # ratchet had drifted on exactly this; the wrapper is a THIRD participant
        # with its own hardcoded default, so with TRAIN_WORK_DIR set it would
        # pause and drain the PRODUCTION run while the ratchet measured the other
        # tree. `test_neither_script_can_disagree_with_the_other` enumerates only
        # two scripts and cannot see it.
        # ⚑ PASS THE TRIAL, not just the work dir. Without it the wrapper runs
        # its own selector and can resolve a DIFFERENT trial from the one whose
        # `iter`, `ck_ready` and snapshot this poll is built on -- so production
        # gets parked to measure a trial that is not the one being parked. They
        # now agree by construction: one selection, passed down.
        bash scripts/pause_window.sh --work-dir "$WORK_DIR" --trial-id "$(basename "$trial")" \
            -- bash scripts/daily_gate_ratchet.sh >> "$LOG" 2>&1
    else
        bash scripts/daily_gate_ratchet.sh >> "$LOG" 2>&1
    fi
    rc=$?
    # Count the WRAPPER's own failures, and only those: exit 7 is reserved for
    # them (pause_window.sh:die), so a ratchet that legitimately returned RETRY
    # or NO_RETRY cannot burn the budget that exists to stop production being
    # parked for nothing.
    if [ "$pause_ok" = "1" ] && [ "$rc" = "$PAUSE_WINDOW_FAILED_RC" ]; then
        echo "$today $((pause_fails + 1))" > "$PAUSE_FAIL_STATE"
        log "pause window FAILED (rc=$rc), $((pause_fails + 1))/$PAUSE_MAX_FAILS for $today"
    elif [ "$pause_ok" = "1" ]; then
        rm -f "$PAUSE_FAIL_STATE"
    fi
    ratchet_outcome "$rc" "$today"
    return "$rc"
}

while true; do
    sleep "$POLL"
    poll_once
    rc=$?
    [ "$ONCE" -eq 1 ] && exit "$rc"
done
