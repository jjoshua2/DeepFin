#!/usr/bin/env bash
# Run a command with live training PAUSED and the selfplay workers DRAINED.
#
# WHY THIS EXISTS. Measured 2026-08-09 (docs/experiment_ledger.md, "the ack-gated
# pause window"), one series, same snapshots, same argv but for --label, from
# runs/arena_results.jsonl:
#
#             label                                games  wall     Elo (95% CI)
#   beside    ratchet_2026-08-09_iter514_vs_prev    106    2520.3s  -46.2 [-108.1,+13.0]
#   parked    pausewindow_2026-08-09_iter514_vs_prev 200   1160.1s  -31.4 [ -73.8,+10.2]
#
# 1.89x the games in 0.46x the wall time = 4.1x games/second, because the arena
# and selfplay contend for the same CPU (Stockfish is ~95% of loop cost). The
# contended run hit its --max-seconds budget at 53/100 opening pairs, so it
# reported at HALF the resolution it was configured for (+/-60.5 Elo against
# +/-42.0); the point is the truncation, not the sign -- both readings are
# nulls.
#
# ⚑ AN EARLIER REVISION OF THIS HEADER SAID "~2.5x more games ... iterations
# stretched 245s -> 611s ... recovered to 297s the moment it ended". None of
# those three survives its source. 200/106 is 1.89x, not 2.5x; 611.3s (iter 518)
# is not the peak, which was 1628.0s (iter 520); and the first post-arena
# iteration was 350.3s, with ~250s not reached again until iter 534, ~50 min
# later. The measured numbers are above and in the ledger entry.
#
# WHAT THE PAUSE COSTS, in the loop's own currency (result.json, same trial):
# the window is ONE stretched iteration -- iter 568, 1732.5s, against a 293.6s
# local baseline -- so 4.9 iterations of training are lost for a complete
# 200-game row. Running beside training instead cost 15.1 (iters 517-523: 7
# iterations in 5651.5s where the 256.2s baseline fits 22.1) and still
# truncated. Parking is the CHEAPER of the two, which is not the intuitive
# direction and is why the arithmetic is in the ledger.
#
# THE ORDER IS THE WHOLE TRICK, and each step exists because the obvious
# version is silently wrong:
#
#   1. snapshot each worker log's BYTE OFFSET
#   2. touch <work_dir>/tune/pause.txt
#   3. WAIT FOR .paused_<LIVE_TRIAL_ID>.ack        <-- never a timer
#   4. SIGTERM the workers
#   5. run the command
#   6. remove the marker
#
# Step 3 is load-bearing. `_wait_if_paused` (tune/trainable.py) parks at the TOP
# of an iteration, and `_revive_fleet` is called INSIDE
# `_ingest_distributed_selfplay` (tune/trainable_phases.py). Revive is therefore
# inert ONLY while the trial is parked. SIGTERM before the ack and the driver
# simply relaunches the workers mid-ingest, and the pause buys nothing.
#
# SIGTERM, not SIGSTOP. SIGTERM runs the production suspend path -- the worker's
# handler sets a flag, the per-ply `_stop_fn` poll banks in-flight games into
# `selfplay_resume/`, and the next session claims them. SIGSTOP would instead
# freeze four workers that hold server leases for the whole window, which is
# untested and adjacent to the A17 lease-steal defect.
set -euo pipefail

# ⚑ EVERY PATH BELOW IS RELATIVE TO THE CALLER'S CWD, deliberately. An earlier
# revision computed a REPO_ROOT and never used it, which reads as "this script
# anchors itself" while it does not. It does not `cd` because its caller may
# legitimately point it at another tree with --work-dir; `ratchet_common.sh`
# has already cd'd to the repo by the time ratchet_loop.sh invokes it.
WORK_DIR="runs/pbt2_small"
TRIAL_ID=""
ACK_TIMEOUT="${CAE_PAUSE_ACK_TIMEOUT:-1800}"
DRAIN_TIMEOUT="${CAE_PAUSE_DRAIN_TIMEOUT:-180}"
POLL="${CAE_PAUSE_POLL_SECONDS:-5}"

usage() {
    cat >&2 <<'USAGE'
usage: pause_window.sh [--work-dir DIR] [--trial-id ID] [--ack-timeout S]
                       [--drain-timeout S] -- <command> [args...]

Pauses the live trial, drains selfplay, runs <command>, then resumes.
The marker is ALWAYS removed, including when <command> fails or this script is
interrupted -- leaving training parked is the one unrecoverable mistake here.
USAGE
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --trial-id) TRIAL_ID="$2"; shift 2 ;;
        --ack-timeout) ACK_TIMEOUT="$2"; shift 2 ;;
        --drain-timeout) DRAIN_TIMEOUT="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) usage ;;
        *) echo "pause_window.sh: unknown option '$1'" >&2; usage ;;
    esac
done
[ $# -ge 1 ] || usage

TUNE_DIR="$WORK_DIR/tune"
MARKER="$TUNE_DIR/pause.txt"

# ── The worker pattern ───────────────────────────────────────────────────────
# ⚑ DUPLICATED FROM scripts/train.sh ON PURPOSE, AND PINNED EQUAL BY A TEST.
# train.sh's stop() defines it as `local wpat='...'` inside a function, so it
# cannot be sourced, and extracting it here at runtime would make this script
# depend on train.sh's formatting.
# tests/test_pause_window.py::test_the_worker_pattern_matches_train_sh_exactly
# asserts the two literals are byte-identical, which is the same guarantee
# tests/test_train_sh_worker_drain.py::
# test_the_worker_pattern_is_defined_once_and_used_by_both_passes gives inside
# train.sh. `--` is REQUIRED for pgrep/pkill: the pattern begins with `-m`.
WORKER_PATTERN='-m chess_anti_engine\.worker( |$)'

log() { echo "[pause-window] $*"; }
# ⚑ EXIT 3 IS "THE WRAPPER FAILED", AND IT IS DISTINCT ON PURPOSE. The job's own
# status is passed through untouched at the end, and the ratchet's vocabulary is
# 0 / RATCHET_EXIT_RETRY=1 / RATCHET_EXIT_NO_RETRY=5. Sharing 1 with the ratchet
# made "I could not pause" indistinguishable from "the arena produced no rows",
# so a wrapper that could never park read as an ordinary retryable failure and
# came back every poll until midnight. 2 stays reserved for usage.
die() { echo "[pause-window] ERROR: $*" >&2; exit 3; }

resolve_trial_id() {
    # `.paused_<trial_id>.ack` is written next to the marker. The trial id is the
    # middle of `train_trial_<id>_<n>_<params>_<date>`; take it from the most
    # recently modified trial dir unless the caller named one.
    #
    # ⚑ A FAILED PARSE MUST NOT BE A TRIAL ID. `sed` prints its input unchanged
    # when the pattern does not match, so a directory this regex does not
    # recognise yields the whole `train_trial_...` name -- which then becomes
    # `.paused_train_trial_....ack`, a file the trial will never write, and the
    # script waits out the full ACK_TIMEOUT holding the marker before dying with
    # a message about the trial not parking. That is the expensive way to
    # discover a naming change, so the shape is checked here, before the marker.
    # Two mutations this pins: deleting the validation, and breaking the regex
    # (`[^_]+_[0-9]+` -> `.*`, which matches and returns garbage).
    local newest id
    newest="$(ls -1dt "$TUNE_DIR"/train_trial_* 2>/dev/null | head -1 || true)"
    [ -n "$newest" ] || die "no train_trial_* under $TUNE_DIR; pass --trial-id"
    id="$(basename "$newest" | sed -E 's/^train_trial_([^_]+_[0-9]+)_.*$/\1/')"
    printf '%s\n' "$id" | grep -Eq '^[A-Za-z0-9]+_[0-9]+$' \
        || die "could not parse a trial id out of '$(basename "$newest")' (got '$id') -- the trial-dir naming has changed; pass --trial-id"
    printf '%s\n' "$id"
}

[ -d "$TUNE_DIR" ] || die "no such tune dir: $TUNE_DIR"

# ⚑ REFUSE TO PILE ONTO SOMEONE ELSE'S PAUSE. Two overlapping windows would race
# on the marker: whichever finished first would resume training under the other.
[ ! -e "$MARKER" ] || die "$MARKER already exists -- another pause window is active"

[ -n "$TRIAL_ID" ] || TRIAL_ID="$(resolve_trial_id)"
ACK="$TUNE_DIR/.paused_${TRIAL_ID}.ack"
log "trial=$TRIAL_ID  marker=$MARKER  ack=$(basename "$ACK")"

# ⚑ A STALE ACK FROM A DEAD TRIAL IS A GUARD THAT CANNOT FAIL.
# `_clear_pause_acks` runs in a `finally`, which a hard kill skips, so the tune
# dir really does accumulate `.paused_<old_trial>.ack` files (one was found on
# 2026-08-09). A wait that polls for "an ack exists" is satisfied INSTANTLY and
# kills the workers while revive is still live.
#
# ⚑ AND DO NOT DELETE IT. An earlier revision removed a pre-existing ack as
# "stale by construction". It is not: window A can release the marker while the
# trial is still parked (it only re-reads at `pause_poll_seconds`, production
# default 60), so a LIVE ack sits there. Deleting it destroys the signal
# `graceful_restart.py` uses as its PRIMARY pause detector, and the trial never
# rewrites it because `_wait_if_paused` guards on an `announced` flag.
#
# The correct test is FRESHNESS, and it already exists 300 lines away:
# `graceful_restart.py:_pause_ack_files` counts only acks touched at/after the
# pause request. Same rule here -- accept an ack whose mtime is at least our
# marker's, never one older, and never remove anything.
PREEXISTING_ACK=0
if [ -e "$ACK" ]; then
    PREEXISTING_ACK=1
    log "NOTE: an ack for this trial already exists (mtime $(stat -c%y "$ACK" 2>/dev/null)):"
    log "      $(tr -d '\n' < "$ACK" 2>/dev/null)"
    log "      it will be IGNORED unless the trial re-acks after our marker."
fi

# ── Baselines, taken BEFORE anything is signalled ────────────────────────────
# The suspend evidence must be read from the byte offset each log had before the
# drain: `selfplay resume: suspended games=` is also emitted by every in-session
# reco restart, so grepping the whole log finds an hours-old line and reads it as
# proof that THIS drain worked.
# ⚑ pgrep's EXIT STATUS IS THE DIFFERENCE BETWEEN "no workers" AND "my pattern
# is broken", and `|| true` erased it. 0 = matched, 1 = no match, >=2 = usage
# or operational error -- which is exactly what a dropped `--` produces
# (`pgrep -f "-m chess..."` => "invalid option -- 'm'", rc 2). Swallowed, that
# reads as "nothing to drain", the job runs beside a full fleet, and the
# contended arena is recorded as a clean strength row.
pg_rc=0
WORKER_PIDS="$(pgrep -f -- "$WORKER_PATTERN" 2>/dev/null)" || pg_rc=$?
[ "$pg_rc" -lt 2 ] || die "pgrep failed (rc=$pg_rc) -- the worker pattern is broken, so the drain would silently match nothing"
OFFSETS="$(mktemp)"; OFFSET_WHY="$(mktemp)"; RESUME_BEFORE="$(mktemp)"
cleanup_tmp() { rm -f "$OFFSETS" "$OFFSET_WHY" "$RESUME_BEFORE"; }

if [ -n "$WORKER_PIDS" ]; then
    for pid in $WORKER_PIDS; do
        # ⚑ THE ANCHORED PARSE, COPIED FROM train.sh:266 -- and its THREE
        # STATES, which matter more than the parse. An earlier revision used
        # `grep -A1 -- '--log-file' | tail -1`, which is weaker three ways: it
        # matches `--log-file-level` and any argv element merely CONTAINING the
        # string, `-A1` prints the line after every match so `tail -1` silently
        # picks the last, and on no match it prints nothing -- indistinguishable
        # from a match whose file was missing. All three land in the same place:
        # $OFFSETS stays empty, the evidence block below is skipped ENTIRELY,
        # and a window in which nothing was banked looks exactly like a clean
        # one. train.sh's comment is explicit that a failed parse must be
        # "wrong-but-loud", so the reason is recorded per worker and printed.
        #
        # `--log-file` and its value are always distinct argv elements
        # (distributed_runtime.py appends them as two list items and Popen gets
        # a LIST), so `--log-file=X` cannot occur; if it were ever LAST,
        # `getline` fails at EOF and awk re-prints `--log-file`, which then
        # fails the `[ -f ]` and degrades to could-not-verify.
        lf="$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
              | awk '/^--log-file$/{getline; print; exit}')" || lf=""
        if [ -z "$lf" ]; then
            # worker.py defaults --log-file to None and every volunteer launch
            # in README.md omits it, so this is reachable in normal use. There
            # is no evidence to read: a THIRD state, not a loss.
            printf '%s no --log-file in its argv\n' "$pid" >> "$OFFSET_WHY"
        elif [ ! -f "$lf" ]; then
            printf '%s log file does not exist: %s\n' "$pid" "$lf" >> "$OFFSET_WHY"
        elif ! sz="$(stat -c%s "$lf" 2>/dev/null)"; then
            printf '%s offset capture failed for %s\n' "$pid" "$lf" >> "$OFFSET_WHY"
        else
            printf '%s %s\n' "$lf" "$sz" >> "$OFFSETS"
        fi
    done
    log "workers: $(echo "$WORKER_PIDS" | tr '\n' ' ')"
    if [ -s "$OFFSET_WHY" ]; then
        log "WARNING: no pre-drain log offset for $(wc -l < "$OFFSET_WHY") worker(s);"
        log "         their suspend evidence CANNOT be read, so a drain that banked"
        log "         nothing will look the same as one that banked everything:"
        sed 's/^/  pid /' "$OFFSET_WHY" | while read -r l; do log "$l"; done
    fi
elif [ "${CAE_PAUSE_ALLOW_NO_WORKERS:-0}" = "1" ]; then
    log "no selfplay workers matched, and CAE_PAUSE_ALLOW_NO_WORKERS=1 -- proceeding"
else
    # ⚑ NOT A WARNING. While training runs there are always workers, so zero
    # matches means either they are gone (and the caller should know) or they
    # are alive under an argv this pattern no longer recognises. The second is
    # indistinguishable from the first here, and it is the expensive one: the
    # job runs against a full fleet and the result is filed as uncontended.
    # Refuse BEFORE the marker, so the cost of being wrong is zero.
    die "no selfplay workers matched '$WORKER_PATTERN' -- refusing to pause. Either the fleet is down (set CAE_PAUSE_ALLOW_NO_WORKERS=1 and re-run) or the pattern has drifted from the workers' argv"
fi

# ⚑ COUNT THE RESUME DIRS FIRST, because the drain does not know what was
# already there. On 2026-08-09 the window banked 93 games, and the rows after it
# carried 2,963 `resumed_inflight_games` in total -- 32x more. The counter is
# per-finalized-game (see the closing note), so those are distinct games, which
# means the resume dirs were already holding a large backlog that this drain did
# not create and cannot take credit for. That gap is still UNEXPLAINED, and it
# is unexplainable without this line: a count taken after the fact cannot say
# what a count taken before would have.
find "$WORK_DIR/server/trials/$TRIAL_ID/workers" -maxdepth 2 -name selfplay_resume -type d \
    -exec sh -c 'printf "%s %s\n" "$1" "$(ls -1 "$1" 2>/dev/null | wc -l)"' _ {} \; \
    > "$RESUME_BEFORE" 2>/dev/null || true
if [ -s "$RESUME_BEFORE" ]; then
    log "selfplay_resume/ BEFORE drain:"; sed 's/^/  /' "$RESUME_BEFORE"
fi

# ── The marker, with an unconditional release ────────────────────────────────
# Leaving training parked is the one unrecoverable mistake this script can make,
# so the trap covers ordinary exit, failure of the command, and interrupts.
released=0
CHILD=""
release() {
    if [ "$released" -eq 0 ]; then
        released=1
        rm -f "$MARKER"
        log "marker cleared -- training resumes at the next poll"
    fi
    cleanup_tmp
}
# ⚑ THE SIGNAL PATH MUST KILL THE JOB FIRST. A trap cannot run while bash waits
# on a FOREGROUND child, so an interrupt would be deferred until the job ended --
# on a 40-minute arena that is 40 minutes of production parked by a script the
# operator already tried to stop. The job therefore runs in the background under
# `wait` (interruptible), and this handler tears it down before releasing.
# tests/test_pause_window.py::...interrupted pins it; it FAILED before this.
#
# ⚑ THE WHOLE PROCESS GROUP, AND THEN WAIT FOR IT. One SIGTERM to the direct
# child is not a teardown: the job is `bash daily_gate_ratchet.sh`, which runs
# the arena under `timeout`, so the signal reaches the wrapper shell and the
# ARENA SURVIVES. Releasing the marker then resumes training beside a live
# 16-concurrent arena, and 600s later the next poll opens a second window and a
# SECOND arena -- which CLAUDE.md forbids outright (paired/compiled arenas OOMed
# training twice). `set -m` before the launch puts the job in its own process
# group whose pgid IS $CHILD, so `kill -TERM -$CHILD` reaches the grandchildren;
# then we WAIT, and escalate to SIGKILL, before letting training back on the GPU.
# Bounded by CAE_PAUSE_JOB_KILL_TIMEOUT so a job that ignores both cannot hold
# the marker forever -- the marker is released either way, and the group's
# survival is reported.
JOB_KILL_TIMEOUT="${CAE_PAUSE_JOB_KILL_TIMEOUT:-30}"
kill_job_group() {
    [ -n "$CHILD" ] || return 0
    kill -TERM -- "-$CHILD" 2>/dev/null || kill -TERM "$CHILD" 2>/dev/null || true
    local waited=0
    while kill -0 -- "-$CHILD" 2>/dev/null; do
        if [ "$waited" -ge "$JOB_KILL_TIMEOUT" ]; then
            log "job group $CHILD survived SIGTERM after ${waited}s -- SIGKILL"
            kill -KILL -- "-$CHILD" 2>/dev/null || true
            sleep 1
            kill -0 -- "-$CHILD" 2>/dev/null &&
                log "WARNING: job group $CHILD is STILL alive; training resumes beside it"
            return 0
        fi
        sleep 1; waited=$((waited + 1))
    done
    log "job group $CHILD is down after ~${waited}s"
}
on_signal() {
    log "interrupted -- stopping the job and resuming training"
    kill_job_group
    release
    exit 130
}
trap release EXIT
trap on_signal INT TERM

# ⚑ THE MARKER SAYS WHO HOLDS IT AND SINCE WHEN. Nothing in the trainer parses
# it (`_resolve_pause_marker_paths` tests existence only), so this costs
# nothing -- and a held marker is the one state an operator has to diagnose
# from outside. `train_watchdog.decide()` reports PAUSED-HELD whenever a marker
# is present and the loop is flat, and this PR makes that a NIGHTLY event; the
# content is what tells "the ratchet is running" apart from "a dead window
# parked production", which the verdict alone cannot.
{ printf 'pause_window.sh pid=%s started=%s\n' "$$" "$(date -Is)"
  printf 'job=%s\n' "$*"; } > "$MARKER"
log "marker set; waiting up to ${ACK_TIMEOUT}s for the trial to park"

# ⚑ WHEN A PRE-EXISTING ACK IS PRESENT, FAIL FAST INSTEAD OF HOLDING FOR HALF
# AN HOUR. The likely cause is the NB1 case above: a previous window released
# while the trial was still inside its park, so the trial will not re-ack and
# no amount of waiting produces one. Holding the default 1800s marker through
# that is 30 minutes of parked production per poll. Bounded to
# CAE_PAUSE_STALE_ACK_TIMEOUT, and the message names the cause.
deadline="$ACK_TIMEOUT"
if [ "$PREEXISTING_ACK" = "1" ]; then
    deadline="${CAE_PAUSE_STALE_ACK_TIMEOUT:-180}"
fi

ack_is_fresh() {
    [ -e "$ACK" ] || return 1
    [ ! "$ACK" -ot "$MARKER" ]   # mtime >= marker's; `-ot` is strictly older
}

waited=0
while ! ack_is_fresh; do
    if [ "$waited" -ge "$deadline" ]; then
        if [ "$PREEXISTING_ACK" = "1" ]; then
            die "no ack NEWER than our marker after ${deadline}s, and a stale one is present -- the trial is probably still parked inside another window's pause and will not re-ack. NOT draining"
        fi
        die "no ack after ${deadline}s -- NOT draining (revive would relaunch the workers)"
    fi
    sleep "$POLL"; waited=$((waited + POLL))
done
log "PARKED after ~${waited}s: $(tr -d '\n' < "$ACK" 2>/dev/null)"

# ── Drain ────────────────────────────────────────────────────────────────────
if [ -n "$WORKER_PIDS" ]; then
    pkill -TERM -f -- "$WORKER_PATTERN" 2>/dev/null || true
    drained=0
    while [ -n "$(pgrep -f -- "$WORKER_PATTERN" 2>/dev/null || true)" ]; do
        if [ "$drained" -ge "$DRAIN_TIMEOUT" ]; then
            log "workers still alive after ${DRAIN_TIMEOUT}s"
            break
        fi
        sleep "$POLL"; drained=$((drained + POLL))
    done
    log "drained after ~${drained}s"

    # ⚑ THE DRAIN IS A GATE, NOT A BEST EFFORT. Checked against the pids taken
    # at baseline rather than by re-running pgrep, so a pattern that stopped
    # matching cannot report a clean drain. A worker that ignored SIGTERM still
    # holds its server lease and still plays; running the job beside it buys
    # the full production pause and delivers the contended measurement this
    # script exists to prevent -- then stamps the day as read.
    survivors=""
    for pid in $WORKER_PIDS; do
        kill -0 "$pid" 2>/dev/null && survivors="$survivors $pid"
    done
    [ -z "$survivors" ] || die "worker(s)$survivors survived SIGTERM after ${DRAIN_TIMEOUT}s -- NOT running the job; the measurement would be contended and indistinguishable from a clean one"

    # Evidence, read from the pre-drain offsets. ⚑ IT MUST BE READ HERE, BEFORE
    # THE MARKER CLEARS -- but not for the reason an earlier revision gave. It
    # claimed "a revived worker TRUNCATES its log file (1.2MB -> 1,560 bytes)".
    # It does not: `logging.FileHandler` opens in APPEND mode (worker.py), and
    # `_rotate_worker_logs` (tune/distributed_runtime.py) RENAMES the previous
    # generation to `worker.log.1` before the replacement process can open
    # anything -- that rotation exists precisely because something did truncate
    # in place during the 2026-08-04 cold start, and it was never identified.
    # The consequence is the same and the conclusion was right: after the revive
    # the path recorded above is a FRESH file, our byte offset is past its end,
    # and a check placed after the resume reads nothing and reports a healthy
    # drain as a failure. The evidence is not lost, it just moves to
    # `worker.log.1`, which is not where we are looking.
    if [ -s "$OFFSETS" ]; then
        while read -r f off; do
            line="$(tail -c "+$((off + 1))" "$f" 2>/dev/null | grep -m1 'selfplay resume: suspended games=' || true)"
            if [ -n "$line" ]; then
                log "  banked: $(echo "$line" | sed -E 's/.*(suspended games=[0-9]+ records=[0-9]+ skipped=[0-9]+).*/\1/')"
            else
                log "  WARNING: no suspend line from $(basename "$(dirname "$f")") -- games may have been DISCARDED"
            fi
        done < "$OFFSETS"
    else
        # ⚑ SILENCE HERE IS THE FAILURE. Every worker's offset capture failed
        # (reasons printed above), so there is no evidence either way -- which
        # must not be reported by printing nothing, because that is also what a
        # clean window looks like.
        log "  NO suspend evidence available: not one worker had a readable pre-drain"
        log "  log offset, so whether the in-flight games were BANKED or DISCARDED"
        log "  is unknown for this window."
    fi
fi

# ── The job ──────────────────────────────────────────────────────────────────
# `set -m` gives the background job its own process group (pgid == $!), which is
# what makes the group kill in on_signal reach the arena rather than only the
# wrapper shell. It is switched back off immediately: monitor mode also changes
# how this shell reports job status, and nothing after this needs it.
log "running: $*"
set +e
set -m
"$@" &
CHILD=$!
set +m
wait "$CHILD"
rc=$?
set -e
CHILD=""
log "command exited rc=$rc"

release
trap - EXIT INT TERM

# ⚑ NAME THE QUANTITY YOU PRINTED. An earlier revision said "compare against the
# baseline above", where the baseline above is FILE COUNTS under
# selfplay_resume/ and the thing to compare is `resumed_inflight_games` -- a
# different quantity, in different units, which the next sentence then said file
# counts cannot substitute for. Worse, it called that key "a TOTAL". It is not:
# `finalize.py` increments it once per FINALIZED game that carried
# `resumed_from_disk`, so it is per-ingest and DECAYS as the backlog clears
# (measured 2026-08-09 around the window: 0 on every row through iter 567, then
# 224, 456, 477, 454, 399 ... 1, reaching 0 at iter 586). Nothing can be
# subtracted from a pre-drain file count; the readable signal is the SHAPE.
cat <<EOF
[pause-window] done. Two DIFFERENT things were printed above; do not mix them.
[pause-window]
[pause-window] 1. "selfplay_resume/ BEFORE drain" is a FILE COUNT per worker dir,
[pause-window]    taken before anything was signalled. It is a baseline for the
[pause-window]    files, and it is NOT comparable to any result.json counter.
[pause-window] 2. "banked: suspended games=N records=M skipped=K" is what each
[pause-window]    worker reported writing on its way out.
[pause-window]
[pause-window] To confirm those games came BACK, read the trial's result.json:
[pause-window]   grep -o 'resumed_inflight_games=[0-9]*' <trial>/result.json | tail -20
[pause-window] It is a PER-ITERATION count (once per finalized game that was
[pause-window] resumed from disk), not a running total. The proof is the shape: 0
[pause-window] on the rows before this window, non-zero from the first row after
[pause-window] it, decaying back to 0 as the backlog finishes. A first post-window
[pause-window] row of 0 means the banked games did NOT come back.
[pause-window] Falling file counts under selfplay_resume/ are NOT proof either way:
[pause-window] they are equally consistent with resume.py's stale-file cleanup.
EOF
exit "$rc"
