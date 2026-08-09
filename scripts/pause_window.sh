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
# ⚑ EXIT 7 IS "THE WRAPPER FAILED", AND IT IS DISTINCT ON PURPOSE. The job's own
# status is passed through untouched at the end. Sharing 1 with the ratchet made
# "I could not pause" indistinguishable from "the arena produced no rows", so a
# wrapper that could never park read as an ordinary retryable failure and came
# back every poll until midnight.
#
# ⚑ AND IT WAS 3, WHICH WAS ALREADY TAKEN. The justification used to read "the
# ratchet's vocabulary is 0 / 1 / 5" -- but `ratchet_common.sh`, the ratchet
# family's OWN single source, says in as many words: "5 avoids 1 (retryable),
# 2 (usage), 3 (the arena's own no-pairs status)". 3 was spoken for, in the file
# this script's comment was paraphrasing. Latent rather than live today
# (`daily_gate_ratchet.sh` returns exactly 0/1/2/5), but it is the identical
# mistake as claiming exit 5 after #371 had taken it for CRASHED: a code picked
# by reading part of the space. 7 is checked against every producer in the
# family -- 0, 1 RETRY, 2 usage/cd, 3 arena no-pairs, 5 NO_RETRY, 130 interrupt.
# `tests/test_pause_window.py::test_the_wrapper_failure_code_is_not_in_the_ratchets_vocabulary`
# derives the taken set FROM THOSE FILES rather than restating it here, so the
# next code someone adds cannot silently collide.
WRAPPER_FAILED_RC=7
die() { echo "[pause-window] ERROR: $*" >&2; exit "$WRAPPER_FAILED_RC"; }

# ⚑ A `set -e` DEATH BEFORE THE MARKER MUST ALSO REPORT "WRAPPER FAILED".
# `mktemp` failing (full /tmp) or a `stat` erroring exits 1 under `set -e`, and 1
# is the ratchet's RETRY: the loop logs "FAILED, retry next poll", does NOT count
# it against CAE_RATCHET_PAUSE_MAX_FAILS, and churns every 600s until midnight.
# Nothing is parked at that point, so the cost is retry noise -- but the fail cap
# exists precisely to bound that, and a failure that dodges the cap is outside
# the cost control this PR added. PHASE is what keeps the remap off the job's own
# status: once the job runs, its rc is passed through verbatim.
PHASE=setup
# ⚑ THE STATUS MUST BE PASSED IN WHEN THIS IS NOT THE FIRST THING THE TRAP RUNS.
# As a bare `trap remap_setup_failure EXIT`, `$?` IS the script's exit status.
# Composed as `release; remap_setup_failure`, `$?` is RELEASE's status -- which
# is 0 on every path -- so the remap silently never fired. A guard that cannot
# fire, in the function whose entire job is to stop a failure being mislabelled.
# Caught by `test_a_failure_AFTER_the_marker_also_reports_the_wrapper_code`,
# written only because the mutation of the composite SURVIVED.
remap_setup_failure() {
    local rc="${1:-$?}"
    [ "$PHASE" = "setup" ] || return 0
    case "$rc" in
        0|2|"$WRAPPER_FAILED_RC") return 0 ;;
    esac
    echo "[pause-window] ERROR: died during setup with rc=$rc (set -e) -- reporting $WRAPPER_FAILED_RC so the loop counts it against CAE_RATCHET_PAUSE_MAX_FAILS instead of retrying all day" >&2
    exit "$WRAPPER_FAILED_RC"
}
trap remap_setup_failure EXIT

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
    # ⚑ N7: POPULATED-PREFERRED, THEN THE DATA FILE'S MTIME -- the SAME rule as
    # `train_watchdog.newest_trial_dir` and `trial_paths._trial_sort_key`, not a
    # third "latest" of this file's own devising.
    #
    # This was `ls -1dt`, i.e. DIRECTORY mtime, and the directory's mtime is not
    # evidence that a trial is alive: it moves whenever ANY entry is created or
    # removed inside it, so Ray writing `checkpoint_NNNNNN/` under a trial --
    # or a stray touch, an rsync, a log rotation -- floats a dead trial above
    # the live one. The cost is not small: the wrong id gives an ACK path
    # nothing will ever write, so the script holds the marker for the full
    # CAE_PAUSE_ACK_TIMEOUT (1800s) and does it again on the next poll, twice,
    # before the fail cap stops it.
    #
    # ⚑ THE REASON THIS IS FIXED HERE AND NOT DEFERRED: it is B2's defect in a
    # second place. B2 was a trial selector that silently picked a DEAD trial
    # and returned plausible, stationary numbers, and it too was "bounded and
    # logged" right up until someone executed it. A selector must be
    # DEMONSTRATED to track the live artefact, not merely to return one --
    # `test_the_trial_selector_prefers_the_populated_trial_over_a_touched_one`
    # is that demonstration.
    #
    # rank 1 = has result.json/progress.csv, rank 0 = has neither; then that
    # file's mtime; then the name, matching `max()` on the same 3-tuple.
    # ⚑ `%.9Y`, NOT `%Y`. Whole-second mtimes make two trials written in the same
    # second compare EQUAL, and the comparison then falls through to the name --
    # so which trial is "live" would be decided alphabetically. `newest_trial_dir`
    # compares float `st_mtime`, so `%Y` is also a silent divergence between the
    # two implementations of one rule. LC_ALL=C so the decimal point is a dot.
    local newest id
    newest="$(
        for d in "$TUNE_DIR"/train_trial_*/; do
            [ -d "$d" ] || continue
            rank=0; m=0
            for f in result.json progress.csv; do
                if [ -f "$d$f" ]; then
                    m="$(stat -c %.9Y "$d$f" 2>/dev/null || echo 0)"; rank=1; break
                fi
            done
            [ "$rank" -eq 1 ] || m="$(stat -c %.9Y "$d" 2>/dev/null || echo 0)"
            printf '%s %s %s\n' "$rank" "$m" "$(basename "$d")"
        done | LC_ALL=C sort -k1,1nr -k2,2nr -k3,3r | head -1 | cut -d' ' -f3-
    )"
    [ -n "$newest" ] || die "no train_trial_* under $TUNE_DIR; pass --trial-id"
    id="$(printf '%s\n' "$newest" | sed -E 's/^train_trial_([^_]+_[0-9]+)_.*$/\1/')"
    printf '%s\n' "$id" | grep -Eq '^[A-Za-z0-9]+_[0-9]+$' \
        || die "could not parse a trial id out of '$newest' (got '$id') -- the trial-dir naming has changed; pass --trial-id"
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
        # ⚑ N6: ONLY REMOVE A MARKER THAT IS STILL OURS. `rm -f "$MARKER"`
        # unconditionally means an operator who `touch`es the root marker by hand
        # DURING our window (to hold the pause past the arena, the obvious thing
        # to do) silently loses their pause the moment the job ends. The marker
        # names its owner precisely so this is decidable; check it. If it is not
        # ours we leave it and say so -- training stays parked, which is the
        # operator's stated intent and is recoverable, whereas resuming against
        # their wishes is not.
        if [ ! -e "$MARKER" ]; then
            log "marker already gone -- nothing to clear"
        elif grep -q "pid=$$\b" "$MARKER" 2>/dev/null; then
            rm -f "$MARKER"
            log "marker cleared -- training resumes at the next poll"
        else
            log "WARNING: $MARKER is no longer ours (pid=$$ not in it) -- LEAVING IT."
            log "         Someone replaced it during our window; training stays"
            log "         PARKED until they clear it. Contents: $(tr -d '\n' < "$MARKER" 2>/dev/null)"
        fi
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
# ⚑ THE REMAP SURVIVES THE HANDOFF. `trap release EXIT` REPLACES the setup trap,
# so without this composite a `set -e` death between the marker and the job would
# release correctly and then report 1 -- back to the uncapped retry N4 describes.
# `release` first (the marker is the unrecoverable part), remap second.
release_then_remap() { local rc=$?; release; remap_setup_failure "$rc"; }
trap release_then_remap EXIT
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

# ⚑ N8: THIS GUARD DEPENDS ON SUB-SECOND TIMESTAMPS, and that is unpinned by
# anything but this comment. `-ot` is STRICTLY older, so at EQUAL mtimes
# `[ ! -ot ]` reads FRESH -- i.e. a stale ack written in the same timestamp tick
# as our marker would be accepted. Unreachable at the nanosecond granularity
# bash+ext4 give here (measured: a 12ms difference resolves correctly), but
# reachable on any 1-second-granularity filesystem (some network mounts, some
# container overlays). If this ever runs somewhere like that, compare with an
# explicit epoch-nanosecond `stat -c %Y`/`%.9Y` instead of `-ot`.
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

    # ⚑ THE DRAIN IS A GATE, NOT A BEST EFFORT -- AND IT NEEDS BOTH HALVES.
    #
    # Half 1, the BASELINE pids: a pattern that stopped matching cannot report a
    # clean drain, because these pids are checked with `kill -0` and not with
    # pgrep at all.
    #
    # Half 2, a FRESH pgrep: the baseline was taken at line ~187, BEFORE the
    # marker and before an ack wait that can run CAE_PAUSE_ACK_TIMEOUT (1800s
    # default). The trial is still running its current iteration for all of that
    # wait, and `_revive_fleet` sits inside `_ingest_distributed_selfplay` -- so
    # revive is live PRECISELY DURING THE WAIT. A worker that dies and is
    # revived there gets a NEW pid, which is in no baseline. Baseline pids all
    # dead + a revived worker alive and ignoring SIGTERM => half 1 finds nothing
    # => the job runs beside a live selfplay worker, rc=0, and `ratchet_outcome`
    # stamps the day as a clean strength reading. That is the exact outcome this
    # gate exists to prevent, and an independent reviewer reproduced it against
    # the real script: "workers still alive after 3s" ... "command exited rc=0"
    # ... "JOB RAN BESIDE THE LIVE WORKER: True".
    #
    # ⚑ The loop above ALREADY SAW IT -- it printed "workers still alive" and
    # then `break`ed into a gate that never consulted the fact. A value accepted
    # and then silently ignored, which is this codebase's signature defect.
    #
    # Neither half subsumes the other: half 1 covers "the pattern drifted", half
    # 2 covers "a worker appeared after the baseline", and half 2 is the one
    # that runs the arena. rc>=2 is fatal here for the same reason it is at
    # baseline; rc==0 (still matching) is fatal too.
    survivors=""
    for pid in $WORKER_PIDS; do
        kill -0 "$pid" 2>/dev/null && survivors="$survivors $pid"
    done

    post_rc=0
    STILL_MATCHING="$(pgrep -f -- "$WORKER_PATTERN" 2>/dev/null)" || post_rc=$?
    [ "$post_rc" -lt 2 ] || die "pgrep failed (rc=$post_rc) after the drain -- the worker pattern is broken, so 'no workers left' is unverifiable; NOT running the job"

    [ -z "$survivors" ] || die "worker(s)$survivors survived SIGTERM after ${DRAIN_TIMEOUT}s -- NOT running the job; the measurement would be contended and indistinguishable from a clean one"
    [ -z "$STILL_MATCHING" ] || die "worker(s) $(echo "$STILL_MATCHING" | tr '\n' ' ')still match '$WORKER_PATTERN' after ${DRAIN_TIMEOUT}s -- NOT running the job. These are NOT in the pre-marker baseline, so they were revived during the ack wait (revive is live until the trial parks); the measurement would be contended and would be filed as clean"

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
PHASE=job
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
