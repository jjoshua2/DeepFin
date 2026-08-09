#!/usr/bin/env bash
# Run a command with live training PAUSED and the selfplay workers DRAINED.
#
# WHY THIS EXISTS. Measured 2026-08-09 (docs/experiment_ledger.md, "the ack-gated
# pause window"): the nightly ratchet run BESIDE training truncated at 106/200
# games and returned a CI spanning zero -- an unreadable result -- while
# iterations stretched 245s -> 611s. The identical series inside a pause window
# got 200/200 in ~21 min. The offline window is not merely safer; it is ~2.5x
# more games in LESS wall time, because the arena and selfplay contend for the
# same CPU (Stockfish is ~95% of loop cost).
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

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

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
# depend on train.sh's formatting. tests/test_pause_window.sh::worker pattern
# asserts the two literals are byte-identical, which is the same guarantee
# `test_the_worker_pattern_is_defined_once` gives inside train.sh.
# `--` is REQUIRED for pgrep/pkill: the pattern begins with `-m`.
WORKER_PATTERN='-m chess_anti_engine\.worker( |$)'

log() { echo "[pause-window] $*"; }
die() { echo "[pause-window] ERROR: $*" >&2; exit 1; }

resolve_trial_id() {
    # `.paused_<trial_id>.ack` is written next to the marker. The trial id is the
    # middle of `train_trial_<id>_<n>_<params>_<date>`; take it from the most
    # recently modified trial dir unless the caller named one.
    local newest
    newest="$(ls -1dt "$TUNE_DIR"/train_trial_* 2>/dev/null | head -1 || true)"
    [ -n "$newest" ] || die "no train_trial_* under $TUNE_DIR; pass --trial-id"
    basename "$newest" | sed -E 's/^train_trial_([^_]+_[0-9]+)_.*$/\1/'
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
# kills the workers while revive is still live. We poll for THIS trial's ack --
# and if one is already sitting there before we have even asked for a pause, it
# is stale by construction, so remove it rather than trust it.
if [ -e "$ACK" ]; then
    log "removing a PRE-EXISTING ack for this trial id (stale: no pause requested yet)"
    rm -f "$ACK"
fi

# ── Baselines, taken BEFORE anything is signalled ────────────────────────────
# The suspend evidence must be read from the byte offset each log had before the
# drain: `selfplay resume: suspended games=` is also emitted by every in-session
# reco restart, so grepping the whole log finds an hours-old line and reads it as
# proof that THIS drain worked.
WORKER_PIDS="$(pgrep -f -- "$WORKER_PATTERN" 2>/dev/null || true)"
OFFSETS="$(mktemp)"; RESUME_BEFORE="$(mktemp)"
cleanup_tmp() { rm -f "$OFFSETS" "$RESUME_BEFORE"; }

if [ -n "$WORKER_PIDS" ]; then
    for pid in $WORKER_PIDS; do
        lf="$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -A1 -- '--log-file' | tail -1 || true)"
        [ -n "$lf" ] && [ -f "$lf" ] && printf '%s %s\n' "$lf" "$(stat -c%s "$lf")" >> "$OFFSETS"
    done
    log "workers: $(echo "$WORKER_PIDS" | tr '\n' ' ')"
else
    log "WARNING: no selfplay workers matched -- nothing to drain"
fi

# ⚑ COUNT THE RESUME DIRS FIRST. `outcome_stats.resumed_inflight_games` is the
# only proof the banked games came BACK (falling file counts are equally
# consistent with resume.py's stale-file cleanup). But the counter is a total,
# so without this baseline it cannot be attributed to THIS drain -- exactly the
# gap in the 2026-08-09 run, where 224 resumed against 93 banked.
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
on_signal() {
    log "interrupted -- stopping the job and resuming training"
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    release
    exit 130
}
trap release EXIT
trap on_signal INT TERM

touch "$MARKER"
log "marker set; waiting up to ${ACK_TIMEOUT}s for the trial to park"

waited=0
while [ ! -e "$ACK" ]; do
    if [ "$waited" -ge "$ACK_TIMEOUT" ]; then
        die "no ack after ${ACK_TIMEOUT}s -- NOT draining (revive would relaunch the workers)"
    fi
    sleep "$POLL"; waited=$((waited + POLL))
done
log "PARKED after ~${waited}s: $(cat "$ACK" 2>/dev/null | tr -d '\n')"

# ── Drain ────────────────────────────────────────────────────────────────────
if [ -n "$WORKER_PIDS" ]; then
    pkill -TERM -f -- "$WORKER_PATTERN" 2>/dev/null || true
    drained=0
    while [ -n "$(pgrep -f -- "$WORKER_PATTERN" 2>/dev/null || true)" ]; do
        if [ "$drained" -ge "$DRAIN_TIMEOUT" ]; then
            log "WARNING: workers still alive after ${DRAIN_TIMEOUT}s; continuing anyway"
            break
        fi
        sleep "$POLL"; drained=$((drained + POLL))
    done
    log "drained after ~${drained}s"

    # Evidence, read from the pre-drain offsets. ⚑ It MUST be read here, before
    # the marker clears: a revived worker TRUNCATES its log file (measured
    # 1.2MB -> 1,560 bytes on 2026-08-09), so after the resume there is nothing
    # left to read and a check placed there reports a false failure.
    if [ -s "$OFFSETS" ]; then
        while read -r f off; do
            line="$(tail -c "+$((off + 1))" "$f" 2>/dev/null | grep -m1 'selfplay resume: suspended games=' || true)"
            if [ -n "$line" ]; then
                log "  banked: $(echo "$line" | sed -E 's/.*(suspended games=[0-9]+ records=[0-9]+ skipped=[0-9]+).*/\1/')"
            else
                log "  WARNING: no suspend line from $(basename "$(dirname "$f")") -- games may have been DISCARDED"
            fi
        done < "$OFFSETS"
    fi
fi

# ── The job ──────────────────────────────────────────────────────────────────
log "running: $*"
set +e
"$@" &
CHILD=$!
wait "$CHILD"
rc=$?
set -e
CHILD=""
log "command exited rc=$rc"

release
trap - EXIT INT TERM

cat <<EOF
[pause-window] done. To confirm the banked games came BACK, read the FIRST new
[pause-window] result.json row and compare against the baseline above:
[pause-window]   outcome_stats.resumed_inflight_games   (must be > 0)
[pause-window] Falling file counts under selfplay_resume/ are NOT proof: they are
[pause-window] equally consistent with resume.py's stale-file cleanup.
EOF
exit "$rc"
