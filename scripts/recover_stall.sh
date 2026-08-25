#!/bin/bash
# GPU-independent force-recovery from a wedged/stalled training stack.
#
# The WSL2 dxg vmbus wedge (see memory: wsl2-gpu-vmbus-wedge-signature) leaves
# the trainer PID ALIVE but every CUDA context blocked forever — no progress,
# no error. `train.sh stop` can't tear it down because `ray stop` itself needs
# the wedged GPU and hangs. The bridge only recovers once the wedged contexts
# are KILLED. So this script:
#   1. force-kills the whole stack with SIGKILL (NEVER ray stop — GPU-free),
#   2. waits for the GPU bridge to come back (poll nvidia-smi),
#   3. restarts via train.sh start (auto-resume from the last checkpoint).
#
# Invoked by watchdog_loop.sh on a confirmed STALLED verdict (90 min flat, PID
# alive, no pause.txt, no intentional-stop marker), or by hand.
#
# ⚑ THE OPERATOR-MARKER GUARD LIVES HERE, NOT IN THE CALLER. watchdog_loop.sh
# still tests the marker before it fires, but that guarded the WATCHDOG route
# only: the BY-HAND route this header invites — `./scripts/recover_stall.sh` —
# went straight to the SIGKILLs and then `train.sh start`, resurrecting a run an
# operator had deliberately stopped. "Idempotent-ish: safe to run when already
# down" (what this header used to promise) is exactly inverted when down is ON
# PURPOSE. The guard is now the first thing this script does, so every route
# through it consults the marker and no future caller can be written without
# the check. See scripts/intentional_stop_guard.sh.
#
# Override with --ignore-intentional-stop (alias: --force). It is loud, it names
# every marker it is ignoring, and it is never the default.
#
# ⚑ THE PAUSE SET IS NOT ONE FILE. The guard enumerates every marker the pause
# machinery itself honours — the tune dir scanned RECURSIVELY (root and
# per-trial markers, the same set `train_watchdog.find_pause_txt` walks) plus a
# `pause_file` configured in the live yaml. Checking only the root
# `runs/pbt2_small/tune/pause.txt` let a PER-TRIAL pause through and recovery
# SIGKILLed a deliberately parked run.
#
# RECOVER_ROOT / RECOVER_STOP_MARKER / RECOVER_TUNE_DIR / RECOVER_PAUSE_FILE /
# RECOVER_PIDFILE are TEST SEAMS (watchdog_loop.sh convention), not operator
# knobs: without them the guard could only be pinned by reading it, and
# exercising the teardown path in a test would pkill the LIVE stack.
set -u

# Sourced BEFORE the cd, off $0, so the path does not depend on the CWD we are
# about to change. It defines a function and constants and does nothing else.
# ⚑ `|| exit 2` because this file has no `set -e`: a guard library that failed to
# load must stop the script, not leave it running with the guard undefined.
# shellcheck source=scripts/intentional_stop_guard.sh
. "$(cd "$(dirname "$0")" && pwd)/intentional_stop_guard.sh" || exit 2

cd "${RECOVER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 2
LOG=scratchpad/recover_stall.log
STOP_MARKER="${RECOVER_STOP_MARKER:-$CAE_STOP_MARKER_DEFAULT}"
TUNE_DIR="${RECOVER_TUNE_DIR:-$CAE_TUNE_DIR_DEFAULT}"
PIDFILE="${RECOVER_PIDFILE:-/tmp/chess_training.pid}"
CONFIG="${TRAIN_CONFIG:-configs/pbt2_small.yaml}"
mkdir -p scratchpad
log(){ echo "$(date '+%m-%d %H:%M:%S') recover_stall: $*" | tee -a "$LOG"; }

# ⚑ `tc.pause_file` IS A LIVE-YAML KEY, AND A FILESYSTEM SCAN CANNOT SEE IT.
# `_resolve_pause_marker_paths` honours it as a pause marker wherever it points,
# so a guard that only walked the tune dir would let a CONFIGURED pause through
# — a value the trainer accepts and the guard silently ignores, which is this
# repo's signature defect pointed at its own safety check.
#
# ⚑ NOT FAIL-CLOSED, ON PURPOSE, AND SAID OUT LOUD. This script exists to
# recover a wedged machine; refusing to run because pyyaml is unhappy would
# disable recovery for a reason unrelated to operator intent. A failed read logs
# a WARN and the filesystem scan (the load-bearing half) still runs. Explicit
# $RECOVER_PAUSE_FILE wins and skips the read entirely — that is the test seam
# and the operator escape hatch.
_read_pause_file='
import sys
from pathlib import Path
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
cfg = flatten_run_config_defaults(load_yaml_file(Path(sys.argv[1])))
print(str(cfg.get("pause_file") or "").strip())
'
PAUSE_FILE="${RECOVER_PAUSE_FILE-}"
if [ -z "${RECOVER_PAUSE_FILE+set}" ]; then
    if ! PAUSE_FILE=$(PYTHONPATH=. python3 -c "$_read_pause_file" "$CONFIG" 2>/dev/null); then
        PAUSE_FILE=""
        log "WARN: could not read pause_file from $CONFIG — scanning $TUNE_DIR only"
    fi
fi
# Relative `pause_file` is anchored on the tune root, matching
# `_resolve_pause_marker_paths`'s `tune_root_ephemeral / p`.
case "$PAUSE_FILE" in
    ""|/*) ;;
    *) PAUSE_FILE="$TUNE_DIR/$PAUSE_FILE" ;;
esac

# ⚑ AN UNKNOWN ARGUMENT IS REJECTED, NOT IGNORED. A typo'd override
# (`--ignore-intentional_stop`) that silently fell through to the default would
# read to the operator as "the guard did not fire" and to this script as "no
# override was asked for" — the accept-then-ignore shape this repo keeps
# reproducing. Fail loudly on argv instead.
OVERRIDE=0
while [ $# -gt 0 ]; do
    case "$1" in
        # `--force` is the spelling 42e72d6cb shipped on the live branch; kept as
        # an alias so an operator's muscle memory does not hit a usage error in
        # the middle of an incident. Both mean the same thing and both are loud.
        --ignore-intentional-stop|--force) OVERRIDE=1; shift ;;
        *)
            echo "usage: $0 [--ignore-intentional-stop|--force]" >&2
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# BEFORE ANY KILL. The refusal must cost nothing, so it happens ahead of the
# SIGKILLs, the pidfile removal and the GPU-bridge wait.
#
# `pause_overridable=1`: this script restarts through `train.sh start`, whose
# `clear_pause_markers` renames every marker aside as `.cleared_<ts>`, so an
# overridden pause here is genuinely carried out rather than left to strand the
# new trial. `watchdog_pbt.sh` passes 0 for exactly the opposite reason.
#
# ⚑ `read -r` INTO AN ARRAY, NOT WORD-SPLITTING: a tune-dir path containing a
# space would otherwise split into two non-existent markers and the guard would
# see neither.
PAUSE_MARKERS=()
while IFS= read -r _m; do
    [ -n "$_m" ] && PAUSE_MARKERS+=("$_m")
done < <(cae_pause_markers "$TUNE_DIR" "$PAUSE_FILE")
intentional_stop_guard "$OVERRIDE" 1 log "$STOP_MARKER" ${PAUSE_MARKERS+"${PAUSE_MARKERS[@]}"} || exit $?

log "BEGIN force teardown (GPU-independent)"
PID=$(cat "$PIDFILE" 2>/dev/null || true)
[ -n "$PID" ] && kill -9 "$PID" 2>/dev/null && log "killed trainer pid=$PID"
# Module names as they actually appear in ps (NOT 'distributed_worker').
for pat in "chess_anti_engine.worker" "chess_anti_engine.inference" \
           "chess_anti_engine.run" "Trainable" "gcs_server" "raylet" \
           "ray/dashboard/agent"; do
    pkill -9 -f "$pat" 2>/dev/null && log "pkilled $pat"
done
rm -f "$PIDFILE"
# ⚑ ONLY AN OVERRIDDEN RUN DELETES A PAUSE MARKER, AND IT DELETES THE WHOLE SET.
# On the ordinary path the guard above already proved no pause marker existed, so
# anything here now appeared in the last few seconds and belongs to a window that
# is just opening — deleting it leaves that window believing it holds a marker
# that is gone. This is NOT the only thing that clears one: `train.sh start`
# moves stale pause markers aside (clear_pause_markers) further down the same
# restart, so a marker that races us is still handled — just by the path that
# knows how to say so.
#
# ⚑ RE-ENUMERATED, NOT REPLAYED FROM $PAUSE_MARKERS. The teardown above takes
# time; the set is re-read so a per-trial marker written during it is cleared
# too. Removing only the root marker — which is what this line used to do — is
# what left a per-trial pause holding the restarted trial.
#
# ⚑ `rm -f`, NEVER `rm -rf`. Every path here comes from a `find`, and a
# recursive delete driven by a glob is the one mistake that is not undoable. A
# marker that is somehow a non-empty DIRECTORY fails loudly and stays — the
# restart then parks, which is recoverable; a recursive delete is not.
if [ "$OVERRIDE" = 1 ]; then
    while IFS= read -r _m; do
        [ -n "$_m" ] || continue
        if rm -f -- "$_m" 2>/dev/null && ! cae_marker_exists "$_m"; then
            log "override: removed pause marker $_m"
        else
            log "override: WARN could not remove pause marker $_m — the restarted trial may park on it"
        fi
    done < <(cae_pause_markers "$TUNE_DIR" "$PAUSE_FILE")
fi

# Wait for the GPU bridge: once the wedged contexts die, dxg recovers on its own.
BACK=0
for i in $(seq 1 30); do
    if timeout 12 nvidia-smi -L >/dev/null 2>&1; then
        log "GPU bridge responsive after check #$i"; BACK=1; break
    fi
    sleep 10
done
[ "$BACK" = 1 ] || log "WARN: GPU bridge still unresponsive after ~5min — starting anyway (start will surface it)"

log "restarting via train.sh start (auto-resume)"
./scripts/train.sh start >>"$LOG" 2>&1
log "END (train.sh start returned $?)"
