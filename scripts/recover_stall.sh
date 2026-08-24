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
# RECOVER_ROOT / RECOVER_STOP_MARKER / RECOVER_PAUSE_TXT / RECOVER_PIDFILE are
# TEST SEAMS (watchdog_loop.sh convention), not operator knobs: without them the
# guard could only be pinned by reading it, and exercising the teardown path in
# a test would pkill the LIVE stack.
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
PAUSE_TXT="${RECOVER_PAUSE_TXT:-$CAE_PAUSE_TXT_DEFAULT}"
PIDFILE="${RECOVER_PIDFILE:-/tmp/chess_training.pid}"
mkdir -p scratchpad
log(){ echo "$(date '+%m-%d %H:%M:%S') recover_stall: $*" | tee -a "$LOG"; }

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
intentional_stop_guard "$OVERRIDE" log "$STOP_MARKER" "$PAUSE_TXT" || exit $?

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
# ⚑ ONLY AN OVERRIDDEN RUN DELETES A PAUSE MARKER. On the ordinary path the
# guard above already proved no pause marker existed, so anything here now
# appeared in the last few seconds and belongs to a window that is just opening
# — deleting it leaves that window believing it holds a marker that is gone.
# This is NOT the only thing that clears one: `train.sh start` moves stale pause
# markers aside (clear_pause_markers) further down the same restart, so a marker
# that races us is still handled — just by the path that knows how to say so.
[ "$OVERRIDE" = 1 ] && rm -f "$PAUSE_TXT"

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
