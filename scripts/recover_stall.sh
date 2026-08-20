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
# ⚑ FAIL-CLOSED (#177, 2026-08-20): this script REFUSES (exit 7) while an
# intentional-stop marker or a pause marker is present, unless passed --force.
# The guard used to live only in the CALLER — and the by-hand path this header
# invites had none at all. On 2026-08-20 the watchdog cleared a live pause
# window's marker on an age bound, after which this script's teardown killed
# the deliberately-parked trial and restarted production beside the job the
# pause protected. "Safe to run when already down" is exactly inverted when
# down is ON PURPOSE, so the operator markers now gate the teardown here,
# where the by-hand path cannot skip them. --force also remains the only path
# that removes a pause marker below.
#
# RECOVER_ROOT / RECOVER_STOP_MARKER / RECOVER_PAUSE_TXT / RECOVER_PIDFILE are
# TEST SEAMS (watchdog_loop.sh convention): without them the guard could only
# be pinned by reading it, and exercising the teardown path in a test would
# pkill the LIVE stack.
set -u
cd "${RECOVER_ROOT:-/home/josh/projects/chess}"
LOG=scratchpad/recover_stall.log
STOP_MARKER="${RECOVER_STOP_MARKER:-/tmp/chess_training.intentional_stop}"
PAUSE_TXT="${RECOVER_PAUSE_TXT:-runs/pbt2_small/tune/pause.txt}"
PIDFILE="${RECOVER_PIDFILE:-/tmp/chess_training.pid}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1
mkdir -p scratchpad
log(){ echo "$(date '+%m-%d %H:%M:%S') recover_stall: $*" | tee -a "$LOG"; }

if [ "$FORCE" != 1 ]; then
    for guard in "$STOP_MARKER" "$PAUSE_TXT"; do
        if [ -e "$guard" ]; then
            log "REFUSING (no --force): operator marker present: $guard"
            exit 7
        fi
    done
fi

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
# Only a FORCED run may remove a pause marker: the unforced path proved no
# marker existed at the guard, and one that appeared since belongs to a pause
# that just started — deleting it would leave that window believing it holds
# a marker that is gone.
[ "$FORCE" = 1 ] && rm -f "$PAUSE_TXT"

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
