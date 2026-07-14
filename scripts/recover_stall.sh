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
# alive, no pause.txt, no intentional-stop marker), or by hand. Idempotent-ish:
# safe to run when already down (kills nothing, just restarts).
set -u
cd /home/josh/projects/chess
LOG=scratchpad/recover_stall.log
mkdir -p scratchpad
log(){ echo "$(date '+%m-%d %H:%M:%S') recover_stall: $*" | tee -a "$LOG"; }

log "BEGIN force teardown (GPU-independent)"
PID=$(cat /tmp/chess_training.pid 2>/dev/null || true)
[ -n "$PID" ] && kill -9 "$PID" 2>/dev/null && log "killed trainer pid=$PID"
# Module names as they actually appear in ps (NOT 'distributed_worker').
for pat in "chess_anti_engine.worker" "chess_anti_engine.inference" \
           "chess_anti_engine.run" "Trainable" "gcs_server" "raylet" \
           "ray/dashboard/agent"; do
    pkill -9 -f "$pat" 2>/dev/null && log "pkilled $pat"
done
rm -f /tmp/chess_training.pid runs/pbt2_small/tune/pause.txt

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
