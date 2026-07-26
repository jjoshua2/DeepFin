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
cd /home/josh/projects/chess
export PYTHONPATH=.

PIDFILE=/tmp/chess_training.pid
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"
STATE=data/ratchet/last_run_date
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

while true; do
    sleep "$POLL"

    trainer_running || continue
    # A paused trial still holds its PID, so PIDFILE alone would let the ratchet
    # run during a deliberate pause.
    paused && continue

    today=$(date +%F)
    [ "$(cat "$STATE" 2>/dev/null)" = "$today" ] && continue

    trial=$(ls -td "$WORK_DIR"/tune/train_trial_*/ 2>/dev/null | head -1)
    [ -n "$trial" ] || continue
    ck=$(ls -td "$trial"checkpoint_* 2>/dev/null | head -1)
    [ -n "$ck" ] || continue
    iter=$(basename "$ck" | sed 's/checkpoint_0*//')
    [ "${iter:-0}" -ge "$MIN_ITER" ] 2>/dev/null || continue

    # Re-check immediately before spending GPU: the gap between the poll above
    # and here is where a stop/pause lands.
    trainer_running || continue
    paused && continue

    log "starting daily ratchet (iter=$iter)"
    if bash scripts/daily_gate_ratchet.sh >> "$LOG" 2>&1; then
        # Stamp only on success, so a failed run retries on the next poll
        # instead of silently skipping the whole day.
        echo "$today" > "$STATE"
        log "daily ratchet done — see data/ratchet/ratchet.csv"
    else
        log "daily ratchet FAILED (rc=$?) — will retry next poll"
    fi
done
