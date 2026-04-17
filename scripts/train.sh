#!/bin/bash
# Usage: ./scripts/train.sh [start|stop|status|log|restart|salvage-export|salvage-restart]
#
# salvage-export [--top-n N] [--out DIR] [--metric KEY] [--no-copy-replay]
#   Export top-N trial seeds (checkpoint + replay) from the current tune run
#   into a salvage pool. Does not touch the running process. If --out is
#   omitted the pool lands under $WORK_DIR/salvage/<run-id>_<timestamp>/.
#
# salvage-restart POOL_DIR [--no-pid] [--no-optimizer] [--reinit-volatility] [--donor-config]
#   Stop training, then start it again pointing at POOL_DIR. Defaults restore
#   the donor PID state and full trainer state but NOT the donor's LR/config.
#   Pass --no-pid / --no-optimizer to flip those defaults.
#
set -e

CONFIG="${TRAIN_CONFIG:-configs/pbt2_small.yaml}"
LOG="/tmp/chess_training.log"
PIDFILE="/tmp/chess_training.pid"
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"

cd "$(dirname "$0")/.."

start() {
    if running; then
        echo "Already running (PID $(cat "$PIDFILE"))"
        return 1
    fi
    local extra_args=("$@")
    echo "Starting training with $CONFIG ${extra_args[*]:+(extra: ${extra_args[*]})}..."
    PYTHONPATH=. nohup python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode tune "${extra_args[@]}" \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "Started PID $! — log: $LOG"
}

stop() {
    if ! running; then
        echo "Not running"
        return 0
    fi
    local pid=$(cat "$PIDFILE")
    echo "Stopping PID $pid ..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing ..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    ray stop 2>/dev/null || true
    sleep 1
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'raylet' 2>/dev/null || true
    echo "Stopped"
}

running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

status() {
    if running; then
        echo "Running (PID $(cat "$PIDFILE"))"
    else
        echo "Not running"
        rm -f "$PIDFILE"
    fi
}

salvage_export() {
    local top_n=3
    local out_dir=""
    local metric="opponent_strength"
    local copy_replay="--salvage-copy-replay"
    while [ $# -gt 0 ]; do
        case "$1" in
            --top-n) top_n="$2"; shift 2 ;;
            --out) out_dir="$2"; shift 2 ;;
            --metric) metric="$2"; shift 2 ;;
            --no-copy-replay) copy_replay="--no-salvage-copy-replay"; shift ;;
            *) echo "Unknown salvage-export arg: $1"; return 1 ;;
        esac
    done

    local out_args=()
    [ -n "$out_dir" ] && out_args=(--salvage-out-dir "$out_dir")

    echo "Exporting top-$top_n salvage seeds from $WORK_DIR (metric=$metric)..."
    PYTHONPATH=. python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode salvage \
        --work-dir "$WORK_DIR" \
        --salvage-top-n "$top_n" \
        --salvage-metric "$metric" \
        "${out_args[@]}" \
        "$copy_replay"
}

salvage_restart() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 salvage-restart POOL_DIR [--no-pid] [--no-optimizer] [--reinit-volatility] [--donor-config]"
        return 1
    fi
    local pool_dir="$1"; shift
    if [ ! -d "$pool_dir" ] || [ ! -f "$pool_dir/manifest.json" ]; then
        echo "Not a salvage pool: $pool_dir (missing manifest.json)"
        return 1
    fi

    # Defaults: restore pid + full trainer, keep GPBT-sampled config, don't reinit volatility.
    local pid_flag="--salvage-restore-pid-state"
    local opt_flag="--salvage-restore-full-trainer-state"
    local donor_flag="--no-salvage-restore-donor-config"
    local volatility_flag="--no-salvage-reinit-volatility-heads"
    while [ $# -gt 0 ]; do
        case "$1" in
            --no-pid) pid_flag="--no-salvage-restore-pid-state"; shift ;;
            --no-optimizer) opt_flag="--no-salvage-restore-full-trainer-state"; shift ;;
            --reinit-volatility) volatility_flag="--salvage-reinit-volatility-heads"; shift ;;
            --donor-config) donor_flag="--salvage-restore-donor-config"; shift ;;
            *) echo "Unknown salvage-restart arg: $1"; return 1 ;;
        esac
    done

    stop
    start \
        --salvage-seed-pool-dir "$pool_dir" \
        "$pid_flag" "$opt_flag" "$donor_flag" "$volatility_flag"
}

case "${1:-status}" in
    start)            shift; start "$@" ;;
    stop)             stop ;;
    restart)          shift; stop; start "$@" ;;
    status)           status ;;
    log)              tail -f "$LOG" ;;
    salvage-export)   shift; salvage_export "$@" ;;
    salvage-restart)  shift; salvage_restart "$@" ;;
    *) echo "Usage: $0 {start|stop|restart|status|log|salvage-export|salvage-restart}" ;;
esac
