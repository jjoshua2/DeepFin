#!/bin/bash
# Usage: ./scripts/train.sh [start|stop|status|log|restart]
set -e

CONFIG="${TRAIN_CONFIG:-configs/pbt2_small.yaml}"
LOG="/tmp/chess_training.log"
PIDFILE="/tmp/chess_training.pid"

start() {
    if running; then
        echo "Already running (PID $(cat "$PIDFILE"))"
        return 1
    fi
    echo "Starting training with $CONFIG ..."
    cd "$(dirname "$0")/.."
    PYTHONPATH=. nohup python3 -m chess_anti_engine.run --config "$CONFIG" --mode tune > "$LOG" 2>&1 &
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
    # Clean up Ray
    ray stop 2>/dev/null || true
    sleep 1
    # Kill any orphan workers/stockfish
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

case "${1:-status}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    log)     tail -f "$LOG" ;;
    *)       echo "Usage: $0 {start|stop|restart|status|log}" ;;
esac
