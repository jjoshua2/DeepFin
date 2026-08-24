#!/bin/bash
# PBT2 watchdog. Run in tmux before leaving a run unattended:
#   tmux new -s watchdog 'bash scripts/watchdog_pbt.sh'
#
# ⚑ THIS IS THE THIRD PATH THAT CAN START PRODUCTION TRAINING, and until now it
# was the only one with no marker check anywhere — not in itself and not in a
# caller, because nothing calls it: it is invoked by a human in a tmux pane and
# then left alone for hours. Its restart branch launches
# `python3 -m chess_anti_engine.run` DIRECTLY, so `train.sh stop`'s
# intentional-stop marker meant nothing to it and a deliberate stop would be
# undone at the next poll. It now consults the same guard as
# `recover_stall.sh` — see scripts/intentional_stop_guard.sh.
#
# ⚑ AND IT STILL BYPASSES `train.sh`. Its restart writes no
# /tmp/chess_training.pid, runs no C-extension freshness check, starts no
# observers and exports neither PYTORCH_NVML_BASED_CUDA_CHECK nor
# CHESS_ANTI_ENGINE_LIVE_CONFIG — so a run it starts is one `train.sh
# status`/`stop` cannot see. The marker guard below closes the policy hole it
# was audited for; the bypass itself is a separate (and larger) call about
# whether this script should route through `train.sh start` or be deleted now
# that `watchdog_loop.sh` covers the same ground. Read that as a known defect,
# not as a clean bill of health.
#
# Override the guard with --ignore-intentional-stop (alias: --force), same as
# recover_stall.sh. WATCHDOG_PBT_MAX_ITERS is a TEST SEAM, not an operator knob:
# without it this loop never terminates and its restart branch could only be
# pinned by reading it.

set -euo pipefail

# Sourced BEFORE the cd, off $0, so the path does not depend on the CWD.
# shellcheck source=scripts/intentional_stop_guard.sh
. "$(cd "$(dirname "$0")" && pwd)/intentional_stop_guard.sh" || exit 2

ROOT="${CHESS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${TRAIN_CONFIG:-configs/pbt2_small.yaml}"
LOG="${TRAIN_MONITOR_LOG:-$ROOT/runs/pbt2_small/monitor.log}"
INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-3600}"
STOP_MARKER="${WATCHDOG_PBT_STOP_MARKER:-$CAE_STOP_MARKER_DEFAULT}"
PAUSE_TXT="${WATCHDOG_PBT_PAUSE_TXT:-$ROOT/$CAE_PAUSE_TXT_DEFAULT}"
MAX_ITERS="${WATCHDOG_PBT_MAX_ITERS:-0}"   # 0 = run forever (production)

# An unknown argument is REJECTED, not ignored: a typo'd override that fell
# through to the default would read as "the guard did not fire".
OVERRIDE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ignore-intentional-stop|--force) OVERRIDE=1; shift ;;
    *)
      echo "usage: $0 [--ignore-intentional-stop|--force]" >&2
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
mkdir -p "$(dirname "$LOG")"
log(){ echo "$*" | tee -a "$LOG"; }

ITER=0
while true; do
  ITER=$((ITER + 1))
  {
    echo "========================================"
    echo "Watchdog check: $(date)"
    echo "========================================"
  } | tee -a "$LOG"

  if ! pgrep -f "chess_anti_engine.run" >/dev/null 2>&1; then
    # ⚑ THE GUARD IS INSIDE THE RESTART BRANCH, evaluated fresh each poll — a
    # marker written while this loop sleeps must stop the NEXT restart, so a
    # once-at-startup check would be a guard with the wrong clock. Repeating
    # the refusal once per $INTERVAL_SECONDS (default 1h) is deliberate: while
    # nothing is running, the refusal line is the only record of WHY, and an
    # operator reading this log needs it more than they need silence.
    if intentional_stop_guard "$OVERRIDE" log "$STOP_MARKER" "$PAUSE_TXT"; then
      log "WARNING: Training process not found. Restarting..."
      PYTHONPATH=. python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode tune --resume >> "$LOG" 2>&1 &
      log "Restarted with PID $!"
      if [ "$MAX_ITERS" -gt 0 ] && [ "$ITER" -ge "$MAX_ITERS" ]; then break; fi
      sleep 120
      continue
    fi
    if [ "$MAX_ITERS" -gt 0 ] && [ "$ITER" -ge "$MAX_ITERS" ]; then break; fi
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  CHESS_ROOT="$ROOT" TRAIN_MONITOR_LOG="$LOG" bash "$ROOT/scripts/monitor_pbt.sh"
  if [ "$MAX_ITERS" -gt 0 ] && [ "$ITER" -ge "$MAX_ITERS" ]; then break; fi
  sleep "$INTERVAL_SECONDS"
done
