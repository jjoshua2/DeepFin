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
# Override the STOP marker with --ignore-intentional-stop (alias: --force). It
# authorizes exactly ONE restart and is consumed there — a long-lived loop must
# not hold standing permission to override every later stop. ⚑ It does NOT
# override a PAUSE marker on this path: see the pause_overridable=0 note in the
# restart branch. WATCHDOG_PBT_MAX_ITERS is a TEST SEAM, not an operator knob:
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
TUNE_DIR="${WATCHDOG_PBT_TUNE_DIR:-$ROOT/$CAE_TUNE_DIR_DEFAULT}"
MAX_ITERS="${WATCHDOG_PBT_MAX_ITERS:-0}"   # 0 = run forever (production)
# Settle time after a restart before the next poll, so a trainer still importing
# is not read as "not running" and launched a second time. A TEST SEAM for the
# same reason as MAX_ITERS: a two-poll test cannot wait 120 real seconds.
RESTART_SETTLE="${WATCHDOG_PBT_RESTART_SETTLE:-120}"

# An unknown argument is REJECTED, not ignored: a typo'd override that fell
# through to the default would read as "the guard did not fire".
#
# ⚑ THE OVERRIDE AUTHORIZES ONE RESTART, NOT A SESSION. This loop runs for days
# in a tmux pane. A flag parsed once at launch and left set would turn a single
# "yes, bring it back up now" into standing permission to override every LATER
# operator stop, days after the human who typed it walked away — the flag would
# outlive the intent that justified it. It is CONSUMED at the first restart
# below (whether or not it was needed), so the second stop is honoured.
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
    #
    # ⚑ THE MARKER SET IS RE-ENUMERATED EACH POLL, for the same reason: a
    # per-trial pause.txt written since the last poll must be seen.
    #
    # ⚑ pause_overridable=0 — THE OVERRIDE DOES NOT REACH A PAUSE HERE. This
    # branch routes through neither `recover_stall.sh`'s teardown nor
    # `train.sh start` (whose `clear_pause_markers` renames markers aside), so
    # it has no way to clear a pause marker correctly. Overriding one would
    # launch a trial that parks at its own pause check within seconds while this
    # log said "Restarted with PID N" — a silent wedge, and a worse failure than
    # the refusal, because the log asserts the opposite of what happened.
    # `recover_stall.sh` passes 1 because it genuinely can clear the set.
    PAUSE_MARKERS=()
    while IFS= read -r _m; do
      [ -n "$_m" ] && PAUSE_MARKERS+=("$_m")
    done < <(cae_pause_markers "$TUNE_DIR" "${WATCHDOG_PBT_PAUSE_FILE-}")
    if intentional_stop_guard "$OVERRIDE" 0 log "$STOP_MARKER" ${PAUSE_MARKERS+"${PAUSE_MARKERS[@]}"}; then
      log "WARNING: Training process not found. Restarting..."
      PYTHONPATH=. python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode tune --resume >> "$LOG" 2>&1 &
      log "Restarted with PID $!"
      # ⚑ CONSUME THE AUTHORIZATION. One --ignore-intentional-stop buys one
      # restart; the next operator stop is honoured like any other.
      if [ "$OVERRIDE" = 1 ]; then
        OVERRIDE=0
        log "override consumed — a later intentional stop will be honoured; re-launch with $CAE_IGNORE_STOP_FLAG to authorize another"
      fi
      if [ "$MAX_ITERS" -gt 0 ] && [ "$ITER" -ge "$MAX_ITERS" ]; then break; fi
      sleep "$RESTART_SETTLE"
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
