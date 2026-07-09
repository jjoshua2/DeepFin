#!/bin/bash
# 512x16 scale-up bootstrap (2026-07-07; multi-phase fix 2026-07-09).
#
# Fresh-init 63.08M net (embed 512 / 16L / h16) on the frozen iter-647 salvage
# window. Width can't migrate from 384, so this is always fresh-init for phase 1.
#
# BUG (pre-fix): a single live-follow call with target_reuse=3 + frozen shards
# spent ~3 epochs of credit then *slept forever* waiting for new positions that
# never arrive, and never ran the planned LR drops.
#
# FIX: phased driver.
#   phase1  lr=3e-4 (config default)  reuse=3  (~3 epochs) then EXIT on idle
#   phase2  lr=1e-4  reuse=1  (~1 epoch) from best/last ckpt
#   phase3  lr=3e-5  reuse=1  (~1 epoch)
# Each phase also stops early on eval plateau (3 consecutive non-improving evals).
#
# Usage:
#   bash scripts/run_bootstrap_512x16.sh            # full chain, fresh phase1
#   bash scripts/run_bootstrap_512x16.sh phases-2-3 # only LR drops from current trainer.pt
#   bash scripts/run_bootstrap_512x16.sh chain-from-running
#       # watch the already-running phase1 (old binary); when it credit-stalls,
#       # kill it and run phases 2-3 with the fixed trainer
set -euo pipefail
cd /home/josh/projects/chess
OUT=runs/scaleup_512x16_bootstrap
CAND=aurora_mlp_out
SHARDS=data/salvage/scaleup_512x16_window_20260707/seeds/slot_000/replay_shards
FFN16="1.5,1.5,1.5,1.5,1.5,1.555556,1.65,1.772222,1.894444,1.666667,1.638889,1.905556,1.783333,1.794444,1.744444,1.5"
MEMCAP=0.30
# Static pool: exit ~2 min after credit is gone (a few idle rescans).
IDLE_EXIT=120
# Stop a phase if eval_loss fails to improve for this many evals (eval every 500).
PLATEAU_EVALS=3
MODE="${1:-full}"

mkdir -p "$OUT"
LOG="$OUT/train.log"
PHASE_LOG="$OUT/phase_driver.log"

log() { echo "[bootstrap $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PHASE_LOG"; }

feed_new_shards() {
  # Hardlink newest live trial shards into the bootstrap pool so live-follow
  # rescan grants credit (or phase 2/3 starts with a fresher static window).
  log "feeding new live shards into $SHARDS"
  PYTHONPATH=. python3 scripts/feed_bootstrap_shards.py 2>&1 | tee -a "$PHASE_LOG"
}

pick_init_ckpt() {
  # Prefer plateau-tracked best, else latest trainer.pt
  local best="$OUT/$CAND/trainer_best_eval.pt"
  local last="$OUT/$CAND/trainer.pt"
  if [[ -f "$best" ]]; then
    echo "$best"
  elif [[ -f "$last" ]]; then
    echo "$last"
  else
    echo ""
  fi
}

run_phase() {
  local phase_name="$1"
  local lr_arg="$2"          # empty = omit --lr (use config default 3e-4)
  local reuse="$3"
  local max_steps="$4"
  local init_ckpt="$5"       # empty = fresh init

  local phase_out="$OUT/${phase_name}.log"
  log "START $phase_name  lr=${lr_arg:-config_default} reuse=$reuse max_steps=$max_steps init=${init_ckpt:-FRESH}"

  local cmd=(
    python3 scripts/bootstrap_memcap_wrapper.py --mem-fraction "$MEMCAP" --
    --config configs/pbt2_small.yaml
    --replay-dir "$SHARDS"
    --out-dir "$OUT"
    --candidates "$CAND"
    --input-extra-features v2_threats
    --embed-dim 512 --num-layers 16 --num-heads 16
    --ffn-mult-by-layer "$FFN16"
    --live-follow
    --live-credit-cap-steps 0
    --live-initial-credit-frac 1.0
    --live-target-reuse "$reuse"
    --live-max-steps "$max_steps"
    --live-eval-every-steps 500
    --live-save-every-steps 1000
    --live-idle-exit-after "$IDLE_EXIT"
    --live-plateau-evals "$PLATEAU_EVALS"
    --batch-size 256
  )
  if [[ -n "$lr_arg" ]]; then
    cmd+=(--lr "$lr_arg")
  fi
  if [[ -n "$init_ckpt" ]]; then
    cmd+=(--init-checkpoint "$init_ckpt")
  fi

  set +e
  PYTHONPATH=. nice -n 15 "${cmd[@]}" >>"$phase_out" 2>&1
  local rc=$?
  set -e
  # Also append phase log into the combined train.log for continuity.
  {
    echo "{\"event\": \"phase_boundary\", \"phase\": \"$phase_name\", \"rc\": $rc, \"ts\": \"$(date -Iseconds)\"}"
    tail -n 5 "$phase_out" 2>/dev/null || true
  } >>"$LOG"
  log "END $phase_name rc=$rc (detail: $phase_out)"
  # Copy final trainer into a phase-tagged name for forensics.
  if [[ -f "$OUT/$CAND/trainer.pt" ]]; then
    cp -f "$OUT/$CAND/trainer.pt" "$OUT/$CAND/trainer_${phase_name}.pt"
  fi
  return "$rc"
}

run_phases_2_3() {
  local init
  init="$(pick_init_ckpt)"
  if [[ -z "$init" ]]; then
    log "ERROR: no checkpoint for phase 2/3 under $OUT/$CAND/"
    return 1
  fi
  # Refresh the pool before each LR-drop phase so we don't fine-tune only on
  # the original frozen iter-647 window.
  feed_new_shards
  log "phase 2/3 warm-start from $init"
  # ~1 epoch each on the (possibly expanded) pool; headroom + plateau early-stop.
  run_phase phase2 0.0001 1 10000 "$init" || true
  feed_new_shards
  init="$(pick_init_ckpt)"
  run_phase phase3 0.00003 1 10000 "$init" || true
  log "LR-drop phases complete"
}

wait_for_sidecars() {
  local waited=0
  while pgrep -f "run_overnight_0706.sh" >/dev/null \
     || pgrep -f "run_uniform_ablation.sh" >/dev/null \
     || pgrep -f "uniform_ablation_wrapper" >/dev/null \
     || pgrep -f "run_lr_probe.sh" >/dev/null \
     || pgrep -f "retarget_retrain" >/dev/null; do
    sleep 600
    waited=$((waited + 600))
    if [[ "$waited" -ge 172800 ]]; then
      log "48h gate cap hit; proceeding despite live processes"
      break
    fi
  done
  log "gate clear after ${waited}s"
}

chain_from_running() {
  # The currently-running phase1 was launched with the OLD binary (no idle-exit).
  # Watch its train.log for credit_exhausted stalls, then kill and LR-drop.
  # Also re-feed live shards periodically so phase1 credit keeps growing.
  log "chain-from-running: watching $LOG for credit_exhausted stalls"
  local stall_hits=0
  local loops=0
  # Immediate feed so the running process can rescan new paths.
  feed_new_shards
  while true; do
    # Avoid matching this script's own argv with pgrep -f: check via pgrep -a + awk.
    if ! ps -eo args | awk '/bootstrap_memcap_wrapper/ && !/awk/ {found=1} END{exit !found}'; then
      log "bootstrap_memcap_wrapper gone; starting phases 2-3 if ckpt exists"
      break
    fi
    loops=$((loops + 1))
    # Re-feed every ~30 min (15 * 120s) while phase1 still trains.
    if (( loops % 15 == 0 )); then
      feed_new_shards
    fi
    if [[ -f "$LOG" ]]; then
      # Count recent credit_exhausted lines in the tail
      local hits
      hits=$(tail -c 200000 "$LOG" 2>/dev/null | grep -c '"reason": "credit_exhausted"' || true)
      local steps
      steps=$(tail -c 100000 "$LOG" 2>/dev/null | grep '"event": "live_progress"' | tail -1 \
        | python3 -c "import sys,json; 
try:
  print(json.loads(sys.stdin.read()).get('steps',0))
except Exception:
  print(0)" 2>/dev/null || echo 0)
      local credit
      credit=$(tail -c 100000 "$LOG" 2>/dev/null | grep '"event": "live_progress"' | tail -1 \
        | python3 -c "import sys,json; 
try:
  print(json.loads(sys.stdin.read()).get('credit_samples',-1))
except Exception:
  print(-1)" 2>/dev/null || echo -1)
      local positions
      positions=$(tail -c 100000 "$LOG" 2>/dev/null | grep '"event": "live_progress"' | tail -1 \
        | python3 -c "import sys,json; 
try:
  print(json.loads(sys.stdin.read()).get('positions',0))
except Exception:
  print(0)" 2>/dev/null || echo 0)
      # Stall heuristic: many credit_exhausted events OR credit nearly zero with
      # steps past ~2.5 epochs (~14.5k on this window).
      if [[ "${hits:-0}" -ge 3 ]] || { [[ "${credit:-1}" -ge 0 ]] && [[ "${credit:-1}" -lt 256 ]] && [[ "${steps:-0}" -ge 14000 ]]; }; then
        stall_hits=$((stall_hits + 1))
        log "stall signal hits=$hits credit=$credit steps=$steps positions=$positions (confirm $stall_hits/2)"
      else
        stall_hits=0
      fi
      if [[ "$stall_hits" -ge 2 ]]; then
        log "phase1 stalled — killing memcap wrapper and chaining LR drops"
        # kill by pid list from ps, not pkill -f (avoids wrapper self-match issues)
        ps -eo pid,args | awk '/bootstrap_memcap_wrapper/ && !/awk/ {print $1}' | while read -r pid; do
          kill "$pid" 2>/dev/null || true
        done
        sleep 5
        break
      fi
      log "watch: steps=$steps credit=$credit positions=$positions exhausted_hits_in_tail=$hits"
    fi
    sleep 120
  done
  # Ensure trainer.pt is present (phase1 should have been saving every 1k).
  if [[ ! -f "$OUT/$CAND/trainer.pt" ]]; then
    log "ERROR: no trainer.pt after phase1; cannot LR-drop"
    return 1
  fi
  run_phases_2_3
  log "chain-from-running DONE"
}

case "$MODE" in
  full)
    wait_for_sidecars
    log "full multi-phase bootstrap starting"
    feed_new_shards
    # Phase 1: ~3 epochs (reuse 3); max_steps headroom ~20k; idle-exit ends it.
    run_phase phase1 "" 3 20000 ""
    run_phases_2_3
    log "FULL chain DONE $(date)"
    ;;
  phases-2-3)
    run_phases_2_3
    ;;
  chain-from-running)
    chain_from_running
    ;;
  feed)
    # One-shot feed for a live phase-1 process (hardlink new live shards now).
    feed_new_shards
    ;;
  *)
    echo "Usage: $0 [full|phases-2-3|chain-from-running|feed]" >&2
    exit 2
    ;;
esac
