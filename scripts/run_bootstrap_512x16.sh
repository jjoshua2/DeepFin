#!/bin/bash
# 512x16 scale-up bootstrap driver (2026-07-07; redesigned 2026-07-09).
#
# Fresh-init 63.08M net (embed 512 / 16L / h16) trained offline on the
# DEDICATED fed pool (data/scaleup_pool_512x16 — seeded from the frozen
# iter-647 salvage window, then continuously fed live shards by
# scripts/feed_bootstrap_shards.py; the salvage pool itself is a ledgered
# frozen revert point and is never mutated).
#
# BUG history (2026-07-08): a single live-follow call with target_reuse=3 on a
# frozen pool spent ~3 epochs of credit then slept forever; --live-idle-exit-after
# fixes termination.
#
# DESIGN (2026-07-09): probe-driven, not pre-scripted. The parity probe
# (value_regret vs the live net, scratchpad/scaleup/parity_probe.log) is the
# experiment's decision signal. Run ONE training process at a time; when it
# stops (idle-exit / plateau / max-steps), read the probe trajectory and choose
# the next lever explicitly:
#   - far from parity, gap closing  -> `continue` (more data, same LR)
#   - close to parity, gap flat     -> `continue 0.0001` then `continue 0.00003`
#     (LR drops to squeeze the tail)
#   - plateaued far short of parity -> kill per the ledger swap-gate rule
# There is NO auto-chained kill-and-drop: an automated wrong-LR phase costs a
# GPU-day (the 07-08 review found exactly that failure mode).
#
# SWAP GATE (pre-committed, ledger 4c): live restart ONLY at audit parity —
# value_regret + audit_targets (2000 pos, paired CIs) within +2cp of the
# then-current live net, panels not worse. Below parity = extend/kill.
set -euo pipefail
cd /home/josh/projects/chess
OUT=runs/scaleup_512x16_bootstrap
CAND=aurora_mlp_out
POOL=data/scaleup_pool_512x16/replay_shards
FFN16="1.5,1.5,1.5,1.5,1.5,1.555556,1.65,1.772222,1.894444,1.666667,1.638889,1.905556,1.783333,1.794444,1.744444,1.5"
MEMCAP=0.30
# Static/fed pool: exit ~10 min after credit is gone (feeds land every ~30 min,
# so a short idle usually means the feeder is between passes — don't flap).
IDLE_EXIT=600
# Stop a run if eval_loss fails to improve for this many evals (eval every 500).
# Env-overridable: LR-drop continuations improve slowly — the 3500-step
# eval_plateau stop on 07-09 fired on value-oscillation noise; use 6+ there.
PLATEAU_EVALS="${PLATEAU_EVALS:-3}"
MODE="${1:-full}"

mkdir -p "$OUT"
LOG="$OUT/train.log"
PHASE_LOG="$OUT/phase_driver.log"

log() { echo "[bootstrap $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PHASE_LOG"; }

feed_new_shards() {
  log "feeding new live shards into $POOL"
  PYTHONPATH=. python3 scripts/feed_bootstrap_shards.py --boot-dir "$POOL" 2>&1 | tee -a "$PHASE_LOG"
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

run_training() {
  local run_name="$1"
  local lr_arg="$2"          # empty = omit --lr (fresh: config default; warm: checkpoint schedule)
  local reuse="$3"
  local max_steps="$4"
  local init_ckpt="$5"       # empty = fresh init

  local run_out="$OUT/${run_name}.log"
  log "START $run_name lr=${lr_arg:-inherit} reuse=$reuse max_steps=$max_steps init=${init_ckpt:-FRESH}"

  local cmd=(
    python3 scripts/bootstrap_memcap_wrapper.py --mem-fraction "$MEMCAP" --
    --config configs/pbt2_small.yaml
    --replay-dir "$POOL"
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
    --live-save-every-steps 500
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
  PYTHONPATH=. nice -n 15 "${cmd[@]}" >>"$run_out" 2>&1
  local rc=$?
  set -e
  {
    echo "{\"event\": \"run_boundary\", \"run\": \"$run_name\", \"rc\": $rc, \"ts\": \"$(date -Iseconds)\"}"
    tail -n 5 "$run_out" 2>/dev/null || true
  } >>"$LOG"
  log "END $run_name rc=$rc (detail: $run_out)"
  # Park a run-tagged copy for forensics; trainer.pt already points at the
  # best checkpoint when the run stopped on an eval plateau.
  if [[ -f "$OUT/$CAND/trainer.pt" ]]; then
    cp -f "$OUT/$CAND/trainer.pt" "$OUT/$CAND/trainer_${run_name}.pt"
  fi
  return "$rc"
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

require_no_running_bootstrap() {
  # Scope the check to THIS run's out-dir so an unrelated memcapped offline
  # job is never collateral.
  if ps -eo args | grep -F "bootstrap_memcap_wrapper" | grep -F -- "--out-dir $OUT" | grep -qv grep; then
    log "ERROR: a bootstrap for $OUT is already running; stop it first (no auto-kill)."
    exit 1
  fi
}

case "$MODE" in
  full)
    require_no_running_bootstrap
    wait_for_sidecars
    log "fresh-init bootstrap starting (probe-driven; no auto LR chain)"
    feed_new_shards
    run_training run1 "" 3 60000 ""
    log "run1 done — read scratchpad/scaleup/parity_probe.log before choosing the next lever"
    ;;
  continue)
    # Warm continuation from the best/last checkpoint. Optional 2nd arg = LR
    # override (e.g. 0.0001); omit to inherit the checkpoint's schedule.
    require_no_running_bootstrap
    wait_for_sidecars
    LR="${2:-}"
    INIT="$(pick_init_ckpt)"
    if [[ -z "$INIT" ]]; then
      log "ERROR: no checkpoint to continue from under $OUT/$CAND/"
      exit 1
    fi
    feed_new_shards
    STAMP=$(date '+%m%d_%H%M')
    run_training "cont_${STAMP}_lr${LR:-inherit}" "$LR" 1 60000 "$INIT"
    log "continuation done — read scratchpad/scaleup/parity_probe.log before choosing the next lever"
    ;;
  feed)
    feed_new_shards
    ;;
  *)
    echo "Usage: $0 [full|continue [LR]|feed]" >&2
    exit 2
    ;;
esac
