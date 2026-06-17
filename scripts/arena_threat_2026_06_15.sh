#!/usr/bin/env bash
# Threat-planes (PR#42) arena pipeline, per docs/eval_protocol.md.
# Phase 1: warm-start two checkpoints from the live iter520 production model over
#          the full window-1 (matched budget) — threat_warm (v2_threats, 175-plane,
#          zero-init new columns) vs baseline_warm (v1, 146-plane).
# Phase 2: 1000-game paired arena, BOTH modes (matched_sims 64 + matched_time 100ms).
#
# Usage: scripts/arena_threat_2026_06_15.sh {plan|run}
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo_root="$(cd "${script_dir}/.." && pwd)"; cd "${repo_root}"
py=".venv/bin/python"
out_base="runs/arena_threat_20260615"
log="${out_base}/pipeline.log"
games="${ARENA_GAMES:-600}"   # 600 first (CI ~±15-20 Elo); add more only if borderline
ckpt="data/best_regret_checkpoints/regret_0.0075_step108246_iter520/trainer.pt"
win="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037"
win_v2="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037_v2threats"
base_out="${out_base}/baseline_warm"; threat_out="${out_base}/threat_warm"
base_ckpt="${base_out}/aurora_mlp_out/trainer.pt"; threat_ckpt="${threat_out}/aurora_mlp_out/trainer.pt"

train_common=( "${py}" scripts/offline_replay_epoch.py --candidates aurora_mlp_out
  --batch-size 512 --seed 0 --prefer-recorded-lc0-root
  --matrix-lr-multiplier 20 --matrix-weight-decay 0 --aux-weight-decay 0.0001
  --aurora-pp-iterations 3 --aurora-pp-beta 0.25 --aurora-polar-steps 8
  --aurora-polar-method polar_express --aurora-polar-dtype fp16 --aurora-polar-safety 1.01
  --smolgen-pooling flatten --eval-positions 4096 --eval-steps 8 --report-every-shards 100
  --init-checkpoint "${ckpt}" )

lg(){ printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${log}"; }

launch_train(){ # $1 name  $2 config  $3 replay-dir  $4 out-dir
  tmux has-session -t "arena_train_$1" 2>/dev/null && return 0
  local cmd; printf -v cmd '%q ' "${train_common[@]}" --config "$2" --replay-dir "$3" --out-dir "$4"
  tmux new-session -d -s "arena_train_$1" \
    "cd $(printf '%q' "${repo_root}") && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${cmd} 2>&1 | tee -a $(printf '%q' "${4}/run.log")"
  lg "LAUNCH train $1 (config=$2)"
}

run_arena(){ # $1 mode  $2 extra-flag-string  $3 label
  lg "ARENA $1 ($games games) start"
  PYTHONPATH=. "${py}" scripts/arena_standard.py \
    --candidate "${threat_ckpt}" --reference "${base_ckpt}" \
    --games "${games}" --mode "$1" $2 --device cuda \
    --out runs/arena_results.jsonl --label "$3" 2>&1 | tee -a "${log}"
  lg "ARENA $1 done"
}

cmd="${1:-plan}"
mkdir -p "${out_base}" "${base_out}" "${threat_out}"
if [[ "$cmd" == plan ]]; then
  echo "Phase1 warm-start (from $ckpt, full window-1):"
  echo "  baseline_warm: config pbt2_small, replay $win"
  echo "  threat_warm:   config exp_threat_planes, replay $win_v2"
  echo "Phase2 arena (candidate=threat_warm, reference=baseline_warm, $games games):"
  echo "  matched_sims --sims 64 ; matched_time --ms-per-move 100"
  exit 0
fi

lg "=== arena-threat pipeline start (games=${games}) ==="
# Phase 1
[[ -s "${base_ckpt}" ]] || launch_train baseline configs/pbt2_small.yaml "${win}" "${base_out}"
[[ -s "${threat_ckpt}" ]] || launch_train threat configs/exp_threat_planes.yaml "${win_v2}" "${threat_out}"
lg "waiting for both warm-start checkpoints..."
while :; do
  [[ -s "${base_ckpt}" && -s "${threat_ckpt}" ]] && { lg "both checkpoints ready"; break; }
  # bail if a training session died without producing its checkpoint
  for n in baseline threat; do
    o="${out_base}/${n}_warm"; c="${o}/aurora_mlp_out/trainer.pt"
    if ! tmux has-session -t "arena_train_${n}" 2>/dev/null && [[ ! -s "$c" ]]; then
      lg "ERROR: ${n} training died without checkpoint — aborting"; exit 1
    fi
  done
  sleep 60
done
# Phase 2: matched_sims FIRST and ONLY — stop for a clarity check before paying
# for matched_time / more games (run scripts/arena_threat_matched_time_20260615.sh
# or re-run with more games once we've seen the sims Elo).
run_arena matched_sims "--sims 64 --max-concurrent-games 256 --syzygy data/syzygy_3-4-5 --max-plies 200" "threat_v2_vs_v1_matched_sims"
lg "=== matched_sims done; STOPPING for clarity check (matched_time deferred) ==="
lg "Elo results (last 2 arena_results rows):"
tail -2 runs/arena_results.jsonl 2>/dev/null | "${py}" -c "import sys,json
for l in sys.stdin:
    d=json.loads(l)
    print('  ',d.get('label'),'Elo',round(d.get('elo',0),1),'+/-',round(d.get('elo_ci95',d.get('ci95',0)),1),'| W/D/L pairs',d.get('pair_counts',d.get('wdl','?')))" 2>/dev/null | tee -a "${log}"
