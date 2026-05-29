#!/usr/bin/env bash
set -euo pipefail

cd /home/josh/projects/chess

duration_s="${1:-21600}"
max_jobs="${MAX_JOBS:-4}"
poll_s="${POLL_S:-45}"
stamp="20260512i"
replay_dir="/home/josh/projects/chess/runs/pbt2_small/replay/train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards"
watch_log="/tmp/offline_replay_epoch_arch_split_qkv_basis_norm.log"
launch_lock="/tmp/offline_replay_epoch_launch.lock"

common_args=(
  .venv/bin/python scripts/offline_replay_epoch.py
  --config configs/bt4_aurora_asha.yaml
  --replay-dir "${replay_dir}"
  --candidates aurora_mlp_out
  --batch-size 512
  --matrix-lr-multiplier 20
  --matrix-weight-decay 0
  --aux-weight-decay 0.0001
  --zclip-max-norm 5
  --zclip-z-thresh 2.0
  --zclip-clip-factor 1.0
  --input-history-encoding legacy
  --input-pos-encoding arc_adapter
  --qkv-projection split
  --smolgen on
  --smolgen-mode per_layer
  --smolgen-bias-scale none
  --smolgen-bias-norm none
  --arc-attention-bias none
  --smolgen-relation-basis on
  --smolgen-relation-norm basis_center_rms
  --smolgen-relation-coeff-norm rms
  --smolgen-relation-scale layer_head
  --deepnorm on
  --eval-positions 2048
  --eval-steps 8
  --report-every-shards 50
)

experiments=(
  "arch_11l_splitqkv_mlp_out_basis_coeff_seed0|runs/offline_replay_epoch_arch_11l_splitqkv_mlp_out_basis_coeff_seed0_${stamp}|--seed 0 --num-layers 11"
  "arch_11l_splitqkv_mlp_out_basis_coeff_seed1|runs/offline_replay_epoch_arch_11l_splitqkv_mlp_out_basis_coeff_seed1_${stamp}|--seed 1 --num-layers 11"
  "arch_10l_splitqkv_mlp_out_basis_coeff_seed0|runs/offline_replay_epoch_arch_10l_splitqkv_mlp_out_basis_coeff_seed0_${stamp}|--seed 0 --num-layers 10"
  "arch_10l_splitqkv_mlp_out_basis_coeff_seed1|runs/offline_replay_epoch_arch_10l_splitqkv_mlp_out_basis_coeff_seed1_${stamp}|--seed 1 --num-layers 10"
)

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${watch_log}"
}

result_done() {
  [[ -s "$1/results.jsonl" ]]
}

session_alive() {
  tmux has-session -t "$1" 2>/dev/null
}

global_active_count() {
  pgrep -af "^\\.venv/bin/python scripts/offline_replay_epoch\\.py" | wc -l
}

own_active_count() {
  local n=0 session out_dir _args
  for spec in "${experiments[@]}"; do
    IFS='|' read -r session out_dir _args <<<"${spec}"
    if session_alive "${session}" && ! result_done "${out_dir}"; then
      n=$((n + 1))
    fi
  done
  echo "${n}"
}

launch_run() {
  local session="$1"
  local out_dir="$2"
  local extra_args="$3"
  local log_path="/tmp/offline_replay_epoch_${session}.log"
  if result_done "${out_dir}" || session_alive "${session}"; then
    return 1
  fi
  tmux new-session -d -s "${session}" \
    "cd /home/josh/projects/chess && ${common_args[*]} --out-dir ${out_dir} ${extra_args} 2>&1 | tee ${log_path}"
  log "launched ${session}: ${out_dir}"
  return 0
}

summarize_results() {
  .venv/bin/python - <<'PY'
import glob, json, os

rows = []
for path in glob.glob("runs/offline_replay_epoch_arch_*_20260512*/results.jsonl"):
    try:
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        if not lines:
            continue
        data = json.loads(lines[-1])
    except Exception:
        continue
    if "eval_loss" in data:
        rows.append((float(data["eval_loss"]), os.path.dirname(path)))
for loss, path in sorted(rows)[:80]:
    print(f"{loss:.6f}\t{path}")
PY
}

log "starting split-QKV basis/coeff relation queue: duration=${duration_s}s max_jobs=${max_jobs}"
start_s="$(date +%s)"
last_summary_s=0

while true; do
  now_s="$(date +%s)"
  if (( now_s - start_s >= duration_s )); then
    log "time budget expired; no new launches"
    summarize_results | tee -a "${watch_log}"
    exit 0
  fi

  launch_status=0
  (
    flock -n 9 || exit 75
    while (( "$(global_active_count)" < max_jobs )); do
      launched=0
      for spec in "${experiments[@]}"; do
        IFS='|' read -r session out_dir extra_args <<<"${spec}"
        if launch_run "${session}" "${out_dir}" "${extra_args}"; then
          launched=1
          break
        fi
      done
      if (( launched == 0 )); then
        exit 2
      fi
    done
  ) 9>"${launch_lock}" || launch_status=$?
  if (( launch_status == 2 )); then
    log "queue exhausted; exiting"
    summarize_results | tee -a "${watch_log}"
    exit 0
  fi

  if (( now_s - last_summary_s >= 1800 )); then
    log "periodic summary; global_active=$(global_active_count) own_active=$(own_active_count)"
    summarize_results | tee -a "${watch_log}"
    last_summary_s="${now_s}"
  fi

  sleep "${poll_s}"
done
