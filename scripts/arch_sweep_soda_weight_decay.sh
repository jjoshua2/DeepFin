#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

duration_s="${1:-21600}"
max_jobs="${MAX_JOBS:-4}"
poll_s="${POLL_S:-45}"
stamp="20260515_soda"
replay_dir="${repo_root}/runs/pbt2_small/replay/train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards"
watch_log="/tmp/offline_replay_epoch_soda_weight_decay.log"

common_args=(
  .venv/bin/python scripts/offline_replay_epoch.py
  --config configs/bt4_aurora_asha.yaml
  --replay-dir "${replay_dir}"
  --candidates aurora_mlp_out
  --batch-size 512
  --matrix-lr-multiplier 20
  --zclip-max-norm 5
  --zclip-z-thresh 2.0
  --zclip-clip-factor 1.0
  --input-history-encoding legacy
  --input-pos-encoding arc_adapter
  --qkv-projection split
  --smolgen on
  --smolgen-mode per_layer
  --smolgen-bias-scale layer_head
  --smolgen-bias-norm center_rms
  --arc-attention-bias none
  --smolgen-relation-basis on
  --deepnorm on
  --num-layers 11
  --eval-positions 2048
  --eval-steps 8
  --report-every-shards 50
)

experiments=(
  "soda_aux_only|runs/offline_replay_epoch_soda_aux_only_${stamp}|--seed 0 --weight-decay-mode soda --matrix-weight-decay 0 --aux-weight-decay 0.0001"
  "soda_matrix_wd|runs/offline_replay_epoch_soda_matrix_wd_${stamp}|--seed 0 --weight-decay-mode weight_decay --matrix-weight-decay 0.0001 --aux-weight-decay 0.0001"
  "soda_matrix_aux|runs/offline_replay_epoch_soda_matrix_aux_${stamp}|--seed 0 --weight-decay-mode soda --matrix-weight-decay 0.0001 --aux-weight-decay 0.0001"
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
  ps -eo comm,args | awk '$1 ~ /python/ && $0 ~ /scripts\/offline_replay_epoch.py/ {n++} END {print n + 0}'
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
    "cd \"${repo_root}\" && ${common_args[*]} --out-dir ${out_dir} ${extra_args} 2>&1 | tee ${log_path}"
  log "launched ${session}: ${out_dir}"
  return 0
}

summarize_results() {
  .venv/bin/python - <<'PY'
import glob, json, os
rows = []
for path in glob.glob("runs/offline_replay_epoch_soda_*_20260515_soda/results.jsonl"):
    try:
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        if not lines:
            continue
        data = json.loads(lines[-1])
    except Exception:
        continue
    if "eval_loss" in data:
        rows.append((float(data["eval_loss"]), os.path.dirname(path), data.get("weight_decay_mode")))
for loss, path, mode in sorted(rows):
    print(f"{loss:.6f}\t{mode}\t{path}")
PY
}

deadline=$((SECONDS + duration_s))
log "watching SODA weight-decay sweep for up to ${duration_s}s"

while (( SECONDS < deadline )); do
  launched=0
  for spec in "${experiments[@]}"; do
    IFS='|' read -r session out_dir extra_args <<<"${spec}"
    if result_done "${out_dir}" || session_alive "${session}"; then
      continue
    fi
    if (( $(global_active_count) >= max_jobs || $(own_active_count) >= max_jobs )); then
      break
    fi
    if (( $(global_active_count) < max_jobs && $(own_active_count) < max_jobs )); then
      launch_run "${session}" "${out_dir}" "${extra_args}" && launched=1
    fi
  done

  remaining=0
  for spec in "${experiments[@]}"; do
    IFS='|' read -r _session out_dir _args <<<"${spec}"
    result_done "${out_dir}" || remaining=$((remaining + 1))
  done
  if (( remaining == 0 )); then
    log "SODA sweep complete; summary:"
    summarize_results | tee -a "${watch_log}"
    exit 0
  fi

  if (( launched == 0 )); then
    sleep "${poll_s}"
  fi
done

log "timeout waiting for SODA sweep; partial summary:"
summarize_results | tee -a "${watch_log}"
