#!/usr/bin/env bash
set -euo pipefail

cd /home/josh/projects/chess

duration_s="${1:-21600}"
max_jobs="${MAX_JOBS:-4}"
poll_s="${POLL_S:-60}"
stamp="20260512b"
replay_dir="/home/josh/projects/chess/runs/pbt2_small/replay/train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards"
watch_log="/tmp/offline_replay_epoch_arch_depth_relbasis.log"

common_args=(
  .venv/bin/python scripts/offline_replay_epoch.py
  --config configs/bt4_aurora_asha.yaml
  --replay-dir "${replay_dir}"
  --candidates aurora_mlp_only
  --batch-size 512
  --matrix-lr-multiplier 20
  --matrix-weight-decay 0
  --aux-weight-decay 0.0001
  --zclip-max-norm 5
  --zclip-z-thresh 2.0
  --zclip-clip-factor 1.0
  --input-history-encoding legacy
  --input-pos-encoding arc_adapter
  --smolgen on
  --smolgen-mode per_layer
  --deepnorm on
  --eval-positions 2048
  --eval-steps 8
  --report-every-shards 50
)

experiments=(
  # Direct depth comparison.
  "arch_control_11layer|runs/offline_replay_epoch_arch_control_11layer_${stamp}|--num-layers 11 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_11layer|runs/offline_replay_epoch_arch_relbasis_rms_11layer_${stamp}|--num-layers 11 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_control_12layer|runs/offline_replay_epoch_arch_control_12layer_${stamp}|--num-layers 12 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_12layer|runs/offline_replay_epoch_arch_relbasis_rms_12layer_${stamp}|--num-layers 12 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"

  # Cheaper variants around the winning relation-basis result.
  "arch_relbasis_center|runs/offline_replay_epoch_arch_relbasis_center_${stamp}|--smolgen-bias-scale layer_head --smolgen-bias-norm center --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_center_combo|runs/offline_replay_epoch_arch_relbasis_center_combo_${stamp}|--smolgen-bias-scale layer_head --smolgen-bias-norm center --arc-attention-bias basic --smolgen-relation-basis on"
  "arch_relbasis_rms_noscale|runs/offline_replay_epoch_arch_relbasis_rms_noscale_${stamp}|--smolgen-bias-scale none --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_rms_layer_scale|runs/offline_replay_epoch_arch_relbasis_rms_layer_scale_${stamp}|--smolgen-bias-scale layer --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"

  # Confirm the winner and compare against control under another seed.
  "arch_relbasis_rms_seed1|runs/offline_replay_epoch_arch_relbasis_rms_seed1_${stamp}|--seed 1 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_control_seed2|runs/offline_replay_epoch_arch_control_seed2_${stamp}|--seed 2 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_seed2|runs/offline_replay_epoch_arch_relbasis_rms_seed2_${stamp}|--seed 2 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
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

active_count() {
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
for loss, path in sorted(rows)[:40]:
    print(f"{loss:.6f}\t{path}")
PY
}

log "starting depth/relation architecture queue: duration=${duration_s}s max_jobs=${max_jobs}"
start_s="$(date +%s)"
last_summary_s=0

while true; do
  now_s="$(date +%s)"
  if (( now_s - start_s >= duration_s )); then
    log "time budget expired; no new launches"
    summarize_results | tee -a "${watch_log}"
    exit 0
  fi

  while (( "$(active_count)" < max_jobs )); do
    launched=0
    for spec in "${experiments[@]}"; do
      IFS='|' read -r session out_dir extra_args <<<"${spec}"
      if launch_run "${session}" "${out_dir}" "${extra_args}"; then
        launched=1
        break
      fi
    done
    if (( launched == 0 )); then
      log "queue exhausted; exiting"
      summarize_results | tee -a "${watch_log}"
      exit 0
    fi
  done

  if (( now_s - last_summary_s >= 1800 )); then
    log "periodic summary; active=$(active_count)"
    summarize_results | tee -a "${watch_log}"
    last_summary_s="${now_s}"
  fi

  sleep "${poll_s}"
done
