#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

duration_s="${1:-43200}"
max_jobs="${MAX_JOBS:-4}"
poll_s="${POLL_S:-60}"
stamp="20260512c"
replay_dir="${repo_root}/runs/pbt2_small/replay/train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards"
watch_log="/tmp/offline_replay_epoch_arch_more_12h.log"
launch_lock="/tmp/offline_replay_epoch_launch.lock"

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
  # Does relation-basis substitute for depth?
  "arch_control_9layer|runs/offline_replay_epoch_arch_control_9layer_${stamp}|--num-layers 9 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_9layer|runs/offline_replay_epoch_arch_relbasis_rms_9layer_${stamp}|--num-layers 9 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"

  # Head-count / specialization. 384 is divisible by 8, 12, and 16.
  "arch_control_8heads|runs/offline_replay_epoch_arch_control_8heads_${stamp}|--num-heads 8 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_8heads|runs/offline_replay_epoch_arch_relbasis_rms_8heads_${stamp}|--num-heads 8 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_control_16heads|runs/offline_replay_epoch_arch_control_16heads_${stamp}|--num-heads 16 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_16heads|runs/offline_replay_epoch_arch_relbasis_rms_16heads_${stamp}|--num-heads 16 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"

  # Winner may want slightly different optimizer/clipping.
  "arch_relbasis_rms_lr16|runs/offline_replay_epoch_arch_relbasis_rms_lr16_${stamp}|--matrix-lr-multiplier 16 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_rms_lr24|runs/offline_replay_epoch_arch_relbasis_rms_lr24_${stamp}|--matrix-lr-multiplier 24 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_rms_z1p75|runs/offline_replay_epoch_arch_relbasis_rms_z1p75_${stamp}|--zclip-z-thresh 1.75 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_rms_z2p25|runs/offline_replay_epoch_arch_relbasis_rms_z2p25_${stamp}|--zclip-z-thresh 2.25 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"

  # Parameter allocation around the winning architecture.
  "arch_relbasis_rms_ffn1p25|runs/offline_replay_epoch_arch_relbasis_rms_ffn1p25_${stamp}|--ffn-mult 1.25 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_rms_ffn1p75|runs/offline_replay_epoch_arch_relbasis_rms_ffn1p75_${stamp}|--ffn-mult 1.75 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"

  # Another seed pair to confirm signal.
  "arch_control_seed3|runs/offline_replay_epoch_arch_control_seed3_${stamp}|--seed 3 --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_relbasis_rms_seed3|runs/offline_replay_epoch_arch_relbasis_rms_seed3_${stamp}|--seed 3 --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
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
    "cd \"${repo_root}\" && ${common_args[*]} --out-dir ${out_dir} ${extra_args} 2>&1 | tee ${log_path}"
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
for loss, path in sorted(rows)[:50]:
    print(f"{loss:.6f}\t{path}")
PY
}

log "starting more architecture queue: duration=${duration_s}s max_jobs=${max_jobs}"
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
