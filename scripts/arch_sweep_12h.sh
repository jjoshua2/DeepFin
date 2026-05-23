#!/usr/bin/env bash
set -euo pipefail

cd /home/josh/projects/chess

duration_s="${1:-43200}"
max_jobs="${MAX_JOBS:-4}"
poll_s="${POLL_S:-60}"
stamp="20260512"
replay_dir="/home/josh/projects/chess/runs/pbt2_small/replay/train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards"
watch_log="/tmp/offline_replay_epoch_arch_12h.log"

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
  --smolgen on
  --smolgen-mode per_layer
  --deepnorm on
  --eval-positions 2048
  --eval-steps 8
  --report-every-shards 50
)

# Format: session|out_dir|extra offline_replay_epoch.py args
# The first four entries are the currently running batch; listing them here lets
# this script treat already-started work as part of the queue and skip it.
experiments=(
  "arch_smolscale_lh|runs/offline_replay_epoch_arch_smolscale_lh_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_smolscale_rms|runs/offline_replay_epoch_arch_smolscale_rms_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis off"
  "arch_arcattn_scale|runs/offline_replay_epoch_arch_arcattn_scale_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias basic --smolgen-relation-basis off"
  "arch_relbasis_combo|runs/offline_replay_epoch_arch_relbasis_combo_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias basic --smolgen-relation-basis on"

  "arch_smolscale_layer|runs/offline_replay_epoch_arch_smolscale_layer_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_smolscale_layer_rms|runs/offline_replay_epoch_arch_smolscale_layer_rms_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis off"
  "arch_arcattn_noadapter|runs/offline_replay_epoch_arch_arcattn_noadapter_${stamp}|--input-pos-encoding none --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias basic --smolgen-relation-basis off"
  "arch_arcattn_rms|runs/offline_replay_epoch_arch_arcattn_rms_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias basic --smolgen-relation-basis off"

  "arch_relbasis_only|runs/offline_replay_epoch_arch_relbasis_only_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_rms|runs/offline_replay_epoch_arch_relbasis_rms_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm center_rms --arc-attention-bias none --smolgen-relation-basis on"
  "arch_relbasis_combo_nonnorm|runs/offline_replay_epoch_arch_relbasis_combo_nonnorm_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias basic --smolgen-relation-basis on"
  "arch_relbasis_noadapter|runs/offline_replay_epoch_arch_relbasis_noadapter_${stamp}|--input-pos-encoding none --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias basic --smolgen-relation-basis on"

  "arch_smolscale_lh_noadapter|runs/offline_replay_epoch_arch_smolscale_lh_noadapter_${stamp}|--input-pos-encoding none --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_smolscale_lh_rawarc|runs/offline_replay_epoch_arch_smolscale_lh_rawarc_${stamp}|--input-pos-encoding arc --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off"
  "arch_arcattn_rawarc|runs/offline_replay_epoch_arch_arcattn_rawarc_${stamp}|--input-pos-encoding arc --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias basic --smolgen-relation-basis off"
  "arch_relbasis_rawarc|runs/offline_replay_epoch_arch_relbasis_rawarc_${stamp}|--input-pos-encoding arc --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias basic --smolgen-relation-basis on"

  "arch_smolscale_lh_repeat|runs/offline_replay_epoch_arch_smolscale_lh_repeat_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale layer_head --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off --seed 1"
  "arch_control_repeat_seed1|runs/offline_replay_epoch_arch_control_repeat_seed1_${stamp}|--input-pos-encoding arc_adapter --smolgen-bias-scale none --smolgen-bias-norm none --arc-attention-bias none --smolgen-relation-basis off --seed 1"
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
  if result_done "${out_dir}"; then
    return 1
  fi
  if session_alive "${session}"; then
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
for path in glob.glob("runs/offline_replay_epoch_arch_*_20260512/results.jsonl"):
    try:
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        if not lines:
            continue
        data = json.loads(lines[-1])
    except Exception:
        continue
    if "eval_loss" in data:
        rows.append((float(data["eval_loss"]), os.path.dirname(path)))
for loss, path in sorted(rows)[:30]:
    print(f"{loss:.6f}\t{path}")
PY
}

log "starting 12h architecture queue: duration=${duration_s}s max_jobs=${max_jobs}"
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
      log "queue exhausted; waiting for existing jobs then exiting"
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
