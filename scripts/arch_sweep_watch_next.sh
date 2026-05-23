#!/usr/bin/env bash
set -euo pipefail

cd /home/josh/projects/chess

deadline_s="${1:-2700}"
poll_s="${POLL_S:-60}"
stamp="20260512"
replay_dir="/home/josh/projects/chess/runs/pbt2_small/replay/train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards"
watch_log="/tmp/offline_replay_epoch_arch_watch_next.log"

current_runs=(
  "runs/offline_replay_epoch_arch_smolscale_lh_${stamp}/results.jsonl"
  "runs/offline_replay_epoch_arch_smolscale_rms_${stamp}/results.jsonl"
  "runs/offline_replay_epoch_arch_arcattn_scale_${stamp}/results.jsonl"
  "runs/offline_replay_epoch_arch_relbasis_combo_${stamp}/results.jsonl"
)

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
  --smolgen-bias-scale layer_head
  --deepnorm on
  --eval-positions 2048
  --eval-steps 8
  --report-every-shards 50
)

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${watch_log}"
}

complete() {
  local path
  for path in "${current_runs[@]}"; do
    [[ -s "${path}" ]] || return 1
  done
  return 0
}

launch_run() {
  local session="$1"
  local out_dir="$2"
  local log_path="$3"
  shift 3
  if tmux has-session -t "${session}" 2>/dev/null; then
    log "session ${session} already exists; not relaunching"
    return
  fi
  if [[ -s "${out_dir}/results.jsonl" ]]; then
    log "result ${out_dir}/results.jsonl already exists; not relaunching"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd /home/josh/projects/chess && ${common_args[*]} --out-dir ${out_dir} $* 2>&1 | tee ${log_path}"
  log "launched ${session}: ${out_dir}"
}

log "watching current architecture sweep for up to ${deadline_s}s"
start_s="$(date +%s)"
while ! complete; do
  now_s="$(date +%s)"
  if (( now_s - start_s >= deadline_s )); then
    log "timeout waiting for current sweep; leaving next batch unlaunched"
    exit 0
  fi
  sleep "${poll_s}"
done

log "current sweep complete; summary:"
.venv/bin/python - <<'PY' | tee -a /tmp/offline_replay_epoch_arch_watch_next.log
import glob, json, os
rows = []
for path in glob.glob("runs/offline_replay_epoch_arch_*_20260512/results.jsonl"):
    try:
        line = open(path, encoding="utf-8").read().strip().splitlines()[-1]
        data = json.loads(line)
    except Exception:
        continue
    if "eval_loss" in data:
        rows.append((float(data["eval_loss"]), os.path.dirname(path)))
for loss, path in sorted(rows):
    print(f"{loss:.6f}\t{path}")
PY

launch_run \
  arch_smolscale_center \
  "runs/offline_replay_epoch_arch_smolscale_center_${stamp}" \
  "/tmp/offline_replay_epoch_arch_smolscale_center.log" \
  --input-pos-encoding arc_adapter \
  --smolgen-bias-norm center \
  --arc-attention-bias none \
  --smolgen-relation-basis off

launch_run \
  arch_arcattn_noadapter \
  "runs/offline_replay_epoch_arch_arcattn_noadapter_${stamp}" \
  "/tmp/offline_replay_epoch_arch_arcattn_noadapter.log" \
  --input-pos-encoding none \
  --smolgen-bias-norm none \
  --arc-attention-bias basic \
  --smolgen-relation-basis off

launch_run \
  arch_relbasis_only \
  "runs/offline_replay_epoch_arch_relbasis_only_${stamp}" \
  "/tmp/offline_replay_epoch_arch_relbasis_only.log" \
  --input-pos-encoding arc_adapter \
  --smolgen-bias-norm none \
  --arc-attention-bias none \
  --smolgen-relation-basis on

launch_run \
  arch_relbasis_combo_nonnorm \
  "runs/offline_replay_epoch_arch_relbasis_combo_nonnorm_${stamp}" \
  "/tmp/offline_replay_epoch_arch_relbasis_combo_nonnorm.log" \
  --input-pos-encoding arc_adapter \
  --smolgen-bias-norm none \
  --arc-attention-bias basic \
  --smolgen-relation-basis on

log "next architecture batch queued"
