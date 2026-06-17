#!/usr/bin/env bash
# Wave-4 of the 2026-06-15 sidecar screen: 2nd-window robustness of the NEW bets
# (so winners are arena-ready, not single-window noise) + sf_policy_sparse_ce
# (PR#45, the last clean new feature). Self-contained: its own baseline on
# window-2 is the reference for every delta here.
#
# Window-2 = current_live_third_group_20260526 (962 shards). Chains after wave-3
# (tmux 'sidecar_sweep_w3'). threat/dynamic need the window-2 v2_threats
# conversion; a prereq-gate holds those two until it exists.
#
# Usage: scripts/sidecar_sweep_2026_06_15_wave4.sh {plan|run|summary}
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${SIDECAR_PYTHON:-.venv/bin/python}"
max_jobs="${MAX_JOBS:-2}"
poll_s="${POLL_S:-30}"
stamp="20260615d"
out_base="runs/sidecar_sweep_20260615d"
sweep_log="${out_base}/sweep.log"
wait_driver="sidecar_sweep_w3"

window="runs/parallel_candidate_replay_snapshots/current_live_third_group_20260526_000000"
window_v2="runs/parallel_candidate_replay_snapshots/current_live_third_group_20260526_000000_v2threats"
v2_min_shards=900    # window-2 has 962; require near-complete conversion

common_args=(
  "${python_bin}" scripts/offline_replay_epoch.py
  --config configs/pbt2_small.yaml
  --replay-dir "${window}"
  --candidates aurora_mlp_out
  --batch-size 512
  --seed 0
  --prefer-recorded-lc0-root
  --matrix-lr-multiplier 20
  --matrix-weight-decay 0
  --aux-weight-decay 0.0001
  --aurora-pp-iterations 3
  --aurora-pp-beta 0.25
  --aurora-polar-steps 8
  --aurora-polar-method polar_express
  --aurora-polar-dtype fp16
  --aurora-polar-safety 1.01
  --smolgen-pooling flatten
  --eval-positions 4096
  --eval-steps 8
  --report-every-shards 100
)

experiments=(
  "baseline|"
  "soft_ablation|--config configs/exp_soft_policy_ablation.yaml"
  "soft_divergent|--config configs/exp_soft_policy_divergent_only.yaml"
  "cat_blend_0p3|--config configs/exp_categorical_continuous_0p3.yaml"
  "cat_blend_0p5|--config configs/exp_categorical_continuous.yaml"
  "cat_blend_0p7|--config configs/exp_categorical_continuous_0p7.yaml"
  "threat_planes|--config configs/exp_threat_planes.yaml --replay-dir ${window_v2}"
  "dynamic_relations|--config configs/exp_dynamic_relations.yaml --replay-dir ${window_v2}"
  "sparse_ce|--config configs/exp_sparse_ce.yaml"
)

session_of() { echo "sweep_${stamp}_$1"; }
outdir_of() { echo "${out_base}/$1"; }
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${sweep_log}"; }
is_done() { [[ -s "$(outdir_of "$1")/results.jsonl" ]]; }
session_alive() { tmux has-session -t "$(session_of "$1")" 2>/dev/null; }
v2_ready() { [[ -d "${window_v2}" ]] && [[ "$(ls "${window_v2}" 2>/dev/null | grep -c '\.zarr$')" -ge "${v2_min_shards}" ]]; }
conversion_alive() { pgrep -af "convert_shards_v2_threats" >/dev/null 2>&1; }
prereq_ready() { case "$1" in *"${window_v2}"*) v2_ready ;; *) return 0 ;; esac; }

launch() {
  local name="$1" extra_str="$2" session outdir
  session="$(session_of "${name}")"; outdir="$(outdir_of "${name}")"; mkdir -p "${outdir}"
  local -a extra; read -ra extra <<<"${extra_str}"
  local -a cmd=("${common_args[@]}" "${extra[@]}" --out-dir "${outdir}")
  local cmd_str; printf -v cmd_str '%q ' "${cmd[@]}"
  tmux new-session -d -s "${session}" \
    "cd $(printf '%q' "${repo_root}") && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${cmd_str} 2>&1 | tee -a $(printf '%q' "${outdir}/run.log")"
  log "LAUNCH ${name}  (session=${session})  extra=[${extra_str}]"
}

print_plan() {
  echo "Sidecar sweep wave-4 (${stamp}) — window-2 robustness + sparse_ce — window=${window}"
  echo "Waits for '${wait_driver}', then runs ${max_jobs} at a time. ref = this wave's own baseline."
  local i=1 name extra
  for spec in "${experiments[@]}"; do
    IFS='|' read -r name extra <<<"${spec}"; printf "  %2d. %-18s %s\n" "${i}" "${name}" "${extra:-（baseline）}"; i=$((i + 1))
  done
}

run_sweep() {
  mkdir -p "${out_base}"
  if tmux has-session -t "${wait_driver}" 2>/dev/null; then
    log "waiting for '${wait_driver}' to finish before starting wave-4..."
    while tmux has-session -t "${wait_driver}" 2>/dev/null; do sleep 60; done
  fi
  log "=== wave-4 (${stamp}) start (max_jobs=${max_jobs}) ==="
  local -a pending=()
  local spec name extra
  for spec in "${experiments[@]}"; do
    IFS='|' read -r name extra <<<"${spec}"
    if is_done "${name}"; then log "SKIP ${name} (already done)"; else pending+=("${spec}"); fi
  done

  while true; do
    local -a alive=(); local p pn pe
    for p in "${pending[@]}"; do IFS='|' read -r pn pe <<<"${p}"; is_done "${pn}" || alive+=("${p}"); done
    pending=("${alive[@]}")
    (( ${#pending[@]} == 0 )) && { log "wave-4 all finished"; break; }
    local running=0 launched_any=0
    for p in "${pending[@]}"; do IFS='|' read -r pn pe <<<"${p}"; session_alive "${pn}" && running=$((running + 1)); done
    for p in "${pending[@]}"; do
      (( running >= max_jobs )) && break
      IFS='|' read -r pn pe <<<"${p}"
      session_alive "${pn}" && continue
      prereq_ready "${pe}" || continue
      launch "${pn}" "${pe}"; running=$((running + 1)); launched_any=1
    done
    if (( running == 0 )) && (( launched_any == 0 )); then
      if ! conversion_alive && ! v2_ready; then log "ABORT remaining (window-2 v2 conversion missing): ${pending[*]}"; break; fi
      log "waiting on window-2 v2 conversion; pending=${#pending[@]}"
    fi
    sleep "${poll_s}"
  done
  log "=== wave-4 done; summarizing ==="
  summarize || true
}

summarize() {
  "${python_bin}" - "${out_base}" <<'PY'
import json, sys, glob, os
base = sys.argv[1]; rows = {}
for rp in glob.glob(os.path.join(base, "*", "results.jsonl")):
    name = os.path.basename(os.path.dirname(rp))
    try: rows[name] = [json.loads(l) for l in open(rp) if l.strip()][-1]
    except Exception: pass
keys = ["eval_policy_loss","eval_wdl_loss","eval_soft_policy_loss","eval_categorical_loss","eval_sf_move_loss","eval_wdl_brier","eval_wdl_ece"]
base_row = rows.get("baseline", {})
hdr = f"{'experiment':18s} " + " ".join(f"{k.replace('eval_',''):>14s}" for k in keys)
print("window-2 robustness (ref = this wave's baseline)" + ("" if base_row else "  (baseline pending)"))
print(hdr); print("-"*len(hdr))
for name in sorted(rows):
    r = rows[name]; cells=[]
    for k in keys:
        v=r.get(k)
        if v is None: cells.append(f"{'-':>14s}"); continue
        if name!="baseline" and base_row.get(k) is not None: cells.append(f"{v:7.4f}({v-base_row[k]:+.4f})")
        else: cells.append(f"{v:14.4f}")
    print(f"{name:18s} " + " ".join(cells))
PY
}

case "${1:-plan}" in
  plan) print_plan ;;
  run) run_sweep ;;
  summary) summarize ;;
  *) echo "usage: $0 {plan|run|summary}" >&2; exit 2 ;;
esac
