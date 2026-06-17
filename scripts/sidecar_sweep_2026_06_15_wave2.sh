#!/usr/bin/env bash
# Wave-2 of the 2026-06-15 sidecar screen: production-arch "is it earning its
# keep?" ablations + w_future settings. Same window/seed/common-args as wave-1,
# so every result compares against wave-1's `baseline` run (referenced below).
#
# Waits for wave-1's driver (tmux session 'sidecar_sweep') to finish so the two
# waves never contend for the GPU. Then runs 2 jobs at a time, same as wave-1.
#
# Usage: scripts/sidecar_sweep_2026_06_15_wave2.sh {plan|run|summary}
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${SIDECAR_PYTHON:-.venv/bin/python}"
max_jobs="${MAX_JOBS:-2}"
poll_s="${POLL_S:-30}"
stamp="20260615b"
out_base="runs/sidecar_sweep_20260615b"
sweep_log="${out_base}/sweep.log"
wave1_driver="sidecar_sweep"
ref_baseline="runs/sidecar_sweep_20260615/baseline/results.jsonl"   # wave-1 baseline = reference

window="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037"

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
  "qkv_fused|--qkv-projection fused"
  "deepnorm_off|--deepnorm off"
  "relbasis_off|--smolgen-relation-basis off"
  "arc_attn_basic|--arc-attention-bias basic"
  "pos_arc|--input-pos-encoding arc"
  "sq_add|--input-square-embedding add"
  "w_future_0|--w-future 0.0"
  "w_future_005|--w-future 0.05"
)

session_of() { echo "sweep_${stamp}_$1"; }
outdir_of() { echo "${out_base}/$1"; }
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${sweep_log}"; }
is_done() { [[ -s "$(outdir_of "$1")/results.jsonl" ]]; }
session_alive() { tmux has-session -t "$(session_of "$1")" 2>/dev/null; }

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
  echo "Sidecar sweep wave-2 (${stamp}) — window=${window}  out_base=${out_base}"
  echo "Waits for wave-1 driver session '${wave1_driver}' to finish, then runs ${max_jobs} at a time."
  local i=13 name extra
  for spec in "${experiments[@]}"; do
    IFS='|' read -r name extra <<<"${spec}"
    printf "  %2d. %-16s %s\n" "${i}" "${name}" "${extra}"; i=$((i + 1))
  done
}

run_sweep() {
  mkdir -p "${out_base}"
  if tmux has-session -t "${wave1_driver}" 2>/dev/null; then
    log "waiting for wave-1 driver '${wave1_driver}' to finish before starting wave-2..."
    while tmux has-session -t "${wave1_driver}" 2>/dev/null; do sleep 60; done
  fi
  log "=== wave-2 (${stamp}) start (max_jobs=${max_jobs}) ==="
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
    (( ${#pending[@]} == 0 )) && { log "wave-2 all finished"; break; }
    local running=0
    for p in "${pending[@]}"; do IFS='|' read -r pn pe <<<"${p}"; session_alive "${pn}" && running=$((running + 1)); done
    for p in "${pending[@]}"; do
      (( running >= max_jobs )) && break
      IFS='|' read -r pn pe <<<"${p}"
      session_alive "${pn}" && continue
      launch "${pn}" "${pe}"; running=$((running + 1))
    done
    sleep "${poll_s}"
  done
  log "=== wave-2 done; summarizing (ref = wave-1 baseline) ==="
  summarize || true
}

summarize() {
  "${python_bin}" - "${out_base}" "${ref_baseline}" <<'PY'
import json, sys, glob, os
base, ref_path = sys.argv[1], sys.argv[2]
rows = {}
for rp in glob.glob(os.path.join(base, "*", "results.jsonl")):
    name = os.path.basename(os.path.dirname(rp))
    try: rows[name] = [json.loads(l) for l in open(rp) if l.strip()][-1]
    except Exception: pass
base_row = {}
if os.path.exists(ref_path):
    try: base_row = [json.loads(l) for l in open(ref_path) if l.strip()][-1]
    except Exception: pass
keys = ["eval_policy_loss", "eval_wdl_loss", "eval_soft_policy_loss",
        "eval_categorical_loss", "eval_sf_move_loss", "eval_wdl_brier", "eval_wdl_ece"]
hdr = f"{'experiment':18s} " + " ".join(f"{k.replace('eval_',''):>14s}" for k in keys)
print("reference = wave-1 baseline" + ("" if base_row else "  (NOT FOUND)"))
print(hdr); print("-"*len(hdr))
for name in sorted(rows):
    r = rows[name]; cells = []
    for k in keys:
        v = r.get(k)
        if v is None: cells.append(f"{'-':>14s}"); continue
        if base_row.get(k) is not None: cells.append(f"{v:7.4f}({v-base_row[k]:+.4f})")
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
