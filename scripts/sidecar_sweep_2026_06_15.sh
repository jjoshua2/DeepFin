#!/usr/bin/env bash
# Overnight sidecar screen of merged-PR bets (2026-06-15).
#
# Each experiment = one from-scratch offline_replay_epoch over the newest frozen
# replay window (seed 0, matched budget), changing exactly one thing vs the
# baseline. Compare results.jsonl eval_* losses across runs; the baseline run is
# the reference for every delta.
#
# Runs MAX_JOBS (default 2) jobs at once, each in its own tmux session, polling
# until the queue drains. v2_threats experiments are gated on the shard
# conversion finishing (their --replay-dir must exist with enough shards).
#
# Usage:
#   scripts/sidecar_sweep_2026_06_15.sh plan      # print the queue, launch nothing
#   scripts/sidecar_sweep_2026_06_15.sh run       # schedule the sweep
#   MAX_JOBS=2 POLL_S=30 scripts/sidecar_sweep_2026_06_15.sh run
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${SIDECAR_PYTHON:-.venv/bin/python}"
max_jobs="${MAX_JOBS:-2}"
poll_s="${POLL_S:-30}"
stamp="20260615"
out_base="runs/sidecar_sweep_${stamp}"
sweep_log="${out_base}/sweep.log"

window="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037"
window_v2="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037_v2threats"
v2_min_shards=1400   # window has 1463; require near-complete conversion before launching v2 jobs

ffn_growth="1.5,1.5,1.5,1.5,1.5833333333333333,1.75,1.9166666666666667,1.5,1.9166666666666667,1.75,1.8333333333333333"

# Shared flags: from-scratch, production optimizer scope + arch baseline, matched
# eval. --config defaults to pbt2_small; target/v2 experiments override it (argparse
# last-wins). --replay-dir defaults to the v1 window; v2 experiments override it.
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

# name | extra args (override common via last-wins). Baseline first (reference).
experiments=(
  "baseline|"
  "soft_ablation|--config configs/exp_soft_policy_ablation.yaml"
  "soft_divergent|--config configs/exp_soft_policy_divergent_only.yaml"
  "smolgen_mean|--smolgen-pooling mean"
  "qk_rmsnorm|--use-qk-rmsnorm on"
  "ma_gate|--input-square-embedding ma_gate"
  "ffn_growth|--ffn-mult-by-layer ${ffn_growth}"
  "cat_blend_0p5|--config configs/exp_categorical_continuous.yaml"
  "cat_blend_0p3|--config configs/exp_categorical_continuous_0p3.yaml"
  "cat_blend_0p7|--config configs/exp_categorical_continuous_0p7.yaml"
  "threat_planes|--config configs/exp_threat_planes.yaml --replay-dir ${window_v2}"
  "dynamic_relations|--config configs/exp_dynamic_relations.yaml --replay-dir ${window_v2}"
)

session_of() { echo "sweep_${stamp}_$1"; }
outdir_of() { echo "${out_base}/$1"; }

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${sweep_log}"; }

is_done() { [[ -s "$(outdir_of "$1")/results.jsonl" ]]; }
session_alive() { tmux has-session -t "$(session_of "$1")" 2>/dev/null; }

v2_shards_ready() {
  [[ -d "${window_v2}" ]] || return 1
  local n; n="$(ls "${window_v2}" 2>/dev/null | grep -c '\.zarr$' || true)"
  [[ "${n}" -ge "${v2_min_shards}" ]]
}

conversion_alive() { pgrep -af "convert_shards_v2_threats" >/dev/null 2>&1; }

prereq_ready() { # $1=extra-args string
  case "$1" in
    *"${window_v2}"*) v2_shards_ready ;;
    *) return 0 ;;
  esac
}

launch() { # $1=name  $2=extra-args
  local name="$1" extra_str="$2" session outdir
  session="$(session_of "${name}")"
  outdir="$(outdir_of "${name}")"
  mkdir -p "${outdir}"
  local -a extra; read -ra extra <<<"${extra_str}"
  local -a cmd=("${common_args[@]}" "${extra[@]}" --out-dir "${outdir}")
  local cmd_str; printf -v cmd_str '%q ' "${cmd[@]}"
  tmux new-session -d -s "${session}" \
    "cd $(printf '%q' "${repo_root}") && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${cmd_str} 2>&1 | tee -a $(printf '%q' "${outdir}/run.log")"
  log "LAUNCH ${name}  (session=${session})  extra=[${extra_str}]"
}

print_plan() {
  echo "Sidecar sweep ${stamp} — window=${window}"
  echo "MAX_JOBS=${max_jobs}  out_base=${out_base}"
  local i=1 name extra
  for spec in "${experiments[@]}"; do
    IFS='|' read -r name extra <<<"${spec}"
    printf "  %2d. %-18s %s\n" "${i}" "${name}" "${extra:-（baseline, common args only）}"
    i=$((i + 1))
  done
}

run_sweep() {
  mkdir -p "${out_base}"
  log "=== sidecar sweep ${stamp} start (max_jobs=${max_jobs}) ==="
  # pending = every experiment not yet finished (running jobs stay here until
  # their results.jsonl appears, so the running-count stays accurate).
  local -a pending=()
  local spec name extra
  for spec in "${experiments[@]}"; do
    IFS='|' read -r name extra <<<"${spec}"
    if is_done "${name}"; then log "SKIP ${name} (results.jsonl already present)"; else pending+=("${spec}"); fi
  done

  while true; do
    # Reap finished experiments.
    local -a alive=(); local p pn pe
    for p in "${pending[@]}"; do
      IFS='|' read -r pn pe <<<"${p}"
      is_done "${pn}" || alive+=("${p}")
    done
    pending=("${alive[@]}")
    (( ${#pending[@]} == 0 )) && { log "all experiments finished"; break; }

    # Count running, then fill free slots from prereq-ready, not-yet-started jobs.
    local running=0 launched_any=0
    for p in "${pending[@]}"; do
      IFS='|' read -r pn pe <<<"${p}"
      session_alive "${pn}" && running=$((running + 1))
    done
    for p in "${pending[@]}"; do
      (( running >= max_jobs )) && break
      IFS='|' read -r pn pe <<<"${p}"
      session_alive "${pn}" && continue       # already running
      prereq_ready "${pe}" || continue          # e.g. v2 conversion not done — skip, retry later
      launch "${pn}" "${pe}"; running=$((running + 1)); launched_any=1
    done

    # Deadlock guard: nothing running, nothing launchable, only blocked jobs left.
    if (( running == 0 )) && (( launched_any == 0 )); then
      if ! conversion_alive && ! v2_shards_ready; then
        log "ABORT remaining (v2 conversion not complete and not running): ${pending[*]}"
        break
      fi
      log "waiting on v2 conversion; pending=${#pending[@]}"
    fi

    sleep "${poll_s}"
  done

  log "=== sweep ${stamp} done; summarizing ==="
  summarize || true
}

summarize() {
  "${python_bin}" - "${out_base}" <<'PY'
import json, sys, glob, os
base = sys.argv[1]
rows = {}
for rp in glob.glob(os.path.join(base, "*", "results.jsonl")):
    name = os.path.basename(os.path.dirname(rp))
    try:
        last = [json.loads(l) for l in open(rp) if l.strip()][-1]
    except Exception:
        continue
    rows[name] = last
if "baseline" not in rows:
    print("(no baseline yet; deltas unavailable)")
keys = ["eval_policy_loss", "eval_wdl_loss", "eval_soft_policy_loss",
        "eval_categorical_loss", "eval_sf_move_loss", "eval_wdl_brier", "eval_wdl_ece"]
base_row = rows.get("baseline", {})
hdr = f"{'experiment':20s} " + " ".join(f"{k.replace('eval_',''):>14s}" for k in keys)
print(hdr); print("-"*len(hdr))
for name in sorted(rows):
    r = rows[name]
    cells = []
    for k in keys:
        v = r.get(k)
        if v is None: cells.append(f"{'-':>14s}"); continue
        if name != "baseline" and k in base_row and base_row.get(k) is not None:
            cells.append(f"{v:7.4f}({v-base_row[k]:+.4f})")
        else:
            cells.append(f"{v:14.4f}")
    print(f"{name:20s} " + " ".join(cells))
PY
}

cmd="${1:-plan}"
case "${cmd}" in
  plan) print_plan ;;
  run) run_sweep ;;
  summary) summarize ;;
  *) echo "usage: $0 {plan|run|summary}" >&2; exit 2 ;;
esac
