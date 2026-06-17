#!/usr/bin/env bash
# Recovery + combo exploration (2026-06-15), crash-safe:
#  Phase 1: 3 offline screens (training only, 2 at a time) on window-1 v2 vs the
#           wave-1 baseline — combo (all-3), v2+soft, v2+cat.
#  Phase 2: the threat-planes arena, run SOLO (no training overlap — the earlier
#           crash was arena+training contending for the CUDA context).
#
# Usage: scripts/combo_screens_then_arena_20260615.sh {plan|run}
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo_root="$(cd "${script_dir}/.." && pwd)"; cd "${repo_root}"
py=".venv/bin/python"; max_jobs="${MAX_JOBS:-2}"; poll_s="${POLL_S:-30}"; stamp="20260615e"
out_base="runs/sidecar_combo_screens_20260615"; log="${out_base}/sweep.log"
v2="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037_v2threats"
ref="runs/sidecar_sweep_20260615/baseline/results.jsonl"          # wave-1 baseline (window-1)
threat_ref="runs/sidecar_sweep_20260615/threat_planes/results.jsonl"

common=( "${py}" scripts/offline_replay_epoch.py --replay-dir "${v2}" --candidates aurora_mlp_out
  --batch-size 512 --seed 0 --prefer-recorded-lc0-root
  --matrix-lr-multiplier 20 --matrix-weight-decay 0 --aux-weight-decay 0.0001
  --aurora-pp-iterations 3 --aurora-pp-beta 0.25 --aurora-polar-steps 8
  --aurora-polar-method polar_express --aurora-polar-dtype fp16 --aurora-polar-safety 1.01
  --smolgen-pooling flatten --eval-positions 4096 --eval-steps 8 --report-every-shards 100 )

experiments=(
  "combo|--config configs/exp_stacked_gainers.yaml"
  "v2_soft|--config configs/exp_threat_soft.yaml"
  "v2_cat|--config configs/exp_threat_cat.yaml"
)

sess(){ echo "screen_${stamp}_$1"; }
od(){ echo "${out_base}/$1"; }
lg(){ printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${log}"; }
done_(){ [[ -s "$(od "$1")/results.jsonl" ]]; }
alive(){ tmux has-session -t "$(sess "$1")" 2>/dev/null; }
launch(){ local n="$1" e="$2" s o; s="$(sess "$n")"; o="$(od "$n")"; mkdir -p "$o"
  local -a x; read -ra x <<<"$e"; local -a c=("${common[@]}" "${x[@]}" --out-dir "$o"); local cs; printf -v cs '%q ' "${c[@]}"
  tmux new-session -d -s "$s" "cd $(printf '%q' "$repo_root") && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${cs} 2>&1 | tee -a $(printf '%q' "$o/run.log")"; lg "LAUNCH $n"; }

if [[ "${1:-plan}" == plan ]]; then echo "Phase1 screens (window-1 v2, 2 at a time): combo, v2_soft, v2_cat"; echo "Phase2 arena (solo) via arena_threat_2026_06_15.sh"; exit 0; fi

mkdir -p "$out_base"; lg "=== combo screens start ==="
# Phase 1
local_pending=(); for sp in "${experiments[@]}"; do IFS='|' read -r n e <<<"$sp"; done_ "$n" && lg "SKIP $n" || local_pending+=("$sp"); done
while ((${#local_pending[@]})); do
  a=(); for sp in "${local_pending[@]}"; do IFS='|' read -r n e <<<"$sp"; done_ "$n" || a+=("$sp"); done; local_pending=("${a[@]}")
  ((${#local_pending[@]}==0)) && break
  r=0; for sp in "${local_pending[@]}"; do IFS='|' read -r n e <<<"$sp"; alive "$n" && r=$((r+1)); done
  for sp in "${local_pending[@]}"; do ((r>=max_jobs)) && break; IFS='|' read -r n e <<<"$sp"; alive "$n" && continue; launch "$n" "$e"; r=$((r+1)); done
  sleep "$poll_s"
done
lg "=== screens done; summary (ref=wave-1 baseline, threat-alone for additivity) ==="
"${py}" - "$out_base" "$ref" "$threat_ref" <<'PY' 2>&1 | tee -a "$log"
import json,sys,glob,os
base,refp,thp=sys.argv[1],sys.argv[2],sys.argv[3]
def last(p):
  try: return [json.loads(l) for l in open(p) if l.strip()][-1]
  except: return {}
b=last(refp); th=last(thp)
rows={os.path.basename(os.path.dirname(p)):last(p) for p in glob.glob(os.path.join(base,"*","results.jsonl"))}
ks=["eval_policy_loss","eval_wdl_loss","eval_categorical_loss","eval_wdl_ece"]
print(f"baseline(wave1): policy {b.get('eval_policy_loss')} wdl {b.get('eval_wdl_loss')}")
print(f"threat_planes  : policy {th.get('eval_policy_loss')} wdl {th.get('eval_wdl_loss')}  (additivity ref)")
for n in sorted(rows):
  r=rows[n]; print(f"{n:10s} " + " ".join(f"{k.replace('eval_','')}={r.get(k):.4f}(Δb {r.get(k)-b[k]:+.4f})" if r.get(k) is not None and b.get(k) is not None else f"{k}=-" for k in ks))
PY
# Phase 2: arena SOLO (checkpoints already exist; arena_threat skips training)
lg "=== starting arena SOLO ==="
bash scripts/arena_threat_2026_06_15.sh run 2>&1 | tee -a "$log"
lg "=== all done ==="
