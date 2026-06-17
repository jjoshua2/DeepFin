#!/usr/bin/env bash
# Combine the winners: threats v2 + 3-way categorical (=WDL floor blend), +/- soft.
# Tests whether the promising pieces STACK (vs threat-alone). From-scratch on
# window-1 v2, 2 at a time. Waits for the threat arena (tmux arena_rerun) to
# finish first so it never overlaps the arena's compiled inference.
#
# Usage: scripts/combo_threat_cat_20260615.sh {plan|run}
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo_root="$(cd "${script_dir}/.." && pwd)"; cd "${repo_root}"
py=".venv/bin/python"; max_jobs="${MAX_JOBS:-2}"; poll_s="${POLL_S:-30}"; stamp="20260615g"
out_base="runs/sidecar_combo_threat_cat_20260615"; log="${out_base}/sweep.log"
wait_driver="arena_rerun"
v2="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037_v2threats"
ref="runs/sidecar_sweep_20260615/baseline/results.jsonl"            # wave-1 baseline
threat_ref="runs/sidecar_sweep_20260615/threat_planes/results.jsonl"  # threat-alone (additivity ref)

common=( "${py}" scripts/offline_replay_epoch.py --replay-dir "${v2}" --candidates aurora_mlp_out
  --batch-size 512 --seed 0 --prefer-recorded-lc0-root
  --matrix-lr-multiplier 20 --matrix-weight-decay 0 --aux-weight-decay 0.0001
  --aurora-pp-iterations 3 --aurora-pp-beta 0.25 --aurora-polar-steps 8
  --aurora-polar-method polar_express --aurora-polar-dtype fp16 --aurora-polar-safety 1.01
  --smolgen-pooling flatten --eval-positions 4096 --eval-steps 8 --report-every-shards 100 )

experiments=(
  "threat_3waycat|--config configs/exp_threat_3waycat.yaml"
  "threat_3waycat_soft|--config configs/exp_threat_3waycat_soft.yaml"
)

sess(){ echo "combo_${stamp}_$1"; }; od(){ echo "${out_base}/$1"; }
lg(){ printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${log}"; }
done_(){ [[ -s "$(od "$1")/results.jsonl" ]]; }; alive(){ tmux has-session -t "$(sess "$1")" 2>/dev/null; }
launch(){ local n="$1" e="$2" s o cs; s="$(sess "$n")"; o="$(od "$n")"; mkdir -p "$o"
  local -a x; read -ra x <<<"$e"; printf -v cs '%q ' "${common[@]}" "${x[@]}" --out-dir "$o"
  tmux new-session -d -s "$s" "cd $(printf '%q' "$repo_root") && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${cs} 2>&1 | tee -a $(printf '%q' "$o/run.log")"; lg "LAUNCH $n"; }

if [[ "${1:-plan}" == plan ]]; then echo "combos (window-1 v2): threat_3waycat, threat_3waycat_soft"; exit 0; fi
mkdir -p "$out_base"
if tmux has-session -t "$wait_driver" 2>/dev/null; then lg "waiting for arena ('$wait_driver') to finish before combos..."; while tmux has-session -t "$wait_driver" 2>/dev/null; do sleep 60; done; fi
lg "=== combo screens start ==="
pend=(); for sp in "${experiments[@]}"; do IFS='|' read -r n e <<<"$sp"; done_ "$n" || pend+=("$sp"); done
while ((${#pend[@]})); do
  a=(); for sp in "${pend[@]}"; do IFS='|' read -r n e <<<"$sp"; done_ "$n" || a+=("$sp"); done; pend=("${a[@]}"); ((${#pend[@]}==0)) && break
  r=0; for sp in "${pend[@]}"; do IFS='|' read -r n e <<<"$sp"; alive "$n" && r=$((r+1)); done
  for sp in "${pend[@]}"; do ((r>=max_jobs)) && break; IFS='|' read -r n e <<<"$sp"; alive "$n" && continue; launch "$n" "$e"; r=$((r+1)); done
  sleep "$poll_s"
done
lg "=== combos done; summary (vs baseline + threat-alone) ==="
"${py}" - "$out_base" "$ref" "$threat_ref" <<'PY' 2>&1 | tee -a "$log"
import json,sys,glob,os
base,refp,thp=sys.argv[1],sys.argv[2],sys.argv[3]
def last(p):
  try: return [json.loads(l) for l in open(p) if l.strip()][-1]
  except: return {}
b=last(refp); th=last(thp)
print(f"baseline: policy {b.get('eval_policy_loss'):.4f} wdl {b.get('eval_wdl_loss'):.4f} ece {b.get('eval_wdl_ece'):.4f}")
print(f"threat-alone: policy {th.get('eval_policy_loss'):.4f} wdl {th.get('eval_wdl_loss'):.4f} ece {th.get('eval_wdl_ece'):.4f}  (additivity ref)")
for p in sorted(glob.glob(os.path.join(base,"*","results.jsonl"))):
  n=os.path.basename(os.path.dirname(p)); r=last(p)
  if r.get("eval_policy_loss") is None: continue
  print(f"{n:22s} policy {r['eval_policy_loss']:.4f} (Δb {r['eval_policy_loss']-b['eval_policy_loss']:+.4f}, Δthreat {r['eval_policy_loss']-th['eval_policy_loss']:+.4f}) "
        f"wdl {r['eval_wdl_loss']:.4f} (Δb {r['eval_wdl_loss']-b['eval_wdl_loss']:+.4f}) ece {r['eval_wdl_ece']:.4f} (Δb {r['eval_wdl_ece']-b['eval_wdl_ece']:+.4f})")
PY
lg "=== done ==="
