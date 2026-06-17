#!/usr/bin/env bash
# Categorical value-target blend-mode screen (2026-06-15): compares how the
# 32-bin value head's target is blended — SF-only vs own-search-only vs the
# 3-way (sf+search, mirroring the main WDL head). From-scratch on window-1 v1
# (146-plane; isolates the value-target change from the threat-plane confound),
# vs the wave-1 baseline. Training only, 2 at a time — safe concurrency.
#
# Waits for the combo+arena pipeline (tmux `combo_arena`) to finish so it never
# overlaps the arena's compiled inference.
#
# Usage: scripts/cat_blend_modes_20260615.sh {plan|run}
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo_root="$(cd "${script_dir}/.." && pwd)"; cd "${repo_root}"
py=".venv/bin/python"; max_jobs="${MAX_JOBS:-2}"; poll_s="${POLL_S:-30}"; stamp="20260615f"
out_base="runs/sidecar_cat_blend_modes_20260615"; log="${out_base}/sweep.log"
wait_driver="combo_arena"
win="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037"
ref="runs/sidecar_sweep_20260615/baseline/results.jsonl"

common=( "${py}" scripts/offline_replay_epoch.py --config configs/pbt2_small.yaml --replay-dir "${win}"
  --candidates aurora_mlp_out --batch-size 512 --seed 0 --prefer-recorded-lc0-root
  --matrix-lr-multiplier 20 --matrix-weight-decay 0 --aux-weight-decay 0.0001
  --aurora-pp-iterations 3 --aurora-pp-beta 0.25 --aurora-polar-steps 8
  --aurora-polar-method polar_express --aurora-polar-dtype fp16 --aurora-polar-safety 1.01
  --smolgen-pooling flatten --eval-positions 4096 --eval-steps 8 --report-every-shards 100 )

experiments=(
  "cat_sf07|--config configs/exp_categorical_continuous_0p7.yaml"
  "cat_search07|--config configs/exp_cat_search.yaml"
  "cat_sfsearch|--config configs/exp_cat_sfsearch.yaml"
)

sess(){ echo "catmode_${stamp}_$1"; }; od(){ echo "${out_base}/$1"; }
lg(){ printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${log}"; }
done_(){ [[ -s "$(od "$1")/results.jsonl" ]]; }; alive(){ tmux has-session -t "$(sess "$1")" 2>/dev/null; }
launch(){ local n="$1" e="$2" s o cs; s="$(sess "$n")"; o="$(od "$n")"; mkdir -p "$o"
  local -a x; read -ra x <<<"$e"
  # --config appears twice (common default + experiment); argparse last-wins keeps the experiment's.
  printf -v cs '%q ' "${common[@]}" "${x[@]}" --out-dir "$o"
  tmux new-session -d -s "$s" "cd $(printf '%q' "$repo_root") && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${cs} 2>&1 | tee -a $(printf '%q' "$o/run.log")"; lg "LAUNCH $n"; }

if [[ "${1:-plan}" == plan ]]; then echo "blend modes (window-1 v1, vs wave-1 baseline): cat_sf07, cat_search07, cat_sfsearch"; exit 0; fi
mkdir -p "$out_base"
if tmux has-session -t "$wait_driver" 2>/dev/null; then lg "waiting for '$wait_driver' to finish..."; while tmux has-session -t "$wait_driver" 2>/dev/null; do sleep 60; done; fi
lg "=== cat blend-mode screen start ==="
pend=(); for sp in "${experiments[@]}"; do IFS='|' read -r n e <<<"$sp"; done_ "$n" || pend+=("$sp"); done
while ((${#pend[@]})); do
  a=(); for sp in "${pend[@]}"; do IFS='|' read -r n e <<<"$sp"; done_ "$n" || a+=("$sp"); done; pend=("${a[@]}"); ((${#pend[@]}==0)) && break
  r=0; for sp in "${pend[@]}"; do IFS='|' read -r n e <<<"$sp"; alive "$n" && r=$((r+1)); done
  for sp in "${pend[@]}"; do ((r>=max_jobs)) && break; IFS='|' read -r n e <<<"$sp"; alive "$n" && continue; launch "$n" "$e"; r=$((r+1)); done
  sleep "$poll_s"
done
lg "=== blend-mode screen done; summary (ref=wave-1 baseline) ==="
"${py}" - "$out_base" "$ref" <<'PY' 2>&1 | tee -a "$log"
import json,sys,glob,os
base,refp=sys.argv[1],sys.argv[2]
def last(p):
  try: return [json.loads(l) for l in open(p) if l.strip()][-1]
  except: return {}
b=last(refp); rows={os.path.basename(os.path.dirname(p)):last(p) for p in glob.glob(os.path.join(base,"*","results.jsonl"))}
ks=["eval_policy_loss","eval_wdl_loss","eval_categorical_loss","eval_wdl_brier","eval_wdl_ece"]
print(f"baseline(wave1): " + " ".join(f"{k.replace('eval_','')}={b.get(k)}" for k in ks))
for n in sorted(rows):
  r=rows[n]; print(f"{n:14s} " + " ".join(f"{k.replace('eval_','')}={r.get(k):.4f}(Δ{r.get(k)-b[k]:+.4f})" if r.get(k) is not None and b.get(k) is not None else f"{k}=-" for k in ks))
PY
lg "=== done ==="
