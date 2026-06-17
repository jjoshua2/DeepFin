#!/usr/bin/env bash
# Gumbel-config self-match: DeepFin iter520 (config A) vs DeepFin iter520
# (reference config B = current defaults), at EQUAL FIXED SIMS. Fixed-node mode
# means no clock => no time-forfeit / compile-on-clock issues, and equal sims
# isolates the gumbel settings. Paired openings + syzygy adjudication keep
# variance low. Use it to rank gumbel knobs (c_scale, c_visit, topk, c_puct, fpu)
# for STRENGTH — A scoring >0.5 means config A is stronger than the default.
#
# Usage: scripts/gumbel_selfmatch.sh {plan|run}
# Env (config A = candidate; B defaults to the production gumbel config):
#   SIMS=8000  GAMES=40
#   CSA=0.4 CVA=10 TOPKA=32 CPA=2.5 FPUA=1.2     # candidate A
#   CSB=0.1 CVB=50 TOPKB=32 CPB=2.5 FPUB=1.2     # reference B
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo_root="$(cd "${script_dir}/.." && pwd)"; cd "${repo_root}"
py=".venv/bin/python"
ckpt="data/best_regret_checkpoints/regret_0.0075_step108246_iter520/trainer.pt"
openings="runs/matches/openings_2move_balanced_100.epd"
syzygy="data/syzygy_3-4-5"
ts="$(date +%Y%m%d_%H%M%S)"

sims="${SIMS:-8000}"; games="${GAMES:-40}"
csa="${CSA:-0.4}"; cva="${CVA:-10}"; topka="${TOPKA:-32}"; cpa="${CPA:-2.5}"; fpua="${FPUA:-1.2}"
csb="${CSB:-0.1}"; cvb="${CVB:-50}"; topkb="${TOPKB:-32}"; cpb="${CPB:-2.5}"; fpub="${FPUB:-1.2}"
la="A_cs${csa}cv${cva}tk${topka}"; lb="B_cs${csb}cv${cvb}tk${topkb}"
out_base="runs/matches/gumbel_selfmatch_${ts}_${la}_vs_${lb}"
log="${out_base}.log"; pgn="${out_base}.pgn"

mk_engine() { # cs cv topk cpuct fpu
  echo "${py} -u -m chess_anti_engine.uci --checkpoint ${ckpt} --device cuda \
--walkers 1 --chunk-sims 512 --max-batch 1024 --compile-mode max-autotune \
--c-scale $1 --c-visit $2 --topk $3 --c-puct $4 --fpu-reduction $5 \
--compile-cache-dir runs/pbt2_small/server/worker_cache"
}

gpu_busy() {
  for s in cheese_h1 cheese_h2 cheese_tc cheese_tc2 cheese_tc3 combo_threat_cat arena_rerun; do
    tmux has-session -t "$s" 2>/dev/null && { echo "session '$s' alive"; return 0; }
  done
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ -n "${used}" && "${used}" -gt 4000 ]]; then echo "VRAM ${used} MiB in use"; return 0; fi
  return 1
}

cmd="${1:-plan}"
if [[ "$cmd" == plan ]]; then
  echo "Gumbel self-match: ${la}  vs  ${lb}"
  echo "  sims=${sims} (fixed, no clock)  games=${games}  syzygy=${syzygy} (both + adjudicate<=5)"
  echo "  A: c_scale=${csa} c_visit=${cva} topk=${topka} c_puct=${cpa} fpu=${fpua}"
  echo "  B: c_scale=${csb} c_visit=${cvb} topk=${topkb} c_puct=${cpb} fpu=${fpub}"
  echo "  out: ${log}"
  exit 0
fi
[[ "$cmd" == run ]] || { echo "usage: $0 {plan|run}" >&2; exit 2; }
if reason="$(gpu_busy)"; then echo "REFUSING: GPU busy (${reason})." >&2; exit 3; fi

mkdir -p "$(dirname "${out_base}")"
echo "[selfmatch] ${la} vs ${lb} — ${games} games @ ${sims} sims — $(date)" | tee "${log}"
PYTHONPATH=. "${py}" scripts/match_vs_uci.py \
  --engine-a "$(mk_engine "${csa}" "${cva}" "${topka}" "${cpa}" "${fpua}")" --label-a "${la}" \
  --engine-b "$(mk_engine "${csb}" "${cvb}" "${topkb}" "${cpb}" "${fpub}")" --label-b "${lb}" \
  --option-a "SyzygyPath=${syzygy}" --option-b "SyzygyPath=${syzygy}" \
  --nodes-a "${sims}" --nodes-b "${sims}" \
  --warmup-nodes-a 3000 --warmup-nodes-b 3000 \
  --openings "${openings}" --max-plies 300 \
  --syzygy-adjudicate-path "${syzygy}" --syzygy-adjudicate-max-pieces 5 \
  --pgn-out "${pgn}" --games "${games}" \
  2>&1 | tee -a "${log}"
echo "[selfmatch] done — A=${la} score is A's; >0.5 means A beats the default" | tee -a "${log}"
