#!/usr/bin/env bash
# DeepFin (strongest live-prod net, iter520) vs Cheese 3.2.1 UCI engine.
# Mirrors the June-1 rofChade match setup (same engine-A flags, 60+1s, 200 games,
# 100 balanced 2-move openings, 1 thread for the opponent). rofChade 3.1 (~3380
# CCRL) shut us out 0/200; Cheese 3.2.1 self-rates ~3000 (UCI_Elo max 3000), so
# this is the "can we actually win one" test.
#
# GPU-bound (our net runs compiled on CUDA): refuses to start while the combos /
# arena / a live trainer hold the GPU, so it can't crash into them. Run it once
# the GPU is free.
#
# Usage:
#   scripts/match_cheese_20260615.sh plan        # show the exact command (default)
#   scripts/match_cheese_20260615.sh run         # run the match (guards GPU first)
# Env overrides:
#   GAMES=200  TC_BASE_MS=60000  TC_INC_MS=1000  CHEESE_THREADS=1
#   CHEESE_ELO=          # empty = full strength; set e.g. 2500 to cap via UCI_Elo
#   NODES_A= NODES_B=    # set BOTH for fixed-node mode (our MCTS sims vs cheese
#                        # alpha-beta nodes), e.g. NODES_A=1000 NODES_B=1000000.
#                        # Overrides the clock. Sweep NODES_B to find iso-strength.
#   SYZYGY=path          # in-search TB (both engines) + adjudication; default
#                        # data/syzygy_3-4-5. Set SYZYGY= to disable.
#   WARMUP_NODES_A=3000  # off-clock warmup that pays torch.compile BEFORE the
#                        # timed games (so we never compile on the clock). 0=off.
#   CLOCK_GRACE_MS=100   # clock mode: tolerate this per-move overrun (carried as
#                        # debt to the next move) instead of forfeiting on jitter.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo_root="$(cd "${script_dir}/.." && pwd)"; cd "${repo_root}"
py=".venv/bin/python"

ckpt="data/best_regret_checkpoints/regret_0.0075_step108246_iter520/trainer.pt"
cheese="/home/josh/local_engines/cheese/cheese-321-linux-pext"
openings="runs/matches/openings_2move_balanced_100.epd"
ts="$(date +%Y%m%d_%H%M%S)"
out_base="runs/matches/deepfin_iter520_vs_cheese321_${ts}"
log="${out_base}.log"; pgn="${out_base}.pgn"; csv="${out_base}_moves.csv"

games="${GAMES:-200}"
tc_base="${TC_BASE_MS:-60000}"; tc_inc="${TC_INC_MS:-1000}"
threads="${CHEESE_THREADS:-1}"
cheese_elo="${CHEESE_ELO:-}"
nodes_a="${NODES_A:-}"; nodes_b="${NODES_B:-}"   # set BOTH for fixed-node mode
# Gumbel search knobs for OUR engine (exposed 2026-06-16). Defaults = current
# match config. C_SCALE is the value-transform scale (0.1 trusts policy, 1.0
# trusts search Q); TOPK_A 16 matches selfplay training, 32 is the match default.
gscale="${C_SCALE:-0.025}"; gtopk="${TOPK_A:-32}"; gcpuct="${C_PUCT:-2.5}"; gfpu="${FPU:-1.2}"
# c_visit = the depth-ramp base in q_scale = c_scale*(c_visit + max_visit): lower
# => search-Q dominates the prior sooner as the tree deepens (LC0 cpuct_base analog).
gcvisit="${C_VISIT:-50.0}"

# Budget: fixed-node (our MCTS sims vs cheese alpha-beta nodes) if NODES_A+NODES_B
# set, else a game clock. cheese gets --enforce-nodes-b as a hard backstop; our
# side honors `go nodes` natively (nodes = total MCTS sims).
if [[ -n "${nodes_a}" && -n "${nodes_b}" ]]; then
  budget=( --nodes-a "${nodes_a}" --nodes-b "${nodes_b}" --enforce-nodes-b )
  budget_label="nodes ${nodes_a}(us) vs ${nodes_b}(cheese)"
  # No clock → chunk-8192/mb-4096 is safe and ~20% faster (avg_batch ~408, the
  # single-game GPU-forward ceiling; see single_game_nps_ceiling memory).
  chunk="${CHUNK_SIMS:-8192}"; mbatch="${MAX_BATCH:-4096}"
else
  budget=( --clock-base-ms "${tc_base}" --clock-inc-ms "${tc_inc}" )
  budget_label="${tc_base}+${tc_inc}ms"
  # Time control: keep chunk 512 so stop latency stays <~35ms; chunk-8192's
  # ~0.46s/chunk would overshoot a ~1s/move budget past the grace and forfeit.
  chunk="${CHUNK_SIMS:-512}"; mbatch="${MAX_BATCH:-1024}"
fi

# Engine A: our net (compiled, 1 walker — walkers>1 is 20x slower, GIL-bound).
engine_a="${py} -u -m chess_anti_engine.uci --checkpoint ${ckpt} --device cuda \
--walkers 1 --chunk-sims ${chunk} --max-batch ${mbatch} --compile-mode max-autotune \
--c-scale ${gscale} --c-visit ${gcvisit} --topk ${gtopk} --c-puct ${gcpuct} --fpu-reduction ${gfpu} \
--compile-cache-dir runs/pbt2_small/server/worker_cache"

# Engine B options: 1 thread (fair single-core, like rofChade), optional Elo cap.
opt_b=( --option-b "Threads=${threads}" )
if [[ -n "${cheese_elo}" ]]; then
  opt_b+=( --option-b "UCI_LimitStrength=true" --option-b "UCI_Elo=${cheese_elo}" )
fi

# Syzygy (default ON): in-search TB on BOTH engines (fair) + harness TB
# adjudication of covered positions — matches what the arena did. Both DeepFin
# and Cheese expose SyzygyPath. Set SYZYGY= to disable. data/syzygy_3-4-5 has
# full WDL up to 5 men, so adjudicate <=5.
syzygy="${SYZYGY-data/syzygy_3-4-5}"
opt_a=(); adj=()
if [[ -n "${syzygy}" ]]; then
  opt_a+=( --option-a "SyzygyPath=${syzygy}" )
  opt_b+=( --option-b "SyzygyPath=${syzygy}" )
  adj=( --syzygy-adjudicate-path "${syzygy}" --syzygy-adjudicate-max-pieces 5 )
fi

# Warmup (default ON) runs an off-clock search that pays the one-time torch.compile
# (max-autotune) BEFORE the timed games — so we never compile on the game clock
# (which would hand game 1 to the opponent). The harness gives it a generous 300s
# deadline (_WARMUP_TIMEOUT_S), so a slow cold compile won't cancel/crash it.
# Set WARMUP_NODES_A=0 to disable. CLOCK_GRACE_MS tolerates per-move GPU/IPC jitter
# (overrun carries as debt to the next move) instead of forfeiting on a blip.
warmup=(); wnodes="${WARMUP_NODES_A:-3000}"
[[ -n "${wnodes}" && "${wnodes}" != 0 ]] && warmup=( --warmup-nodes-a "${wnodes}" )
grace=( --clock-grace-ms "${CLOCK_GRACE_MS:-100}" )

gpu_busy() {  # true if any GPU-holding session is alive or VRAM is heavily used
  for s in combo_threat_cat arena_rerun arena_train_baseline arena_train_threat; do
    tmux has-session -t "$s" 2>/dev/null && { echo "session '$s' alive"; return 0; }
  done
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ -n "${used}" && "${used}" -gt 4000 ]]; then echo "VRAM ${used} MiB in use"; return 0; fi
  return 1
}

cmd="${1:-plan}"
[[ -f "${ckpt}" ]]   || { echo "ERROR: net checkpoint missing: ${ckpt}" >&2; exit 1; }
[[ -x "${cheese}" ]] || { echo "ERROR: cheese binary missing/not exec: ${cheese}" >&2; exit 1; }

if [[ "${cmd}" == plan ]]; then
  echo "DeepFin iter520  vs  Cheese 3.2.1 (${cheese##*/})"
  echo "  games=${games}  budget=${budget_label}  cheese_threads=${threads}  cheese_elo=${cheese_elo:-full}"
  echo "  syzygy=${syzygy:-OFF} (in-search both sides + adjudicate<=5)  warmup_a=${WARMUP_NODES_A:-3000}(off-clock)  clock_grace_ms=${CLOCK_GRACE_MS:-100}"
  echo "  gumbel(ours): c_scale=${gscale}  c_visit=${gcvisit}  topk=${gtopk}  c_puct=${gcpuct}  fpu=${gfpu}"
  echo "  openings=${openings}"
  echo "  out: ${log} / ${pgn}"
  echo
  echo "engine-a: ${engine_a}"
  echo "engine-b: ${cheese} ${opt_b[*]}"
  exit 0
fi
[[ "${cmd}" == run ]] || { echo "usage: $0 {plan|run}" >&2; exit 2; }

if reason="$(gpu_busy)"; then
  echo "REFUSING to start: GPU busy (${reason}). Wait for it to free, then re-run." >&2
  exit 3
fi

mkdir -p "$(dirname "${out_base}")"
echo "[match] DeepFin iter520 vs Cheese 3.2.1 — ${games} games, ${budget_label} — $(date)" | tee "${log}"
PYTHONPATH=. "${py}" scripts/match_vs_uci.py \
  --engine-a "${engine_a}" --label-a "DeepFin-iter520" \
  --engine-b "${cheese}"   --label-b "Cheese-3.2.1" \
  "${opt_a[@]}" "${opt_b[@]}" "${adj[@]}" "${warmup[@]}" "${grace[@]}" \
  --games "${games}" \
  "${budget[@]}" \
  --openings "${openings}" --max-plies 300 \
  --pgn-out "${pgn}" --move-log-out "${csv}" \
  2>&1 | tee -a "${log}"
echo "[match] done — see ${log}" | tee -a "${log}"
