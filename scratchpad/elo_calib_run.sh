#!/usr/bin/env bash
# Elo calibration of the #438 control net -- prereg 3174ba574. Queued behind the
# dose ladder (waits for its completion line). Final-only reads, fixed blocks.
set -uo pipefail
cd /home/josh/projects/chess
export PYTHONPATH=.
LOG=scratchpad/elo_calib.log
SYZ=/home/josh/projects/chess/data/syzygy_3-4-5:/home/josh/projects/chess/data/syzygy_6
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

say "waiting for dose ladder completion"
while ! grep -q "ladder complete" scratchpad/dose_ladder.log 2>/dev/null; do sleep 300; done
say "ladder done — match A (control-LAST vs iter-595, 400 games)"

python3 scripts/arena_standard.py \
  --candidate runs/lc0_control_20260820/checkpoint.pt \
  --reference data/salvage/pre_lc0_control_20260819/seeds/slot_000/trainer.pt \
  --mode matched_sims --sims 100 --games 400 --no-rolling --seed 42 \
  --search-shape play --syzygy "$SYZ" \
  --label lc0ctl_vs_iter595 \
  --out scratchpad/elo_calib/A_vs_iter595.json \
  --pgn-out scratchpad/elo_calib/A.pgn >> scratchpad/elo_calib/A.log 2>&1
say "match A rc=$?"

say "match B (control LAST vs MID, 400 games)"
python3 scripts/arena_standard.py \
  --candidate runs/lc0_control_20260820/checkpoint.pt \
  --reference runs/lc0_control_20260820/checkpoint_mid.pt \
  --mode matched_sims --sims 100 --games 400 --no-rolling --seed 42 \
  --search-shape play --syzygy "$SYZ" \
  --label lc0ctl_last_vs_mid \
  --out scratchpad/elo_calib/B_last_vs_mid.json \
  --pgn-out scratchpad/elo_calib/B.pgn >> scratchpad/elo_calib/B.log 2>&1
say "match B rc=$?"

say "match C (control-LAST vs FULL-STRENGTH Cheese, 60s+1s, FULL 60-game block)"
python3 scripts/match_vs_uci.py \
  --engine-a "python3 -m chess_anti_engine.uci --checkpoint runs/lc0_control_20260820/checkpoint.pt --device cuda --walkers 1 --chunk-sims 512 --max-batch 1024 --compile-mode max-autotune" \
  --engine-b /home/josh/local_engines/cheese/cheese-321-linux-pext \
  --label-a lc0ctl_last --label-b cheese321_full \
  --games 60 --clock-base-ms 60000 --clock-inc-ms 1000 --clock-grace-ms 100 \
  --warmup-nodes-a 3000 \
  --move-log-out scratchpad/elo_calib/C_movelog.jsonl \
  --pgn-out scratchpad/elo_calib/C.pgn >> scratchpad/elo_calib/C.log 2>&1
say "match C rc=$?"
say "elo calibration complete"
