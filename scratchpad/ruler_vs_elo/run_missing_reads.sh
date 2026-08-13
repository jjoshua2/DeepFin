#!/bin/bash
# Task #198 — does a 20-minute ruler predict a 5-hour arena?
#
# We hold DIRECT paired arena Elo vs boot512 for six lineage checkpoints measured
# under IDENTICAL conditions (same 200 pairs, sims 32, --search-shape training,
# and a reference file that is byte-identical across the two paths it was named
# by: md5 08d2e9a3be0cb9b31dc2ef4b02297a46). value_regret exists for only three
# of them. This fills the other three, giving n=6 paired (ruler, Elo) points.
#
# EXCLUDED ON PURPOSE, do not "fix" by adding them back:
#   iter400 — arena ran --search-shape PLAY, not training (different regime)
#   iter346 — arena row predates the search_candidate field; shape unverifiable
#
# RUN ONLY IN A PAUSE WINDOW (training stopped). ~4 min/checkpoint on an idle GPU.
# Ruler settings are pinned to the banked ladder's so the new dumps are
# comparable with the existing three (scratchpad/valreg_ladder_20260811/).
set -euo pipefail
cd /home/josh/projects/chess

if [ -f /tmp/chess_training.pid ] && kill -0 "$(cat /tmp/chess_training.pid)" 2>/dev/null; then
    echo "REFUSING: training is up. This is pause-window work." >&2
    exit 1
fi

OUT=scratchpad/ruler_vs_elo
mkdir -p "$OUT"

run() {  # run <name> <checkpoint>
    local name="$1" ckpt="$2"
    if [ ! -e "$ckpt" ]; then echo "MISSING $ckpt" >&2; exit 1; fi
    if [ -s "$OUT/${name}_dump.jsonl" ]; then echo "skip $name (already done)"; return; fi
    echo "=== $name"
    PYTHONPATH=. python3 scripts/value_regret.py \
        --checkpoint "$ckpt" \
        --max-positions 2000 --min-pieces 8 \
        --batch-size 128 --input-encoding fen_only \
        --gpu-mem-fraction 0.35 \
        --dump-per-position "$OUT/${name}_dump.jsonl" \
        2>&1 | tee "$OUT/${name}.log" | grep -E "OVERALL|kept" || true
}

run iter477 scratchpad/arena_20260803/ckpt/iter477.pt
run iter768 data/ratchet/snapshots/ck_2026-08-10_iter768.pt
run iter862 data/ratchet/snapshots/ck_2026-08-10_iter862_postmerge/trainer.pt

echo
echo "now: PYTHONPATH=. python3 scratchpad/ruler_vs_elo/correlate.py"
