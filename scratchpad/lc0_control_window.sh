#!/usr/bin/env bash
# Runs INSIDE scripts/pause_window.sh, with training parked and workers drained.
# Two steps, fail-fast: purity receipt, then the frozen-budget training run.
set -euo pipefail

LIVE=/home/josh/projects/chess
WT=/home/josh/chess-438-merge-review
FROZEN=$LIVE/data/lc0_control_heldout_frozen_v2.json
RECEIPT=$LIVE/data/lc0_control_purity_receipt.json
CACHE=$LIVE/data/lc0_control_purity_cache
OUT=$LIVE/runs/lc0_control_20260820
STEPS=38544

cd "$WT"
export PYTHONPATH=.

echo "=== [1/2] purity receipt  $(date '+%H:%M:%S') ==="
python3 scripts/lc0_control_heldout.py purity \
    --frozen "$FROZEN" \
    --train-shards "$LIVE"/data/lc0_rows/*/ \
    --receipt "$RECEIPT" \
    --exposed-out "$LIVE"/data/lc0_control_exposed_latest.json \
    --cache-dir "$CACHE" \
    --workers 12

echo "=== [2/2] lc0_control_train --steps $STEPS   $(date '+%H:%M:%S') ==="
python3 scripts/lc0_control_train.py \
    --config configs/lc0_positive_control.yaml \
    --shards "$LIVE"/data/lc0_rows/*/ \
    --out-dir "$OUT" \
    --steps "$STEPS" \
    --purity-receipt "$RECEIPT"

echo "=== window work complete  $(date '+%H:%M:%S') ==="
