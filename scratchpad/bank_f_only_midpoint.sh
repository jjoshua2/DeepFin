#!/usr/bin/env bash
# Bank the F-only MIDPOINT checkpoint (~iter 338 = +37 from the flip at 301).
# Peer review 2026-08-18: a midpoint + final bank gives a TRAJECTORY instead of
# another two-dot story, and Ray's ~6-checkpoint window makes it unrecoverable
# after the fact.
# --metric training_iteration is REQUIRED: the default metric picks the
# best-metric row, not current state.
set -uo pipefail
D="runs/pbt2_small/tune/train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11/progress.csv"
COL=$(head -1 "$D" | tr ',' '\n' | grep -n '^training_iteration$' | cut -d: -f1)
while :; do
  IT=$(tail -1 "$D" | cut -d, -f"$COL")
  case "$IT" in ''|*[!0-9]*) sleep 60; continue;; esac
  [ "$IT" -ge 338 ] && break
  sleep 60
done
echo "reached iter $IT -- banking midpoint"
./scripts/train.sh salvage-export --top-n 1 --metric training_iteration \
  --out data/salvage/f_only_midpoint_20260818
echo "EXIT=$?"
du -sh data/salvage/f_only_midpoint_20260818 2>/dev/null
