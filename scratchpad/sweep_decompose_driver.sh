#!/usr/bin/env bash
# Did the ROOT move, or is the parameter just relaxing toward a fixed one?
#
# The frozen curve is an iter-265 object computed on the iter-265 replay window.
# Two more sweeps on the CURRENT window, back to back, give a clean 2x2 -- the row
# sampler is seeded (default_rng(0)) so both runs below see the SAME rows, and the
# only difference between them is the MODEL:
#
#   data effect  = G_265(current rows) - G_265(iter-265 rows)   [have the second]
#   model effect = G_487(current rows) - G_265(current rows)
# c338 is included because the DECELERATION happened across 297->338->482: if the
# root had already moved by 338 the model leg is not a single monotone drift.
set -u
cd /home/josh/projects/chess
rm -f scratchpad/.sweepdec.done
R="runs/pbt2_small/replay/train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11/replay_shards"
run () {
  PYTHONPATH=. python3 scratchpad/logtemp_sweep.py \
    --replay-dir "$R" --checkpoint "$2" --rows 12000 --window-shards 750 \
    --out "scratchpad/sweepdec_$1.json" > "scratchpad/sweepdec_$1.log" 2>&1
  echo "$1 exit $?" >> scratchpad/sweep_decompose.log
}
: > scratchpad/sweep_decompose.log
run c265 scratchpad/pinned_ckpt
run c338 data/salvage/f_only_midpoint_20260818/seeds/slot_000
run c487 data/salvage/f_only_readout_iter487_20260819
echo DONE > scratchpad/.sweepdec.done
