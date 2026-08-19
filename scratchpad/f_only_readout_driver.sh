#!/usr/bin/env bash
# F-ONLY DECIDING READOUT. Settings copied verbatim from base4k_driver.sh so the
# paired comparison shares its instrument (batch 32, same frozen audit set).
set -u
cd /home/josh/projects/chess
rm -f scratchpad/.freadout.done
run () {  # $1=label  $2=checkpoint path
  PYTHONPATH=. python3 scripts/audit_targets.py \
    --config configs/pbt2_small.yaml \
    --checkpoint "$2" \
    --audit-set data/audit_set_v1.jsonl \
    --max-positions 4000 --batch-size 32 --gpu-mem-fraction 0.10 \
    --out-dir "scratchpad/f4k_$1" \
    --dump-per-position "scratchpad/f4k_$1.jsonl" --dump-distributions \
    > "scratchpad/f4k_$1.log" 2>&1
  echo "$1 exit $?" >> scratchpad/f_only_readout.log
}
: > scratchpad/f_only_readout.log
run i297 data/salvage/apf_endpoint_checkpoint_000297_20260818/checkpoint_000297
run i487 data/salvage/f_only_readout_iter487_20260819
echo DONE > scratchpad/.freadout.done
