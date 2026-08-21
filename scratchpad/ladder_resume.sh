#!/usr/bin/env bash
# Dose ladder, take 2 — fixed rig (52127f817): all 5 arms, ONE invocation
# (within-invocation shard/draw pairing), fresh out-dir, rig_active proofs on.
set -uo pipefail
LIVE=/home/josh/projects/chess
WT=/home/josh/projects/chess-doseladder
OUT=$LIVE/scratchpad/dose_ladder2
LOG=$LIVE/scratchpad/dose_ladder2.log
mkdir -p "$OUT"
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

say "ladder2 start — 5 arms x 6000 steps, fixed variant wiring"
cd "$WT"; export PYTHONPATH=.
python3 scripts/retarget_retrain.py \
  --config "$LIVE/configs/pbt2_small.yaml" \
  --checkpoint "$LIVE/data/salvage/pre_lc0_control_20260819/seeds/slot_000/trainer.pt" \
  --replay-dir "$LIVE/data/salvage/pre_lc0_control_20260819/seeds/slot_000/replay_shards" \
  --steps 6000 --batch-size 512 --gpu-mem-fraction 0.5 \
  --out-dir "$OUT" \
  --no-rebuild-sf-targets \
  --variant "a000:" \
  --variant "a025:sf_p0_blend_alpha=0.25" \
  --variant "a050:sf_p0_blend_alpha=0.5" \
  --variant "a070:sf_p0_blend_alpha=0.7" \
  --variant "a100:sf_p0_blend_alpha=1.0" \
  >> "$LOG.train" 2>&1
say "driver exited rc=$?"

say "audit yardstick on all arms"
cd "$LIVE"; export PYTHONPATH=.
for arm in a000 a025 a050 a070 a100; do
  ck=$(ls "$OUT"/*"$arm"*/*.pt "$OUT"/"$arm"*.pt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then say "ARM $arm: NO CHECKPOINT FOUND — skipping"; continue; fi
  nice -n 15 python3 scripts/audit_targets.py --checkpoint "$ck" \
    --audit-set data/audit_set_v1.jsonl --config configs/pbt2_small.yaml \
    --gpu-mem-fraction 0.25 \
    > "$OUT/audit_$arm.json" 2>> "$LOG.audit"
  say "audit $arm rc=$? ckpt=$ck"
done
say "ladder2 complete"
