#!/usr/bin/env bash
# #438 primary readout -- prereg docs/lc0_positive_control_prereg.md, Amendments 3+4.
# Amendment 4 fixes: held-out legs scan the HELD-OUT dirs; train legs read the
# TRAIN frozen artifact and scan the TRAIN corpus; the train freeze runs
# concurrently and its completion is signalled by a .done marker (poll a FILE).
set -euo pipefail
LIVE=/home/josh/projects/chess
WT=/home/josh/chess-438-merge-review
OUT=$LIVE/runs/lc0_control_20260820
FROZEN_H=$LIVE/data/lc0_control_heldout_frozen_v2.json
FROZEN_T=$LIVE/data/lc0_control_train_frozen_v1.json
R=$LIVE/scratchpad/lc0_readout
mkdir -p "$R"
cd "$WT"; export PYTHONPATH=.
export CHESS_LIVE_PRODUCTION_CONFIG=$LIVE/configs/pbt2_small.yaml

SCORE_H=(python3 scripts/lc0_control_eval.py score
       --config configs/lc0_positive_control.yaml
       --frozen "$FROZEN_H" --summary "$OUT/summary.json"
       --shards "$LIVE"/data/lc0_rows_heldout/*/ --batch-size 512)
SCORE_T=(python3 scripts/lc0_control_eval.py score
       --config configs/lc0_positive_control.yaml
       --frozen "$FROZEN_T" --summary "$OUT/summary.json"
       --shards "$LIVE"/data/lc0_rows/*/ --batch-size 512
       --population train)

echo "=== [0] random-init seed band (owed amendment; BEFORE the primary) ==="
for s in 101 102 103 104 105; do
  "${SCORE_H[@]}" --seed "$s" --out "$R/band_seed${s}.json"
done

echo "=== [1] negative control: shuffled targets on LAST/heldout (exits 1 on rig leak) ==="
"${SCORE_H[@]}" --checkpoint "$OUT/checkpoint.pt" --shuffle-targets \
    --out "$R/negctl_last_heldout.json"

echo "=== [2a] primary held-out scores ==="
"${SCORE_H[@]}" --checkpoint "$OUT/checkpoint_mid.pt" --population heldout --out "$R/mid_heldout.json"
"${SCORE_H[@]}" --checkpoint "$OUT/checkpoint.pt"     --population heldout --out "$R/last_heldout.json"

echo "=== [2b] wait for the train freeze, then train scores ==="
while [ ! -f "$LIVE/scratchpad/.train_freeze.done" ]; do sleep 60; done
echo "train freeze marker seen $(date '+%H:%M:%S')"
"${SCORE_T[@]}" --checkpoint "$OUT/checkpoint_mid.pt" --out "$R/mid_train.json"
"${SCORE_T[@]}" --checkpoint "$OUT/checkpoint.pt"     --out "$R/last_train.json"

echo "=== [3] compares (Amendment 3 bar: 0.409 pp at n=91,842) ==="
python3 scripts/lc0_control_eval.py compare \
    --a "$R/mid_heldout.json" --b "$R/last_heldout.json" --max-halfwidth-pp 0.409
python3 scripts/lc0_control_eval.py compare \
    --a "$R/mid_train.json" --b "$R/last_train.json" \
    --population train --max-halfwidth-pp 0.409
echo "=== readout complete $(date '+%H:%M:%S') ==="
