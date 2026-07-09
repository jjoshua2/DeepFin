#!/bin/bash
# TCEC throughput autotune — find the best nps (sims/s) configuration for
# THIS machine in ~5-10 minutes, including an optional multi-GPU pass.
#
#   bash scripts/tcec_autotune.sh --checkpoint /path/to/trainer.pt
#   bash scripts/tcec_autotune.sh --checkpoint /path/to/trainer.pt --devices cuda:0,cuda:1
#
# Phases (each prints per-config sims/s; the summary at the end names the winner):
#   A  batching sweep     — predefined (chunk_sims, topk, max_batch) grid, single GPU
#   B  walker sweep       — search-parallelism {1,2,4,8} at the production batching
#   C  multi-GPU sweep    — only when --devices lists 2+ GPUs (PUCV gather modes)
#
# The FIRST search triggers a one-time torch.compile (~2 min on a cold cache)
# — that is expected, included in the budget, and cached for later runs.
# Please run on an otherwise idle GPU and paste the full output back to us.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CHECKPOINT=""
DEVICES=""
NODES="${NODES:-20000}"
REPEATS="${REPEATS:-2}"
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --devices)    DEVICES="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ] || { echo "--checkpoint /path/to/trainer.pt is required"; exit 2; }

OUT="tcec_autotune_$(date '+%Y%m%d_%H%M').log"
say() { echo "[autotune $(date '+%H:%M:%S')] $*" | tee -a "$OUT"; }
say "checkpoint=$CHECKPOINT devices=${DEVICES:-cuda:0} nodes=$NODES repeats=$REPEATS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$OUT"

say "=== phase A: batching sweep (includes one-time compile on the first config) ==="
python3 scripts/bench_uci_engine.py --checkpoint "$CHECKPOINT" --device cuda \
  --sweep --nodes "$NODES" --repeats "$REPEATS" 2>&1 | tee -a "$OUT"

say "=== phase B: walker sweep ==="
python3 scripts/bench_uci_engine.py --checkpoint "$CHECKPOINT" --device cuda \
  --walker-sweep --nodes "$NODES" --repeats "$REPEATS" 2>&1 | tee -a "$OUT"

if [ -n "$DEVICES" ] && echo "$DEVICES" | grep -q ","; then
  say "=== phase C: multi-GPU PUCV sweep ($DEVICES) ==="
  python3 scripts/bench_uci_engine.py --checkpoint "$CHECKPOINT" \
    --devices "$DEVICES" --pucv-sweep --nodes "$NODES" --repeats "$REPEATS" 2>&1 | tee -a "$OUT"
else
  say "phase C skipped (pass --devices cuda:0,cuda:1 to test multi-GPU)"
fi

say "=== done — full log: $OUT ==="
say "Best-effort summary (highest sims/s lines):"
grep -iE "sims/s|sims_per_s|nps" "$OUT" | sort -t= -k2 -rn 2>/dev/null | head -8 || true
say "Please send the whole $OUT file back; we will reply with the exact engine command line to register."
