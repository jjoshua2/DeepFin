#!/bin/bash
set -euo pipefail

ROOT="${CHESS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SOURCE_CONFIG="${BENCH_BATCH_WAIT_CONFIG:-$ROOT/configs/pbt2_small.yaml}"
BROKER_LOG="${BENCH_BATCH_WAIT_BROKER_LOG:-$ROOT/runs/pbt2_small/server/shared_broker.out}"
TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/bench_batch_wait.XXXXXX.yaml")"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

cp "$SOURCE_CONFIG" "$TMP_CONFIG"
sed -i "s/distributed_inference_use_compile:.*/distributed_inference_use_compile: false/" "$TMP_CONFIG"
sed -i "s/distributed_worker_use_compile:.*/distributed_worker_use_compile: false/" "$TMP_CONFIG"

cd "$ROOT"
echo "wait_ms | avg_batch | pos/s"
echo "--------|-----------|------"

for WAIT_MS in 0.0 0.5 1.0 2.0 5.0 10.0; do
  sed -i "s/distributed_inference_batch_wait_ms:.*/distributed_inference_batch_wait_ms: $WAIT_MS/" "$TMP_CONFIG"

  PYTHONPATH=. python3 -m chess_anti_engine.run --config "$TMP_CONFIG" --mode tune >/dev/null 2>&1 &
  PID=$!

  sleep 15
  sleep 10
  AVG=$(tail -3 "$BROKER_LOG" 2>/dev/null | awk 'match($0, /avg [0-9.]+/) {v=substr($0, RSTART+4, RLENGTH-4); sum+=v; n++} END {if(n>0) printf "%.0f", sum/n; else print "?"}')
  PPS=$(tail -3 "$BROKER_LOG" 2>/dev/null | awk 'match($0, /[0-9.]+ pos\/s/) {v=substr($0, RSTART, RLENGTH-6); sum+=v; n++} END {if(n>0) printf "%.0f", sum/n; else print "?"}')

  printf "%5s   | %6s    | %s\n" "$WAIT_MS" "$AVG" "$PPS"

  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  sleep 2
done

echo "Done. Source config was not modified: $SOURCE_CONFIG"
