#!/usr/bin/env bash
set -euo pipefail

LIVE=/home/josh/projects/chess
WT=/home/josh/projects/chess/.dev/worktree/wise-cloud
OPS=$LIVE/scratchpad/bt4_g10_raw
OUT=$LIVE/data/lc0/bt4_policy_sidecars/g10_raw
RUN06=$LIVE/data/nnue_bootstrap/run06_g10
RUN07=$LIVE/data/nnue_bootstrap/run07_g10_companion4
ONNX=$LIVE/data/lc0/onnx/BT4-it332-vanilla-winner.onnx
GPU_LOCK=$LIVE/scratchpad/gpu0_experiment.lock
CODE_COMMIT=021124eeaff67a465e000a06a4a2e3b977afc801
CODE_BLOB=eb18a4065e43da3fd5e71e7a5d267443434942d1

mkdir -p "$OPS"
exec 9>"$OPS/driver.lock"
if ! flock -n 9; then
  echo "another G10 BT4 driver holds $OPS/driver.lock" >&2
  exit 75
fi
exec >>"$OPS/driver.log" 2>&1
echo $$ >"$OPS/driver.pid"

failed=1
paused=0
finish() {
  local rc=$?
  if (( paused )); then
    touch "$OPS/driver.paused"
    echo "[bt4-g10 $(date -u +%FT%TZ)] paused cleanly"
  elif (( failed )); then
    touch "$OPS/driver.fail"
    echo "[bt4-g10 $(date -u +%FT%TZ)] failed rc=$rc"
  fi
}
trap finish EXIT

verify_code() {
  test "$(git -C "$WT" rev-parse "$CODE_COMMIT:scripts/bt4_raw_corpus_sidecar.py")" = "$CODE_BLOB"
  test "$(git -C "$WT" hash-object scripts/bt4_raw_corpus_sidecar.py)" = "$CODE_BLOB"
  git -C "$WT" merge-base --is-ancestor "$CODE_COMMIT" HEAD
}

run_group() {
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nice -n 10 python3 scripts/bt4_raw_corpus_sidecar.py \
    --source run06_g10="$RUN06" \
    --source run07_g10_companion4="$RUN07" \
    --out-root "$OUT" \
    --onnx "$ONNX" \
    --batch-size 1024 \
    --threads 16 \
    --gpu-mem-gb 24 \
    --max-shards 16 \
    --gpu-lock "$GPU_LOCK" \
    --min-free-gib 150
}

verify_final() {
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nice -n 10 python3 scripts/bt4_raw_corpus_sidecar.py \
    --source run06_g10="$RUN06" \
    --source run07_g10_companion4="$RUN07" \
    --out-root "$OUT" \
    --onnx "$ONNX" \
    --batch-size 1024 \
    --threads 16 \
    --gpu-mem-gb 24 \
    --max-shards 16 \
    --gpu-lock "$GPU_LOCK" \
    --min-free-gib 150 \
    --verify-all
}

final_inventory_caught_up() {
  python3 - "$OUT/bt4_raw_sidecar.status.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
status = json.loads(path.read_text())
sources = status.get("sources", {})
if set(sources) != {"run06_g10", "run07_g10_companion4"}:
    raise SystemExit(1)
for source in sources.values():
    if source.get("closed_source_shards") != source.get("complete_sidecars"):
        raise SystemExit(1)
    if source.get("unlisted_in_flight"):
        raise SystemExit(1)
PY
}

echo "[bt4-g10 $(date -u +%FT%TZ)] armed commit=$CODE_COMMIT"
test ! -e "$OPS/driver.done"
test ! -e "$OPS/driver.fail"
test -f "$ONNX"
cd "$WT"

while true; do
  verify_code
  if test -e "$OPS/pause.request"; then
    paused=1
    failed=0
    exit 0
  fi
  run_group
  if test -f "$RUN06/summary.json" && test -f "$RUN07/summary.json" \
      && final_inventory_caught_up; then
    verify_final
    python3 - "$OUT/bt4_raw_sidecar.verify.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text())
if receipt.get("verdict") != "PASS" or receipt.get("snapshot_only") is not False:
    raise SystemExit(f"final BT4 sidecar verification failed: {receipt}")
print(
    "[bt4-g10] final PASS: "
    f"{receipt['total_verified_rows']} rows / "
    f"{receipt['total_verified_shards']} shards"
)
PY
    touch "$OPS/driver.done"
    failed=0
    echo "[bt4-g10 $(date -u +%FT%TZ)] complete"
    exit 0
  fi
  # Give an awaiting trainer or arena an uncontested boundary to take the
  # shared GPU lease. A pause request is observed before the next group.
  sleep 15
done
