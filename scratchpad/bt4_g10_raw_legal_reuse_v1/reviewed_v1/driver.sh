#!/usr/bin/env bash
set -euo pipefail

LIVE=/home/josh/projects/chess
WT=/tmp/deepfin-bt4-label-runtime-pr537
OLD_OPS=$LIVE/scratchpad/bt4_g10_raw
OPS=$LIVE/scratchpad/bt4_g10_raw_legal_reuse_v1
OUT=$LIVE/data/lc0/bt4_policy_sidecars/g10_raw
RUN06=$LIVE/data/nnue_bootstrap/run06_g10
RUN07=$LIVE/data/nnue_bootstrap/run07_g10_companion4
ONNX=$LIVE/data/lc0/onnx/BT4-it332-vanilla-winner.onnx
GPU_LOCK=$LIVE/scratchpad/gpu0_experiment.lock
CODE_COMMIT=4e600902e1fd08727e1d83e009d6e241be0c7cba
RAW_SHA=6a58dab2deddc8cae3d801ec988d25b4e80c16f530fb3fc3df756440f2748c23
DUMP_SHA=d6b62559952aee3b0baeeadfdb584aeb9a6fd88b0d2285ab162ee39776be2cf8
HANDOFF_SHA=54388b8f5a684b9b840b7489ff1e5b8a257cd8ab5de133d43a6e4c718ea8ffce
export PATH=/usr/bin:/bin:/home/josh/.local/bin
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "$OPS"
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
  test "$(sha256sum "$OPS/verify_runtime.py" | cut -d ' ' -f1)" = "8ac3f13d293b74e7b80ff3880d0e659e77bcae1e8573b7da9dea76c251c99b18"
  /usr/bin/python3 "$OPS/verify_runtime.py"
  test "$(git -C "$WT" rev-parse HEAD)" = "$CODE_COMMIT"
  test -z "$(git -C "$WT" status --porcelain --untracked-files=no)"
  test "$(sha256sum "$WT/scripts/bt4_raw_corpus_sidecar.py" | cut -d ' ' -f1)" = "$RAW_SHA"
  test "$(sha256sum "$WT/scripts/bt4_policy_dump.py" | cut -d ' ' -f1)" = "$DUMP_SHA"
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

verify_code
# Share the old driver's lease: finish its already queued group naturally.
exec 9>"$OLD_OPS/driver.lock"
echo "[bt4-g10 $(date -u +%FT%TZ)] waiting for previous driver boundary"
if ! flock -w 21600 9; then
  echo "old driver did not release its lease within six hours" >&2
  exit 75
fi
test -f "$OLD_OPS/driver.paused"
test ! -e "$OLD_OPS/driver.fail"
test ! -e "$OLD_OPS/driver.done"
if test -e "$OPS/pause.request"; then
  paused=1
  failed=0
  exit 0
fi
# Consume only this handoff's exact request; preserve any later operator pause.
if ! test -f "$OLD_OPS/pause.request" ||    test "$(sha256sum "$OLD_OPS/pause.request" | cut -d ' ' -f1)" != "$HANDOFF_SHA"; then
  echo "handoff request missing or changed; preserving paused state" >&2
  paused=1
  failed=0
  exit 0
fi
mv -- "$OLD_OPS/pause.request" "$OPS/handoff.request.consumed.json"
echo "[bt4-g10 $(date -u +%FT%TZ)] previous driver paused; handoff acquired"

echo "[bt4-g10 $(date -u +%FT%TZ)] armed commit=$CODE_COMMIT"
test ! -e "$OPS/driver.done"
test ! -e "$OPS/driver.fail"
test -f "$ONNX"
cd "$WT"

while true; do
  verify_code
  if test -e "$OPS/pause.request" || test -e "$OLD_OPS/pause.request"; then
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
