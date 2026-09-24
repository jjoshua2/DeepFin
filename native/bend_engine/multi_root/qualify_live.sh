#!/usr/bin/env bash
# Source-only full-runner gate. Component passes cannot substitute for this gate.
# No model export, training, old generated-C artifact or ordinary-pytest invocation.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: qualify_live.sh VERIFIED_COMPILER_SOURCE NEW_OUTPUT' >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler="$(cd -- "$1" && pwd)"; out="$2"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
[[ ! -e "$out" ]]
mkdir -p -- "$(dirname -- "$out")"; mkdir -- "$out"
out="$(cd -- "$out" && pwd)"
printf '{"status":"running","live_runner_qualified":false}\n' > "$out/status.json"
trap 'rc=$?; if [[ $rc != 0 ]]; then printf "{\"status\":\"failed\",\"exit\":%d,\"live_runner_qualified\":false}\n" "$rc" > "$out/status.json"; fi' EXIT
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.json"
# Keep validation and native generation in the pinned canonical compiler path.
# A killed/failed generation cannot be replaced by old or hand-patched C.
for entry in main live live_registry_probe; do
  BEND_NO_TELEMETRY=1 "$bun" "$compiler/bend2/main.ts" "$here/$entry.bend" -o "$out/$entry.c" > "$out/$entry-build.txt" 2>&1
done
for mode in normal ubsan; do
  flags=(); [[ "$mode" == ubsan ]] && flags=(-fsanitize=undefined -fno-sanitize-recover=all)
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" "$out/live_registry_probe.c" -pthread -lm -o "$out/registry-$mode"
  "$out/registry-$mode" --threads 1 > "$out/registry-$mode.txt"
  for channels in 146 175; do
    dir="$out/$mode-c$channels-b4"; mkdir "$dir"
    profile=2; [[ "$channels" == 175 ]] && profile=4
    printf '#define DEEPFIN_MODEL_BATCH 4\n#define DEEPFIN_MODEL_CHANNELS %d\n#define DEEPFIN_MODEL_PROFILE %d\n' "$channels" "$profile" > "$dir/model_contract.h"
    for entry in main live; do
      "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$dir" -c "$out/$entry.c" -o "$dir/$entry.o"
    done
    for backend in test control; do
      "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" -I "$dir" -c "$here/${backend}_backend.c" -o "$dir/$backend.o"
      "$cxx" -std=c++20 -O1 -ffp-contract=off "${flags[@]}" -I "$dir" "$dir/live.o" "$dir/$backend.o" "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$dir/live-$backend"
    done
    "$cxx" -std=c++20 -O1 -ffp-contract=off "${flags[@]}" -I "$dir" "$dir/main.o" "$dir/test.o" "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$dir/reference"
    python -m native.bend_engine.multi_root.verify_live --binary "$dir/live-test" --gate "$dir/live-control" --reference "$dir/reference" --report "$dir/qualification.json"
  done
done
sha256sum "$out"/*.c > "$out/generated-sha256.txt"
printf '{"status":"passed","live_runner_qualified":true,"scope":"deterministic callback only","neural_model_qualified":false,"gpu_qualified":false}\n' > "$out/status.json"
