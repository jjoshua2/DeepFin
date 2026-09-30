#!/usr/bin/env bash
# Explicit full-application qualification; never called by ordinary pytest.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: qualify_search.sh COMPILER_SOURCE GENERATED_ENGINE.c NEW_OUTPUT' >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler="$(cd -- "$1" && pwd)"; engine="$(realpath "$2")"; out="$3"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
[[ ! -e "$out" ]]
mkdir -p "$out"
out="$(cd -- "$out" && pwd)"
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.txt"
for name in work_probe epoch_probe; do
  "$bun" "$compiler/bend2/main.ts" "$here/$name.bend" -o "$out/$name.c"
done
for mode in normal ubsan; do
  flags=()
  if [[ "$mode" == ubsan ]]; then flags=(-fsanitize=undefined -fno-sanitize-recover=all); fi
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" -DDEEPFIN_BEND_NATIVE_MODEL -c "$engine" -o "$out/engine-$mode.o"
  "$cxx" -std=c++20 -O1 -ffp-contract=off "${flags[@]}" -pthread "$out/engine-$mode.o" \
    "$here/../standalone/async_model.cpp" "$here/search_gate.cpp" -lm -o "$out/gate-$mode"
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" "$out/work_probe.c" -pthread -lm -o "$out/work-$mode"
  "$out/work-$mode" > "$out/work-$mode.txt"
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" "$out/epoch_probe.c" -pthread -lm -o "$out/epoch-$mode"
  status=0
  "$out/epoch-$mode" > "$out/epoch-$mode.txt" 2> "$out/epoch-$mode.err" || status=$?
  test "$status" = 2
  grep -qx 'epoch 4294967295' "$out/epoch-$mode.txt"
  grep -q 'search epoch exhausted; restart required' "$out/epoch-$mode.err"
  python -m native.bend_engine.async_probe.verify_search --report "$out/$mode.json" \
    --accounting-output "$out/work-$mode.txt" --command "$out/gate-$mode" --threads 1
 done
sha256sum "$engine" "$out/gate-normal" "$out/gate-ubsan" > "$out/executables.txt"
