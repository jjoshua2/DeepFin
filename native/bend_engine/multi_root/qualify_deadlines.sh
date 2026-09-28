#!/usr/bin/env bash
# Explicit deadline-specific native tests. Input matrix comes from qualify_async.sh.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: qualify_deadlines.sh VERIFIED_COMPILER ASYNC_MATRIX NEW_OUTPUT' >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler="$(realpath "$1")"; matrix="$(realpath "$2")"; out="$3"
[[ ! -e "$out" ]]; mkdir -p "$out"; out="$(realpath "$out")"
bun="${BUN:-bun}"; cc="${CC:-clang}"
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.txt"
"$bun" "$compiler/bend2/main.ts" "$here/deadline_probe.bend" -o "$out/probe.c"
for mode in normal ubsan; do
  flags=(); [[ "$mode" == ubsan ]] && flags=(-fsanitize=undefined -fno-sanitize-recover=all)
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" "$out/probe.c" -pthread -lm -o "$out/probe-$mode"
  "$out/probe-$mode" --threads 1 > "$out/probe-$mode.txt"
  python - "$out/probe-$mode.txt" <<'PY'
import sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text().splitlines()
expected=['timers 10']
for width in (9344,11200):
    for mask in range(16):
        expected.append(f'case {mask} {width} {4-mask.bit_count()}')
        row=0
        for root in range(1,5):
            if mask & (1 << (root-1)):
                expected.append(f'drop {root}')
            else:
                expected.append(f'keep {root} {row}')
                row+=1
expected.append('compaction 32')
assert lines==expected
PY
  for channels in 146 175; do
    dir="$matrix/c${channels}-b4"
    gate="$dir/gate"; reference="$dir/runner"
    if [[ "$mode" == ubsan ]]; then gate="$dir/control-ubsan"; reference="$dir/callback-ubsan"; fi
    python -m native.bend_engine.multi_root.verify_deadlines --binary "$gate" \
      --reference "$reference" --report "$out/c${channels}-$mode.json"
  done
done
