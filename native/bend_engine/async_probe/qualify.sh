#!/usr/bin/env bash
# No network, model export or source mutation. The compiler must already exist.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler="${1:?provide the checked Bend compiler directory}"
out="${2:?provide a NEW output directory}"
bun="${BUN:-bun}"
cc="${CC:-clang}"
cxx="${CXX:-clang++}"
python="${PYTHON:-python}"
if [[ -e "$out" ]]; then echo 'Use a new output directory' >&2; exit 2; fi
mkdir -p -- "$out"
out="$(cd -- "$out" && pwd)"
compiler="$(cd -- "$compiler" && pwd)"
export BEND_NO_TELEMETRY=1
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" | tee "$out/compiler.txt"
"$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$out/probe.c" 2>"$out/bend-check.txt"
"$cc" -std=c11 -O1 -ffp-contract=off "$out/probe.c" -pthread -lm -o "$out/no-native"
for mode in normal sanitized; do
  san=()
  bend_san=()
  if [[ "$mode" == sanitized ]]; then
    san=(-fsanitize=address,undefined -fno-sanitize-recover=all)
    # Clang 17 cannot compile ASan's dynamic stack realignment with this pinned
    # generated musttail code. Keep full worker ASan+UBSan; Bend C gets UBSan.
    # No compiler/runtime source is patched to hide the limitation.
    bend_san=(-fsanitize=undefined -fno-sanitize-recover=all)
  fi
  "$cxx" -std=c++20 -O1 -g -Wall -Wextra -Werror "${san[@]}" \
    "$here/slot_test.cpp" -pthread -o "$out/slot-$mode"
  "$cc" -std=c11 -O1 -ffp-contract=off "${bend_san[@]}" -DDEEPFIN_BEND_NATIVE_MODEL \
    -c "$out/probe.c" -o "$out/probe-$mode.o"
  "$cxx" -std=c++20 -O1 -g -Wall -Wextra -Werror "${san[@]}" \
    "$out/probe-$mode.o" "$here/../standalone/async_model.cpp" "$here/test_backend.cpp" \
    -pthread -lm -o "$out/bend-$mode"
  "$python" "$here/verify.py" --slot "$out/slot-$mode" --bend "$out/bend-$mode" \
    --material "$out/no-native" --report "$out/$mode.json"
done
{
  echo 'normal: generated Bend/C and C++ worker without sanitizers'
  echo 'sanitized: generated Bend/C UBSan; C++ worker and adapter ASan+UBSan'
  "$cc" --version; "$cxx" --version; "$python" --version; "$bun" --version
  sha256sum "$out/probe.c" "$out"/slot-* "$out"/bend-normal "$out"/bend-sanitized
} > "$out/build.txt"
