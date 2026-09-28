#!/usr/bin/env bash
# Opt-in real CPU-model qualification target, not the standalone chess engine.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: build_model_probe.sh NEW_OUTPUT PACKAGE.pt2 LIBTORCH_CMAKE_PREFIX' >&2; exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out="$1"; package="$2"; prefix="$3"
bun="${BUN:-bun}"
[[ ! -e "$out" ]] || { echo 'Use a new output directory' >&2; exit 2; }
"$bun" "$here/../standalone/bind_model.js" --check "$package"
mkdir -p -- "$out"
out="$(cd -- "$out" && pwd)"
"$bun" "$here/../standalone/bind_model.js" --header "$package" "$out/model_contract.h" > "$out/binding.json"
cmake -S "$here" -B "$out/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" -DCMAKE_PREFIX_PATH="$prefix" -DBEND_CONTRACT_DIR="$out"
cmake --build "$out/build" --parallel 1
ldd "$out/build/native-async-model-probe" > "$out/libraries.txt"
if grep -qi libpython "$out/libraries.txt"; then echo 'Unexpected libpython dependency' >&2; exit 2; fi
sha256sum "$out/build/native-async-model-probe" > "$out/executable.txt"
