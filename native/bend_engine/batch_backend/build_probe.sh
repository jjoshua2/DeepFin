#!/usr/bin/env bash
# Explicit backend-only build. Does not create or replace the UCI executable.
set -euo pipefail
if [[ $# != 4 && $# != 5 ]]; then
  echo 'usage: build_probe.sh NEW_OUTPUT PACKAGE.pt2 LIBTORCH_CMAKE_PREFIX VERIFIED_COMPILER_SOURCE [--cuda-bf16]' >&2
  exit 2
fi
mode=--batch
cuda=OFF
if [[ $# == 5 ]]; then
  [[ "$5" == --cuda-bf16 ]] || { echo 'expected --cuda-bf16' >&2; exit 2; }
  mode=--cuda-batch
  cuda=ON
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
single="$here/../standalone"
bun="${BUN:-bun}"
output="$1"; package="$2"; torch_prefix="$3"; compiler="$4"
command -v "$bun" >/dev/null
command -v cmake >/dev/null
command -v "${CC:-clang}" >/dev/null
command -v "${CXX:-clang++}" >/dev/null
[[ -d "$torch_prefix" && ! -e "$output" ]]
"$bun" "$single/verify_compiler.js" "$compiler"
"$bun" "$single/bind_model.js" "$mode-check" "$package"
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
output="$(cd -- "$output" && pwd)"
"$bun" "$single/bind_model.js" "$mode-header" "$package" "$output/model_contract.h" > "$output/model-binding.json"
"$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$output/probe.c"
cmake -S "$single" -B "$output/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DCMAKE_PREFIX_PATH="$torch_prefix" -DBEND_GENERATED_C="$output/probe.c" \
  -DBEND_CUDA_MODEL="$cuda" -DBEND_CONTRACT_DIR="$output" -DBEND_TARGET_NAME=deepfin-bend-batch-probe
cmake --build "$output/build" --parallel 1
ldd "$output/build/deepfin-bend-batch-probe" > "$output/libraries.txt"
if grep -qi libpython "$output/libraries.txt"; then echo 'unexpected interpreter dependency' >&2; exit 2; fi
sha256sum "$output/probe.c" "$output/build/deepfin-bend-batch-probe" > "$output/build.txt"
printf 'Backend probe only: %q --threads 1\n' "$output/build/deepfin-bend-batch-probe"
