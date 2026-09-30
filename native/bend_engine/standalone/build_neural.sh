#!/usr/bin/env bash
# Python is not a build/launch dependency here. The package exporter is separate.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'usage: build_neural.sh NEW_OUTPUT PACKAGE.pt2 LIBTORCH_CMAKE_PREFIX [COMPILER_SOURCE]' >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bun="${BUN:-bun}"
output="$1"; package="$2"; torch_prefix="$3"
command -v cmake >/dev/null
command -v "${CXX:-clang++}" >/dev/null
[[ -d "$torch_prefix" ]]
"$bun" "$here/bind_model.js" --check "$package"
args=("$output")
if [[ $# == 4 ]]; then args+=("$4"); fi
bash "$here/build.sh" "${args[@]}"
output="$(cd -- "$output" && pwd)"
"$bun" "$here/bind_model.js" --header "$package" "$output/model_contract.h" > "$output/model-binding.json"
cmake -S "$here" -B "$output/neural" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DCMAKE_PREFIX_PATH="$torch_prefix" -DBEND_GENERATED_C="$output/engine.c" -DBEND_CONTRACT_DIR="$output"
cmake --build "$output/neural" --parallel 1
ldd "$output/neural/deepfin-bend-neural" > "$output/neural-libraries.txt"
if grep -qi libpython "$output/neural-libraries.txt"; then echo 'unexpected interpreter dependency' >&2; exit 2; fi
sha256sum "$output/engine.c" "$output/neural/deepfin-bend-neural" > "$output/neural-build.txt"
printf 'Run with DEEPFIN_BEND_MODEL_PACKAGE set to the bound package: %q --threads 1\n' "$output/neural/deepfin-bend-neural"
