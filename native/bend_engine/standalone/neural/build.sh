#!/usr/bin/env bash
# Explicit trusted exporter package and installed LibTorch. No Python build/run.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base="$(cd -- "$here/.." && pwd)"
repo="$(cd -- "$base/../../.." && pwd)"
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'usage: build.sh PACKAGE.pt2 NEW_OUTPUT_DIRECTORY TORCH_CMAKE_PREFIX [COMPILER_SOURCE]' >&2; exit 2
fi
package="$(realpath "$1")"; output="$2"; torch_prefix="$(realpath "$3")"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
for tool in "$bun" "$cc" "$cxx" cmake; do command -v "$tool" >/dev/null; done
[[ ! -e "$output" ]] || { echo 'output already exists; use a new path' >&2; exit 2; }
[[ -f "$torch_prefix/Torch/TorchConfig.cmake" ]] || { echo 'missing explicit Torch CMake prefix' >&2; exit 2; }
# Reject incompatible/changed input before creating any output or downloading tools.
contract="$("$bun" "$here/contract.js" "$package")"
revision="$("$bun" "$base/verify_compiler.js" --revision)"
repository="$("$bun" "$base/verify_compiler.js" --repository)"
compiler="${4:-$repo/build/bend_standalone_toolchain/$revision/source}"
if [[ ! -d "$compiler" ]]; then
  mkdir -p -- "$(dirname -- "$compiler")"
  git init "$compiler"
  git -C "$compiler" remote add origin "$repository"
  git -C "$compiler" fetch --depth=1 origin "$revision"
  git -C "$compiler" checkout --detach FETCH_HEAD
fi
compiler="$(cd -- "$compiler" && pwd)"
"$bun" "$base/verify_compiler.js" "$compiler"
mkdir -p -- "$output"; output="$(cd -- "$output" && pwd)"
printf '%s\n' "$contract" > "$output/model_contract.h"
cp "${package%.pt2}.json" "$output/export_manifest.json"
BEND_NO_TELEMETRY=1 "$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$output/engine.c"
cmake -S "$here" -B "$output/cmake" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
  -DCMAKE_PREFIX_PATH="$torch_prefix" -DBEND_GENERATED_C="$output/engine.c" -DMODEL_CONTRACT_DIR="$output"
cmake --build "$output/cmake" --parallel 1
cp "$output/cmake/deepfin-bend-neural" "$output/deepfin-bend-neural"
ldd "$output/deepfin-bend-neural" > "$output/linked-libraries.txt"
if grep -qi 'libpython\|not found' "$output/linked-libraries.txt"; then echo 'invalid runtime dependencies' >&2; exit 2; fi
{
  printf 'compiler_revision=%s\n' "$revision"
  "$bun" "$base/verify_compiler.js" "$compiler"
  printf 'bun='; "$bun" --version
  "$cc" --version
  sha256sum "$output/engine.c" "$output/model_contract.h" "$output/deepfin-bend-neural" "$package"
} > "$output/build.txt"
printf 'Run with DEEPFIN_BEND_PACKAGE set to the exact immutable package: %q --threads 1\n' "$output/deepfin-bend-neural"
