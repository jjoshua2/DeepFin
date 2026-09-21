#!/usr/bin/env bash
# Pre-exported trusted CPU model only. No Python compiler discovery or runtime.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent="$(cd -- "$here/.." && pwd)"
repo="$(cd -- "$parent/../../.." && pwd)"
if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo 'Usage: build.sh NEW_OUTPUT PACKAGE LIBTORCH_CMAKE_PREFIX [COMPILER_SOURCE] [generic|portable|native|ubsan]' >&2; exit 2
fi
output="$1"; package="$(realpath -s "$2")"; prefix="$(realpath "$3")"; mode="${5:-generic}"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
case "$mode" in generic|portable|native|ubsan) ;; *) echo 'unsupported model build mode' >&2; exit 2;; esac
for exe in "$bun" "$cc" "$cxx" cmake; do command -v "$exe" >/dev/null; done
[[ ! -e "$output" && ! -L "$output" ]] || { echo 'Output exists; use a fresh directory' >&2; exit 2; }
# Reject unsupported metadata and mismatched bytes before any compilation.
"$bun" "$here/bind_model.js" "$package"
revision="$("$bun" "$parent/verify_compiler.js" --revision)"
repository="$("$bun" "$parent/verify_compiler.js" --repository)"
compiler="${4:-$repo/build/bend_standalone_toolchain/$revision/source}"
if [[ ! -d "$compiler" ]]; then
  mkdir -p -- "$(dirname -- "$compiler")"
  git init "$compiler"; git -C "$compiler" remote add origin "$repository"
  git -C "$compiler" fetch --depth=1 origin "$revision"
  git -C "$compiler" checkout --detach FETCH_HEAD
fi
compiler="$(cd -- "$compiler" && pwd)"
"$bun" "$parent/verify_compiler.js" "$compiler"
mkdir -p -- "$output"; output="$(cd -- "$output" && pwd)"
"$bun" "$here/bind_model.js" "$package" "$output"
BEND_NO_TELEMETRY=1 "$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$output/engine.c"
cmake -S "$here" -B "$output/cmake" -DCMAKE_PREFIX_PATH="$prefix" \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
  -DBEND_GENERATED_C="$output/engine.c" -DMODEL_CONFIG_DIR="$output" -DBEND_MODE="$mode"
cmake --build "$output/cmake" --parallel 1
cp "$output/cmake/deepfin-bend-neural" "$output/deepfin-bend-neural"
{
  printf 'compiler_revision=%s\nmode=%s\n' "$revision" "$mode"
  "$bun" "$parent/verify_compiler.js" "$compiler"
  "$bun" "$here/bind_model.js" "$package"
  "$cc" --version
  sha256sum "$output/engine.c" "$output/deepfin-bend-neural" "$output/leaf_model_config.h"
} > "$output/build.txt"
printf 'Run: DEEPFIN_BEND_PACKAGE=%q %q --threads 1\n' "$package" "$output/deepfin-bend-neural"
