#!/usr/bin/env bash
# Native backend benchmark only: no Bend generation, Python-in-engine or live changes.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "usage: build.sh NEW_OUTPUT TRUSTED_CPU_PACKAGE.pt2 LIBTORCH_CMAKE_PREFIX" >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
single="$here/../standalone"
out="$1"; package="$(realpath "$2")"; prefix="$3"
[[ ! -e "$out" ]]
mkdir -p "$out"
out="$(cd -- "$out" && pwd)"
"${BUN:-bun}" "$single/bind_model.js" --batch-header "$package" "$out/model_contract.h" > "$out/binding.json"
cmake -S "$single" -B "$out/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DCMAKE_PREFIX_PATH="$prefix" -DBEND_CONTRACT_DIR="$out" \
  -DBEND_GENERATED_C="$here/probe.cpp" -DBEND_TARGET_NAME=deepfin-service-probe \
  -DBEND_CUDA_MODEL=OFF -DBEND_ASYNC_BATCH=OFF
cmake --build "$out/build" --parallel 1
sha256sum "$here/probe.cpp" "$out/build/deepfin-service-probe" > "$out/build-identities.txt"
ldd "$out/build/deepfin-service-probe" > "$out/libraries.txt"
if grep -qi libpython "$out/libraries.txt"; then
  echo "service probe unexpectedly links libpython" >&2; exit 2
fi
