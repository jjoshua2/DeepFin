#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBE_DIR="$ROOT/native/bend_engine/cuda_parity"
BUILD_DIR="${BEND_CUDA_PARITY_BUILD_DIR:-$ROOT/build/bend_cuda_parity}"
BEND_BIN="${BEND_BIN:-bend}"
PYTHON_BIN="${PYTHON:-python}"
CMAKE_BIN="${CMAKE:-cmake}"
CC_BIN="${BEND_CUDA_PARITY_CC:-clang}"
CXX_BIN="${BEND_CUDA_PARITY_CXX:-clang++}"

for tool in "$BEND_BIN" "$PYTHON_BIN" "$CMAKE_BIN" "$CC_BIN" "$CXX_BIN"; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: required tool not found: $tool" >&2
    exit 2
  }
done

export BEND_NO_TELEMETRY="${BEND_NO_TELEMETRY:-1}"
if [ -d /usr/local/cuda/bin ]; then
  PATH="/usr/local/cuda/bin:$PATH"
fi

mkdir -p "$BUILD_DIR"
GENERATED="$BUILD_DIR/cuda_parity.generated.c"
CMAKE_BUILD="$BUILD_DIR/cmake"

echo "bend CUDA parity: $("$BEND_BIN" --version 2>/dev/null || echo unknown)"
"$BEND_BIN" "$PROBE_DIR/main.bend" -o "$GENERATED"

TORCH_PREFIX="$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

echo "bend CUDA parity: C=$CC_BIN CXX=$CXX_BIN"
"$CMAKE_BIN" \
  -S "$PROBE_DIR" \
  -B "$CMAKE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DBEND_GENERATED_C="$GENERATED"

"$CMAKE_BIN" --build "$CMAKE_BUILD" --config Release --parallel 2

BINARY="$CMAKE_BUILD/deepfin_bend_cuda_parity"
test -x "$BINARY"
echo "$BINARY"
