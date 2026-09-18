#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBE_DIR="$ROOT/native/bend_engine/aoti_probe"
BUILD_DIR="${BEND_AOTI_PROBE_BUILD_DIR:-$ROOT/build/bend_aoti_probe}"
BEND_BIN="${BEND_BIN:-bend}"
PYTHON_BIN="${PYTHON:-python}"
CMAKE_BIN="${CMAKE:-cmake}"
CC_BIN="${BEND_AOTI_CC:-clang}"
CXX_BIN="${BEND_AOTI_CXX:-clang++}"

command -v "$BEND_BIN" >/dev/null 2>&1 || {
  echo "error: Bend compiler not found: $BEND_BIN" >&2
  exit 2
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "error: Python not found for build-time Torch discovery: $PYTHON_BIN" >&2
  exit 2
}
command -v "$CMAKE_BIN" >/dev/null 2>&1 || {
  echo "error: CMake not found: $CMAKE_BIN" >&2
  exit 2
}
command -v "$CC_BIN" >/dev/null 2>&1 || {
  echo "error: C compiler not found: $CC_BIN" >&2
  exit 2
}
command -v "$CXX_BIN" >/dev/null 2>&1 || {
  echo "error: C++ compiler not found: $CXX_BIN" >&2
  exit 2
}

export BEND_NO_TELEMETRY="${BEND_NO_TELEMETRY:-1}"

mkdir -p "$BUILD_DIR"
GENERATED="$BUILD_DIR/aoti_probe.generated.c"
CMAKE_BUILD="$BUILD_DIR/cmake"

echo "bend AOTI probe: $("$BEND_BIN" --version 2>/dev/null || echo unknown)"
echo "bend AOTI probe: generating C -> $GENERATED"
"$BEND_BIN" "$PROBE_DIR/main.bend" -o "$GENERATED"

TORCH_PREFIX="$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

echo "bend AOTI probe: configuring LibTorch from $TORCH_PREFIX"
echo "bend AOTI probe: C=$CC_BIN CXX=$CXX_BIN"
"$CMAKE_BIN" \
  -S "$PROBE_DIR" \
  -B "$CMAKE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DBEND_GENERATED_C="$GENERATED"

"$CMAKE_BIN" --build "$CMAKE_BUILD" --config Release --parallel 2

BINARY="$CMAKE_BUILD/deepfin_bend_aoti_probe"
test -x "$BINARY"
echo "$BINARY"
