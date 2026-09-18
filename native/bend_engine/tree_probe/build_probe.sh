#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBE_DIR="$ROOT/native/bend_engine/tree_probe"
BUILD_DIR="${BEND_TREE_PROBE_BUILD_DIR:-$ROOT/build/bend_tree_probe}"
BEND_BIN="${BEND_BIN:-bend}"
CC_BIN="${BEND_TREE_CC:-clang}"

command -v "$BEND_BIN" >/dev/null 2>&1 || {
  echo "error: Bend compiler not found: $BEND_BIN" >&2
  exit 2
}
command -v "$CC_BIN" >/dev/null 2>&1 || {
  echo "error: C compiler not found: $CC_BIN" >&2
  exit 2
}

export BEND_NO_TELEMETRY="${BEND_NO_TELEMETRY:-1}"

mkdir -p "$BUILD_DIR"
GENERATED="$BUILD_DIR/tree_probe.generated.c"
BINARY="$BUILD_DIR/deepfin_bend_tree_probe"

echo "bend tree probe: $("$BEND_BIN" --version 2>/dev/null || echo unknown)"
"$BEND_BIN" "$PROBE_DIR/main.bend" -o "$GENERATED"

"$CC_BIN"   -std=c11   -O3   -I"$ROOT"   "$GENERATED"   "$ROOT/native/bend_engine/probe/chess_bridge.c"   -lpthread   -lm   -o "$BINARY"

echo "$BINARY"
