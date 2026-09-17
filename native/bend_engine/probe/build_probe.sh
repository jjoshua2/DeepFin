#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBE_DIR="$ROOT/native/bend_engine/probe"
BUILD_DIR="${BEND_PROBE_BUILD_DIR:-$ROOT/build/bend_chess_probe}"
BEND_BIN="${BEND_BIN:-bend}"
CC_BIN="${CC:-clang}"

command -v "$BEND_BIN" >/dev/null 2>&1 || {
  echo "error: Bend compiler not found: $BEND_BIN" >&2
  exit 2
}
command -v "$CC_BIN" >/dev/null 2>&1 || {
  echo "error: C compiler not found: $CC_BIN" >&2
  exit 2
}

export BEND_NO_TELEMETRY="${BEND_NO_TELEMETRY:-1}"
WANT_VERSION="$(sed -n '1p' "$ROOT/native/bend_engine/BEND_VERSION")"
GOT_VERSION="$("$BEND_BIN" --version 2>/dev/null || true)"
case "$GOT_VERSION" in
  *"$WANT_VERSION"*) ;;
  *)
    echo "error: Bend $WANT_VERSION required, got: ${GOT_VERSION:-unknown}" >&2
    exit 2
    ;;
esac

mkdir -p "$BUILD_DIR"
GENERATED="$BUILD_DIR/probe.generated.c"
BINARY="$BUILD_DIR/deepfin_bend_chess_probe"

echo "bend probe: $("$BEND_BIN" --version 2>/dev/null || echo unknown)"
echo "bend probe: generating C -> $GENERATED"
"$BEND_BIN" "$PROBE_DIR/main.bend" -o "$GENERATED"

echo "bend probe: linking generated Bend C + DeepFin CBoard bridge -> $BINARY"
"$CC_BIN"   -std=c11   -O3   -I"$ROOT"   "$GENERATED"   "$PROBE_DIR/chess_bridge.c"   -lpthread   -lm   -o "$BINARY"

echo "$BINARY"
