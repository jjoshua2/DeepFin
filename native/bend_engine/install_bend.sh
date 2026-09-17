#!/usr/bin/env bash
# Install the Bend version recorded in BEND_VERSION. The upstream installer
# always fetches latest; this script downloads the pinned tarball and wraps it
# so a later 2.0.x cannot silently qualify the probe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION_FILE="$ROOT/native/bend_engine/BEND_VERSION"
PREFIX="${BEND_PROBE_HOME:-$ROOT/build/bend_toolchain}"
WANT_VERSION="$(sed -n '1p' "$VERSION_FILE")"
WANT_SHA="$(sed -n 's/^sha256 //p' "$VERSION_FILE" | head -n1)"
WANT_URL="$(sed -n 's/^url //p' "$VERSION_FILE" | head -n1)"

test -n "$WANT_VERSION" && test -n "$WANT_SHA" && test -n "$WANT_URL"

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    command -v bun
    return
  fi
  if [ -x "$HOME/.bun/bin/bun" ]; then
    echo "$HOME/.bun/bin/bun"
    return
  fi
  echo "error: bun is required to run Bend $WANT_VERSION" >&2
  echo "install it with: curl -fsSL https://bun.sh/install | bash" >&2
  exit 2
}

BUN_BIN="$(ensure_bun)"

APP="$PREFIX/app/$WANT_VERSION"
BIN="$PREFIX/bin"
mkdir -p "$APP" "$BIN"
TGZ="$PREFIX/bend-$WANT_VERSION.tar.gz"

echo "bend probe: fetching $WANT_URL"
curl -fsSL -o "$TGZ" "$WANT_URL"
GOT_SHA="$(sha256sum "$TGZ" | awk '{print $1}')"
if [ "$GOT_SHA" != "$WANT_SHA" ]; then
  echo "error: Bend tarball sha256 mismatch" >&2
  echo "  want $WANT_SHA" >&2
  echo "  got  $GOT_SHA" >&2
  exit 2
fi

rm -rf "$APP/src"
mkdir -p "$APP/src"
tar -xzf "$TGZ" -C "$APP/src"
test -f "$APP/src/bend2/main.ts"
ln -sfn "app/$WANT_VERSION/src" "$PREFIX/current"

cat > "$BIN/bend" <<EOF
#!/bin/sh
export BEND_NO_TELEMETRY="\${BEND_NO_TELEMETRY:-1}"
exec "$(readlink -f "$BUN_BIN")" "$(readlink -f "$PREFIX/current/bend2/main.ts")" "\$@"
EOF
chmod +x "$BIN/bend"

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$BIN" >> "$GITHUB_PATH"
  echo "$(dirname "$BUN_BIN")" >> "$GITHUB_PATH"
fi

echo "bend probe: installed $("$BIN/bend" --version 2>/dev/null || echo unknown) at $BIN/bend"
echo "$BIN/bend"
