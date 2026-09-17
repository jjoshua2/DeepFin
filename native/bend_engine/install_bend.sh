#!/usr/bin/env bash
# Install the current Bend release from https://bend-lang.com/dl/latest.json.
# The parity probe is the compatibility gate: Bend is young and latest moves
# quickly, so this does not pin a patch version. The sha256 in latest.json is
# still checked so a truncated download cannot silently qualify.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${BEND_PROBE_HOME:-$ROOT/build/bend_toolchain}"
LATEST_URL="${BEND_LATEST_URL:-https://bend-lang.com/dl/latest.json}"

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    command -v bun
    return
  fi
  if [ -x "$HOME/.bun/bin/bun" ]; then
    echo "$HOME/.bun/bin/bun"
    return
  fi
  echo "error: bun is required to run Bend" >&2
  echo "install it with: curl -fsSL https://bun.sh/install | bash" >&2
  exit 2
}

BUN_BIN="$(ensure_bun)"

echo "bend probe: fetching $LATEST_URL"
META="$(curl -fsSL "$LATEST_URL")"
read -r WANT_VERSION WANT_SHA WANT_URL <<EOF
$(printf '%s' "$META" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ver"], d["sha256"], d["url"])')
EOF

case "$WANT_VERSION" in
  *[!0-9A-Za-z._-]*|.|..|"")
    echo "error: invalid Bend version in latest.json: $WANT_VERSION" >&2
    exit 2
    ;;
esac
case "$WANT_SHA" in
  *[!0-9a-f]*|"")
    echo "error: invalid Bend sha256 in latest.json" >&2
    exit 2
    ;;
esac
test "${#WANT_SHA}" -eq 64
case "$WANT_URL" in
  https://bend-lang.com/dl/*) ;;
  *)
    echo "error: unexpected Bend tarball URL: $WANT_URL" >&2
    exit 2
    ;;
esac

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
