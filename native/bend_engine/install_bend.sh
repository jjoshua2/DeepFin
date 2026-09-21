#!/usr/bin/env bash
# Install the current Bend release named by https://bend-lang.com/dl/latest.json.
# The parity probe is the compatibility gate: Bend is young and latest moves
# quickly, so this does not pin a patch version. The archive is still
# sha256-checked. The old feed carried {ver, sha256, url}; the current feed is
# {ver, notice} only. Use the matching GitHub asset digest, or the official
# installer's matching version/platform checksum if the API is unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${BEND_PROBE_HOME:-$ROOT/build/bend_toolchain}"
LATEST_URL="${BEND_LATEST_URL:-https://bend-lang.com/dl/latest.json}"
RESOLVE="$ROOT/native/bend_engine/resolve_latest.py"

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    command -v bun
    return
  fi
  if [ -x "$HOME/.bun/bin/bun" ]; then
    echo "$HOME/.bun/bin/bun"
    return
  fi
  echo "error: bun is required to run a Bend source tarball" >&2
  echo "install it with: curl -fsSL https://bun.sh/install | bash" >&2
  exit 2
}

install_official_binary() {
  local src="$1"
  test -x "$src/bin/bend"
  test -f "$src/bend2/base.bend"
  chmod +x "$src/bin/bend"
  ln -sfn "app/$WANT_VERSION/src/bend" "$PREFIX/current"
  local official
  official="$(readlink -f "$src/bin/bend")"
  cat > "$BIN/bend" <<EOF
#!/bin/sh
export BEND_NO_TELEMETRY="\${BEND_NO_TELEMETRY:-1}"
exec "$official" "\$@"
EOF
}

install_source_wrapper() {
  local src="$1"
  test -f "$src/bend2/main.ts"
  local bun_bin
  bun_bin="$(ensure_bun)"
  cat > "$BIN/bend" <<EOF
#!/bin/sh
export BEND_NO_TELEMETRY="\${BEND_NO_TELEMETRY:-1}"
exec "$(readlink -f "$bun_bin")" "$(readlink -f "$src/bend2/main.ts")" "\$@"
EOF
}

echo "bend probe: fetching $LATEST_URL"
META="$(curl --proto '=https' --tlsv1.2 -fsSL "$LATEST_URL")"
RESOLVE_ARGS=()
if [ -n "${BEND_PLATFORM:-}" ]; then
  RESOLVE_ARGS+=(--platform "$BEND_PLATFORM")
fi
# Capture the resolver status directly; a read/heredoc masks its failure.
RESOLVED="$(printf '%s' "$META" | python3 "$RESOLVE" "${RESOLVE_ARGS[@]}")"
read -r WANT_VERSION WANT_SHA WANT_URL <<< "$RESOLVED"

case "$WANT_VERSION" in
  *[!0-9A-Za-z._-]*|.|..|"")
    echo "error: invalid Bend version in latest.json: $WANT_VERSION" >&2
    exit 2
    ;;
esac
case "$WANT_SHA" in
  *[!0-9a-f]*|"")
    echo "error: invalid Bend sha256" >&2
    exit 2
    ;;
esac
test "${#WANT_SHA}" -eq 64
GH_PREFIX="https://github.com/bendlang/bend/releases/download/v${WANT_VERSION}/bend-${WANT_VERSION}-"
case "$WANT_URL" in
  https://bend-lang.com/dl/*) ;;
  "${GH_PREFIX}linux-x64.tar.gz"|"${GH_PREFIX}linux-arm64.tar.gz"|"${GH_PREFIX}darwin-x64.tar.gz"|"${GH_PREFIX}darwin-arm64.tar.gz") ;;
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
curl --proto '=https' --tlsv1.2 -fsSL -o "$TGZ" "$WANT_URL"
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

if [ -x "$APP/src/bend/bin/bend" ] && [ -f "$APP/src/bend/bend2/base.bend" ]; then
  install_official_binary "$APP/src/bend"
elif [ -f "$APP/src/bend2/main.ts" ]; then
  ln -sfn "app/$WANT_VERSION/src" "$PREFIX/current"
  install_source_wrapper "$APP/src"
elif [ -f "$APP/src/bend/bend2/main.ts" ]; then
  ln -sfn "app/$WANT_VERSION/src/bend" "$PREFIX/current"
  install_source_wrapper "$APP/src/bend"
else
  echo "error: Bend tarball has neither official bin/bend nor bend2/main.ts" >&2
  exit 2
fi
chmod +x "$BIN/bend"

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$BIN" >> "$GITHUB_PATH"
fi

# Do not report a successful installation of a non-working executable.
GOT_VERSION="$("$BIN/bend" version)"
if [ "$GOT_VERSION" != "bend $WANT_VERSION" ]; then
  echo "error: installed Bend version mismatch: $GOT_VERSION" >&2
  exit 2
fi
echo "bend probe: installed $GOT_VERSION at $BIN/bend"
echo "$BIN/bend"
