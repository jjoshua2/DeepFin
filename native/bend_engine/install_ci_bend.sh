#!/usr/bin/env bash
# Reproducible PR compiler, independent of the moving upstream download service.
# Matches the currently qualified U64 fork, not upstream-only main. Updating this
# pin requires requalifying the native probes. Never reset an existing checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${BEND_PROBE_HOME:-$ROOT/build/bend_toolchain}"
REVISION=aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae
SOURCE_SHA256=d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4
SOURCE="$PREFIX/sources/$REVISION"
BUN_BIN="$(command -v bun || true)"
if [ -z "$BUN_BIN" ] && [ -x "$HOME/.bun/bin/bun" ]; then BUN_BIN="$HOME/.bun/bin/bun"; fi
if [ -z "$BUN_BIN" ]; then echo 'error: Bun is required for the source-pinned CI compiler' >&2; exit 2; fi
BUN_BIN="$(readlink -f "$BUN_BIN")"
mkdir -p "$PREFIX/sources"
if [ ! -e "$SOURCE" ]; then
  TEMP="$(mktemp -d "$PREFIX/sources/.download.XXXXXX")"
  trap 'rm -rf "$TEMP"' EXIT
  git -C "$TEMP" init -q
  git -C "$TEMP" fetch -q --depth=1 https://github.com/jjoshua2/bend.git "$REVISION"
  git -C "$TEMP" checkout -q --detach FETCH_HEAD
  test "$(git -C "$TEMP" rev-parse HEAD)" = "$REVISION"
  # -T prevents nesting if a concurrent installer created the destination.
  mv -T "$TEMP" "$SOURCE"
  trap - EXIT
fi
if [ -L "$SOURCE" ]; then echo 'error: CI compiler directory is a symlink' >&2; exit 2; fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$REVISION"
# Verify the same 84 compiler/Base/effect inputs as the standalone source gate.
# No dependency on the standalone directory: older PRs also run these probes.
SOURCE_DIR="$SOURCE" EXPECTED_SHA="$SOURCE_SHA256" "$BUN_BIN" -e '
import {createHash} from "node:crypto";
import {lstatSync, readFileSync, readdirSync} from "node:fs";
import {join} from "node:path";
const root = process.env.SOURCE_DIR;
function files(path) {
  const stat = lstatSync(join(root, path));
  if (stat.isSymbolicLink()) throw new Error("unexpected compiler symlink: " + path);
  if (stat.isFile()) return [path];
  if (!stat.isDirectory()) throw new Error("unexpected compiler input: " + path);
  return readdirSync(join(root, path)).flatMap(name => files(path + "/" + name));
}
const paths = ["bend2/base.bend", "bend2/bend.ts", "bend2/comp.ts", "bend2/main.ts", "bend2/effs"].flatMap(files).sort();
const hash = createHash("sha256");
for (const path of paths) {
  hash.update(path + "\0");
  hash.update(createHash("sha256").update(readFileSync(join(root, path))).digest());
}
const actual = hash.digest("hex");
if (paths.length !== 84 || actual !== process.env.EXPECTED_SHA) {
  throw new Error("CI compiler source mismatch; preserve this checkout and use a fresh BEND_PROBE_HOME");
}
console.log("Verified CI compiler: 84 inputs, sha256=" + actual);
'
GOT_VERSION="$(BEND_NO_TELEMETRY=1 "$BUN_BIN" "$SOURCE/bend2/main.ts" version)"
if [ "$GOT_VERSION" != 'bend 2.0.21' ]; then echo "error: CI compiler version mismatch: $GOT_VERSION" >&2; exit 2; fi
mkdir -p "$PREFIX/bin"
WRAPPER="$(mktemp "$PREFIX/bin/.bend.XXXXXX")"
trap 'rm -f "$WRAPPER"' EXIT
printf '#!/usr/bin/env bash\nexport BEND_NO_TELEMETRY="${BEND_NO_TELEMETRY:-1}"\nexec %q %q "$@"\n' "$BUN_BIN" "$(readlink -f "$SOURCE/bend2/main.ts")" > "$WRAPPER"
chmod +x "$WRAPPER"
mv -f "$WRAPPER" "$PREFIX/bin/bend"
trap - EXIT
if [ -n "${GITHUB_PATH:-}" ]; then echo "$PREFIX/bin" >> "$GITHUB_PATH"; fi
echo "bend probe: installed $GOT_VERSION + U64 ($REVISION) at $PREFIX/bin/bend"
