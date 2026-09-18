#!/usr/bin/env bash
# Install ONLY this experiment's pinned fork; never replace the release toolchain.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PREFIX="${BEND_U64_HOME:-$ROOT/build/bend_u64_toolchain}"
PYTHON="${PYTHON:-python3}"

read -r REPOSITORY REVISION <<EOF_PIN
$("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["repository"], p["revision"])' "$HERE/toolchain.json")
EOF_PIN
mkdir -p "$PREFIX"
if [ ! -d "$PREFIX/source" ]; then
  TEMP="$(mktemp -d "$PREFIX/.download.XXXXXX")"
  trap 'rm -rf "$TEMP"' EXIT
  git -C "$TEMP" init -q
  git -C "$TEMP" fetch -q --depth=1 "$REPOSITORY" "$REVISION"
  git -C "$TEMP" checkout -q --detach FETCH_HEAD
  test "$(git -C "$TEMP" rev-parse HEAD)" = "$REVISION"
  # Refuse a concurrent install rather than nesting/overwriting an existing tree.
  "$PYTHON" - "$TEMP" "$PREFIX/source" <<'PY'
from pathlib import Path
import sys
source, target = map(Path, sys.argv[1:])
if target.exists():
    raise SystemExit("toolchain destination appeared during install; retry")
source.rename(target)
PY
fi
# Recheck contents on every invocation, including installations restored from cache.
"$PYTHON" - "$HERE" "$PREFIX/source" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from run_probe import check_compiler
pin = check_compiler(Path(sys.argv[2]))
print("Verified U64 compiler", pin["revision"], "at", sys.argv[2])
PY
