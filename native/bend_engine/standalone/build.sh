#!/usr/bin/env bash
# Build tools (Bun/Clang) are not deployed with the engine. No Python here.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../../.." && pwd)"
revision=d9b9bce9ce4c02583ca1a548cfd239b368c3fc77
output="${1:-$repo/build/bend_standalone}"
compiler="${2:-$repo/build/bend_standalone_toolchain/source}"
mode="${3:-generic}"
bun="${BUN:-bun}"
cc="${CC:-clang}"
flags=(-O1 -ffp-contract=off -Werror=shift-count-overflow)
case "$mode" in
  generic) ;;
  native) flags+=(-march=native) ;;
  portable) flags+=(-DBEND_U64_PORTABLE) ;;
  ubsan) flags+=(-fsanitize=undefined -fno-sanitize-recover=all) ;;
  static) flags+=(-static) ;;
  *) echo "mode must be generic, native, portable, ubsan or static" >&2; exit 2 ;;
esac
command -v "$bun" >/dev/null
command -v "$cc" >/dev/null
if [[ -e "$output" ]]; then
  echo "Output directory already exists; use a new path: $output" >&2
  exit 2
fi
if [[ ! -d "$compiler" ]]; then
  mkdir -p -- "$(dirname -- "$compiler")"
  git init "$compiler"
  git -C "$compiler" remote add origin https://github.com/jjoshua2/bend.git
  git -C "$compiler" fetch --depth=1 origin "$revision"
  git -C "$compiler" checkout --detach FETCH_HEAD
fi
compiler="$(cd -- "$compiler" && pwd)"
# Includes the full effect-file list; extra effect files change the fingerprint.
"$bun" "$here/verify_compiler.js" "$compiler"
mkdir -p -- "$output"
output="$(cd -- "$output" && pwd)"
"$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$output/engine.c"
"$cc" -std=c11 "${flags[@]}" "$output/engine.c" -pthread -lm -o "$output/deepfin-bend"
{
  printf 'compiler_revision=%s\nmode=%s\n' "$revision" "$mode"
  printf 'bun='; "$bun" --version
  "$cc" --version
  printf 'flags='; printf '%q ' -std=c11 "${flags[@]}" -pthread -lm; printf '\n'
  sha256sum "$output/engine.c" "$output/deepfin-bend"
} > "$output/build.txt"
printf 'Run: %q --threads 1\n' "$output/deepfin-bend"
