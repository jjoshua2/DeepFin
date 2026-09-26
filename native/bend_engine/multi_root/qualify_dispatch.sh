#!/usr/bin/env bash
# Reuse THIS source build from qualify_live.sh; never a historical binary.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: VERIFIED_COMPILER CURRENT_LIVE_BUILD NEW_OUTPUT' >&2; exit 2; }
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler="$(cd -- "$1" && pwd)"; built="$(cd -- "$2" && pwd)"; out="$3"
[[ ! -e "$out" ]]
mkdir -p "$out"; out="$(cd -- "$out" && pwd)"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.json"
"$bun" "$compiler/bend2/main.ts" "$here/dispatch_probe.bend" -o "$out/probe.c" > "$out/probe-build.txt" 2>&1
for mode in normal ubsan; do
  flags=(); [[ "$mode" == ubsan ]] && flags=(-fsanitize=undefined -fno-sanitize-recover=all)
  "$cc" -std=c11 -O1 "${flags[@]}" "$out/probe.c" -pthread -lm -o "$out/probe-$mode"
  "$cxx" -std=c++20 -O1 "${flags[@]}" "$here/../standalone/dispatch_binding_test.cpp" -o "$out/binding-$mode"
  "$out/binding-$mode" > "$out/binding-$mode.json"
  python - "$out/probe-$mode" <<'PY'
import os, subprocess, sys
binary=sys.argv[1]
clean={k:v for k,v in os.environ.items() if not k.startswith('DEEPFIN_')}
def run(value):
    env=dict(clean)
    if value is not None: env['DEEPFIN_COHORT_DISPATCH']=value
    return subprocess.run([binary,'--threads','1'],env=env,text=True,capture_output=True,timeout=10,check=False)
for cap in (1,2,3,4):
    table=[min(n,cap) for n in range(1,17)]
    text=' '.join(map(str,[4,*table]))
    r=run(text)
    assert r.returncode==0 and not r.stderr,(text,r.stderr)
    assert r.stdout.splitlines()[1:]==[f'{n} {4 if n==0 else table[n-1]}' for n in range(17)]
for text in ('', '4', '3 '+'1 '*16, '8 '+'1 '*16, '4 '+'0 '*16, '4 '+'5 '*16,
             '4 '+'1 '*15, '4 '+'1 '*17, '4 true '+'1 '*15, '4 -1 '+'1 '*15,
             '4 2 '+'1 '*15, ' '*65, '4 4294967296 '+'1 '*15):
    r=run(text)
    assert r.returncode==2 and not r.stdout,(text,r.stdout,r.stderr)
r=run(None)
assert r.returncode==0 and r.stdout.splitlines()==[f'{n} 4' for n in range(17)]
PY
  for channels in 146 175; do
    d="$built/$mode-c$channels-b4"
    python -m native.bend_engine.multi_root.verify_dispatch --gate "$d/live-control" --reference "$d/reference" --report "$out/$mode-$channels.json"
  done
done
printf '{"status":"passed","model_qualified":false,"speed_qualified":false}\n' > "$out/status.json"
