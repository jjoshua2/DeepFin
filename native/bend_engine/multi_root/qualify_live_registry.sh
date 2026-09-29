#!/usr/bin/env bash
# Actual small Bend component checks. This does NOT qualify the live runner.
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: qualify_live_registry.sh VERIFIED_COMPILER NEW_OUTPUT' >&2; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compiler="$(cd -- "$1" && pwd)"; out="$2"; bun="${BUN:-bun}"; cc="${CC:-clang}"
[[ ! -e "$out" ]]; mkdir -p -- "$(dirname -- "$out")"; mkdir -- "$out"
out="$(cd -- "$out" && pwd)"
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.json"
for variant in baseline ignore-generation retain-expired-timer; do
  dir="$out/$variant"; mkdir "$dir"
  cp "$here/LiveRegistry.bend" "$here/Deadlines.bend" "$here/live_registry_probe.bend" "$dir/"
  python - "$dir" "$variant" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'LiveRegistry.bend'
s = p.read_text()
changes = {'ignore-generation': ('U32.is_eq(gen,generation(slot,book))', 'True{}'),
           'retain-expired-timer': ('D.Timers{limits_without(slot,limits),clear(expired,slot)}', 'D.Timers{limits_without(slot,limits),expired}')}
if sys.argv[2] in changes:
    old, new = changes[sys.argv[2]]
    assert s.count(old) == 1
    p.write_text(s.replace(old,new))
PY
  BEND_NO_TELEMETRY=1 "$bun" "$compiler/bend2/main.ts" "$dir/live_registry_probe.bend" -o "$dir/probe.c" > "$dir/build.txt" 2>&1
  for mode in normal ubsan; do
    flags=(); [[ "$mode" == ubsan ]] && flags=(-fsanitize=undefined -fno-sanitize-recover=all)
    "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" "$dir/probe.c" -pthread -lm -o "$dir/$mode"
    rc=0
    "$dir/$mode" --threads 1 > "$dir/$mode.txt" 2> "$dir/$mode.err" || rc=$?
    if [[ "$variant" == baseline ]]; then
      test "$rc" = 0; test ! -s "$dir/$mode.err"
      grep -Fxq 'live registry: capacity, stale identity, 128 replacements, exhaustion and timer clearing passed' "$dir/$mode.txt"
    else
      test "$rc" = 2; test ! -s "$dir/$mode.txt"
      grep -Fxq 'live registry invariant failed' "$dir/$mode.err"
    fi
  done
done
python - "$out" <<'PY'
import hashlib, json, sys
from pathlib import Path
out=Path(sys.argv[1])
report={'status':'passed','scope':'compiled registry component only','live_runner_qualified':False,
        'baseline_modes':['normal','ubsan'],'replacement_cycles_per_mode':128,
        'compiled_executed_mutations':['ignore-generation','retain-expired-timer'],
        'mutation_modes_each':['normal','ubsan'],
        'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.bend','.c','.txt','.err')}}
(out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Registry normal/UBSan and both compiled mutation controls passed; full runner unqualified.')
PY
