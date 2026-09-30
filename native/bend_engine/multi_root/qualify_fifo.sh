#!/usr/bin/env bash
# Opt-in deterministic scheduler qualification. No model, GPU or deployment.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: qualify_fifo.sh VERIFIED_COMPILER PINNED_PARENT.c NEW_OUTPUT' >&2
  exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$here/../../.." && pwd)"
compiler="$(realpath "$1")"; parent="$(realpath "$2")"; out="$3"
out="$(realpath -m "$out")"
case "$out/" in "$root/native/bend_engine/"*)
  echo 'qualification output must be outside the native source tree' >&2; exit 2;;
esac
[[ ! -e "$out" ]]; mkdir -p "$out"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
export BUN="$bun" CC="$cc" CXX="$cxx"
cd "$root"
# Exact parent #863 generated C, retained in artifact 10775514198. Refuse an
# accidental candidate-as-control comparison or different compiler output.
echo "fcb787241c29f10c0fa33fbd8983f9630d3ab52914c109de2449b52a93251a24  $parent" | sha256sum -c -
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.txt"
"$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$out/runner.c" 2> "$out/generation.log"
"$cc" -std=c11 -O3 -DLEGAL_ORACLE -I "$root" "$here/../legal_probe/support.c" -pthread -lm -o "$out/oracle"
bash "$here/qualify_async.sh" "$out/runner.c" "$out/oracle" "$out/matrix"
bash "$here/qualify_deadlines.sh" "$compiler" "$out/matrix" "$out/deadlines"
"$bun" "$compiler/bend2/main.ts" "$here/fifo_probe.bend" -o "$out/fifo-probe.c"
for mode in normal ubsan; do
  flags=(); [[ "$mode" == ubsan ]] && flags=(-fsanitize=undefined -fno-sanitize-recover=all)
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" "$out/fifo-probe.c" -pthread -lm -o "$out/fifo-probe-$mode"
  "$out/fifo-probe-$mode" --threads 1 > "$out/fifo-probe-$mode.txt"
  python - "$out/fifo-probe-$mode.txt" <<'PY'
import sys
from pathlib import Path
from native.bend_engine.multi_root.verify_fifo import check_probe
check_probe(Path(sys.argv[1]).read_text())
PY
done
# Reverse requeue order in a disposable copy of the actual adapters. The
# mutant must compile and run normally, then fail the unchanged order oracle.
python - "$here/.." "$out/mutant" <<'PY'
from pathlib import Path
import shutil, sys
src, dest = map(Path, sys.argv[1:])
for path in src.rglob('*.bend'):
    target = dest / path.relative_to(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
p = dest/'multi_root/Cohort.bend'
s = p.read_text()
old = 'requeue(rest, Q.push(&1, Root, q, root))'
assert s.count(old) == 1
p.write_text(s.replace(old, 'Q.push(&1, Root, requeue(rest, q), root)'))
PY
"$bun" "$compiler/bend2/main.ts" "$out/mutant/multi_root/fifo_probe.bend" -o "$out/mutant.c"
"$cc" -std=c11 -O1 -ffp-contract=off "$out/mutant.c" -pthread -lm -o "$out/mutant-probe"
"$out/mutant-probe" --threads 1 > "$out/mutant.txt"
python - "$out/mutant.txt" <<'PY'
import sys
from pathlib import Path
from native.bend_engine.multi_root.verify_fifo import check_probe
try:
    check_probe(Path(sys.argv[1]).read_text())
except ValueError:
    print('reversed-requeue mutation rejected')
else:
    raise AssertionError('reversed-requeue mutation survived')
PY
for channels in 146 175; do
  for batch in 1 2 4 8 16; do
    dir="$out/matrix/c${channels}-b${batch}"
    "$cc" -std=c11 -O1 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$dir" -c "$parent" -o "$dir/parent.o"
    "$cxx" -std=c++20 -O1 -ffp-contract=off -I "$dir" "$dir/parent.o" "$dir/callback.o" \
      "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$dir/parent"
    python -m native.bend_engine.multi_root.verify_fifo --reference "$dir/parent" --candidate "$dir/runner" \
      --batch "$batch" --report "$dir/fifo-equivalence.json"
    if [[ "$batch" == 4 ]]; then
      python -m native.bend_engine.multi_root.verify_fifo --reference "$dir/parent" --candidate "$dir/callback-ubsan" \
        --batch "$batch" --report "$dir/fifo-equivalence-ubsan.json"
    fi
  done
done
python - "$out" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
reports = sorted((out/'matrix').glob('*/fifo-equivalence*.json'))
assert len(reports) == 12
pairs = 0
for path in reports:
    r = json.loads(path.read_text())
    assert r['status'] == 'passed' and len(r['observations']) == 10
    pairs += len(r['observations'])
summary = {'status': 'passed', 'scheduler_equivalence_pairs': pairs,
           'normal_configurations': 10, 'ubsan_configurations': 2,
           'modes_each': ['sync','async'], 'cases_each': 5,
           'adapter_cases_per_mode': 120, 'adapter_rows_per_mode': 840,
           'reversed_requeue_mutation_rejected': True,
           'candidate_c_sha256': hashlib.sha256((out/'runner.c').read_bytes()).hexdigest(),
           'parent_c_sha256': 'fcb787241c29f10c0fa33fbd8983f9630d3ab52914c109de2449b52a93251a24',
           'scope': 'deterministic coordinator equivalence; no model, speed or strength claim'}
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
PY
