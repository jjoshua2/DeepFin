#!/usr/bin/env bash
# Opt-in complete callback qualification. No model/GPU or live runtime access.
set -euo pipefail
if [[ $# != 4 ]]; then echo 'usage: qualify_wait.sh COMPILER PARENT.c PARENT_SOURCE NEW_OUTPUT' >&2; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$here/../../.." && pwd)"
compiler="$(realpath "$1")"; parent="$(realpath "$2")"; baseline="$(realpath "$3")"
[[ ! -e "$4" ]]; mkdir -p "$4"; out="$(realpath "$4")"
bun="${BUN:-bun}"; cc="${CC:-clang}"; cxx="${CXX:-clang++}"
export BUN="$bun" CC="$cc" CXX="$cxx"
cd "$root"
echo "7f348a6ad58c7e3abebbf876f4ada8862c9f6dbc5856f445e1e2310346d68b74  $parent" | sha256sum -c -
echo "2b3f060c99223c362ae5975291522ef739aee1070953cd68e187ba606a1eccff  $baseline/native/bend_engine/batch_backend/async_batch.h" | sha256sum -c -
echo "9c4626e56640a0d55ce6f3b6928c39ec8443c550d95f1ff217f179944fb83828  $baseline/native/bend_engine/batch_backend/async_batch.cpp" | sha256sum -c -
echo "062611f98a7661ebebef82b31f1c2d326a2cf1f3376ebc103fab59cbccc67033  $here/test_backend.c" | sha256sum -c -
"$bun" "$here/../standalone/verify_compiler.js" "$compiler" > "$out/compiler.txt"
for mode in normal sanitized; do
  flags=(); [[ "$mode" == sanitized ]] && flags=(-fsanitize=address,undefined -fno-sanitize-recover=all)
  "$cxx" -std=c++20 -O1 -pthread -Wall -Wextra -Werror "${flags[@]}" "$here/../batch_backend/async_wait_test.cpp" -o "$out/wait-$mode"
  timeout 30 "$out/wait-$mode" | tee "$out/wait-$mode.json"
done
mkdir "$out/mutant"
cp "$here/../batch_backend/async_wait_test.cpp" "$out/mutant/"
python - "$here/../batch_backend/async_batch.h" "$out/mutant/async_batch.h" <<'PY'
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text()
assert source.count('ready_.notify_one();')==1
Path(sys.argv[2]).write_text(source.replace('ready_.notify_one();','(void)0;'))
PY
"$cxx" -std=c++20 -O1 -pthread -Wall -Wextra -Werror "$out/mutant/async_wait_test.cpp" -o "$out/mutant/test"
set +e
timeout 30 "$out/mutant/test" > "$out/mutant.stdout" 2> "$out/mutant.stderr"
status=$?
set -e
[[ "$status" == 1 ]]; grep -Fx 'completion notification did not wake waiter' "$out/mutant.stderr"
"$bun" "$compiler/bend2/main.ts" "$here/main.bend" -o "$out/runner.c" 2> "$out/generation.log"
"$cc" -std=c11 -O3 -DLEGAL_ORACLE -I "$root" "$here/../legal_probe/support.c" -pthread -lm -o "$out/oracle"
bash "$here/qualify_async.sh" "$out/runner.c" "$out/oracle" "$out/matrix"
bash "$here/qualify_deadlines.sh" "$compiler" "$out/matrix" "$out/deadlines"
for channels in 146 175; do
  dir="$out/matrix/c${channels}-b4"
  "$cc" -std=c11 -O1 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$dir" -c "$parent" -o "$dir/parent.o"
  "$cxx" -std=c++20 -O1 -ffp-contract=off -I "$dir" "$dir/parent.o" "$dir/callback.o" "$baseline/native/bend_engine/batch_backend/async_batch.cpp" -pthread -lm -o "$dir/parent"
  python -m native.bend_engine.multi_root.verify_fifo --reference "$dir/parent" --candidate "$dir/runner" --batch 4 --report "$dir/wait-equivalence.json"
done
opt="$out/optimized"; mkdir "$opt"
cp "$out/matrix/c146-b4/model_contract.h" "$opt/"
"$cc" -std=c11 -O3 -ffp-contract=off -I "$opt" -c "$here/test_backend.c" -o "$opt/callback.o"
for arm in sleep notify; do
  source="$parent"; bridge="$baseline/native/bend_engine/batch_backend/async_batch.cpp"
  if [[ "$arm" == notify ]]; then source="$out/runner.c"; bridge="$here/../batch_backend/async_batch.cpp"; fi
  "$cc" -std=c11 -O3 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$opt" -c "$source" -o "$opt/$arm.o"
  "$cxx" -std=c++20 -O3 -ffp-contract=off -I "$opt" "$opt/$arm.o" "$opt/callback.o" "$bridge" -pthread -lm -o "$opt/$arm"
done
"$cc" --version > "$opt/compiler.txt"; "$cxx" --version >> "$opt/compiler.txt"
lscpu > "$opt/cpu.txt"
python -m native.bend_engine.multi_root.benchmark_wait --reference "$opt/sleep" --candidate "$opt/notify" --report "$out/timing.json"
python - "$out" "$parent" <<'PY'
from pathlib import Path
import hashlib,json,sys
out,parent=map(Path,sys.argv[1:])
assert json.loads((out/'matrix/matrix.json').read_text())['status']=='passed'
assert json.loads((out/'timing.json').read_text())['status']=='passed'
summary={'status':'passed','no_notify_mutation_rejected':True,'parent_c_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
 'candidate_c_sha256':hashlib.sha256((out/'runner.c').read_bytes()).hexdigest(),
 'normal_matrix_configurations':10,'ubsan_matrix_configurations':2,'equivalence_pairs':20,
 'native_wait_cases_per_mode':12,'native_wait_assertions_per_mode':326}
for path in out.glob('matrix/*/wait-equivalence.json'):
 report=json.loads(path.read_text());assert report['status']=='passed' and len(report['observations'])==10
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
