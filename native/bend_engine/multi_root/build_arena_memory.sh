#!/usr/bin/env bash
# Rebuild the exact #874 program, not a re-generated or modified historical control.
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: build_arena_memory.sh QUALIFIED_ARENA.c NEW_OUTPUT' >&2; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$here/../../.." && pwd)"
source="$(realpath "$1")"
[[ ! -e "$2" ]]; mkdir -p "$2"; out="$(realpath "$2")"
cc="${CC:-clang}"; cxx="${CXX:-clang++}"
echo "8f9d1712af22a1ec40914ce2e53e60d5a3fde264b92c34a8561d4c3edb254b72  $source" | sha256sum -c -
echo "062611f98a7661ebebef82b31f1c2d326a2cf1f3376ebc103fab59cbccc67033  $here/test_backend.c" | sha256sum -c -
echo "17adb8be00f86c532b1029a074824c03b3fece5d5a5a1f481029b726f4401029  $here/../batch_backend/async_batch.cpp" | sha256sum -c -
echo "79ee5ac5e22040065ed82f82157e7eaf166d163ce640af4768480ee110955f6e  $here/../batch_backend/async_batch.h" | sha256sum -c -
printf '#define DEEPFIN_MODEL_BATCH 4\n#define DEEPFIN_MODEL_CHANNELS 146\n#define DEEPFIN_MODEL_PROFILE 2\n' > "$out/model_contract.h"
"$cc" --version > "$out/compiler.txt"; "$cxx" --version >> "$out/compiler.txt"
"$cc" -std=c11 -O3 -ffp-contract=off -I "$out" -c "$here/test_backend.c" -o "$out/callback.o"
"$cc" -std=c11 -O3 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$out" -c "$source" -o "$out/runner.o"
"$cxx" -std=c++20 -O3 -ffp-contract=off -I "$out" "$out/runner.o" "$out/callback.o" \
  "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$out/runner"
"$cc" -std=c11 -O3 -DLEGAL_ORACLE -I "$root" "$here/../legal_probe/support.c" -pthread -lm -o "$out/oracle"
python - "$source" "$out" "$here" <<'PY'
from pathlib import Path
import hashlib, json, sys
source, out, here = map(Path, sys.argv[1:])
files = {'generated_c':source, 'binary':out/'runner', 'oracle':out/'oracle',
         'callback':here/'test_backend.c', 'worker':here/'../batch_backend/async_batch.cpp',
         'worker_header':here/'../batch_backend/async_batch.h', 'contract':out/'model_contract.h',
         'benchmark':here/'benchmark_arena_memory.py', 'builder':here/'build_arena_memory.sh',
         'parser':here/'verify.py', 'semantic_view':here/'verify_fifo.py',
         'reference':here/'../session_probe/run_probe.py', 'oracle_source':here/'../legal_probe/support.c'}
report = {'status':'built', 'sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},
          'compiler':(out/'compiler.txt').read_text(), 'flags':'-O3 -ffp-contract=off',
          'program_revision':'de48b98e1e3343033041f622407c0542ea550c03',
          'batch':4, 'channels':146, 'model':'deterministic callback; no neural model or GPU'}
(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
PY
