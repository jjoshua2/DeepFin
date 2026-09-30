#!/usr/bin/env bash
# Rebuild exact qualified native outputs, not a patched/reconstructed control.
set -euo pipefail
if [[ $# != 3 ]]; then echo 'usage: build_fifo_benchmark.sh PARENT.c FIFO.c NEW_OUTPUT' >&2; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
reference="$(realpath "$1")"; candidate="$(realpath "$2")"
[[ ! -e "$3" ]]; mkdir -p "$3"; out="$(cd -- "$3" && pwd)"
cc="${CC:-clang}"; cxx="${CXX:-clang++}"
echo "fcb787241c29f10c0fa33fbd8983f9630d3ab52914c109de2449b52a93251a24  $reference" | sha256sum -c -
echo "7f348a6ad58c7e3abebbf876f4ada8862c9f6dbc5856f445e1e2310346d68b74  $candidate" | sha256sum -c -
echo "062611f98a7661ebebef82b31f1c2d326a2cf1f3376ebc103fab59cbccc67033  $here/test_backend.c" | sha256sum -c -
echo "9c4626e56640a0d55ce6f3b6928c39ec8443c550d95f1ff217f179944fb83828  $here/../batch_backend/async_batch.cpp" | sha256sum -c -
printf '#define DEEPFIN_MODEL_BATCH 4\n#define DEEPFIN_MODEL_CHANNELS 146\n#define DEEPFIN_MODEL_PROFILE 2\n' > "$out/model_contract.h"
"$cc" --version > "$out/compiler.txt"
"$cxx" --version >> "$out/compiler.txt"
"$cc" -std=c11 -O3 -ffp-contract=off -I "$out" -c "$here/test_backend.c" -o "$out/callback.o"
for arm in list fifo; do
  source="$reference"; [[ "$arm" == fifo ]] && source="$candidate"
  "$cc" -std=c11 -O3 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$out" -c "$source" -o "$out/$arm.o"
  "$cxx" -std=c++20 -O3 -ffp-contract=off -I "$out" "$out/$arm.o" "$out/callback.o" \
    "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$out/$arm"
done
python - "$reference" "$candidate" "$out" "$here" <<'PY'
import hashlib, json, sys
from pathlib import Path
reference, candidate, out, here = map(Path, sys.argv[1:])
files = {'reference_c':reference,'candidate_c':candidate,'list_binary':out/'list','fifo_binary':out/'fifo',
         'callback':here/'test_backend.c','worker':here/'../batch_backend/async_batch.cpp',
         'model_contract':out/'model_contract.h','benchmark':here/'benchmark_fifo.py',
         'builder':here/'build_fifo_benchmark.sh','parser':here/'verify.py','equivalence':here/'verify_fifo.py'}
report = {'sha256':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()},
          'compiler':(out/'compiler.txt').read_text(), 'optimization':'-O3 -ffp-contract=off',
          'batch':4,'channels':146,'model':'unchanged deterministic callback, not a neural model'}
(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
PY
