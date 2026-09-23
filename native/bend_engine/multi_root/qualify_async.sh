#!/usr/bin/env bash
# Opt-in complete coordinator matrix; never invoked from ordinary pytest.
set -euo pipefail
if [[ $# != 3 ]]; then echo 'usage: qualify_async.sh GENERATED.c ORACLE NEW_OUTPUT' >&2; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source="$(realpath "$1")"; oracle="$(realpath "$2")"; out="$3"
[[ ! -e "$out" ]]; mkdir -p "$out"; out="$(cd -- "$out" && pwd)"
cc="${CC:-clang}"; cxx="${CXX:-clang++}"
for mode in normal sanitized; do
  flags=(); [[ "$mode" == sanitized ]] && flags=(-fsanitize=address,undefined -fno-sanitize-recover=all)
  "$cxx" -std=c++20 -O1 -pthread -Wall -Wextra -Werror "${flags[@]}" \
    "$here/../batch_backend/async_batch_test.cpp" -o "$out/worker-$mode"
  "$out/worker-$mode" | tee "$out/worker-$mode.json"
done
for channels in 146 175; do
  for batch in 1 2 4 8 16; do
    dir="$out/c${channels}-b${batch}"; mkdir "$dir"
    profile=2; [[ "$channels" == 175 ]] && profile=4
    printf '#define DEEPFIN_MODEL_BATCH %s\n#define DEEPFIN_MODEL_CHANNELS %s\n#define DEEPFIN_MODEL_PROFILE %s\n' "$batch" "$channels" "$profile" > "$dir/model_contract.h"
    "$cc" -std=c11 -O1 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$dir" -c "$source" -o "$dir/runner.o"
    "$cc" -std=c11 -O1 -ffp-contract=off -I "$dir" -c "$here/test_backend.c" -o "$dir/callback.o"
    "$cxx" -std=c++20 -O1 -ffp-contract=off -I "$dir" "$dir/runner.o" "$dir/callback.o" \
      "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$dir/runner"
    for mode in sync async; do
      flags=(); [[ "$mode" == async ]] && flags=(--asynchronous)
      python -m native.bend_engine.multi_root.verify --binary "$dir/runner" --oracle "$oracle" \
        --batch "$batch" --channels "$channels" "${flags[@]}" --report "$dir/$mode.json"
    done
  done
  dir="$out/c${channels}-b4"
  "$cc" -std=c11 -O1 -ffp-contract=off -I "$dir" -c "$here/control_backend.c" -o "$dir/control.o"
  "$cxx" -std=c++20 -O1 -ffp-contract=off -I "$dir" "$dir/runner.o" "$dir/control.o" \
    "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$dir/gate"
  python -m native.bend_engine.multi_root.verify_control --binary "$dir/gate" --reference "$dir/runner" --report "$dir/control.json"
  flags=(-fsanitize=undefined -fno-sanitize-recover=all)
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" -DDEEPFIN_BEND_NATIVE_MODEL -DDEEPFIN_ASYNC_BATCH -I "$dir" -c "$source" -o "$dir/ubsan.o"
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" -I "$dir" -c "$here/control_backend.c" -o "$dir/control-ubsan.o"
  "$cc" -std=c11 -O1 -ffp-contract=off "${flags[@]}" -I "$dir" -c "$here/test_backend.c" -o "$dir/callback-ubsan.o"
  for variant in control callback; do
    "$cxx" -std=c++20 -O1 -ffp-contract=off "${flags[@]}" -I "$dir" "$dir/ubsan.o" "$dir/$variant-ubsan.o" \
      "$here/../batch_backend/async_batch.cpp" -pthread -lm -o "$dir/$variant-ubsan"
  done
  python -m native.bend_engine.multi_root.verify_control --binary "$dir/control-ubsan" --reference "$dir/callback-ubsan" --report "$dir/control-ubsan.json"
  python -m native.bend_engine.multi_root.verify --binary "$dir/callback-ubsan" --oracle "$oracle" \
    --batch 4 --channels "$channels" --asynchronous --report "$dir/ubsan.json"
done
python - "$out" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
for channels in (146,175):
    baseline=json.loads((out/f'c{channels}-b1/sync.json').read_text())
    for batch in (1,2,4,8,16):
        for mode in ('sync','async'):
            candidate=json.loads((out/f'c{channels}-b{batch}/{mode}.json').read_text())
            for ep,r in baseline['root_results'].items():
                assert all(candidate['root_results'][ep][k]==v for k,v in r.items())
            assert candidate['complete_tree_sha256']==baseline['complete_tree_sha256']
    candidate=json.loads((out/f'c{channels}-b4/ubsan.json').read_text())
    assert candidate['complete_tree_sha256']==baseline['complete_tree_sha256']
(out/'matrix.json').write_text(json.dumps({'status':'passed','normal_configurations':10,
    'modes_each':2,'ubsan_configurations':2,'control_configurations':4,
    'all_final_tree_bits_match_serial':True})+'\n')
PY
