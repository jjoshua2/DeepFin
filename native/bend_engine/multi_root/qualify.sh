#!/usr/bin/env bash
# Explicit normal/UBSan full cohort checks; never part of ordinary pytest.
set -euo pipefail
if [[ $# != 3 ]]; then echo 'usage: qualify.sh GENERATED_RUNNER.c ORACLE_BINARY NEW_OUTPUT' >&2; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source="$(realpath "$1")"; oracle="$(realpath "$2")"; out="$3"
[[ ! -e "$out" ]]; mkdir -p "$out"; out="$(cd -- "$out" && pwd)"
cc="${CC:-clang}"
for channels in 146 175; do
  for batch in 1 2 4 8 16; do
    dir="$out/c${channels}-b${batch}"; mkdir "$dir"
    profile=2; [[ "$channels" == 175 ]] && profile=4
    printf '#define DEEPFIN_MODEL_BATCH %s\n#define DEEPFIN_MODEL_CHANNELS %s\n#define DEEPFIN_MODEL_PROFILE %s\n' "$batch" "$channels" "$profile" > "$dir/model_contract.h"
    "$cc" -std=c11 -O1 -ffp-contract=off -DDEEPFIN_BEND_NATIVE_MODEL -I "$dir" \
      "$source" "$here/test_backend.c" -pthread -lm -o "$dir/runner"
    python -m native.bend_engine.multi_root.verify --binary "$dir/runner" --oracle "$oracle" \
      --batch "$batch" --channels "$channels" --report "$dir/report.json"
  done
  dir="$out/c${channels}-b4"
  "$cc" -std=c11 -O1 -ffp-contract=off -fsanitize=undefined -fno-sanitize-recover=all \
    -DDEEPFIN_BEND_NATIVE_MODEL -I "$dir" "$source" "$here/test_backend.c" -pthread -lm -o "$dir/ubsan"
  python -m native.bend_engine.multi_root.verify --binary "$dir/ubsan" --oracle "$oracle" \
    --batch 4 --channels "$channels" --report "$dir/ubsan.json"
done
python - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for channels in (146,175):
    baseline = json.loads((out/f'c{channels}-b1/report.json').read_text())
    for batch in (2,4,8,16):
        candidate = json.loads((out/f'c{channels}-b{batch}/report.json').read_text())
        assert candidate['root_results'] == baseline['root_results']
        assert candidate['complete_tree_sha256'] == baseline['complete_tree_sha256']
    instrumented = json.loads((out/f'c{channels}-b4/ubsan.json').read_text())
    assert instrumented['root_results'] == baseline['root_results']
    assert instrumented['complete_tree_sha256'] == baseline['complete_tree_sha256']
(out/'matrix.json').write_text(json.dumps({'status':'passed','normal_configurations':10,
    'ubsan_configurations':2,'all_final_tree_bits_match_serial':True})+'\n')
PY
