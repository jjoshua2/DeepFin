#!/usr/bin/env bash
# Opt-in CPU qualification. Python/CBoard are EXTERNAL test oracles only.
# No checkpoint, Torch install, model export, deeper perft, or production change.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../../.." && pwd)"
output="${1:?usage: qualify_toolchain.sh NEW_OUTPUT_DIRECTORY PINNED_COMPILER_DIRECTORY}"
compiler="${2:?provide the pinned compiler checkout}"
bun="${BUN:-bun}"
cc="${CC:-clang}"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
compiler="$(cd -- "$compiler" && pwd)"
mkdir -p -- "$output"
output="$(cd -- "$output" && pwd)"
cd -- "$repo"
export BEND_NO_TELEMETRY=1
"$bun" "$here/verify_compiler.js" "$compiler"
"$bun" test "$here/verify_compiler.test.js"

# Build just the existing independent CBoard reference extension, not Torch or
# the training package. Match setup.py's actual slider macros and Python ABI.
python - <<'PY'
import ast
from pathlib import Path
import subprocess
import sysconfig
import numpy
module = ast.parse(Path('setup.py').read_text())
macros = next(ast.literal_eval(n.value) for n in module.body
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
              and n.target.id == '_CBOARD_FAST_SLIDER_MACROS')
subprocess.run(['cc', '-shared', '-fPIC', '-O1', '-I' + sysconfig.get_paths()['include'],
                '-I' + numpy.get_include(), *[f'-D{k}={v}' for k, v in macros],
                'chess_anti_engine/encoding/_lc0_ext.c', '-pthread', '-lm', '-o',
                'chess_anti_engine/encoding/_lc0_ext' + sysconfig.get_config_var('EXT_SUFFIX')], check=True)
PY
bash "$here/build.sh" "$output/engine" "$compiler" generic
cp "$output/engine/build.txt" "$output/build.txt"
"$bun" "$compiler/bend2/main.ts" "$here/table_dump.bend" -o "$output/table.c"
"$cc" -std=c11 -O1 -I "$repo" "$here/table_reference.c" "$here/../legal_probe/support.c" \
  -pthread -lm -o "$output/table-reference"
"$output/table-reference" > "$output/table-reference.txt"
test "$(wc -l < "$output/table-reference.txt")" = 108160
flags=(-std=c11 -O1 -ffp-contract=off -Werror=shift-count-overflow)
qualify() {
  local mode="$1"
  shift
  python -m native.bend_engine.standalone.verify_encoding --require-c \
    --report "$output/encoding-$mode.json" --command "$@"
  python -m native.bend_engine.standalone.verify_rules \
    --report "$output/rules-$mode.json" --command "$@"
  python "$here/verify.py" --report "$output/engine-$mode.json" --command "$@"
}
for mode in generic portable native ubsan static; do
  extra=()
  case "$mode" in
    portable) extra=(-DBEND_U64_PORTABLE) ;;
    native) extra=(-march=native) ;;
    ubsan) extra=(-fsanitize=undefined -fno-sanitize-recover=all) ;;
    static) extra=(-static) ;;
  esac
  if [[ "$mode" == generic ]]; then
    binary="$output/engine/deepfin-bend"
  else
    binary="$output/engine/$mode"
    "$cc" "${flags[@]}" "${extra[@]}" "$output/engine/engine.c" -pthread -lm -o "$binary"
  fi
  "$cc" "${flags[@]}" "${extra[@]}" "$output/table.c" -pthread -lm -o "$output/table-$mode"
  "$output/table-$mode" --threads 1 > "$output/table-$mode.txt"
  cmp "$output/table-reference.txt" "$output/table-$mode.txt"
  if [[ "$mode" == static ]]; then
    if readelf -l "$binary" | grep -q INTERP; then echo 'Dynamic interpreter remains' >&2; exit 1; fi
    runtime="$output/empty-runtime"
    mkdir "$runtime"
    cp "$binary" "$runtime/deepfin-bend"
    test "$(find "$runtime" -mindepth 1 | wc -l)" = 1
    qualify "$mode" sudo chroot "$runtime" /deepfin-bend --threads 1
    find "$runtime" -mindepth 1 -printf '%P\n' > "$output/runtime-inventory.txt"
  else
    qualify "$mode" "$binary" --threads 1
  fi
  objdump -d "$binary" > "$output/instructions-$mode.txt"
done
# ISA claims are about emitted instructions, not an inferred compiler flag.
grep -E '\bpext\b' "$output/instructions-generic.txt" > "$output/pext-generic.txt"
grep -E '\bpext\b' "$output/instructions-native.txt" > "$output/pext-native.txt"
if grep -Eq '\b(pext|pdep)\b' "$output/instructions-portable.txt"; then
  echo 'BMI2 instruction leaked into forced-portable build' >&2; exit 1
fi
python - "$output" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
root = Path(sys.argv[1])
modes = ['generic', 'portable', 'native', 'ubsan', 'static']
expected_tensor = 'cb7a7e6688a73c5bfbf734769cd6c5f1d3a04609167bf019a4cbfa79d5eb6bbf'
for mode in modes:
    enc = json.loads((root / f'encoding-{mode}.json').read_text())
    eng = json.loads((root / f'engine-{mode}.json').read_text())
    assert enc['status'] == 'passed' and enc['tensor_comparisons_python'] == 505
    assert enc['tensor_comparisons_c'] == 501 and enc['c_clock_range_skips'] == 4
    assert enc['ordered_tensor_sha256'] == expected_tensor
    assert eng['perft_counts'] == [8902, 97862, 43238]
    assert eng['exact_children'] == 137 and eng['searched_roots'] == 51
    assert eng['rejected_transactions'] == 23
reference = (root / 'table-reference.txt').read_bytes()
assert all((root / f'table-{mode}.txt').read_bytes() == reference for mode in modes)
summary = {'status': 'PASS', 'modes': modes, 'table_entries_per_mode': 108160,
           'table_sha256': hashlib.sha256(reference).hexdigest(),
           'ordered_tensor_sha256': expected_tensor, 'encoding_python_per_mode': 505,
           'encoding_c_per_mode': 501, 'python_only_clock_cases_per_mode': 4,
           'unchanged_perft': [8902, 97862, 43238], 'static_runtime_files': ['deepfin-bend'],
           'proofs_are_separate': True, 'model_or_training_work': False}
(root / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
# Keep compact evidence separate; full generated files remain in this explicit
# local output directory, not in commits or routine uploaded reports.
mkdir "$output/reports"
cp "$output"/*.json "$output/build.txt" "$output/runtime-inventory.txt" \
   "$output/pext-generic.txt" "$output/pext-native.txt" "$output/reports/"
sha256sum "$output/engine/engine.c" "$output/engine/deepfin-bend" "$output/engine/static" \
  > "$output/reports/build-sha256.txt"
printf 'Qualification reports: %s/reports\n' "$output"
