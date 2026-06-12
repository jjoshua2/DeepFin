#!/usr/bin/env bash
# Fuzzing entry point for the CBoard C implementation.
#
#   ./scripts/fuzz/run_fuzz.sh libfuzzer [seconds]   # coverage-guided C fuzz (clang)
#   ./scripts/fuzz/run_fuzz.sh diff [games]          # differential vs python-chess,
#                                                    # under ASAN/UBSAN-built extensions
#
# libFuzzer crashes land in scripts/fuzz/corpus/crash-*; reproduce with
#   ./scripts/fuzz/cboard_fuzz <crash-file>
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-libfuzzer}"

case "$MODE" in
  libfuzzer)
    SECONDS_BUDGET="${2:-120}"
    mkdir -p scripts/fuzz/corpus
    clang -g -O1 -fsanitize=fuzzer,address,undefined \
      -Ichess_anti_engine/encoding \
      scripts/fuzz/cboard_libfuzzer.c -lm \
      -o scripts/fuzz/cboard_fuzz
    ./scripts/fuzz/cboard_fuzz scripts/fuzz/corpus \
      -max_total_time="$SECONDS_BUDGET" -print_final_stats=1
    ;;
  diff)
    GAMES="${2:-500}"
    CAE_EXT_SANITIZE=address,undefined python3 setup.py build_ext --inplace --force
    LIBASAN="$(gcc -print-file-name=libasan.so)"
    LD_PRELOAD="$LIBASAN" ASAN_OPTIONS=detect_leaks=0 \
      UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
      PYTHONPATH=. python3 scripts/fuzz_cboard_diff.py --games "$GAMES"
    echo "NOTE: extensions are still sanitizer-built; rebuild for normal use:"
    echo "  python3 setup.py build_ext --inplace --force"
    ;;
  *)
    echo "usage: $0 {libfuzzer [seconds] | diff [games]}" >&2
    exit 2
    ;;
esac
