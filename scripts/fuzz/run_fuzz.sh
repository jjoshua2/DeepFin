#!/usr/bin/env bash
# Fuzzing entry point for the CBoard C implementation.
#
#   ./scripts/fuzz/run_fuzz.sh libfuzzer [seconds]   # coverage-guided C fuzz (clang)
#   ./scripts/fuzz/run_fuzz.sh diff [games] [encode-every]
#                                                    # differential vs python-chess
#                                                    # (state + encoded planes, production
#                                                    # regime), under ASAN/UBSAN extensions
#   ./scripts/fuzz/run_fuzz.sh batch [games]         # _mcts_tree batch encoders vs the
#                                                    # single-board oracle, sanitized
#
# Cost at the defaults, measured UNSANITIZED on one core at nice 19 (the
# sanitized build these run under is several times slower, plus one build_ext):
#   diff  500 games, encode-every 4, v2_threats/175 : 18 s
#   batch 120 games x 2 rep-fix phases, 175 planes  : 13 s  (146 was 13 s -- the
#                                                     plane count is not the cost)
# The encode oracle is what makes `diff` cost more than it used to; it was off
# before, which is the whole point of turning it on. Neither entry point runs in
# CI -- the CI budget is tests/test_fuzz_smoke.py, ~1 s.
#
# libFuzzer crashes land in scripts/fuzz/corpus/crash-*; reproduce with
#   ./scripts/fuzz/cboard_fuzz <crash-file>
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-libfuzzer}"

# ⚑ MUST STAY IN LOCK-STEP WITH _CBOARD_FAST_SLIDER_MACROS IN setup.py.
# tests/test_slider_tables.py parses both and fails if they diverge — without
# the defines the libFuzzer harness compiles CBoard's LEGACY ray walkers, so it
# would fuzz an implementation production does not run while reporting coverage
# of "the CBoard C implementation". The `diff` and `batch` modes get these for
# free: they go through setup.py.
CBOARD_FAST_SLIDER_DEFINES=(
  -DDEEPFIN_FAST_SLIDERS=1
  -Dinit_attack_tables=init_attack_tables_reference
  -Dslider_attacks=slider_attacks_reference
  -Dbishop_attacks=bishop_attacks_reference
  -Drook_attacks=rook_attacks_reference
  -Dqueen_attacks=queen_attacks_reference
  -Dis_attacked_by=is_attacked_by_reference
)

# Backend selection matches setup.py's: plain clang defines no __BMI2__, so this
# fuzzes the MAGIC arm — the one the portable worker wheels and CI ship. Set
# FUZZ_NATIVE=1 to fuzz whatever arm this host's -march=native selects instead
# (PEXT on Zen 3+/Haswell+). Both arms are worth fuzzing; neither covers the
# other, because the index computation is the part that differs.
FUZZ_ARCH_FLAGS=()
if [ "${FUZZ_NATIVE:-0}" = "1" ]; then
  FUZZ_ARCH_FLAGS=(-march=native)
fi

run_sanitized_py() {
  CAE_EXT_SANITIZE=address,undefined python3 setup.py build_ext --inplace --force
  local LIBASAN
  LIBASAN="$(gcc -print-file-name=libasan.so)"
  LD_PRELOAD="$LIBASAN" ASAN_OPTIONS=detect_leaks=0 \
    UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
    PYTHONPATH=. python3 "$@"
  echo "NOTE: extensions are still sanitizer-built; rebuild for normal use:"
  echo "  python3 setup.py build_ext --inplace --force"
}

case "$MODE" in
  libfuzzer)
    SECONDS_BUDGET="${2:-120}"
    mkdir -p scripts/fuzz/corpus
    clang -g -O1 -fsanitize=fuzzer,address,undefined \
      "${FUZZ_ARCH_FLAGS[@]}" "${CBOARD_FAST_SLIDER_DEFINES[@]}" \
      -Ichess_anti_engine/encoding \
      scripts/fuzz/cboard_libfuzzer.c -lm \
      -o scripts/fuzz/cboard_fuzz
    ./scripts/fuzz/cboard_fuzz scripts/fuzz/corpus \
      -max_total_time="$SECONDS_BUDGET" -print_final_stats=1
    ;;
  diff)
    GAMES="${2:-500}"
    # --encode-every is passed explicitly (not left to the script default) so
    # this entry point states the plane oracle it runs. The oracle used to be
    # off here entirely, which left the production encoding regime checked by
    # nothing (encoding audit E1). Override the cadence with a 3rd argument.
    ENCODE_EVERY="${3:-4}"
    run_sanitized_py scripts/fuzz_cboard_diff.py --games "$GAMES" \
      --encode-every "$ENCODE_EVERY"
    ;;
  batch)
    GAMES="${2:-120}"
    run_sanitized_py scripts/fuzz_batch_encode_diff.py --games "$GAMES"
    ;;
  *)
    echo "usage: $0 {libfuzzer [seconds] | diff [games] [encode-every] | batch [games]}" >&2
    exit 2
    ;;
esac
