#!/usr/bin/env bash
# Run the project's static-analysis tools over given paths (or `--changed`).
#
# Default GATE: ruff + basedpyright + vulture (wall time ~= basedpyright,
# a few seconds). The gate is kept at ZERO findings repo-wide — no
# basedpyright baseline (burned down to zero and deleted 2026-07-02).
# pylint was removed 2026-07-02 — its narrow checks migrated to ruff rules
# (ARG/B006/G004/SIM115, see pyproject) and basedpyright's flow checks
# (reportPossiblyUnboundVariable/reportUnreachable).
# Known false positives are suppressed inline — see CLAUDE.md "Static
# analysis". When a FP appears on a new symbol, fix or suppress in the same
# commit.
#
# --fast skips vulture (negligible now; kept for compat).
#
# --deep adds the OCCASIONAL-CLEANUP tier (advisory, slow — run before a
# cleanup pass, not per edit):
#   * skylos (dead code + circular imports; ~40s, grep-verified tables)
#   * ruff cleanup report over the opt-in groups (B, SIM, PERF, NPY, RUF,
#     PT, PLW) — a shopping list, not a gate. Promote a group into the
#     pyproject gate once it reaches zero findings (PIE/UP/C4 + B009/B010/
#     B023 were promoted 2026-07-02).
#
# --slop runs scb-check (verbosity/erosion/clone detection + ast-grep anti-pattern
# rules). Opt-in because its output is noisy and many findings are style
# opinions; treat the score as a drift signal, not a commit gate. Baseline
# scores are saved in scripts/scb-baseline.json.
#
# Usage:
#   scripts/lint.sh                     # gate on package + tests + scripts
#   scripts/lint.sh path/a.py path/b.py # gate on given files
#   scripts/lint.sh --changed           # gate on changed/untracked .py files
#   scripts/lint.sh --fast [paths...]   # skip vulture
#   scripts/lint.sh --deep [paths...]   # + skylos + ruff cleanup report (advisory)
#   scripts/lint.sh --slop [paths...]   # also scb-check (verbosity/erosion)
#   scripts/lint.sh --all               # alias for default
set -euo pipefail

cd "$(dirname "$0")/.."

LINT_TMP="$(mktemp -d)"
trap 'rm -rf "$LINT_TMP"' EXIT

RUN_DEAD=1
RUN_DEEP=0
RUN_SLOP=0
USE_CHANGED=0
PATHS=()
USER_SET_PATHS=0

for arg in "$@"; do
    case "$arg" in
        --fast) RUN_DEAD=0 ;;
        --dead|--deep) RUN_DEEP=1 ;;
        --slop) RUN_SLOP=1 ;;
        --changed) USE_CHANGED=1 ;;
        --all) ;;  # noop: default behavior
        --help|-h)
            sed -n '2,38p' "$0"
            exit 0
            ;;
        *) PATHS+=("$arg"); USER_SET_PATHS=1 ;;
    esac
done

if [[ $USE_CHANGED -eq 1 ]]; then
    while IFS= read -r f; do
        [[ -f "$f" && "$f" == *.py ]] && PATHS+=("$f")
    done < <(git diff --name-only HEAD)
    while IFS= read -r f; do
        [[ -f "$f" && "$f" == *.py ]] && PATHS+=("$f")
    done < <(git ls-files --others --exclude-standard)
    USER_SET_PATHS=1
fi

if [[ ${#PATHS[@]} -eq 0 ]]; then
    PATHS=(chess_anti_engine tests scripts)
fi

JOB_NAMES=()
JOB_PIDS=()
JOB_LOGS=()
JOB_ADVISORY=()

tool() {
    if [[ -x ".venv/bin/$1" ]]; then
        printf '.venv/bin/%s\n' "$1"
    else
        printf '%s\n' "$1"
    fi
}

start_job() {
    local advisory="$1"
    local name="$2"
    shift 2
    local idx="${#JOB_NAMES[@]}"
    local safe_name="${name//[^A-Za-z0-9_.-]/_}"
    local log="$LINT_TMP/${idx}-${safe_name}.log"
    JOB_NAMES+=("$name")
    JOB_LOGS+=("$log")
    JOB_ADVISORY+=("$advisory")
    ("$@" >"$log" 2>&1) &
    JOB_PIDS+=("$!")
}

run_basedpyright() {
    # basedpyright scope is defined in pyrightconfig.json (package + tests + scripts).
    # No baseline: the repo is kept at zero findings. Only override scope when the
    # user explicitly names paths, so ad-hoc invocations on specific files still work.
    if [[ $USER_SET_PATHS -eq 1 ]]; then
        "$(tool basedpyright)" "${PATHS[@]}"
    else
        "$(tool basedpyright)"
    fi
}

run_slop() {
    # Summary scores to stderr for the drift signal. Full findings to stdout.
    # uvx runs the transient tool environment under LINT_TMP.
    local target="chess_anti_engine"
    local report="$LINT_TMP/scb-report.json"
    local err="$LINT_TMP/scb-report.err"
    if [[ $USER_SET_PATHS -eq 1 ]]; then
        if [[ ${#PATHS[@]} -ne 1 ]]; then
            echo "  scb-check accepts exactly one path; pass one path with --slop"
            return 2
        fi
        target="${PATHS[0]}"
    fi
    if ! UV_TOOL_DIR="$LINT_TMP/uv-tools" uvx --from scb-check scb-check check "$target" --report >"$report" 2>"$err"; then
        sed 's/^/  /' "$err"
        return 1
    fi
    "$(tool python3)" - "$report" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    print(f'  verbosity={r["verbosity"]:.3f}  erosion={r["erosion"]:.3f}  cog_erosion={r["cog_erosion"]:.3f}')
    print(f'  {r["verbosity_flagged_loc"]}/{r["total_loc"]} LOC flagged, {r["high_cc_functions"]}/{r["total_functions"]} high-CC fns')
except Exception as e:
    print(f'  scb-check report failed: {e}')
    raise
PY
}

run_group() {
    local failed=0
    local status=0
    for idx in "${!JOB_PIDS[@]}"; do
        if wait "${JOB_PIDS[$idx]}"; then
            status=0
        else
            status=$?
        fi

        echo
        echo "::: ${JOB_NAMES[$idx]}"
        cat "${JOB_LOGS[$idx]}"
        if [[ $status -ne 0 ]]; then
            if [[ "${JOB_ADVISORY[$idx]}" == "1" ]]; then
                echo "::: ${JOB_NAMES[$idx]} exited $status (advisory; continuing)"
            else
                failed=1
            fi
        fi
    done
    JOB_NAMES=()
    JOB_PIDS=()
    JOB_LOGS=()
    JOB_ADVISORY=()
    return "$failed"
}

echo
echo "::: running lint tools in parallel"
start_job 0 "ruff check" env RUFF_CACHE_DIR="$LINT_TMP/ruff-cache" "$(tool ruff)" check "${PATHS[@]}"
start_job 0 "basedpyright" run_basedpyright

if [[ $RUN_DEAD -eq 1 ]]; then
    # Vulture exits non-zero on findings; under set -e that fails the
    # run, which is what we want — unignored dead code should gate.
    start_job 0 "vulture (dead code, min confidence 80)" "$(tool vulture)" --min-confidence 80 "${PATHS[@]}"
fi

if [[ $RUN_DEEP -eq 1 ]]; then
    # Skylos exits 0 even with findings (its grep-verify rescues most
    # false positives). Its output tables remain the signal; treat as
    # advisory. Review the tables by eye, add `# skylos: ignore` for
    # FPs. Explicitly advisory — do not quietly gate on it.
    start_job 1 "skylos (advisory — dead code + circular imports)" "$(tool skylos)" "${PATHS[@]}"

    # Occasional-cleanup shopping list: opt-in ruff groups, advisory.
    # Every finding here has ONE of three fates (2026-07-02 triage):
    #   FIX-OVER-TIME (stays listed; burn down, promote group at zero):
    #     PT011/PT017 (test quality: raises-match + assert-in-except),
    #     RUF012 (ClassVar annotations), SIM102/108/117 (structure),
    #     PERF401/403 (comprehension rewrites), RUF005/007 (misc)
    #   KNOW-AND-BE-CAREFUL (stays listed; needs judgment, never autofix):
    #     NPY002 (legacy np.random -> Generator CHANGES RNG STREAMS),
    #     B905 (zip strict= changes semantics per site),
    #     B008 (call-in-default: often intentional interface defaults)
    #   WON'T-FIX (ignored below with reasons; keeps the report signal):
    #     RUF001/2/3  em-dashes/unicode in strings+comments are house style
    #     PERF203     try/except-in-loop is deliberate robustness in game loops
    #     PLW2901     loop-var reassignment (line = line.strip()) is idiom here
    #     SIM105      try/except/pass in hot worker loops; suppress() adds
    #                 per-iteration overhead and hides intent no better
    #     PLW0603     module-singleton globals (warn-once flags) are deliberate
    # The gate rules ride along so RUF100 (unused-noqa) is judged against the
    # REAL rule set — without them every gate-rule noqa reads as "unused".
    start_job 1 "ruff cleanup report (advisory — B,SIM,PERF,NPY,RUF,PT,PLW)" \
        env RUFF_CACHE_DIR="$LINT_TMP/ruff-cache" "$(tool ruff)" check \
        --extend-select B,SIM,PERF,NPY,RUF,PT,PLW \
        --ignore E741,ARG005,RUF001,RUF002,RUF003,PERF203,PLW2901,SIM105,PLW0603 \
        --statistics "${PATHS[@]}"
fi

if [[ $RUN_SLOP -eq 1 ]]; then
    start_job 1 "scb-check (verbosity + erosion + slop patterns)" run_slop
fi

if [[ ${#JOB_PIDS[@]} -gt 0 ]]; then
    run_group
fi

echo
echo "lint: OK"
