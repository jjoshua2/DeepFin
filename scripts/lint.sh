#!/usr/bin/env bash
# Run the project's static-analysis tools over given paths (or `--changed`).
#
# Default: ruff + basedpyright + pylint + vulture + skylos. The active gate is
# clean (ruff 0, basedpyright 0 after the package baseline, pylint 10/10,
# vulture 0, skylos A+ with 0 dead); keep it that way. Tests and scripts are
# clean without baseline entries. Known false positives are suppressed inline
# — see the directives listed in CLAUDE.md's "Static analysis" section. When a
# FP appears on a new symbol, either fix the symbol or add the suppression in
# the same commit.
#
# --fast skips vulture + skylos (they're the slower + noisier ones) for quick
# per-edit checks where the dead-code sweep isn't needed.
#
# --slop runs scb-check (verbosity/erosion/clone detection + ast-grep anti-pattern
# rules). Opt-in because its output is noisy and many findings are style
# opinions; treat the score as a drift signal, not a commit gate. Baseline
# scores are saved in scripts/scb-baseline.json.
#
# Usage:
#   scripts/lint.sh                     # all tools on package + tests + scripts
#   scripts/lint.sh path/a.py path/b.py # all tools on given files
#   scripts/lint.sh --changed           # all tools on changed/untracked .py files
#   scripts/lint.sh --fast [paths...]   # skip vulture + skylos
#   scripts/lint.sh --slop [paths...]   # also scb-check (verbosity/erosion)
#   scripts/lint.sh --all               # alias for default (all tools, full paths)
set -euo pipefail

cd "$(dirname "$0")/.."

LINT_TMP="$(mktemp -d)"
trap 'rm -rf "$LINT_TMP"' EXIT

RUN_DEAD=1
RUN_SLOP=0
USE_CHANGED=0
PATHS=()
USER_SET_PATHS=0

for arg in "$@"; do
    case "$arg" in
        --fast) RUN_DEAD=0 ;;
        --dead) RUN_DEAD=1 ;;  # noop: default behavior (kept for backwards compat)
        --slop) RUN_SLOP=1 ;;
        --changed) USE_CHANGED=1 ;;
        --all) ;;  # noop: default behavior
        --help|-h)
            sed -n '2,22p' "$0"
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
    # basedpyright scope is defined in pyrightconfig.json (package + tests + scripts). It
    # also respects .basedpyright/baseline.json so existing package drift doesn't
    # fail CI; refresh the baseline with `basedpyright --writebaseline` after a fix
    # campaign. Only override scope when the user explicitly names paths, so ad-hoc
    # invocations on specific files still work.
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
start_job 0 "pylint (narrow config: 7 semantic checks, see pyproject.toml)" env PYLINTHOME="$LINT_TMP/pylint" "$(tool pylint)" "${PATHS[@]}"

if [[ $RUN_DEAD -eq 1 ]]; then
    # Vulture exits non-zero on findings; under set -e that fails the
    # run, which is what we want — unignored dead code should gate.
    start_job 0 "vulture (dead code, min confidence 80)" "$(tool vulture)" --min-confidence 80 "${PATHS[@]}"

    # Skylos exits 0 even with findings (its grep-verify rescues most
    # false positives). Its output tables remain the signal; treat as
    # advisory. Review the tables by eye, add `# skylos: ignore` for
    # FPs. Explicitly advisory — do not quietly gate on it.
    start_job 1 "skylos (advisory — dead code + circular imports)" "$(tool skylos)" "${PATHS[@]}"
fi

if [[ $RUN_SLOP -eq 1 ]]; then
    start_job 1 "scb-check (verbosity + erosion + slop patterns)" run_slop
fi

if [[ ${#JOB_PIDS[@]} -gt 0 ]]; then
    run_group
fi

echo
echo "lint: OK"
