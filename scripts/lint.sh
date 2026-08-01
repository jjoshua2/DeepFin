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
#   * ruff cleanup report over the not-yet-gated groups (B, NPY) — a
#     shopping list, not a gate. SIM/PERF/RUF/PT/PLW + B007/B904 were burned
#     to zero and promoted into the pyproject gate 2026-07-02; what remains
#     here is only the needs-judgment tail (B905/B008/NPY002) and the
#     B009/B010 WON'T-GATE (see pyproject).
#   * shellcheck over scripts/*.sh when shellcheck is installed (advisory —
#     the ops scripts are load-bearing but findings need judgment).
#
# --slop runs scb-check (verbosity/erosion/clone detection + ast-grep anti-pattern
# rules). Opt-in because its output is noisy and many findings are style
# opinions; treat the score as a drift signal, not a commit gate. Baseline
# scores are saved in scripts/scb-baseline.json.
#
# basedpyright SCOPE is not the same as ruff/vulture scope. NAMING PATHS is what
# narrows it -- and only that. Every invocation without paths (bare, --changed,
# --fast, --deep, --slop, --all) runs it at pyrightconfig.json's whole-repo scope,
# which is what CI runs. A path-scoped run cannot see breakage the change caused in
# a file it never opened, which is how main went red (PR #295).
#
# With --changed, ruff and vulture do narrow to the changed .py files; when the
# change collected none, PATHS falls back to the full default list and they run
# whole-repo too.
#
# Usage:
#   scripts/lint.sh                     # gate on package + tests + scripts
#   scripts/lint.sh path/a.py path/b.py # gate on given files (basedpyright scoped too)
#   scripts/lint.sh --changed           # changed/untracked .py (basedpyright: whole repo)
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
            sed -n '2,49p' "$0"
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
    # No baseline: the repo is kept at zero findings. Only EXPLICITLY NAMED paths
    # narrow that scope, so ad-hoc invocations on specific files stay fast
    # (measured 2026-08-01: 2 files analyzed / ~1.4s vs 544 / ~32s whole-repo).
    #
    # `--changed` deliberately does NOT narrow it. A change breaks the type gate in
    # files it never opened -- PR #295 added parameters to
    # DiskReplayBuffer._load_refresh_chunks and turned main red via
    # tests/test_replay_disk_buffer.py, which that PR did not touch. A scoped run
    # structurally cannot see that. --changed is the commit-time form, where the
    # whole-repo scan is worth the 34s.
    if [[ $USER_SET_PATHS -eq 1 && $USE_CHANGED -eq 0 ]]; then
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

    # Occasional-cleanup shopping list: the not-yet-gated ruff groups,
    # advisory. After the 2026-07-02 burn-down promoted SIM/PERF/RUF/PT/PLW
    # + B007/B904 into the gate, what remains is only the needs-judgment
    # tail — never autofix these:
    #   NPY002 (legacy np.random -> Generator CHANGES RNG STREAMS),
    #   B905 (zip strict= changes semantics per site),
    #   B008 (call-in-default: often intentional interface defaults)
    # WON'T-FIX (ignored below): B009/B010 — getattr/setattr-with-constant
    # is the deliberate idiom for dynamic attrs on torch Modules (WON'T-GATE,
    # see pyproject).
    start_job 1 "ruff cleanup report (advisory — B,NPY judgment tail)" \
        env RUFF_CACHE_DIR="$LINT_TMP/ruff-cache" "$(tool ruff)" check \
        --extend-select B,NPY \
        --ignore E741,ARG005,SIM105,PERF203,RUF001,RUF002,RUF003,PLW2901,PLW0603,B009,B010 \
        --statistics "${PATHS[@]}"

    # Ops shell scripts (train.sh, salvage, arena drivers) are load-bearing;
    # shellcheck them when available. Advisory: findings need judgment.
    if command -v shellcheck >/dev/null 2>&1; then
        start_job 1 "shellcheck (advisory — scripts/*.sh)" shellcheck scripts/*.sh
    fi
fi

if [[ $RUN_SLOP -eq 1 ]]; then
    start_job 1 "scb-check (verbosity + erosion + slop patterns)" run_slop
fi

if [[ ${#JOB_PIDS[@]} -gt 0 ]]; then
    run_group
fi

echo
echo "lint: OK"
