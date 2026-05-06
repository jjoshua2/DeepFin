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

echo "::: ruff check"
ruff check "${PATHS[@]}"

echo
echo "::: basedpyright"
# basedpyright scope is defined in pyrightconfig.json (package + tests + scripts). It
# also respects .basedpyright/baseline.json so existing package drift doesn't
# fail CI; refresh the baseline with `basedpyright --writebaseline` after a fix
# campaign. Only override scope when the user explicitly names paths, so ad-hoc
# invocations on specific files still work.
if [[ $USER_SET_PATHS -eq 1 ]]; then
    basedpyright "${PATHS[@]}"
else
    basedpyright
fi

echo
echo "::: pylint (narrow config: 7 semantic checks, see pyproject.toml)"
pylint "${PATHS[@]}"

if [[ $RUN_DEAD -eq 1 ]]; then
    echo
    echo "::: vulture (dead code, min confidence 80)"
    # Vulture exits non-zero on findings; under set -e that fails the
    # run, which is what we want — unignored dead code should gate.
    vulture --min-confidence 80 "${PATHS[@]}"

    echo
    echo "::: skylos (advisory — dead code + circular imports)"
    # Skylos exits 0 even with findings (its grep-verify rescues most
    # false positives). Its output tables remain the signal; treat as
    # advisory. Review the tables by eye, add `# skylos: ignore` for
    # FPs. Explicitly advisory — do not quietly gate on it.
    skylos "${PATHS[@]}" || true
fi

if [[ $RUN_SLOP -eq 1 ]]; then
    echo
    echo "::: scb-check (verbosity + erosion + slop patterns)"
    # Summary scores to stderr for the drift signal. Full findings to stdout —
    # uvx runs on demand; installs into ~/.cache/uv on first call.
    uvx scb-check check "${PATHS[@]}" --report 2>/dev/null | python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read())
    print(f'  verbosity={r[\"verbosity\"]:.3f}  erosion={r[\"erosion\"]:.3f}  cog_erosion={r[\"cog_erosion\"]:.3f}')
    print(f'  {r[\"verbosity_flagged_loc\"]}/{r[\"total_loc\"]} LOC flagged, {r[\"high_cc_functions\"]}/{r[\"total_functions\"]} high-CC fns')
except Exception as e:
    print(f'  scb-check report failed: {e}')
"
fi

echo
echo "lint: OK"
