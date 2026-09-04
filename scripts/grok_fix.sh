#!/usr/bin/env bash
# Run explicitly requested Grok implementation in its own worktree and preserve
# the output, diff and validation results for the caller's independent judgment.
set -euo pipefail
umask 077

REPO="$(git rev-parse --show-toplevel)"
SPEC="" FINDINGS="" BRANCH="" BASE="origin/main" TESTCMD=""
DURATION="${GROK_FIX_TIMEOUT_SECONDS:-1800}"
while [ "$#" -gt 0 ]; do
    if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "argument requires a value: $1" >&2; exit 64
    fi
    case "$1" in
        --spec) SPEC="$2" ;;
        --findings) FINDINGS="$2" ;;
        --branch) BRANCH="$2" ;;
        --base) BASE="$2" ;;
        --test) TESTCMD="$2" ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
    shift 2
done
[ -n "$BRANCH" ] || { echo "--branch is required" >&2; exit 64; }
if { [ -z "$SPEC" ] && [ -z "$FINDINGS" ]; } ||
    { [ -n "$SPEC" ] && [ -n "$FINDINGS" ]; }; then
    echo "choose exactly one of --spec or --findings" >&2; exit 64
fi
INPUT="${SPEC:-$FINDINGS}"
[ -r "$INPUT" ] && [ -f "$INPUT" ] || { echo "cannot read input: $INPUT" >&2; exit 65; }
case "$DURATION" in
    ""|*[!0-9]*) echo "invalid timeout" >&2; exit 64 ;;
    *[1-9]*) ;;
    *) echo "timeout must be positive" >&2; exit 64 ;;
esac
for tool in grok timeout; do
    command -v "$tool" >/dev/null || { echo "$tool is unavailable" >&2; exit 69; }
done
BASE_SHA="$(git rev-parse --verify "${BASE}^{commit}")"
OUT="${GROK_FIX_OUT:-${TMPDIR:-/tmp}/grok_fix}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
RUN_DIR="$(mktemp -d "$OUT/run.XXXXXXXX")"
WT="$RUN_DIR/worktree"
RAW="$RUN_DIR/raw.txt"
PROMPT="$RUN_DIR/prompt.md"
{
    printf 'Source base: %s (%s)\n\n' "$BASE_SHA" "$BASE"
    cat "$INPUT"
    cat <<'PROMPT_END'

Read the shared project guidance. Implement the requested scope in this assigned
worktree, preserving unrelated work. Verify any supplied finding before fixing it;
explain rejected findings instead of inventing a change. Choose tests appropriate to
the affected behavior. Do not commit, push, open a PR, deploy or affect live jobs.
Leave the diff for the caller's independent review and report what remains uncertain.
PROMPT_END
} > "$PROMPT"
git -C "$REPO" worktree add -b "$BRANCH" "$WT" "$BASE_SHA"
printf 'worktree: %s\nbranch: %s\nbase: %s\nraw_output: %s\n' "$WT" "$BRANCH" "$BASE_SHA" "$RAW"

if timeout --kill-after=5s "${DURATION}s" grok --cwd "$WT" --prompt-file "$PROMPT" \
    --permission-mode acceptEdits --max-turns 40 --no-subagents --output-format plain \
    > "$RAW" 2>&1; then
    GROK_EXIT=0
else
    GROK_EXIT=$?
fi
printf 'grok exit: %s\n' "$GROK_EXIT"
cat "$RAW"
git -C "$WT" status --short
git -C "$WT" diff --stat "$BASE_SHA"
if [ "$GROK_EXIT" -ne 0 ]; then
    echo "Grok failed; worktree and raw output retained. No success verdict." >&2
    exit 70
fi

FAILED=0
if { git -C "$WT" diff --name-only "$BASE_SHA"; git -C "$WT" ls-files --others --exclude-standard; } |
    awk '/\.(py|pyi|c|h|sh|toml|yaml|yml)$/ { found=1 } END { exit !found }'; then
    if (cd "$WT" && ./scripts/lint.sh) > "$RUN_DIR/lint.txt" 2>&1; then
        echo "lint exit: 0"
    else
        CODE=$?; echo "lint exit: $CODE"; FAILED=1
    fi
    echo "lint log: $RUN_DIR/lint.txt"
else
    echo "lint: not run (no code/config changes detected)"
fi
if [ -n "$TESTCMD" ]; then
    if (cd "$WT" && bash -c "$TESTCMD") > "$RUN_DIR/test.txt" 2>&1; then
        echo "test exit: 0"
    else
        CODE=$?; echo "test exit: $CODE"; FAILED=1
    fi
    echo "test log: $RUN_DIR/test.txt"
else
    echo "tests: not run (no --test command supplied)"
fi
echo "Inspect the complete diff and validation logs before deciding what to commit."
[ "$FAILED" -eq 0 ] || exit 70
