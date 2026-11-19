#!/usr/bin/env bash
# Run explicitly requested Grok implementation in its own worktree and preserve
# the output, diff and validation results for the caller's independent judgment.
set -euo pipefail
umask 077

REPO="$(git rev-parse --show-toplevel)"
SPEC="" FINDINGS="" BRANCH="" BASE="origin/main" TESTCMD=""
# Two independent caps end a run, whichever is hit first: the turn budget
# (GROK_FIX_MAX_TURNS) and the wall-clock timeout (GROK_FIX_TIMEOUT_SECONDS).
# A turn has been observed to take ~25-40 s, so 300 turns can need 2-3.3 h: the
# 3 h timeout default is usually the binding cap. Raise both together.
DURATION="${GROK_FIX_TIMEOUT_SECONDS:-10800}"
MAX_TURNS="${GROK_FIX_MAX_TURNS:-300}"
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
case "$MAX_TURNS" in
    ""|*[!0-9]*) echo "GROK_FIX_MAX_TURNS must be a positive integer" >&2; exit 64 ;;
    *[1-9]*) ;;
    *) echo "GROK_FIX_MAX_TURNS must be greater than zero" >&2; exit 64 ;;
esac
for tool in grok timeout setsid python3; do
    command -v "$tool" >/dev/null || { echo "$tool is unavailable" >&2; exit 69; }
done
BASE_SHA="$(git rev-parse --verify "${BASE}^{commit}")"
OUT="${GROK_FIX_OUT:-${TMPDIR:-/tmp}/grok_fix}"
HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUN_DIR="$(python3 -S -B "$HELPER_DIR/grok_artifact_dir.py" "$OUT" run)"
WT="$RUN_DIR/worktree"
RAW="$RUN_DIR/raw.txt"
PROMPT="$RUN_DIR/prompt.md"
GROK_PGID=""
kill_grok_group() {
    [ -z "$GROK_PGID" ] || kill -KILL -- "-$GROK_PGID" 2>/dev/null || true
    # Never signal this id again: once the group is gone the kernel may reuse it.
    GROK_PGID=""
}
# Kill only Grok's process group on exit; the worktree, branch and raw output
# are deliberately retained for the caller.
trap kill_grok_group EXIT
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

# Grok runs in its own session so this script owns the kill of everything it
# started (same scheme as grok_review.sh). Which processes `timeout` signals is
# implementation-specific: GNU signals the whole group, while the timeout
# shipped with Ubuntu 26.04 left a TERM-ignoring grandchild alive. The
# background child is not a group leader, so setsid makes it one without
# forking and $! is the group id. That needs job control off (with it on, the
# job already leads a group, setsid forks and returns at once), so turn it off
# and verify the result rather than assume it.
set +m
setsid timeout --kill-after=5s "${DURATION}s" grok --cwd "$WT" --prompt-file "$PROMPT" \
    --permission-mode acceptEdits --max-turns "$MAX_TURNS" --no-subagents --output-format plain \
    > "$RAW" 2>&1 &
GROK_PGID=$!
# setsid runs in the child after the fork, so the group appears a moment after
# $! does: poll for it. An empty answer means the child already exited, which
# leaves nothing to isolate or kill.
for _ in $(seq 1 100); do
    if ! GROK_ACTUAL_PGID="$(ps -o pgid= -p "$GROK_PGID" 2>/dev/null | tr -d ' ')"; then
        if kill -0 "$GROK_PGID" 2>/dev/null; then
            kill -KILL "$GROK_PGID" 2>/dev/null || true
            echo "GROK FAILED: could not inspect the Grok process group" >&2
            exit 70
        fi
        GROK_ACTUAL_PGID=""
    fi
    if [ -z "$GROK_ACTUAL_PGID" ] || [ "$GROK_ACTUAL_PGID" = "$GROK_PGID" ]; then
        break
    fi
    sleep 0.05
done
if [ -n "$GROK_ACTUAL_PGID" ] && [ "$GROK_ACTUAL_PGID" != "$GROK_PGID" ]; then
    kill -KILL "$GROK_PGID" 2>/dev/null || true
    GROK_PGID=""
    echo "GROK FAILED: could not isolate the Grok process group" >&2
    exit 70
fi
if wait "$GROK_PGID"; then
    GROK_EXIT=0
else
    GROK_EXIT=$?
fi
kill_grok_group
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
