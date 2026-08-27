#!/usr/bin/env bash
# Have Grok APPLY fixes in an isolated worktree, then run this repo's gates and hand the
# caller the raw diff and raw exit codes. NO MODEL JUDGES ANYTHING IN THIS SCRIPT.
#
# ⚑ THAT IS THE POINT. The sibling `grok-implementer` subagent is `model: sonnet` and its
# spec tells it to "read the diff yourself for spec violations" and report "deviations from
# spec" -- i.e. a Sonnet decides what the calling session gets told about work it did not
# see. This script removes that layer: grok writes, the gates run, the DIFF and the EXIT
# CODES come back verbatim, and the calling session (Fable/Opus) makes every call.
#
# Usage:
#   scripts/grok_fix.sh --spec <file> --branch <name> [--base origin/main] [--test "<cmd>"]
#   scripts/grok_fix.sh --findings <file> --branch <name> [...]   # a grok_review.sh block
#
# Leaves the worktree in place, UNCOMMITTED, for the caller to inspect and commit.
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "not in a git repo" >&2; exit 64; }
OUT_DIR="${GROK_FIX_OUT:-${TMPDIR:-/tmp}/grok_fix}"
SPEC="" FINDINGS="" BRANCH="" BASE="origin/main" TESTCMD=""

while [ $# -gt 0 ]; do
    case "$1" in
        --spec)     SPEC="${2:-}"; shift 2 ;;
        --findings) FINDINGS="${2:-}"; shift 2 ;;
        --branch)   BRANCH="${2:-}"; shift 2 ;;
        --base)     BASE="${2:-}"; shift 2 ;;
        --test)     TESTCMD="${2:-}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
[ -n "$BRANCH" ] || { echo "--branch is required" >&2; exit 64; }
[ -n "$SPEC" ] || [ -n "$FINDINGS" ] || { echo "need --spec or --findings" >&2; exit 64; }
command -v grok >/dev/null 2>&1 || { echo "GROK UNAVAILABLE: not on PATH" >&2; exit 69; }

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WT="$OUT_DIR/wt_${BRANCH//\//_}_${STAMP}"
RAW="$OUT_DIR/grokfix_${BRANCH//\//_}_${STAMP}.txt"
PROMPT="$OUT_DIR/spec_${BRANCH//\//_}_${STAMP}.md"

# ⚑ A WORKTREE, NEVER THE LIVE TREE. Training re-reads the live tree every iteration, and
# grok cannot be made read-only by any flag (measured 2026-08-26: --disallowed-tools,
# --permission-mode plan and --deny were each accepted and each wrote the file anyway).
# Here we WANT writes -- the containment is the worktree, not a permission setting.
git -C "$REPO" worktree add -b "$BRANCH" "$WT" "$BASE" >/dev/null 2>&1 || {
    echo "could not create worktree $WT on $BRANCH from $BASE" >&2; exit 65; }

{
    if [ -n "$SPEC" ]; then cat "$SPEC"; else
        echo "# Fix these reviewed findings"
        echo
        echo "Each line is: index | file:line | severity | claim | failure | settling observation"
        echo '```'
        cat "$FINDINGS"
        echo '```'
    fi
    cat <<'SPEC_TAIL'

## How to work in this repo
Match the surrounding code's style, naming and comment density. Comments are for
constraints the code cannot show, not narration.

⚑ FIX ONLY WHAT IS LISTED. Do not opportunistically refactor, reformat, rename, or "tidy"
anything outside the listed items. Unrelated edits are the single fastest way to get this
whole change rejected, because they make the diff unreviewable.

⚑ If a listed item turns out NOT to be a real defect, DO NOT invent a change to satisfy it.
Leave the code alone and say so in your final message, naming the item index and why. A
finding that dissolves under inspection is a useful result; a defensive edit that silences
it is a corrupted record.

Add or update a test with every behaviour change. A new test is presumed vacuous until you
have made the breaking change, watched the test fail, and reverted it.
SPEC_TAIL
} > "$PROMPT"

grok --cwd "$WT" --prompt-file "$PROMPT" --permission-mode acceptEdits \
     --max-turns 40 --no-subagents --output-format plain > "$RAW" 2>&1
GROK_EXIT=$?

echo "===== GROK FIX RUN ====="
echo "authored_by: grok-cli"
echo "worktree:   $WT"
echo "branch:     $BRANCH  (base $BASE)"
echo "raw_output: $RAW"
echo "grok exit:  $GROK_EXIT"
echo
echo "----- grok's own final message (raw, unedited) -----"
tail -c 4000 "$RAW"
echo
echo "----- files changed -----"
git -C "$WT" status --short
echo
echo "----- diff --stat -----"
git -C "$WT" diff --stat HEAD

# ⚑ GATES RUN HERE, AND ONLY THEIR EXIT CODES ARE REPORTED. No interpretation: this repo
# has already had an agent call lint green while it exited 1, by grepping output instead of
# reading the status. Bare lint (no path arguments) because a path-scoped run structurally
# cannot see breakage the change caused in a file it did not open.
echo
echo "----- gates -----"
( cd "$WT" && ./scripts/lint.sh > "$OUT_DIR/lint_${STAMP}.txt" 2>&1 )
echo "lint exit: $?   (log: $OUT_DIR/lint_${STAMP}.txt)"
if [ -n "$TESTCMD" ]; then
    ( cd "$WT" && eval "$TESTCMD" > "$OUT_DIR/test_${STAMP}.txt" 2>&1 )
    echo "test exit: $?   (log: $OUT_DIR/test_${STAMP}.txt)  cmd: $TESTCMD"
else
    echo "test exit: SKIPPED — no --test given. This is NOT a pass."
fi

cat <<'CALLER_TAIL'

----- for the CALLING session, which is the only thing here that judges -----
Nothing above has been assessed. Specifically still owed by you:
  * read the full diff (`git -C <worktree> diff HEAD`) for edits outside the listed items
  * compare lint's exit code against the BASE branch's — this repo's suite is red by
    design, so an absolute number means nothing and only a DELTA does
  * for each new test, run its mutant and confirm it fails
  * decide each item: fixed / not-a-defect / still-open — and record the not-a-defects,
    never silently absorb them
Nothing is committed and no PR is open; both are yours.
CALLER_TAIL
exit 0
