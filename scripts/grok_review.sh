#!/usr/bin/env bash
# Independent code review from a different model family, with NO Claude in the path.
#
# ⚑ THE POINT OF THIS SCRIPT IS THAT IT IS NOT AN AGENT.
# This replaced a Claude-Sonnet subagent wrapper that was measured (2026-08-26, over 36
# spawns) doing two things a reviewer must never do: rendering grok's findings as its own
# triage list instead of passing them through, and -- on PR #453 -- reviewing the diff
# ITSELF and returning zero grok output under a "GROK REVIEW" header, so the caller
# believed it had a second model family when it had a second Claude. Both are failures of
# instruction-following, and neither is possible here: a pipe cannot summarise, and a shell
# script cannot decide to write the review itself. The guarantee is structural.
#
# Usage:
#   scripts/grok_review.sh --pr <N>   [--focus "..."] [--repo <path>]
#   scripts/grok_review.sh --diff <base>...<head> [--focus "..."] [--repo <path>]
#   scripts/grok_review.sh --worktree <path> [--focus "..."]   # review uncommitted work
#
# Writes grok's COMPLETE raw stdout to a file AND to this script's stdout. Exits non-zero
# if grok is missing, unauthenticated, or returns nothing usable twice -- a failed run is
# an honest result and must never be papered over with a substitute review.
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
OUT_DIR="${GROK_REVIEW_OUT:-${TMPDIR:-/tmp}/grok_review}"
MODE="" TARGET="" FOCUS="" WORKTREE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --pr)       MODE=pr;       TARGET="${2:-}"; shift 2 ;;
        --diff)     MODE=diff;     TARGET="${2:-}"; shift 2 ;;
        --worktree) MODE=worktree; WORKTREE="${2:-}"; shift 2 ;;
        --focus)    FOCUS="${2:-}"; shift 2 ;;
        --repo)     REPO="${2:-}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
[ -n "$MODE" ] || { echo "need one of --pr N | --diff A...B | --worktree PATH" >&2; exit 64; }

command -v grok >/dev/null 2>&1 || {
    echo "GROK UNAVAILABLE: the grok CLI is not on PATH." >&2
    echo "Do NOT substitute a Claude review for this -- that collapses the two lanes." >&2
    exit 69
}

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SLUG="$(printf '%s' "${TARGET:-$WORKTREE}" | tr -c 'A-Za-z0-9' '_' | cut -c1-40)"
RAW="$OUT_DIR/grok_${MODE}_${SLUG}_${STAMP}.txt"
PROMPT="$OUT_DIR/prompt_${MODE}_${SLUG}_${STAMP}.md"

# ⚑⚑ SNAPSHOT, ALWAYS — IT IS THE ONLY THING THAT WORKS. NO GROK FLAG MAKES HEADLESS GROK
# READ-ONLY. Measured 2026-08-26, four mechanisms, each asked to overwrite a file; ALL
# FOUR were accepted and ALL FOUR wrote it anyway:
#   --disallowed-tools search_replace,write        -> wrote the file
#   --permission-mode plan                         -> wrote the file
#   --deny bash --deny write --deny search_replace -> wrote the file
#   (default, no --always-approve)                 -> wrote 4 files in a live worktree,
#                                                     measured 2026-08-22
# The --deny run narrated "by whatever path is available" and then did it. These flags are
# this repo's own signature defect wearing someone else's logo: a value accepted and then
# silently ignored. Do NOT swap the snapshot for a flag, however convincing its name --
# `--worktree` is no help either, the CLI states outright that headless mode ignores it.
# The live tree is re-read by training every iteration, so a stray write there is a
# production incident, not an inconvenience.
SNAP="$(mktemp -d)"
cleanup() { rm -rf "$SNAP"; }
trap cleanup EXIT

case "$MODE" in
    pr)
        gh pr view "$TARGET" --json title,body \
            --jq '"# PR #'"$TARGET"'\n## " + .title + "\n\n" + (.body // "")' > "$PROMPT" 2>/dev/null \
            || { echo "could not read PR $TARGET" >&2; exit 65; }
        git -C "$REPO" archive HEAD 2>/dev/null | tar -x -C "$SNAP" || true
        { echo; echo "## DIFF UNDER REVIEW"; echo '```diff'
          gh pr diff "$TARGET"; echo '```'; } >> "$PROMPT"
        ;;
    diff)
        { echo "# Diff under review: $TARGET"; echo '```diff'
          git -C "$REPO" diff "$TARGET"; echo '```'; } > "$PROMPT"
        git -C "$REPO" archive HEAD 2>/dev/null | tar -x -C "$SNAP" || true
        ;;
    worktree)
        [ -d "$WORKTREE" ] || { echo "no such worktree: $WORKTREE" >&2; exit 65; }
        git -C "$WORKTREE" archive HEAD 2>/dev/null | tar -x -C "$SNAP" || true
        # Uncommitted work is usually the whole point of reviewing a worktree.
        git -C "$WORKTREE" diff HEAD | git -C "$SNAP" apply --allow-empty 2>/dev/null || true
        { echo "# Uncommitted + committed work in a worktree"; echo '```diff'
          git -C "$WORKTREE" diff HEAD; echo '```'; } > "$PROMPT"
        ;;
esac

# The standing bias is included verbatim on every run: it is the single most useful thing
# we can tell an outside reviewer about this codebase, and leaving it to the caller means
# it gets forgotten exactly when the review matters.
cat >> "$PROMPT" <<'PROMPT_TAIL'

## How to review this codebase
This codebase's signature defect is a value that is accepted and then silently ignored --
a knob that never reaches the production path, a metric that does not mean its name, a
gate that cannot fail. So the question is not "is this code correct" but "does this take
effect on the production path, and what observation would prove it did".

Do not fix anything; this is a read-only review.

## Required output format
Write whatever analysis you want first. Then end your response with a findings block in
EXACTLY this form, so it can be parsed and tracked item by item:

BEGIN_FINDINGS
1 | <file>:<line> | <severity: BLOCKER|MAJOR|MINOR|NIT> | <one-line claim> | <the concrete failure: inputs or state -> wrong behaviour> | <the single observation that would prove or disprove this>
2 | ...
END_FINDINGS

One finding per line, pipe-separated, no wrapping. If a finding spans several sites, list
the primary one and say so in the claim. If you found NOTHING, emit the block with the
single line `NONE` between the markers -- a clean pass is a real result and must be
stated, not implied by silence.

The last field is required and is the point: name the check that settles it. A finding
nobody can test is a guess, and this repo has been burned by confident guesses that read
like measurements.
PROMPT_TAIL

[ -n "$FOCUS" ] && { echo; echo "## Caller's review focus"; echo "$FOCUS"; } >> "$PROMPT"

run_grok() {
    grok --cwd "$SNAP" --prompt-file "$PROMPT" \
         --max-turns 25 --no-subagents --output-format plain 2>&1
}

# One retry: empty or near-empty output is the observed transient failure mode.
OUTPUT="$(run_grok)"
if [ "$(printf '%s' "$OUTPUT" | wc -c)" -lt 200 ]; then
    OUTPUT="$(run_grok)"
fi

printf '%s\n' "$OUTPUT" > "$RAW"

if [ "$(printf '%s' "$OUTPUT" | wc -c)" -lt 200 ]; then
    echo "GROK FAILED: no usable output after two attempts. Raw: $RAW" >&2
    echo "Report this as a FAILED review with zero findings. Do NOT write one yourself." >&2
    exit 70
fi

echo "===== GROK REVIEW (raw, complete, unedited) ====="
echo "authored_by: grok-cli"
echo "raw_output: $RAW"
echo "mode: $MODE  target: ${TARGET:-$WORKTREE}"
echo "================================================="
printf '%s\n' "$OUTPUT"
echo "===== END GROK REVIEW ====="

# The parsed findings are a CONVENIENCE INDEX, printed after the raw text and never
# instead of it. If grok ignored the format, say so rather than reporting zero findings --
# an unparseable review is not a clean review, and the two must never look alike.
FINDINGS="$(printf '%s\n' "$OUTPUT" | sed -n '/^BEGIN_FINDINGS/,/^END_FINDINGS/p' \
            | sed '1d;$d' | sed '/^[[:space:]]*$/d')"
echo
if [ -z "$FINDINGS" ]; then
    echo "PARSED FINDINGS: NONE FOUND IN THE OUTPUT — grok did not emit the block."
    echo "⚑ This is UNPARSEABLE, not CLEAN. Read the raw text above before concluding"
    echo "  anything; do not record this run as a zero-finding pass."
elif [ "$(printf '%s' "$FINDINGS" | tr -d '[:space:]')" = "NONE" ]; then
    echo "PARSED FINDINGS: 0 (grok explicitly reported a clean pass)"
else
    echo "PARSED FINDINGS: $(printf '%s\n' "$FINDINGS" | wc -l)"
    printf '%s\n' "$FINDINGS"
fi
echo
echo "⚑ These are LEADS, not verdicts. Verify each before acting; record the ones that"
echo "  turn out wrong AS false positives rather than fixing them defensively."

# ⚑ Verify the REAL tree was untouched. Report, never repair -- a script that quietly
# reverts grok's stray writes would hide the very behaviour this check exists to catch.
if [ -d "$REPO/.git" ]; then
    DIRT="$(git -C "$REPO" status --short 2>/dev/null | head -20)"
    [ -n "$DIRT" ] && { echo; echo "NOTE: working tree is not clean (pre-existing or otherwise):"; echo "$DIRT"; }
fi
exit 0
