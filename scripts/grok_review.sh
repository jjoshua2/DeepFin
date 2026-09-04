#!/usr/bin/env bash
# Independent code review from Grok via `grok -p`, with no summarizing model in
# the path. Grok always works against a disposable snapshot, never the live tree.
set -euo pipefail
umask 077

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
OUT_DIR="${GROK_REVIEW_OUT:-${TMPDIR:-/tmp}/grok_review}"
GROK_REVIEW_TIMEOUT_SECONDS="${GROK_REVIEW_TIMEOUT_SECONDS:-1800}"
MODE=""
TARGET=""
FOCUS=""
WORKTREE=""

while [ "$#" -gt 0 ]; do
    if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "argument requires a value: $1" >&2
        exit 64
    fi
    case "$1" in
        --pr|--diff|--worktree)
            if [ -n "$MODE" ]; then
                echo "choose exactly one review mode" >&2
                exit 64
            fi
            ;;
    esac
    case "$1" in
        --pr)
            MODE="pr"
            TARGET="${2:-}"
            shift 2
            ;;
        --diff)
            MODE="diff"
            TARGET="${2:-}"
            shift 2
            ;;
        --worktree)
            MODE="worktree"
            WORKTREE="${2:-}"
            shift 2
            ;;
        --focus)
            FOCUS="${2:-}"
            shift 2
            ;;
        --repo)
            REPO="${2:-}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 64
            ;;
    esac
done

[ -n "$MODE" ] || {
    echo "need one of --pr N | --diff A...B | --worktree PATH" >&2
    exit 64
}

command -v grok >/dev/null 2>&1 || {
    echo "GROK UNAVAILABLE: the grok CLI is not on PATH." >&2
    echo "Do not substitute another model and label it as the Grok lane." >&2
    exit 69
}
command -v timeout >/dev/null 2>&1 || {
    echo "GROK UNAVAILABLE: the timeout command is not on PATH." >&2
    exit 69
}
case "$GROK_REVIEW_TIMEOUT_SECONDS" in
    ""|*[!0-9]*)
        echo "GROK_REVIEW_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 64
        ;;
    *[1-9]*) ;;
    *)
        echo "GROK_REVIEW_TIMEOUT_SECONDS must be greater than zero" >&2
        exit 64
        ;;
esac

if [ "$MODE" = "worktree" ]; then
    [ -d "$WORKTREE" ] || {
        echo "no such worktree: $WORKTREE" >&2
        exit 65
    }
    REPO="$(cd "$WORKTREE" && pwd -P)"
else
    [ -d "$REPO" ] || {
        echo "no such repository path: $REPO" >&2
        exit 65
    }
    REPO="$(cd "$REPO" && pwd -P)"
fi

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "not a git worktree: $REPO" >&2
    exit 65
}

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
RUN_DIR="$(mktemp -d "$OUT_DIR/review.XXXXXXXX")"
RAW="$RUN_DIR/raw.txt"
PROMPT="$RUN_DIR/prompt.md"

SNAP="$(mktemp -d)"
TMP_ARCHIVE=""
TMP_DIFF=""
TMP_UNTRACKED=""
cleanup() {
    [ -z "$TMP_ARCHIVE" ] || rm -f -- "$TMP_ARCHIVE"
    [ -z "$TMP_DIFF" ] || rm -f -- "$TMP_DIFF"
    [ -z "$TMP_UNTRACKED" ] || rm -f -- "$TMP_UNTRACKED"
    rm -rf -- "$SNAP"
}
trap cleanup EXIT

extract_local_ref() {
    local ref="$1"
    git -C "$REPO" archive "$ref" 2>/dev/null | tar -x -C "$SNAP"
}

init_snapshot_repo() {
    git -C "$SNAP" init -q &&
        git -C "$SNAP" add -A &&
        git -C "$SNAP" -c user.name="Grok Review Snapshot" \
            -c user.email="snapshot@invalid" commit --allow-empty -qm "review base"
}

worktree_fingerprint() {
    (
        cd "$REPO" || exit 65
        git rev-parse HEAD || exit 65
        git status --porcelain=v1 --untracked-files=all || exit 65
        git diff --binary HEAD || exit 65
        git ls-files --others --exclude-standard -z \
            | tar --null --files-from=- -cf -
    ) | sha256sum | awk '{ print $1 }'
}

SOURCE_HEAD="$(git -C "$REPO" rev-parse HEAD)"
BEFORE_STATUS="$(git -C "$REPO" status --short --untracked-files=all 2>/dev/null)"
BEFORE_FINGERPRINT="$(worktree_fingerprint)" || {
    echo "could not fingerprint the live worktree before review" >&2
    exit 65
}
case "$MODE" in
    pr)
        REPO_SLUG="$(cd "$REPO" && gh repo view --json nameWithOwner --jq .nameWithOwner)" || {
            echo "could not resolve repository name" >&2
            exit 65
        }
        (
            cd "$REPO" || exit 65
            gh pr view "$TARGET" --json title,body \
                --jq '"# PR #'"$TARGET"'\n## " + .title + "\n\n" + (.body // "")'
        ) > "$PROMPT" 2>/dev/null || {
            echo "could not read PR $TARGET" >&2
            exit 65
        }
        PR_REVISIONS="$(cd "$REPO" && gh api "repos/$REPO_SLUG/pulls/$TARGET" --jq '[.base.sha, .head.sha] | join(" ")')" || {
            echo "could not resolve PR $TARGET head" >&2
            exit 65
        }
        read -r PR_BASE PR_HEAD <<< "$PR_REVISIONS"
        if ! [[ "$PR_BASE" =~ ^[0-9a-f]{40}$ && "$PR_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
            echo "invalid PR revision metadata" >&2
            exit 65
        fi
        if git -C "$REPO" cat-file -e "${PR_HEAD}^{commit}" 2>/dev/null; then
            extract_local_ref "$PR_HEAD" || {
                echo "could not snapshot PR $TARGET head $PR_HEAD" >&2
                exit 65
            }
        else
            TMP_ARCHIVE="$(mktemp)"
            (cd "$REPO" && gh api "repos/$REPO_SLUG/tarball/$PR_HEAD") > "$TMP_ARCHIVE" || {
                echo "could not download PR $TARGET head $PR_HEAD" >&2
                exit 65
            }
            tar -xzf "$TMP_ARCHIVE" --strip-components=1 -C "$SNAP" || {
                echo "could not extract PR $TARGET snapshot" >&2
                exit 65
            }
        fi
        init_snapshot_repo || {
            echo "could not initialize PR snapshot" >&2
            exit 65
        }
        TMP_DIFF="$(mktemp)"
        (cd "$REPO" && gh pr diff "$TARGET") > "$TMP_DIFF" || {
            echo "could not read PR $TARGET diff" >&2
            exit 65
        }
        PR_AFTER="$(cd "$REPO" && gh api "repos/$REPO_SLUG/pulls/$TARGET" --jq '[.base.sha, .head.sha] | join(" ")')"
        if [ "$PR_AFTER" != "$PR_REVISIONS" ]; then
            echo "PR base/head changed while collecting review input; retry against a stable revision" >&2
            exit 65
        fi
        {
            echo "Snapshot head: $PR_HEAD; PR base: $PR_BASE"
            echo
            echo "## DIFF UNDER REVIEW"
            echo '```diff'
            sed -n '1,$p' "$TMP_DIFF"
            echo '```'
        } >> "$PROMPT"
        ;;
    diff)
        case "$TARGET" in
            *...*) HEAD_REF="${TARGET##*...}" ;;
            *)
                echo "--diff requires a three-dot range: <base>...<head>" >&2
                exit 64
                ;;
        esac
        [ -n "$HEAD_REF" ] || {
            echo "--diff range has no head revision: $TARGET" >&2
            exit 64
        }
        BASE_REF="${TARGET%...*}"
        BASE_SHA="$(git -C "$REPO" rev-parse --verify "${BASE_REF}^{commit}")"
        HEAD_REF="$(git -C "$REPO" rev-parse --verify "${HEAD_REF}^{commit}")"
        TMP_DIFF="$(mktemp)"
        git -C "$REPO" diff "$BASE_SHA...$HEAD_REF" > "$TMP_DIFF" || {
            echo "could not read diff range $TARGET" >&2
            exit 65
        }
        {
            echo "# Diff under review: $TARGET"
            echo "Resolved base: $BASE_SHA; snapshot head: $HEAD_REF"
            echo '```diff'
            sed -n '1,$p' "$TMP_DIFF"
            echo '```'
        } > "$PROMPT"
        extract_local_ref "$HEAD_REF" || {
            echo "could not snapshot diff head $HEAD_REF" >&2
            exit 65
        }
        init_snapshot_repo || {
            echo "could not initialize diff snapshot" >&2
            exit 65
        }
        ;;
    worktree)
        extract_local_ref "$SOURCE_HEAD" || {
            echo "could not snapshot worktree HEAD" >&2
            exit 65
        }
        init_snapshot_repo || {
            echo "could not initialize worktree snapshot" >&2
            exit 65
        }
        TMP_DIFF="$(mktemp)"
        git -C "$REPO" diff --binary HEAD > "$TMP_DIFF" || {
            echo "could not read worktree diff" >&2
            exit 65
        }
        if [ -s "$TMP_DIFF" ]; then
            git -C "$SNAP" apply "$TMP_DIFF" || {
                echo "could not apply tracked worktree changes to the snapshot" >&2
                exit 65
            }
        fi
        TMP_UNTRACKED="$(mktemp)"
        git -C "$REPO" ls-files --others --exclude-standard -z > "$TMP_UNTRACKED" || {
            echo "could not list untracked worktree files" >&2
            exit 65
        }
        if [ -s "$TMP_UNTRACKED" ]; then
            (
                cd "$REPO" || exit 65
                tar --null --files-from="$TMP_UNTRACKED" -cf -
            ) | tar -xf - -C "$SNAP" || {
                echo "could not copy untracked files to the snapshot" >&2
                exit 65
            }
        fi
        {
            echo "# Uncommitted changes in a worktree"
            echo "Source HEAD: $SOURCE_HEAD"
            echo "Source fingerprint (HEAD, status, tracked diff, untracked archive): $BEFORE_FINGERPRINT"
            echo '```diff'
            sed -n '1,$p' "$TMP_DIFF"
            echo '```'
            echo
            echo "## Untracked files included in the snapshot"
            git -C "$REPO" ls-files --others --exclude-standard
        } > "$PROMPT"
        ;;
esac

cat >> "$PROMPT" <<'PROMPT_TAIL'

## How to review this codebase

This codebase's signature defect is a value that is accepted and then silently ignored:
a knob that never reaches the production path, a metric that does not mean its name, or a
gate that cannot fail. Ask whether this change takes effect on the production path and
which observation proves that it does.

Do not fix anything. This is a review only.

## Required output format

Write any analysis first. End with a findings block in exactly this form:

BEGIN_FINDINGS
1 | <file>:<line> | <severity: BLOCKER|MAJOR|MINOR|NIT> | <one-line claim> | <concrete failure: inputs or state -> wrong behavior> | <single observation that proves or disproves this>
2 | ...
END_FINDINGS

Use one pipe-separated, non-wrapping line per finding. If there are no findings, put the
single line `NONE` between the markers. The last field must name a concrete check.
PROMPT_TAIL

if [ -n "$FOCUS" ]; then
    {
        echo
        echo "## Caller's review focus"
        echo "$FOCUS"
    } >> "$PROMPT"
fi

PROMPT_BYTES="$(wc -c < "$PROMPT")"
if [ "$PROMPT_BYTES" -gt 1000000 ]; then
    echo "GROK FAILED: prompt is too large for a reliable grok -p invocation ($PROMPT_BYTES bytes)." >&2
    echo "Narrow the diff or review it in bounded pieces." >&2
    exit 65
fi

if [ "$(git -C "$REPO" rev-parse HEAD)" != "$SOURCE_HEAD" ] ||
    [ "$(worktree_fingerprint)" != "$BEFORE_FINGERPRINT" ]; then
    echo "source changed while collecting the snapshot; no review was started" >&2
    exit 71
fi

# A file avoids operating-system argv limits on larger diffs. Inherit Grok's
# configured model/effort rather than duplicating machine defaults in this repo.
if timeout --kill-after=5s "${GROK_REVIEW_TIMEOUT_SECONDS}s" \
    grok --prompt-file "$PROMPT" --cwd "$SNAP" --max-turns 25 \
        --no-subagents --output-format plain > "$RAW" 2>&1; then
    GROK_STATUS=0
else
    GROK_STATUS=$?
fi
printf 'raw_output: %s\n' "$RAW"

AFTER_STATUS="$(git -C "$REPO" status --short --untracked-files=all 2>/dev/null)"
AFTER_FINGERPRINT="$(worktree_fingerprint)" || {
    echo "could not fingerprint the live worktree after review" >&2
    exit 71
}
if [ "$AFTER_FINGERPRINT" != "$BEFORE_FINGERPRINT" ]; then
    echo "LIVE WORKTREE CHECK: CHANGED WHILE GROK RAN" >&2
    echo "Fingerprint before: $BEFORE_FINGERPRINT" >&2
    echo "Fingerprint after:  $AFTER_FINGERPRINT" >&2
    echo "Before:" >&2
    printf '%s\n' "$BEFORE_STATUS" >&2
    echo "After:" >&2
    printf '%s\n' "$AFTER_STATUS" >&2
    exit 71
fi

if [ "$GROK_STATUS" -ne 0 ]; then
    echo "GROK FAILED: grok -p exited $GROK_STATUS. Raw: $RAW" >&2
    echo "Report this as a failed review, not a clean verdict." >&2
    echo "LIVE WORKTREE CHECK: unchanged"
    exit 70
fi

echo "===== GROK REVIEW (raw, complete, unedited) ====="
echo "authored_by: grok-cli"
echo "raw_output: $RAW"
echo "mode: $MODE  target: ${TARGET:-$WORKTREE}"
echo "model: configured Grok default"
echo "reasoning_effort: configured Grok default"
echo "================================================="
cat "$RAW"
echo "===== END GROK REVIEW ====="

# Require one complete findings block. Short explicit NONE is a valid result;
# missing/partial/duplicate markers and malformed finding rows are not a pass.
if ! awk '
    /^BEGIN_FINDINGS$/ { if (seen++) exit 1; inside=1; next }
    /^END_FINDINGS$/ { if (!inside) exit 1; inside=0; ended=1; next }
    inside && NF {
        if ($0 == "NONE") clean++
        else if ($0 ~ /^[0-9]+ \| .+ \| (BLOCKER|MAJOR|MINOR|NIT) \| .+ \| .+ \| .+$/) findings++
        else invalid=1
    }
    END { if (!ended || inside || invalid || (clean && findings) || clean > 1 || !(clean || findings)) exit 1 }
' "$RAW"; then
    echo "GROK FAILED: output is unparseable, not clean. Raw: $RAW" >&2
    exit 70
fi

echo "These are leads, not verdicts. Verify each before acting."
echo "LIVE WORKTREE CHECK: unchanged"
