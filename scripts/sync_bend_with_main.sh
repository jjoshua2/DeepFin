#!/usr/bin/env bash
# Merge main into the `bend` branch without losing Bend's files.
#
# Bend lives only on `bend`; `main` does not carry it. A plain `git merge
# origin/main` into `bend` deletes every Bend file that `bend` has not changed
# since the merge base, because on main's side those files were deleted. That
# happens silently, with no conflict. This script runs the merge without
# committing and then, for every Bend path (patterns in
# native/bend_engine/branch_paths.txt) whose main side differs from the merge
# base, does exactly one of:
#
#   main DELETED it        -> restore bend's copy (the "Bend left main" case)
#   main changed or ADDED it, bend did not touch it
#                          -> keep main's version (the ordinary merge result)
#   main and bend both changed it, to different contents
#                          -> leave it for the operator; nothing is committed
#
# and prints one summary line per such path. Non-Bend paths get the ordinary
# merge result. The merge is committed only when nothing is left for the
# operator.
#
# Pattern syntax (branch_paths.txt): a pattern ending in '/' matches everything
# under that directory; any other pattern is a shell glob in which '*' also
# matches '/'. Blank lines and '#' comments are ignored.
#
# Usage, from a clean checkout:
#   scripts/sync_bend_with_main.sh [--branch NAME] [MAIN_REF]
#
#   MAIN_REF       ref to merge, default origin/main (fetch first).
#   --branch NAME  confirm that the current branch NAME is the intended sync
#                  target. Without it the current branch must be `bend`.
#                  Recommended: a sync branch cut from bend, merged by PR:
#                    git fetch origin
#                    git switch -c sync/main-YYYYMMDD origin/bend
#                    scripts/sync_bend_with_main.sh --branch sync/main-YYYYMMDD
#
# Exit codes: 0 merged and committed, or already up to date; 1 refused (detached
# HEAD, wrong branch, dirty tree, bad ref, not on a bend checkout); 2 merge left
# in progress for the operator: listed Bend paths changed on both sides and/or
# conflicts outside the Bend paths. Bend deletions are already restored. Review,
# resolve, then `git commit --no-edit` (or `git merge --abort`).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

main_ref="origin/main"
confirm_branch=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            [[ $# -ge 2 ]] || { echo "sync_bend_with_main: --branch needs a name" >&2; exit 1; }
            confirm_branch="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        -*)
            echo "sync_bend_with_main: unknown option $1" >&2
            exit 1
            ;;
        *)
            main_ref="$1"
            shift
            ;;
    esac
done

die() { echo "sync_bend_with_main: $*" >&2; exit 1; }

current="$(git symbolic-ref --quiet --short HEAD)" \
    || die "HEAD is detached; switch to bend or to a sync branch cut from it"
if [[ -n "$confirm_branch" ]]; then
    [[ "$current" == "$confirm_branch" ]] \
        || die "current branch is '$current', not '$confirm_branch' (--branch)"
elif [[ "$current" != "bend" ]]; then
    die "current branch is '$current'; run on bend, or confirm a sync branch with --branch $current"
fi
if git rev-parse --verify --quiet MERGE_HEAD >/dev/null; then die "a merge is already in progress"; fi
[[ -z "$(git status --porcelain --untracked-files=no)" ]] \
    || die "working tree has uncommitted changes; commit or stash first"

paths_file="native/bend_engine/branch_paths.txt"
[[ -f "$paths_file" ]] || die "$paths_file not found; run this on the bend branch"
git rev-parse --verify --quiet "$main_ref^{commit}" >/dev/null || die "unknown ref $main_ref"

patterns=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -n "$line" ]]; then patterns+=("$line"); fi
done < "$paths_file"
[[ ${#patterns[@]} -gt 0 ]] || die "$paths_file lists no patterns"

is_bend_path() {
    local p="$1" pat
    for pat in "${patterns[@]}"; do
        if [[ "$pat" == */ ]]; then
            [[ "$p" == "$pat"* ]] && return 0
        else
            # shellcheck disable=SC2053  # unquoted on purpose: glob match
            [[ "$p" == $pat ]] && return 0
        fi
    done
    return 1
}

ours="$(git rev-parse HEAD)"
theirs="$(git rev-parse "$main_ref^{commit}")"
base="$(git merge-base "$ours" "$theirs")" || die "$main_ref shares no history with $current"

if [[ "$base" == "$theirs" ]]; then
    echo "sync_bend_with_main: already up to date with $main_ref"
    exit 0
fi

# Blob id of every Bend path on each side (missing = absent on that side).
declare -A b_base b_ours b_theirs
load_side() {
    local -n side="$1"
    local commit="$2" meta path
    while IFS=$'\t' read -r -d '' meta path; do
        if is_bend_path "$path"; then side["$path"]="${meta##* }"; fi
    done < <(git ls-tree -r -z "$commit")
}
load_side b_base "$base"
load_side b_ours "$ours"
load_side b_theirs "$theirs"

merge_rc=0
git merge --no-ff --no-commit "$main_ref" >/dev/null 2>&1 || merge_rc=$?
git rev-parse --verify --quiet MERGE_HEAD >/dev/null \
    || die "git merge failed before starting a merge (exit $merge_rc)"

declare -A seen=()
for p in "${!b_base[@]}" "${!b_theirs[@]}"; do seen["$p"]=1; done

restore=()
summary=()
both=()
while IFS= read -r -d '' p; do
    if [[ -z "$p" ]]; then continue; fi   # printf emits one empty item for an empty set
    base_blob="${b_base[$p]:-}"
    ours_blob="${b_ours[$p]:-}"
    theirs_blob="${b_theirs[$p]:-}"
    if [[ "$theirs_blob" == "$base_blob" ]]; then continue; fi   # main did not touch it
    if [[ -z "$theirs_blob" ]]; then
        if [[ -n "$ours_blob" ]]; then
            restore+=("$p")
            summary+=("restored from bend  (main deleted)     $p")
        else
            summary+=("deleted on both     (nothing to do)    $p")
        fi
    elif [[ "$ours_blob" == "$base_blob" ]]; then
        if [[ -z "$base_blob" ]]; then
            summary+=("took main's version (main added)       $p")
        else
            summary+=("took main's version (main changed)     $p")
        fi
    elif [[ "$ours_blob" == "$theirs_blob" ]]; then
        summary+=("same on both sides  (nothing to do)    $p")
    else
        both+=("$p")
        summary+=("LEFT FOR OPERATOR   (changed on both)  $p")
    fi
done < <(printf '%s\0' "${!seen[@]}" | sort -z)

if [[ ${#restore[@]} -gt 0 ]]; then
    printf '%s\0' "${restore[@]}" | xargs -0 git checkout "$ours" --
fi

echo "sync_bend_with_main: Bend paths that main changed relative to the merge base: ${#summary[@]}"
if [[ ${#summary[@]} -gt 0 ]]; then
    printf '  %s\n' "${summary[@]}"
fi

other_deleted="$(git diff --cached --name-only --diff-filter=D "$ours")"
if [[ -n "$other_deleted" ]]; then
    echo "sync_bend_with_main: the merge deletes these non-Bend files (main deleted them):"
    printf '%s\n' "$other_deleted" | sed 's/^/  /'
fi

unmerged="$(git diff --name-only --diff-filter=U)"
if [[ ${#both[@]} -gt 0 || -n "$unmerged" ]]; then
    if [[ ${#both[@]} -gt 0 ]]; then
        echo "sync_bend_with_main: Bend paths changed on both sides (git's merge result or conflict left in place):" >&2
        printf '  %s\n' "${both[@]}" >&2
    fi
    if [[ -n "$unmerged" ]]; then
        echo "sync_bend_with_main: unresolved conflicts:" >&2
        printf '%s\n' "$unmerged" | sed 's/^/  /' >&2
    fi
    echo "Nothing committed. Review and resolve, then: git commit --no-edit (or git merge --abort)" >&2
    exit 2
fi

git commit --no-edit -q
echo "sync_bend_with_main: merged $main_ref into $current at $(git rev-parse --short HEAD)"
