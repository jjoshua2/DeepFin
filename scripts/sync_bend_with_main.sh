#!/usr/bin/env bash
# Merge main into the `bend` branch without losing Bend's files.
#
# Bend lives only on `bend`; `main` does not carry it. A plain `git merge
# origin/main` into `bend` deletes every Bend file that `bend` has not changed
# since the merge base, because on main's side those files were deleted. That
# happens silently, with no conflict. This script performs the merge without
# committing, restores every Bend path from the pre-merge `bend` commit, and
# commits only when nothing else is left unresolved.
#
# Bend paths are the patterns in native/bend_engine/branch_paths.txt:
# a pattern ending in '/' matches everything under that directory; any other
# pattern is a shell glob in which '*' also matches '/'.
#
# Usage (from a clean checkout of the `bend` branch):
#   git fetch origin
#   scripts/sync_bend_with_main.sh [MAIN_REF]    # default origin/main
#   git push origin bend                         # after reviewing the result
#
# Exit codes: 0 merged and committed (or already up to date); 1 usage or state
# error; 2 merge left conflicts outside the Bend paths, which you resolve and
# commit by hand (Bend paths are already restored).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

main_ref="${1:-origin/main}"
paths_file="native/bend_engine/branch_paths.txt"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "sync_bend_with_main: working tree has uncommitted changes; commit or stash first" >&2
    exit 1
fi
if [[ ! -f "$paths_file" ]]; then
    echo "sync_bend_with_main: $paths_file not found; run this on the bend branch" >&2
    exit 1
fi
git rev-parse --verify --quiet "$main_ref^{commit}" >/dev/null || {
    echo "sync_bend_with_main: unknown ref $main_ref" >&2
    exit 1
}

patterns=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "$line" ]] && patterns+=("$line")
done < "$paths_file"

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

pre_merge="$(git rev-parse HEAD)"
merge_rc=0
git merge --no-ff --no-commit "$main_ref" || merge_rc=$?

if ! git rev-parse --verify --quiet MERGE_HEAD >/dev/null; then
    if [[ $merge_rc -ne 0 ]]; then
        echo "sync_bend_with_main: git merge failed before starting a merge" >&2
        exit 1
    fi
    echo "sync_bend_with_main: already up to date with $main_ref"
    exit 0
fi

# Every Bend file the pre-merge commit had that the merge removed, left
# conflicted or changed goes back to bend's copy. A restore from a tree-ish
# writes stage 0, which also resolves modify/delete conflicts.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git ls-tree -r --name-only -z "$pre_merge" | tr '\0' '\n' | sort > "$tmp/before"
git ls-files --stage -z | tr '\0' '\n' | awk -F'\t' '{split($1, a, " "); if (a[3] == 0) print $2}' \
    | sort > "$tmp/merged"
{
    comm -23 "$tmp/before" "$tmp/merged"
    git diff --name-only "$pre_merge"
} | sort -u > "$tmp/changed"
: > "$tmp/restore"
while IFS= read -r path; do
    if is_bend_path "$path" && grep -qxF -- "$path" "$tmp/before"; then
        printf '%s\0' "$path" >> "$tmp/restore"
    fi
done < "$tmp/changed"
restored="$(tr -cd '\0' < "$tmp/restore" | wc -c)"
if [[ "$restored" -gt 0 ]]; then
    xargs -0 git checkout "$pre_merge" -- < "$tmp/restore"
fi
echo "sync_bend_with_main: restored $restored Bend file(s) from bend"

other_deleted="$(git diff --cached --name-only --diff-filter=D "$pre_merge")"
if [[ -n "$other_deleted" ]]; then
    echo "sync_bend_with_main: the merge deletes these non-Bend files (main deleted them):"
    printf '%s\n' "$other_deleted" | sed 's/^/  /'
fi

if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
    echo "sync_bend_with_main: conflicts remain outside the Bend paths:" >&2
    git diff --name-only --diff-filter=U | sed 's/^/  /' >&2
    echo "Resolve them, then: git commit --no-edit" >&2
    exit 2
fi

git commit --no-edit -q
echo "sync_bend_with_main: merged $main_ref into $(git symbolic-ref --short HEAD) at $(git rev-parse --short HEAD)"
