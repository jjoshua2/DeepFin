# Definitions shared by scripts/daily_gate_ratchet.sh and scripts/ratchet_loop.sh.
# Sourced, never executed.
#
# WHY THIS FILE EXISTS. The two scripts have to agree about four things — which
# repo, which run directory, what an exit status means, and how an iteration is
# parsed out of a checkpoint name — and every one of those had been duplicated
# into both files. Duplication that a test merely SYNCS is still duplication:
# it drifts the moment someone edits one copy, and it had already drifted here.
# `ratchet_loop.sh` honoured $TRAIN_WORK_DIR while `daily_gate_ratchet.sh`
# hard-coded runs/pbt2_small, so with TRAIN_WORK_DIR=runs/other the loop
# selected iter900 from one tree and the ratchet then snapshotted iter478 from
# the other and wrote THAT into the CSV — a row whose iter column names a
# checkpoint the decision was not made from. The same class produced the
# checkpoint_000000 parse living in two places with only one of them covered.
#
# So the fix is not "keep them equal", it is "make them one". A definition here
# cannot diverge from itself; tests/test_ratchet_search_shape.py additionally
# fails if either script re-defines any of these locally.
#
# RATCHET_ROOT and TRAIN_PIDFILE are TEST SEAMS, not operator knobs: without
# them nothing can execute these scripts' bodies, and a body that cannot be
# executed is one whose wiring is pinned only by reading it.

# The repo. Every path in both scripts is relative to it.
cd "${RATCHET_ROOT:-/home/josh/projects/chess}" || exit 2
export PYTHONPATH=.

# The run directory whose newest trial is measured. ONE reader of the env var,
# so the loop and the ratchet cannot pick different trees.
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"

# Exit statuses of scripts/daily_gate_ratchet.sh. ratchet_loop.sh branches on
# them, so a private copy in either file is a status the other does not
# recognise — i.e. a give-up silently demoted to a retry.
#   0                      at least one CSV row exists for today
#   RATCHET_EXIT_RETRY     no row, but another attempt today might produce one
#   RATCHET_EXIT_NO_RETRY  no row, and no further attempt today can help
# 5 avoids 1 (retryable), 2 (usage), 3 (the arena's own no-pairs status) and
# 124/137 (timeout / SIGKILL).
RATCHET_EXIT_RETRY=1
RATCHET_EXIT_NO_RETRY=5

# The iteration number from a checkpoint directory name.
#
# `sed 's/checkpoint_0*//'` — what both scripts used to do — yields the EMPTY
# STRING for checkpoint_000000, because the zeros it strips ARE the number.
# Strip the prefix, then leading zeros only down to the last digit.
ratchet_iter_from_checkpoint () {   # $1 = .../checkpoint_000478
    basename "$1" | sed -E 's/^checkpoint_//; s/^0+([0-9])/\1/'
}

# The trial whose data is freshest. ⚑ NOT `ls -td`, which ranks by DIRECTORY
# mtime -- and a directory's mtime moves whenever any entry is created or
# removed inside it, so Ray writing `checkpoint_NNNNNN/` under a DEAD trial
# floats it above the live one. Both scripts had that rule, and
# `pause_window.sh` was fixed while these two were not, which made the
# disagreement WORSE than the original bug: the wrapper parked the trial IT
# chose while `iter`, `MIN_ITER`, `ck_ready` and the snapshot all came from the
# trial the LOOP chose. Production parked to measure a dead trial.
#
# Same rule as `train_watchdog.newest_trial_dir` / `trial_paths._trial_sort_key`
# / `pause_window.sh:resolve_trial_id`: populated-preferred (rank 1 = carries
# result.json or progress.csv), then that file's mtime, then the name.
# `%.9Y` not `%Y`: whole-second mtimes tie for two trials written in the same
# second -- what a restart does -- and the comparison then falls through to the
# NAME, deciding alphabetically. LC_ALL=C so the decimal separator is a dot.
# Prints the directory WITH a trailing slash, matching what `ls -td .../*/` gave
# its callers.
ratchet_newest_trial_dir () {   # $1 = tune dir (default $WORK_DIR/tune)
    local tune="${1:-$WORK_DIR/tune}" d f rank m best
    best="$(
        for d in "$tune"/train_trial_*/; do
            [ -d "$d" ] || continue
            rank=0; m=0
            for f in result.json progress.csv; do
                if [ -f "$d$f" ]; then
                    m="$(stat -c %.9Y "$d$f" 2>/dev/null || echo 0)"; rank=1; break
                fi
            done
            [ "$rank" -eq 1 ] || m="$(stat -c %.9Y "$d" 2>/dev/null || echo 0)"
            printf '%s %s %s\n' "$rank" "$m" "$d"
        done | LC_ALL=C sort -k1,1nr -k2,2nr -k3,3r | head -1 | cut -d' ' -f3-
    )"
    [ -n "$best" ] || return 1
    printf '%s\n' "$best"
}
