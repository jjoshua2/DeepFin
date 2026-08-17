#!/usr/bin/env bash
# Copy every Nth live checkpoint somewhere Ray cannot prune it.
#
# WHY THIS EXISTS: Ray prunes live checkpoints on a rolling basis, so the OLDEST
# points of any series are the first to go. That is backwards from what analysis
# needs. Audit L16 found a policy-only holdout drift over iters 165-192 -- and by
# the time the trend was significant enough to notice, checkpoints 165-186 were
# already gone. The flat first segment can never be re-scored against a different
# holdout, so "was it always rising?" is permanently unanswerable.
#
# A single readout has an obvious artefact to bank. A TREND accumulates silently
# and is only recognised once enough points exist to fit a slope -- by which time
# its early checkpoints are the oldest and therefore already deleted. So bank
# continuously and cheaply, rather than at the moment someone realises they care.
#
# Safe by construction: cp only, nice 19, never writes to the trial dir, never
# constructs a DiskReplayBuffer (that deletes shards -- audit G12).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REPO_ROOT/data/salvage/rolling"
EVERY=${EVERY:-5}        # keep checkpoints whose index is a multiple of this
KEEP=${KEEP:-24}         # cap on retained dirs (~656M each)
SLEEP=${SLEEP:-900}
# One pass, then exit. The loop is unbounded by design (this is a daemon), and a
# test that has to wait for `timeout` to kill it pays the full duration on EVERY
# case even though the work finished in the first millisecond -- ~80s per serial
# suite run for four cases. A seam that ENDS the loop is cheaper than a shorter
# timeout and, unlike one, it also proves the pass completed rather than that it
# was cut short. Unset in production, where the daemon must never exit.
ONCE=${ONCE:-0}

# ⚑⚑ FAIL LOUDLY FROM THE WRONG TREE — #441 review N4.
#
# Deriving TUNE_DIR from the checkout is right, and it makes this daemon a
# SILENT NO-OP anywhere the trial does not live. Measured from a `git worktree`
# before this guard: `ONCE=1 bash scripts/bank_rolling_checkpoints.sh` printed
# nothing, created an empty `data/salvage/rolling`, and exited 0. Unset ONCE --
# i.e. production -- and it loops on that forever, banking nothing.
#
# That is worse here than for the other derived-root scripts. This is the
# rolling half of the REVERT POINTS the experiment protocol depends on: an
# operator who launches it from a worktree gets a green daemon, an existing
# destination directory, and no checkpoints, and only finds out when a rollback
# is needed and the series is gone. Ray has meanwhile pruned the originals.
#
# `scripts/feed_bootstrap_shards.py` already handles the identical case the
# right way (`ERROR: ... not found` on stderr, exit 2); this is that shape.
# `TUNE_DIR` stays overridable for a test fixture and for a non-default layout.
#
# ⚑ And it is the DIRECTORY, not a glob string. Holding the pattern in a variable
# forces the loop below to expand it UNQUOTED, which also word-splits on IFS and
# applies pathname expansion to the whole path -- so a checkout under a directory
# with a space or a glob metachar in its name silently iterates over fragments.
# Keeping the prefix quoted at the point of use and letting only the trailing `*`
# glob has neither problem, and matches what the inner `checkpoint_0*` loop
# already does. (Before the path scrub the root was a hardcoded literal, so this
# could not bite; deriving it from the checkout is exactly what arms it.)
TUNE_DIR=${TUNE_DIR:-"$REPO_ROOT/runs/pbt2_small/tune"}
if [ ! -d "$TUNE_DIR" ]; then
    echo "[bank] ERROR: no tune dir at $TUNE_DIR — nothing to bank." >&2
    echo "[bank] This script banks the LIVE trial's checkpoints and operates on the" >&2
    echo "[bank] tree it lives in ($REPO_ROOT). Run it from the main checkout, or set" >&2
    echo "[bank] TUNE_DIR=<path>. Exiting rather than looping silently." >&2
    exit 2
fi

mkdir -p "$DST"

while :; do
    for trial in "$TUNE_DIR"/train_trial_*; do
        [ -d "$trial" ] || continue
        for ck in "$trial"/checkpoint_0*; do
            [ -d "$ck" ] || continue
            name=$(basename "$ck")
            idx=$((10#${name#checkpoint_}))
            [ $((idx % EVERY)) -eq 0 ] || continue
            [ -e "$DST/$name" ] && continue
            # Copy to a temp name first so a prune mid-copy cannot leave a
            # half-written dir that later reads as a complete banked checkpoint.
            if nice -n 19 cp -r "$ck" "$DST/.tmp_$name" 2>/dev/null; then
                mv "$DST/.tmp_$name" "$DST/$name"
                echo "[bank] $(date '+%F %T') banked $name"
            else
                rm -rf "$DST/.tmp_$name"
            fi
        done
    done

    # Trim oldest by checkpoint index, not mtime -- mtime is the COPY time and
    # would evict in arbitrary order.
    mapfile -t have < <(ls "$DST" 2>/dev/null | grep '^checkpoint_' | sort -t_ -k2 -n)
    n=${#have[@]}
    if [ "$n" -gt "$KEEP" ]; then
        for ((i=0; i<n-KEEP; i++)); do
            echo "[bank] trimming ${have[$i]}"
            rm -rf "${DST:?}/${have[$i]}"
        done
    fi

    [ "$ONCE" = 1 ] && break
    sleep "$SLEEP"
done
