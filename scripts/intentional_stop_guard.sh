# The operator-marker guard, shared by every script that can (re)START training.
# Sourced, never executed. Defines a function and three constants; it has NO
# side effects — no `cd`, no output — so a caller can source it before it has
# decided anything about its own root.
#
# WHY THIS FILE EXISTS. `train.sh stop` touches $CAE_STOP_MARKER_DEFAULT BEFORE
# it kills, precisely so nothing in the stack resurrects a run the operator
# stopped on purpose. That contract only holds if every restarting path CHECKS
# the marker, and until now the check lived in the CALLER: `watchdog_loop.sh`
# tests `[ ! -f "$MARKER" ]` before it invokes `recover_stall.sh`, so the
# watchdog route was covered and the BY-HAND route — the mode recover_stall.sh's
# own header invites, `./scripts/recover_stall.sh` — had no check at all. A
# guard in the caller guards the caller, not the operation.
#
# A guard in the caller had already failed once for real. On 2026-08-20
# (fixed in 42e72d6cb) the watchdog cleared a live pause window's marker on an
# age bound and then force-restarted production beside the job that pause
# protected. That fix taught the watchdog which markers it may clear; this file
# is the other half — the operation refuses for itself, so no future caller can
# be written without the check.
#
# ⚑ STANDING POLICY: production training is NEVER auto-started. A self-restarting
# path that ignores this marker is a policy violation vector, not a convenience.
#
# THE OVERRIDE IS DELIBERATE AND LOUD, NEVER THE DEFAULT. A human who really
# means it passes --ignore-intentional-stop, and gets a line naming every marker
# being overridden and what it holds. "Loud" is the point: the default of a
# force-recovery tool must be to leave a deliberate stop alone.

# Exit status for a refusal. 7 avoids 1 (generic), 2 (usage), 3/5/6 (the
# watchdog verdict statuses in train_watchdog.py) and 124/137 (timeout/SIGKILL),
# so a caller can tell "refused on a marker" from any other failure.
INTENTIONAL_STOP_EXIT=7

# ⚑ THE ONE DEFINITION OF THE MARKER PATHS. `scripts/train.sh` (STOP_MARKER),
# `scripts/watchdog_loop.sh` (MARKER) and `scripts/train_watchdog.py`
# (DEFAULT_STOP_MARKER) each carry their own copy of the stop-marker literal for
# reasons of their own; `tests/test_recover_stall_guard.py` pins all four equal,
# so a path edit that lands in one file and not the others fails there rather
# than silently disarming a guard.
CAE_STOP_MARKER_DEFAULT="/tmp/chess_training.intentional_stop"
CAE_PAUSE_TXT_DEFAULT="runs/pbt2_small/tune/pause.txt"

# The canonical spelling of the override, so the flag a caller ACCEPTS and the
# flag the refusal ADVERTISES cannot drift apart.
CAE_IGNORE_STOP_FLAG="--ignore-intentional-stop"

# intentional_stop_guard <override:0|1> <log-fn> <marker>...
#
# Returns 0 when no marker is present, or when $1 is 1 (override) — in which
# case it first logs one line PER overridden marker, naming the path and the
# text the marker holds. Returns $INTENTIONAL_STOP_EXIT otherwise, after logging
# one line per marker that caused the refusal.
#
# The log function is passed in rather than assumed: each caller already owns a
# `log()` that tees to its own file, and a guard that printed to stdout would be
# swallowed by the `>> "$LOG" 2>&1` its callers are invoked under.
#
# ⚑ `-e`, not `-f`: a marker replaced by a directory or a dangling symlink is
# still an operator saying "do not restart this". Fail-closed on the shape too.
intentional_stop_guard() {
    local override="$1" logfn="$2"
    shift 2
    local guard held found=0
    for guard in "$@"; do
        [ -e "$guard" ] || continue
        found=1
        held=$(tr '\n' ' ' < "$guard" 2>/dev/null) || held=""
        [ -n "$held" ] || held="(empty)"
        if [ "$override" = 1 ]; then
            "$logfn" "OVERRIDE $CAE_IGNORE_STOP_FLAG: ignoring operator marker $guard — held: $held"
        else
            "$logfn" "REFUSING (exit $INTENTIONAL_STOP_EXIT): operator marker present: $guard — held: $held"
        fi
    done
    [ "$found" = 1 ] || return 0
    if [ "$override" = 1 ]; then
        "$logfn" "OVERRIDE $CAE_IGNORE_STOP_FLAG: RESTARTING PRODUCTION TRAINING against a deliberate stop, on explicit operator instruction."
        return 0
    fi
    "$logfn" "REFUSING: training was stopped or paused ON PURPOSE and will NOT be restarted. Remove the marker(s) named above, or re-run with $CAE_IGNORE_STOP_FLAG to override deliberately."
    return "$INTENTIONAL_STOP_EXIT"
}
