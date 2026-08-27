# The operator-marker guard, shared by every script that can (re)START training.
# Sourced, never executed. Defines functions and constants; it has NO side
# effects — no `cd`, no output — so a caller can source it before it has decided
# anything about its own root.
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
#
# TWO MARKER CLASSES, BECAUSE THEY ARE NOT EQUALLY OVERRIDABLE.
#   STOP  — `train.sh stop` wrote it. Overriding it means "start anyway", which
#           every caller can actually carry out.
#   PAUSE — a pause window / graceful restart holds the trial. Overriding it
#           means "start anyway AND deal with the marker", and a caller that
#           cannot clear the marker correctly must NOT be allowed to override
#           it: the trial it starts parks at its own pause check while the
#           caller's log claims a successful restart. So `pause_overridable` is
#           per-caller, not global. See `intentional_stop_guard` below.

# Exit status for a refusal. 7 avoids 1 (generic), 2 (usage), 3/5/6 (the
# watchdog verdict statuses in train_watchdog.py) and 124/137 (timeout/SIGKILL),
# so a caller can tell "refused on a marker" from any other failure.
# ⚑ Kept in step with scripts/watchdog_loop.sh's EXIT_RECOVER_REFUSED, which
# rolls back its cooldown stamp on exactly this status. Pinned equal by
# tests/test_recover_stall_guard.py.
INTENTIONAL_STOP_EXIT=7

# ⚑ THE ONE DEFINITION OF THE MARKER PATHS. `scripts/train.sh` (STOP_MARKER),
# `scripts/watchdog_loop.sh` (MARKER) and `scripts/train_watchdog.py`
# (DEFAULT_STOP_MARKER) each carry their own copy of the stop-marker literal for
# reasons of their own; `tests/test_recover_stall_guard.py` pins all four equal,
# so a path edit that lands in one file and not the others fails there rather
# than silently disarming a guard.
CAE_STOP_MARKER_DEFAULT="/tmp/chess_training.intentional_stop"
CAE_TUNE_DIR_DEFAULT="runs/pbt2_small/tune"

# The canonical spelling of the override, so the flag a caller ACCEPTS and the
# flag the refusal ADVERTISES cannot drift apart.
CAE_IGNORE_STOP_FLAG="--ignore-intentional-stop"

# cae_marker_exists <path>
#
# ⚑ `-e` ALONE IS NOT FAIL-CLOSED. `[ -e ]` FOLLOWS the link and is FALSE for a
# DANGLING symlink, so a marker whose target had been removed — or one an
# operator pointed at a not-yet-created file — read as "no marker" and silently
# permitted recovery. That is the exact inversion this guard exists to prevent,
# and an earlier revision of this file claimed `-e` covered it. `-L` is the half
# that sees the link itself. A directory is already covered by `-e`.
cae_marker_exists() {
    [ -e "$1" ] || [ -L "$1" ]
}

# cae_marker_held <path> — the marker's contents, flattened to one line, for the
# log. Never fails: an unreadable or dangling marker still has to be REPORTED.
cae_marker_held() {
    local held
    if [ -L "$1" ] && [ ! -e "$1" ]; then
        printf '(dangling symlink -> %s)' "$(readlink "$1" 2>/dev/null || printf '?')"
        return 0
    fi
    if [ -d "$1" ]; then
        printf '(a directory)'
        return 0
    fi
    held=$(tr '\n' ' ' < "$1" 2>/dev/null) || held=""
    [ -n "$held" ] || held="(empty or unreadable)"
    printf '%s' "$held"
}

# cae_pause_markers <tune_dir> [extra_pause_file]
#
# Prints, one per line, every path the PAUSE MACHINERY itself treats as a pause
# marker and that is visible from outside the Ray actor.
#
# ⚑ THE ROOT pause.txt IS NOT THE SET. `_resolve_pause_marker_paths`
# (tune/trainable_config_ops.py) honours FOUR sources — `tc.pause_file`, the
# per-trial `trial_dir/pause.txt`, the ephemeral Ray-session tune root, and
# `<work_dir>/tune/pause.txt` — and `train_watchdog.find_pause_txt` searches the
# tune dir RECURSIVELY (`rglob("pause.txt")`). A guard that checked only the
# root marker therefore let a PER-TRIAL pause through, and recovery SIGKILLed a
# deliberately parked run. The recursive scan below is the same set
# `find_pause_txt` walks; `tests/test_recover_stall_guard.py` pins that
# relationship directly rather than by assertion in a comment.
#
# ⚑ DELIBERATELY STRICTER THAN `find_pause_txt` IN ONE DIRECTION: no `-type f`,
# so a directory or a DANGLING SYMLINK named pause.txt is also a marker here.
# `find_pause_txt` requires `is_file()`. Stricter is the safe direction for a
# guard — it can only ever refuse more.
#
# The ephemeral Ray-session tune root is NOT reachable from here (it lives under
# /tmp/ray/session_*/artifacts/.../driver_artifacts/ and is named by the actor).
# Say so rather than implying full coverage: a pause held ONLY at the ephemeral
# path is not seen by this guard — nor by `train_watchdog.find_pause_txt`, which
# is the instrument that produces the STALLED verdict in the first place.
cae_pause_markers() {
    local tune_dir="${1:-}" extra="${2:-}"
    if [ -n "$extra" ]; then
        printf '%s\n' "$extra"
    fi
    if [ -n "$tune_dir" ] && [ -d "$tune_dir" ]; then
        find "$tune_dir" -name pause.txt 2>/dev/null
    fi
}

# intentional_stop_guard <override:0|1> <pause_overridable:0|1> <logfn> \
#                        <stop_marker> [pause_marker...]
#
# Returns 0 when nothing blocks, $INTENTIONAL_STOP_EXIT otherwise, after logging
# one line per marker that caused the refusal (naming the path AND what it
# holds, because "refused" without "which file" leaves the operator guessing).
#
# `pause_overridable` is the caller's answer to "can I clear a pause marker
# correctly if I proceed?". `recover_stall.sh` says 1: it restarts through
# `train.sh start`, whose `clear_pause_markers` renames markers aside properly.
# `watchdog_pbt.sh` says 0: it launches the trainer directly, so an overridden
# pause would park the new trial instantly while its log said "Restarted".
#
# The log function is passed in rather than assumed: each caller already owns a
# `log()` that tees to its own file, and a guard that printed to stdout would be
# swallowed by the `>> "$LOG" 2>&1` its callers are invoked under.
intentional_stop_guard() {
    local override="$1" pause_overridable="$2" logfn="$3" stop_marker="$4"
    shift 4
    local guard refuse=0 overrode=0 pause_blocked=0

    if cae_marker_exists "$stop_marker"; then
        if [ "$override" = 1 ]; then
            overrode=1
            "$logfn" "OVERRIDE $CAE_IGNORE_STOP_FLAG: ignoring INTENTIONAL-STOP marker $stop_marker — held: $(cae_marker_held "$stop_marker")"
        else
            refuse=1
            "$logfn" "REFUSING (exit $INTENTIONAL_STOP_EXIT): INTENTIONAL-STOP marker present: $stop_marker — held: $(cae_marker_held "$stop_marker")"
        fi
    fi

    for guard in "$@"; do
        [ -n "$guard" ] || continue
        cae_marker_exists "$guard" || continue
        if [ "$override" = 1 ] && [ "$pause_overridable" = 1 ]; then
            overrode=1
            "$logfn" "OVERRIDE $CAE_IGNORE_STOP_FLAG: ignoring PAUSE marker $guard — held: $(cae_marker_held "$guard")"
        else
            # ⚑ WITH override=1 AND pause_overridable=0 THIS IS THE BRANCH THAT
            # FIRES, and saying so is the point: a caller that cannot clear the
            # marker must not pretend it restarted. Silent success here is the
            # "accepted then ignored" shape one level up from the knob.
            refuse=1
            [ "$override" = 1 ] && pause_blocked=1
            "$logfn" "REFUSING (exit $INTENTIONAL_STOP_EXIT): PAUSE marker present: $guard — held: $(cae_marker_held "$guard")"
        fi
    done

    if [ "$refuse" = 1 ]; then
        if [ "$pause_blocked" = 1 ]; then
            "$logfn" "REFUSING: $CAE_IGNORE_STOP_FLAG does NOT override a PAUSE on this path — this restart cannot clear a pause marker safely, so the trial it started would park at its own pause check while this log claimed success. Release the pause (scripts/pause_window.sh / scripts/graceful_restart.py) or remove the marker(s) above, then re-run."
        else
            "$logfn" "REFUSING: training was stopped or paused ON PURPOSE and will NOT be restarted. Remove the marker(s) named above, or re-run with $CAE_IGNORE_STOP_FLAG to override deliberately."
        fi
        return "$INTENTIONAL_STOP_EXIT"
    fi

    # ⚑ ONLY WHEN SOMETHING WAS ACTUALLY OVERRIDDEN. A run that passed the flag
    # and met no marker did not restart "against a deliberate stop", and logging
    # that it did would put a false incident line in the operator's record.
    if [ "$overrode" = 1 ]; then
        "$logfn" "OVERRIDE $CAE_IGNORE_STOP_FLAG: RESTARTING PRODUCTION TRAINING against a deliberate stop, on explicit operator instruction."
    fi
    return 0
}
