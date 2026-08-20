#!/usr/bin/env bash
# Worker discovery + drain classification for scripts/pause_window.sh.
#
# ⚑ WHY THIS FILE EXISTS — TWO DEFECTS IN ONE INCIDENT (2026-08-20, the aborted
# lc0-control window):
#
#   1. `kill -0` CANNOT TELL A ZOMBIE FROM A LIVE PROCESS. A parked trial stops
#      reaping its children (reaping happens in the iteration loop, which is
#      exactly what "parked" suspends), so a worker that exits cleanly on
#      SIGTERM stays visible as state Z until release. All four workers drained
#      correctly, `kill -0` read all four zombies as survivors, and the window
#      aborted a fully clean drain — a deadlock BY CONSTRUCTION, since the
#      zombies could not be reaped until the very release the gate was blocking.
#   2. Discovery was `pgrep -f` — the pattern-match-then-signal shape that
#      self-matched a wait-loop's own shell on 2026-08-10 (4h46m outage) and is
#      banned in this repo. It also cannot bind the match to THE trial: any
#      worker of any trial, or an unrelated process quoting the module name in
#      its argv, matched.
#
# Discovery here is READ-ONLY /proc walking: a process is a worker of trial T
# iff its cmdline matches WORKER_PATTERN *and* carries `--trial-id T`, and all
# discovered workers must share one parent (the Ray trainable). Nothing is ever
# signalled by name: signals go to explicit PIDs, and only after re-verifying
# the PID's kernel start time against the banked value, so a recycled PID can
# never receive our TERM.
#
# The drain rule:
#     gone                          => drained
#     same start time + state Z     => drained (exited; parked parent can't reap)
#     same start time + state != Z  => survivor
#     start time changed            => the original worker is gone and the PID
#                                      was recycled => drained; NEVER signal it
#
# Injection points, FOR TESTS ONLY — production never sets either:
#   CAE_PAUSE_PROC_ROOT  fake /proc tree (dirs with cmdline/stat/status files)
#   CAE_PAUSE_KILL_CMD   records/simulates the TERM instead of sending it

PAUSE_PROC_ROOT="${CAE_PAUSE_PROC_ROOT:-/proc}"
PAUSE_KILL_CMD="${CAE_PAUSE_KILL_CMD:-kill}"

# starttime = /proc/<pid>/stat field 22, in clock ticks since boot — the kernel's
# own identity stamp for this incarnation of the PID. Parsed AFTER stripping
# everything through the LAST ") ", because field 2 (comm) is parenthesised and
# may itself contain spaces or parentheses; a naive awk split misreads any such
# comm. After the strip, field 1 is state (overall 3), so overall 22 is 20.
pause_proc_start_time() {
    local stat rest
    stat="$(cat "$PAUSE_PROC_ROOT/$1/stat" 2>/dev/null)" || return 1
    rest="${stat##*) }"
    # shellcheck disable=SC2086
    set -- $rest
    [ -n "${20:-}" ] || return 1
    printf '%s\n' "${20}"
}

pause_proc_state() {
    awk '/^State:/{print $2; found=1} END{exit !found}' \
        "$PAUSE_PROC_ROOT/$1/status" 2>/dev/null
}

pause_proc_ppid() {
    local stat rest
    stat="$(cat "$PAUSE_PROC_ROOT/$1/stat" 2>/dev/null)" || return 1
    rest="${stat##*) }"
    # shellcheck disable=SC2086
    set -- $rest
    [ -n "${2:-}" ] || return 1
    printf '%s\n' "$2"
}

# Emit one line per worker of trial $1: "pid starttime ppid". Read-only.
# ⚑ ZOMBIES ARE INVISIBLE HERE, AND THAT IS CORRECT. The kernel frees a
# zombie's argv, so its /proc/<pid>/cmdline is EMPTY and no pattern can match
# it (verified by the synthetic test: an unreaped zombie worker vanished from
# the snapshot while its stat/status stayed readable). Discovery feeds only
# "what to signal" and "what counts as live contention" — a zombie is neither.
# Classification of already-banked PIDs goes through pause_worker_drain_state
# with the BANKED start time and never needs cmdline, which is why the banked
# value must come from a snapshot taken while the worker was alive.
# ⚑ The trial-id match is on DISTINCT argv WORDS (space-delimited after the NUL
# transform), so trial `dea5e_0000` cannot match a worker of `dea5e_00000`.
# Our own process is excluded by PID, not by pattern — the pattern lives in
# this FILE, not in our argv, so self-match is structurally impossible, but the
# exclusion costs nothing and survives a future caller that does quote it.
pause_worker_snapshot() {
    local trial="$1" d pid cmd start ppid
    for d in "$PAUSE_PROC_ROOT"/[0-9]*; do
        pid="${d##*/}"
        [ "$pid" = "$$" ] && continue
        cmd="$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)" || continue
        [[ "$cmd" =~ $WORKER_PATTERN ]] || continue
        case " $cmd" in
            *" --trial-id $trial "*) ;;
            *) continue ;;
        esac
        start="$(pause_proc_start_time "$pid")" || continue
        ppid="$(pause_proc_ppid "$pid")" || continue
        printf '%s %s %s\n' "$pid" "$start" "$ppid"
    done
    return 0
}

# gone | zombie | reused | alive — see the drain rule in the header.
pause_worker_drain_state() {
    local pid="$1" banked="$2" start state
    start="$(pause_proc_start_time "$pid")" || { echo gone; return; }
    if [ "$start" != "$banked" ]; then echo reused; return; fi
    state="$(pause_proc_state "$pid")" || { echo gone; return; }
    if [ "$state" = "Z" ]; then echo zombie; else echo alive; fi
}

# TERM the ORIGINAL incarnation only: re-read the start time at signal moment
# and refuse a recycled PID. $3 is the logger (defaults to `:`).
pause_term_if_same_start() {
    local pid="$1" banked="$2" logfn="${3:-:}" start
    start="$(pause_proc_start_time "$pid")" || return 0     # already gone
    if [ "$start" != "$banked" ]; then
        "$logfn" "pid $pid was RECYCLED (start $start != banked $banked) -- not signalling"
        return 0
    fi
    "$PAUSE_KILL_CMD" -TERM "$pid" 2>/dev/null || true
}

# ── Marker lease ─────────────────────────────────────────────────────────────
# `setsid`/`nohup` protect the launcher from a lost shell, but nothing protects
# the MARKER from a lost launcher: the cleanup trap does not fire on SIGKILL or
# a host reboot, and an orphaned marker parks production indefinitely — the
# script's own "one unrecoverable mistake". This watchdog is a detached process
# that removes the marker when (a) the launcher PID is gone while the marker
# remains, or (b) a hard deadline passes. Both paths are LOUD (appended to the
# lease log) and both are lose-lose tradeoffs made in the same direction: a
# possibly-contended measurement is recoverable, an indefinitely parked
# production run is not.
# The launcher kills the watchdog BY EXPLICIT PID during normal release.
pause_start_lease_watchdog() {
    local marker="$1" owner="$2" deadline="$3" wlog="$4"
    setsid bash -c '
        marker="$1"; owner="$2"; deadline="$3"; wlog="$4"
        start="$(date +%s)"
        while sleep 30; do
            [ -e "$marker" ] || exit 0
            if ! kill -0 "$owner" 2>/dev/null; then
                rm -f "$marker"
                printf "[pause-lease] %s launcher pid %s is GONE with the marker still present -- marker removed; if the window job survived it, training resumes BESIDE it\n" \
                    "$(date -Is)" "$owner" >> "$wlog"
                exit 0
            fi
            now="$(date +%s)"
            if [ $((now - start)) -ge "$deadline" ]; then
                rm -f "$marker"
                printf "[pause-lease] %s hard deadline %ss reached with launcher pid %s still alive -- marker removed; the window job may still be running and training resumes BESIDE it\n" \
                    "$(date -Is)" "$deadline" "$owner" >> "$wlog"
                exit 0
            fi
        done
    ' lease-watchdog "$marker" "$owner" "$deadline" "$wlog" >/dev/null 2>&1 &
    printf '%s\n' "$!"
}
