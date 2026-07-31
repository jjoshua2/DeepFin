"""`train.sh stop` must SIGTERM the selfplay workers and wait for them, so their
handler can suspend in-flight games before ray is torn down — and must then
report what actually happened, read off the workers' own logs.

These tests execute the blocks EXTRACTED FROM THE REAL FILE, not a copy of
them, so an edit to `scripts/train.sh` that breaks the contract fails here. The
blocks are run against real short-lived processes with `pgrep` stubbed to name
them.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

TRAIN_SH = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"

_BEGIN = "── Drain selfplay BEFORE anything is killed"
_END = 'echo "Stopping PID $pid ..."'

# The second pass, which runs AFTER ray teardown against orphans. It lives
# outside the first block's range, so it needs its own extraction or it is
# exactly the untested-second-caller shape this whole change is about.
_ORPHAN_BEGIN = "── Second drain pass: ORPHANS"

# The verdict is emitted after BOTH passes, because the orphan pass sends a
# second SIGTERM and is therefore a worker's LAST chance to bank. It is a third
# extractable block, run in the same function as the other two so it sees their
# `local`s exactly the way it does inside stop().
_VERDICT_BEGIN = "── Selfplay drain VERDICT"
_VERDICT_END = 'echo "Stopped"'
_ORPHAN_END = _VERDICT_BEGIN


def _block(begin: str, end: str) -> str:
    lines = TRAIN_SH.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if begin in ln]
    ends = [i for i, ln in enumerate(lines) if end in ln]
    assert len(starts) == 1, f"begin marker not unique ({begin!r}): {starts}"
    assert len(ends) == 1, f"end marker not unique ({end!r}): {ends}"
    assert starts[0] < ends[0]
    return "\n".join(lines[starts[0]:ends[0]])


def _drain_block() -> str:
    return _block(_BEGIN, _END)


def _worker_pattern() -> str:
    """The single `local wpat=...` definition, read out of the real file.

    Read rather than duplicated: a copy here would pass while production's
    pattern was broken, which is the defect this file exists to catch.
    """
    found = re.findall(r"^\s*local wpat='(.+)'$", TRAIN_SH.read_text(), re.M)
    assert len(found) == 1, f"expected exactly one wpat definition, got {found}"
    return found[0]


def _run(script: str, *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


# ---------------------------------------------------------------------------
# The pgrep stub
# ---------------------------------------------------------------------------
#
# `pgrep` is stubbed rather than real because the real one would match — and
# then SIGTERM — the four production workers running on this machine.
#
# The stub honours BOTH of the things the production call depends on: liveness
# and the PATTERN. Liveness, because a fixed list cannot tell a pid-capture
# drain from a pgrep-polling one (both converge), and it is what lets a test
# publish a process partway through the way the driver's
# `revive_dead_selfplay_processes` does. The pattern, because it is the drain's
# ONLY production selector.
#
# 2026-07-31: the pattern half DID NOT WORK and this comment claimed it did.
# The old parser was
#     for a in "$@"; do case "$a" in --) ;; -*) ;; *) pat="$a" ;; esac; done
# and the production pattern is `-m chess_anti_engine\.worker( |$)`, which
# STARTS WITH `-`. So `-*)` ate it, `pat` stayed empty, and the match degraded
# to `grep -qE -- ""` — which matches every non-empty cmdline. Mutating the
# production pattern to a module name that cannot exist left every
# process-executing test in this file green. A written-down guard that cannot
# fire is worse than no guard, so the parser now mirrors the real pgrep: `--`
# ends option processing and everything after it is the pattern even when it
# starts with `-`, and an EMPTY pattern is a hard error rather than a wildcard.
# `test_the_pgrep_stub_honours_the_pattern_it_is_given` pins that directly, so
# the next regression here fails without needing a production-side mutation.
_PGREP_STUB = r"""
pgrep() {
    local a p pat="" endopts=0
    for a in "$@"; do
        if [ "$endopts" -eq 0 ]; then
            case "$a" in
                --) endopts=1; continue ;;
                -*) continue ;;
            esac
        fi
        pat="$a"
    done
    if [ -z "$pat" ]; then
        echo "PGREP_STUB_NO_PATTERN in argv: $*" >&2
        return 2
    fi
    while read -r p; do
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
            if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE -- "$pat"; then
                echo "$p"
            fi
        fi
    done < "$POOL_FILE"
}
"""


_FAKE_WORKER_ARGV = (
    "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker"
)

# The two lines the verdict reads. Both are real production strings:
#   selfplay/resume.py:549  "selfplay resume: suspended games=%d records=%d ..."
#   worker.py:1069          "shutdown requested (signal %d); exiting after suspend"
#   worker.py:3249          "selfplay resume: suspend failed; games abandoned"
_SUSPEND_LINE = (
    "INFO chess_anti_engine.selfplay.resume selfplay resume: "
    "suspended games={n} records=7 skipped=0 empty_slots=0 [] label_refetch=0 dir=/x"
)
_GRACEFUL_LINE = (
    "INFO chess_anti_engine.worker shutdown requested (signal 15); "
    "exiting after suspend"
)
_FAILED_LINE = (
    "ERROR chess_anti_engine.worker selfplay resume: suspend failed; games abandoned"
)

# What the victim's SIGTERM handler does. These are the states the verdict has
# to tell apart, and every one of them was observed on the 2026-07-31 teardown
# or is directly reachable from worker.py.
_ACTIONS = {
    # No handler at all: the default disposition kills it. THE worker_00 case —
    # it dies having banked nothing, and it is the case the warning exists for.
    "die": "    pass",
    # Survives the whole grace period without recording anything.
    "ignore": "    pass",
    # The common real case: banks and exits before the deadline.
    "bank_and_exit": "    log(SUSPEND)\n    log(GRACEFUL)\n    sys.exit(0)",
    # The 2026-07-31 regression: banked, but still tearing down its CUDA
    # context when the deadline passed, so liveness called it a discard.
    "bank_and_stay": "    log(SUSPEND)",
    # A worker between sessions: reaches the graceful exit with nothing in
    # flight. Never logs a suspend line, and is NOT a loss.
    "graceful_empty": "    log(GRACEFUL)\n    sys.exit(0)",
    # worker.py's blanket handler around the suspend. Partial or total, it is a
    # loss and must be loud.
    "suspend_failed": "    log(FAILED)\n    sys.exit(0)",
    # Ignores the first pass entirely and banks on the orphan pass's SIGTERM.
    "bank_on_second": (
        "    if len(SEEN) < 2:\n"
        "        return\n"
        "    log(SUSPEND)\n"
        "    log(GRACEFUL)\n"
        "    sys.exit(0)"
    ),
}


def _victim(
    n: int,
    *,
    action: str = "die",
    suspend: int = 0,
    stale: int = 0,
    log_file: bool = True,
    missing_log: bool = False,
    publish: bool = True,
) -> str:
    """A worker-shaped victim process, gated on readiness before it is published
    to the stub's pool.

    THE GATE IS THE POINT. The victims used to be
    `bash -c 'exec -a "<argv>" sleep 300' & echo $! >> "$POOL_FILE"`, which
    registers the pid BEFORE `execve` completes — and /proc/<pid>/cmdline is
    EMPTY during that window (measured 2026-07-31: 84/200 = 42% when sampled
    immediately after fork). An empty cmdline does not match the worker
    pattern, so the victim was invisible to `pgrep`, `wpids` came back empty
    and the drain drained nothing: two tests could pass VACUOUSLY, and did
    flake under load when the window widened.

    So the victim touches a READY file of its own after startup, the harness
    waits for it, and only then appends the pid to `$POOL_FILE` — the pool is
    the stub's whole universe, so publishing last makes visibility atomic and
    removes the race rather than papering over it. The pid is appended even if
    the gate times out, so the reaper still gets it, and `VICTIM_NEVER_READY`
    fails the test loudly instead.

    Not `exec -a`: that sets argv[0] to one string, so `--log-file` would not
    be a separate cmdline element and the drain's `awk '/^--log-file$/{getline}'`
    parse could never find a value — an early version of these tests failed for
    exactly that reason against production code that was correct. A real
    worker's argv has them as distinct elements: `distributed_runtime.py:863`
    appends `"--log-file", str(worker_log)` as two list items and `Popen` gets a
    LIST, so `--log-file=X` cannot occur.

    ``stale`` writes a suspend line BEFORE the drain starts (the negative
    control). ``log_file=False`` omits `--log-file` entirely, which is what
    every volunteer launch in README.md does. ``missing_log`` names a log that
    is deleted before the drain. ``publish=False`` withholds the pid from the
    pool so the caller can publish it later, simulating a mid-drain revival.
    """
    assert action in _ACTIONS, action
    install = {
        "die": "pass",
        "ignore": "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
    }.get(action, "signal.signal(signal.SIGTERM, on_term)")
    stale_stmt = f'log(SUSPEND_FMT.format(n={stale}))' if stale else "pass"
    logarg = "" if not log_file else f'--log-file "$WLOG{n}" '
    rm_log = f'rm -f "$WLOG{n}"' if missing_log else ""
    publish_line = f'echo "$V{n}" >> "$POOL_FILE"' if publish else ""
    return f"""
PYF{n}=$(mktemp --suffix=.py)
READY{n}=$(mktemp -u)
WLOG{n}=$(mktemp)
cat > "$PYF{n}" <<'PYEOF'
import signal, sys, time

argv = sys.argv
READY = argv[argv.index("--ready") + 1]
LOG = argv[argv.index("--log-file") + 1] if "--log-file" in argv else None
SUSPEND_FMT = {_SUSPEND_LINE!r}
SUSPEND = SUSPEND_FMT.format(n={suspend})
GRACEFUL = {_GRACEFUL_LINE!r}
FAILED = {_FAILED_LINE!r}
SEEN = []


def log(msg):
    if LOG is None:
        return
    with open(LOG, "a") as fh:
        fh.write(msg + "\\n")
        fh.flush()


def on_term(signum, frame):
    SEEN.append(signum)
{_ACTIONS[action]}


{stale_stmt}
{install}
open(READY, "w").close()
time.sleep(300)
PYEOF
python3 "$PYF{n}" -m chess_anti_engine.worker --work-dir /tmp/fake_worker \
    {logarg}--ready "$READY{n}" >/dev/null 2>&1 </dev/null &
V{n}=$!
for _ in $(seq 200); do
    if [ -f "$READY{n}" ]; then break; fi
    sleep 0.05
done
if [ ! -f "$READY{n}" ]; then echo "VICTIM_NEVER_READY {n}"; fi
{rm_log}
{publish_line}
"""


# What a REPLACEMENT process writes into the dead worker's log. Every launch
# emits `worker starting version=` (worker.py:669) before it can emit anything
# else, which is what makes it usable as the identity boundary.
_WORKER_STARTING_LINE = (
    "INFO chess_anti_engine.worker worker starting version=0.0.2 protocol=2"
)


def _replacement_appender(n: int, *, lines: list[str]) -> str:
    """Simulate `revive_dead_selfplay_processes` relaunching into the SAME
    `worker.log` while the drain is watching it.

    Gated on the victim's DEATH, not on a timer: the appended bytes must land
    after the drain recorded its byte offset (which happens before the SIGTERM)
    and before the verdict reads it. Waiting for the pid to disappear puts the
    append strictly inside that window instead of hoping a `sleep` lands there.
    `REPLACEMENT_NEVER_APPENDED` fails the test if it did not happen at all.
    """
    body = "\n".join(
        f'    printf \'%s\\n\' "$(date +%%F) {ln}" >> "$WLOG{n}"' for ln in lines
    )
    return f"""
REPL_DONE{n}=$(mktemp -u)
( while kill -0 "$V{n}" 2>/dev/null; do sleep 0.05; done
{body}
  : > "$REPL_DONE{n}"
) >/dev/null 2>&1 </dev/null &
"""


def _harness(
    victim: str, *, grace: int, post: str = "", post_drain: str = "",
    orphan_pass: bool = False, orphan_grace: int = 4,
) -> str:
    """Run the real drain block, then the real VERDICT block, under the same
    `set -e` train.sh uses, against real processes.

    Both blocks run inside ONE function, as they do inside `stop()`, so the
    verdict sees the drain's `local` variables (`wpids`, `dstate`, `grace`)
    exactly the way production does. Extracting them into separate functions
    would give the verdict an empty `$wpids` and make every assertion vacuous.

    What sits BETWEEN them in production — `kill "$pid"`, `ray stop`,
    `pkill -9 -f 'ray::'` — is deliberately absent: it is outside every marker,
    and running it here would tear down this machine's real ray.

    ``orphan_pass`` splices in the second drain pass, which is where a worker's
    LAST chance to bank is. Off by default so the common tests do not pay its
    grace period. ``post_drain`` injects shell between the drain and the
    verdict — the window in which an early exit would leak the snapshot dir.
    """
    orphan = _block(_ORPHAN_BEGIN, _ORPHAN_END) if orphan_pass else ""
    return f"""
set -e
export CAE_STOP_GRACE_SECONDS={grace}
export CAE_ORPHAN_GRACE_SECONDS={orphan_grace}
POOL_FILE=$(mktemp)
{_PGREP_STUB}
{victim}
drain() {{
{_drain_block()}
# Production has `echo "Stopping PID $pid ..."` and the driver kill here. That
# echo is load-bearing for `set -e` — it is what keeps the drain's non-zero
# for-loop status from ending the function — so it is modelled, not skipped.
echo "REACHED_DRIVER_KILL"
{post_drain}
{orphan}
{_block(_VERDICT_BEGIN, _VERDICT_END)}
}}
drain
echo "REACHED_END_OF_STOP"
{post}
# Reap the stubs. `wait` would block forever on a victim that survived the
# drain, which is the expected state in several of these tests. The stubs also
# redirect stdout, or a surviving grandchild would hold the pipe open and
# subprocess.run() would block on EOF long after bash exited.
for j in $(jobs -p); do kill -9 "$j" 2>/dev/null || true; done
for p in $(cat "$POOL_FILE"); do kill -9 "$p" 2>/dev/null || true; done
rm -f "$POOL_FILE"
"""


# ---------------------------------------------------------------------------
# The harness's own guards
# ---------------------------------------------------------------------------


def test_the_pgrep_stub_honours_the_pattern_it_is_given() -> None:
    """A CONTROL ON THE CONTROL.

    Every process-executing test in this file selects its victims through the
    stub, so a stub that matches everything makes all of them assert nothing
    about the production pattern. This drives the stub directly: the real
    pattern must find the worker-shaped victim, a pattern that cannot match
    must find nothing, and a call with no pattern at all must ERROR rather than
    quietly behave like a wildcard.
    """
    script = f"""
set -e
POOL_FILE=$(mktemp)
{_PGREP_STUB}
{_victim(1)}
echo "REAL: $(pgrep -f -- {shlex.quote(_worker_pattern())} | tr '\\n' ' ')"
echo "NOMATCH: $(pgrep -f -- '-m chess_anti_engine\\.nosuchmodule( |$)' || true)"
echo "NOPAT: $(pgrep -f || echo "rc=$?")"
echo "V1=$V1"
for p in $(cat "$POOL_FILE"); do kill -9 "$p" 2>/dev/null || true; done
rm -f "$POOL_FILE"
"""
    r = _run(script)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    pid = r.stdout.split("V1=")[1].strip()
    assert f"REAL: {pid}" in r.stdout, r.stdout + r.stderr
    assert "NOMATCH: \n" in r.stdout + "\n", (
        f"a non-matching pattern found something:\n{r.stdout}"
    )
    assert "NOPAT: rc=2" in r.stdout, f"a pattern-less call did not error:\n{r.stdout}"
    assert "PGREP_STUB_NO_PATTERN" in r.stderr, r.stderr


# ---------------------------------------------------------------------------
# First pass: the workers get signalled and waited for
# ---------------------------------------------------------------------------


def test_workers_are_signalled_and_the_wait_ends_when_they_exit() -> None:
    victim = (
        _victim(1, action="bank_and_exit", suspend=5)
        + _victim(2, action="bank_and_exit", suspend=6)
    )
    r = _run(_harness(victim, grace=30))
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "Draining selfplay workers" in r.stdout, r.stdout + r.stderr
    assert "Selfplay drained cleanly: 11 in-flight game(s) suspended" in r.stdout, (
        r.stdout + r.stderr
    )
    assert "WARNING" not in r.stdout, r.stdout
    # THE POINT: a clean drain must not stop stop(). The block runs under
    # `set -e` and the only thing keeping its non-zero for-loop status from
    # aborting the function is that another command follows it — so this
    # assertion, not the drain output above, is what catches a rearrangement
    # that leaves training half-stopped with the driver still up.
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 0, r.stderr


def test_a_worker_that_ignores_sigterm_warns_instead_of_hanging() -> None:
    r = _run(_harness(_victim(1, action="ignore"), grace=4))
    # If the victim were signalled before it installed SIG_IGN it would die and
    # this would silently become a different case, asserting nothing.
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "still running after 4s" in r.stdout, r.stdout + r.stderr
    assert "NO suspend recorded" in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr


def test_a_revived_worker_does_not_extend_the_wait() -> None:
    """The driver's `revive_dead_selfplay_processes` relaunches a worker that
    exits mid-iteration. The drain waits on the pids captured ONCE; if it
    re-ran `pgrep` instead it would keep finding the replacement and burn the
    whole grace period, then warn falsely.

    The revival becomes VISIBLE 1s in — after the drain captured its pid list
    (t~0) and before its first liveness check (t~2). It is never signalled and
    outlives the wait, exactly like a worker the driver relaunched mid-teardown.
    A revival landing after the first check would not discriminate the two loop
    shapes, because the captured list has already gone empty by then.

    Visibility, not launch, is what is delayed. The process is started and
    READY-gated up front and merely withheld from `$POOL_FILE` — the stub's
    entire universe — until t=1s. Launching it at t=1s instead would put python
    startup inside the 1s budget and, worse, could leave it never visible at
    all, in which case a pgrep-polling drain would also converge and this test
    would pass while discriminating nothing. `REVIVAL_UNPUBLISHED` asserts the
    publish actually happened.

    The revival is not in `$wpids`, so the verdict says nothing about it — by
    design: it is seconds old with nothing in flight, and the orphan pass is
    what stops it reclaiming the files this drain just banked.
    """
    victim = (
        _victim(1, action="bank_and_exit", suspend=3)
        + _victim(2, publish=False)
        + """
( sleep 1; echo "$V2" >> "$POOL_FILE"; sleep 60 ) >/dev/null 2>&1 </dev/null &
"""
    )
    post = """
if ! grep -qx "$V2" "$POOL_FILE"; then echo "REVIVAL_UNPUBLISHED"; fi
"""
    r = _run(_harness(victim, grace=30, post=post), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "REVIVAL_UNPUBLISHED" not in r.stdout, r.stdout
    assert "Selfplay drained cleanly: 3 in-flight game(s) suspended" in r.stdout, (
        r.stdout + r.stderr
    )
    assert "WARNING" not in r.stdout, r.stdout
    # A pgrep-until-empty loop would have spent the full 30s and then warned.
    waited = int(r.stdout.split("all workers exited within ")[1].split("s")[0])
    assert waited <= 6, f"drain waited {waited}s; it should not track revivals"


def test_no_workers_is_a_no_op() -> None:
    r = _run(_harness("", grace=30))
    assert "Draining selfplay workers" not in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout
    assert "WARNING" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# Second pass: orphans
# ---------------------------------------------------------------------------


def _orphan_harness(victim: str, *, grace: int) -> str:
    """The SECOND pass alone, which runs after ray teardown. `wpat` is injected
    from the real file's single definition, not restated here."""
    return f"""
set -e
export CAE_ORPHAN_GRACE_SECONDS={grace}
POOL_FILE=$(mktemp)
wpat='{_worker_pattern()}'
{_PGREP_STUB}
{victim}
second_pass() {{
{_block(_ORPHAN_BEGIN, _ORPHAN_END)}
}}
second_pass
echo "REACHED_END_OF_STOP"
# SIGKILL is asynchronous: the pid stays visible until the parent reaps it, so
# checking immediately would report a survivor that is already dead. Give each
# one up to 2s to disappear before calling it alive.
for p in $(cat "$POOL_FILE"); do
    for _ in $(seq 20); do
        if ! kill -0 "$p" 2>/dev/null; then break; fi
        sleep 0.1
    done
    if kill -0 "$p" 2>/dev/null; then echo "STILL_ALIVE $p"; fi
    kill -9 "$p" 2>/dev/null || true
done
for j in $(jobs -p); do kill -9 "$j" 2>/dev/null || true; done
rm -f "$POOL_FILE"
"""


def test_orphaned_workers_are_drained_after_ray_teardown() -> None:
    """Nothing before this point kills a worker — verified 2026-07-31 against
    /proc/<pid>/cmdline, no worker matches `ray::` or `raylet`. So a worker the
    driver revived DURING the first pass survives ray teardown, reuses the dead
    worker's work_dir, and would claim-by-rename the selfplay_resume/ files the
    first pass just banked."""
    r = _run(_orphan_harness(_victim(1), grace=30))
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "Draining orphaned workers" in r.stdout, r.stdout + r.stderr
    assert "STILL_ALIVE" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 0, r.stderr


def test_an_orphan_that_ignores_sigterm_is_killed_not_left_running() -> None:
    """Unlike the first pass, this one must never leave a survivor: there is no
    later stage to catch it, and `train.sh start` would race it."""
    r = _run(_orphan_harness(_victim(1, action="ignore"), grace=4))
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "Force killing orphaned workers" in r.stdout, r.stdout + r.stderr
    assert "STILL_ALIVE" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


def test_second_pass_is_a_no_op_on_a_clean_teardown() -> None:
    """It runs on every stop, so it must cost nothing when the first pass
    already drained everything."""
    r = _run(_orphan_harness("", grace=30))
    assert "Draining orphaned workers" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# The pattern itself
# ---------------------------------------------------------------------------


def test_the_worker_pattern_is_defined_once_and_used_by_both_passes() -> None:
    """Both passes must select workers the SAME way. A literal duplicated
    between them could be edited in one place, leaving the suite green while
    the second pass silently matched nothing — the untested-second-caller
    shape this whole change exists to fix."""
    text = TRAIN_SH.read_text()
    pattern = _worker_pattern()   # asserts exactly one definition

    assert text.count(f"'{pattern}'") == 1, "the pattern literal is duplicated"
    for block, name in (
        (_drain_block(), "first pass"),
        (_block(_ORPHAN_BEGIN, _ORPHAN_END), "second pass"),
    ):
        assert 'pgrep -f -- "$wpat"' in block, f"{name} does not use $wpat"
        assert pattern not in block.replace(f"local wpat='{pattern}'", ""), (
            f"{name} restates the pattern instead of using $wpat"
        )


def test_the_worker_pattern_excludes_neighbouring_module_names() -> None:
    """`chess_anti_engine\\.worker` unanchored also matches `.worker_pool`,
    `.worker_config` and `.worker_buffer`; a volunteer pool would be killed by
    a stop it has nothing to do with."""
    pattern = _worker_pattern()
    should_match = [
        _FAKE_WORKER_ARGV,
        "/usr/bin/python3 -m chess_anti_engine.worker",
    ]
    should_not_match = [
        "/usr/bin/python3 -m chess_anti_engine.worker_pool --respawn",
        "/usr/bin/python3 -m chess_anti_engine.worker_config",
        "python3 -m chess_anti_engine.inference.broker",
    ]
    for argv in should_match:
        r = _run(f"printf '%s' {shlex.quote(argv)} | grep -qE -- {shlex.quote(pattern)}")
        assert r.returncode == 0, f"pattern must match: {argv}"
    for argv in should_not_match:
        r = _run(f"printf '%s' {shlex.quote(argv)} | grep -qE -- {shlex.quote(pattern)}")
        assert r.returncode != 0, f"pattern must NOT match: {argv}"


def test_train_sh_still_parses() -> None:
    r = _run(f"bash -n {TRAIN_SH}")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "marker", [_BEGIN, _END, _ORPHAN_BEGIN, _VERDICT_BEGIN, _VERDICT_END],
)
def test_extraction_markers_exist(marker: str) -> None:
    assert marker in TRAIN_SH.read_text()


# ---------------------------------------------------------------------------
# THE VERDICT: three states, and liveness is evidence for none of them
# ---------------------------------------------------------------------------
#
# 2026-07-31, the first real teardown with the #291 handler. stop() printed
#     WARNING: workers still alive after 90s ( 2351838 2351840 2356433); their
#     in-flight games will be DISCARDED.
# Mapping those pids to workers afterwards:
#   worker_02 (2351838) banked 174 games, worker_03 (2351840) banked 206 —
#     both FALSE alarms; `kill -0` had caught them tearing down CUDA.
#   worker_00 (2356433) logged ZERO suspend lines and zero "exiting after
#     suspend" in its whole session and really did drop ~24 games — a TRUE
#     alarm.
# So the old rule (alive ⇒ discarded) was wrong 2 times in 3. The first attempt
# to fix it required a worker to be ALIVE to call it a loss, which would have
# reported worker_00 as a clean drain had it exited two seconds earlier: a
# noisy true-positive traded for a silent false-negative in the one case that
# matters. Both errors are the same error — asking liveness a question about
# data. These tests pin all three states of the replacement.


def test_a_worker_that_banked_its_games_is_not_reported_as_discarded() -> None:
    """THE 2026-07-31 REGRESSION. Still running at the deadline, but it
    suspended 23 games — that is a success, not a loss."""
    r = _run(_harness(_victim(1, action="bank_and_stay", suspend=23), grace=6),
             timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "suspended 23 in-flight game(s)" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "WARNING" not in r.stdout, r.stdout
    assert "Selfplay drained cleanly: 23 in-flight game(s) suspended" in r.stdout, (
        r.stdout
    )
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


def test_a_worker_that_dies_having_banked_nothing_is_a_loud_loss() -> None:
    """THE SHIP-BLOCKER, and the primary case the warning exists for.

    This is worker_00: it takes the default SIGTERM disposition, dies at once,
    and records nothing. The previous implementation's loss branch was
    `elif kill -0 "$w"`, so a worker that died having banked nothing fell off
    the end and printed `Workers drained after 2s (0 in-flight game(s)
    suspended)` with NO warning — real, certain data loss reported as a clean
    teardown. A loss verdict must NEVER require the worker to still be alive.
    """
    r = _run(_harness(_victim(1, action="die"), grace=6), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "all workers exited within" in r.stdout, r.stdout + r.stderr
    assert "NO suspend recorded" in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


def test_a_suspend_that_raised_is_reported_as_a_loss_even_though_it_banked() -> None:
    """`worker.py:3249` logs `suspend failed; games abandoned` from the blanket
    handler around the suspend, after which the rest of the table is dropped.
    Partial success is still a loss."""
    r = _run(_harness(_victim(1, action="suspend_failed"), grace=6), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "suspend FAILED" in r.stdout, r.stdout + r.stderr
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout


def test_a_worker_with_nothing_in_flight_is_not_reported_as_losing_games() -> None:
    """A worker between sessions reaches the graceful exit point without ever
    suspending anything. `suspended games=0` and "no suspend line at all" are
    indistinguishable by count, so the verdict keys on the graceful-exit line
    that `worker.py:1069` emits exactly once. Calling this data loss would be
    the same defect as the one being fixed, one layer down."""
    r = _run(_harness(_victim(1, action="graceful_empty"), grace=6), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "exited cleanly with no games in flight" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "Selfplay drained cleanly: 0 in-flight game(s) suspended" in r.stdout, (
        r.stdout
    )


def test_a_suspend_line_from_BEFORE_the_drain_is_not_credited() -> None:
    """NEGATIVE CONTROL. `suspended games=` is emitted by every in-session reco
    restart too, so a whole-file grep would find an hours-old line and read it
    as proof THIS drain worked. The block records each log's byte offset before
    signalling; this test fails if that offset is dropped."""
    r = _run(_harness(_victim(1, action="ignore", stale=99), grace=6), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    # NOT a bare `"99" not in r.stdout`, which is what this assertion used to
    # be: every line of the output carries a pid, and pid 2399696 failed it
    # against correct behaviour (observed 2026-07-31). A negative control that
    # fires on the pid it was handed is not a control. Assert the strings that
    # would only appear if the stale line HAD been credited.
    assert "suspended 99 in-flight game(s)" not in r.stdout, (
        f"stale pre-drain suspend line was credited:\n{r.stdout}"
    )
    assert "in-flight game(s) suspended" not in r.stdout, (
        f"a loss was summarised as a clean drain:\n{r.stdout}"
    )
    assert "NO suspend recorded" in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout


# --- the third state: could not verify -------------------------------------


def test_a_worker_with_no_log_file_is_could_not_verify_not_a_loss() -> None:
    """`worker.py` defaults `--log-file` to None and every volunteer launch in
    README.md omits it, so there is simply no evidence to read. Reporting that
    as DISCARDED is a lie an operator cannot check; reporting it as a clean
    drain is worse. It gets its own state and its own message."""
    r = _run(_harness(_victim(1, action="ignore", log_file=False), grace=6),
             timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "COULD NOT VERIFY" in r.stdout, r.stdout + r.stderr
    assert "no --log-file in its argv" in r.stdout, r.stdout
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


def test_a_worker_whose_log_is_missing_is_could_not_verify() -> None:
    """Second route into the same state: `--log-file` is in the argv but the
    file is not there when the offset is taken."""
    r = _run(_harness(_victim(1, action="ignore", missing_log=True), grace=6),
             timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "COULD NOT VERIFY" in r.stdout, r.stdout + r.stderr
    assert "log file does not exist" in r.stdout, r.stdout
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout


# --- ordering: the verdict comes after the LAST chance to bank --------------


def test_a_worker_that_banks_on_the_orphan_pass_is_credited() -> None:
    """The orphan pass sends a SECOND SIGTERM and waits again before SIGKILL,
    so it is a worker's last chance. If the verdict were printed at the end of
    the first pass — where it used to be — this worker would be named as a loss
    and nothing would ever retract it. Not a corner case: on 2026-07-31
    worker_02 finished suspending with 4s of margin on a 90s grace.
    """
    victim = _victim(1, action="bank_on_second", suspend=7)
    r = _run(_harness(victim, grace=4, orphan_pass=True, orphan_grace=10),
             timeout=60)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "still running after 4s" in r.stdout, r.stdout + r.stderr
    assert "Draining orphaned workers" in r.stdout, r.stdout + r.stderr
    assert "suspended 7 in-flight game(s)" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
    assert "Selfplay drained cleanly: 7 in-flight game(s) suspended" in r.stdout, (
        r.stdout
    )


def test_the_verdict_is_emitted_after_the_orphan_pass_not_before() -> None:
    """Pins the ORDER, not just the outcome. A rearrangement that moved the
    verdict back into the first pass would still pass the test above whenever
    the worker happened to bank in time, so the position is asserted directly.
    """
    text = TRAIN_SH.read_text()
    assert text.index(_ORPHAN_BEGIN) < text.index(_VERDICT_BEGIN), (
        "the verdict must be emitted after the orphan pass, which is the "
        "worker's last chance to bank"
    )
    assert _drain_block().count("DISCARDED") == 0, (
        "the first pass must not render a verdict; it runs before the orphan "
        "pass has given the worker its last chance"
    )


# --- identity: the offset separates TIME, not the process ------------------
#
# `revive_dead_selfplay_processes` relaunches with the SAME `worker_index`
# (distributed_runtime.py:1253), the artifact root and `worker.log` path are
# rebuilt identically (:722,:724), and `logging.FileHandler` appends. The
# driver survives the whole first drain pass, so a worker that dies during the
# grace period is revived INTO THE FILE THIS BLOCK IS READING, and its lines
# land after the recorded offset.
#
# Observed in production on the very teardown this feature was built from:
# worker_01/worker.log line 817 is a second `worker starting version=` at
# 13:51:29 — stop() began 13:50:16, deadline 13:51:46. It cost nothing only
# because that replacement was SIGKILLed while still importing.


def test_a_replacements_graceful_exit_is_not_credited_to_the_worker_that_died() -> None:
    """THE SHIP-BLOCKER OF THE SECOND ROUND, and the same silent false-negative
    as M1 arriving through a different door.

    The victim dies having banked nothing. Its replacement reaches the poll
    loop, is SIGTERMed, and logs `exiting after suspend` with nothing in flight
    — into the dead worker's log, after the offset. Reading that as the dead
    worker's evidence prints `exited cleanly with no games in flight` and
    `Selfplay drained cleanly` for a worker whose table was dropped.
    """
    victim = _victim(1, action="die") + _replacement_appender(
        1, lines=[_WORKER_STARTING_LINE, _GRACEFUL_LINE],
    )
    post = """
if [ ! -f "$REPL_DONE1" ]; then echo "REPLACEMENT_NEVER_APPENDED"; fi
"""
    r = _run(_harness(victim, grace=6, post=post), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "REPLACEMENT_NEVER_APPENDED" not in r.stdout, r.stdout
    assert "exited cleanly" not in r.stdout, (
        f"a replacement's graceful exit was credited to the dead worker:\n{r.stdout}"
    )
    assert "died and was REPLACED mid-drain having recorded nothing" in r.stdout, (
        r.stdout + r.stderr
    )
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout


def test_a_replacements_suspend_line_is_not_credited_to_the_worker_that_died() -> None:
    """Same door, larger lie: the replacement banks games of its own and the
    dead worker is credited with them. `suspended 42` for a worker that
    suspended nothing would also inflate the run-wide total an operator reads
    as the price of the teardown."""
    victim = _victim(1, action="die") + _replacement_appender(
        1, lines=[_WORKER_STARTING_LINE, _SUSPEND_LINE.format(n=42)],
    )
    post = """
if [ ! -f "$REPL_DONE1" ]; then echo "REPLACEMENT_NEVER_APPENDED"; fi
"""
    r = _run(_harness(victim, grace=6, post=post), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "REPLACEMENT_NEVER_APPENDED" not in r.stdout, r.stdout
    # Specific strings, never a bare `"42" not in r.stdout`: every line carries
    # a pid and pid 2399696 already broke one control in this file that way.
    assert "suspended 42 in-flight game(s)" not in r.stdout, (
        f"a replacement's suspend line was credited to the dead worker:\n{r.stdout}"
    )
    assert "in-flight game(s) suspended" not in r.stdout, (
        f"a loss was summarised as a clean drain:\n{r.stdout}"
    )
    assert "died and was REPLACED mid-drain having recorded nothing" in r.stdout, (
        r.stdout + r.stderr
    )
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout


def test_a_worker_that_banked_and_was_then_replaced_keeps_its_own_count() -> None:
    """The truncation must not throw away what the ORIGINAL recorded. This is
    worker_01 on 2026-07-31: it banked, exited, and was revived into the same
    log 23s later. Its 251 games are real and must still be credited — and only
    its own."""
    victim = _victim(1, action="bank_and_exit", suspend=11) + _replacement_appender(
        1, lines=[_WORKER_STARTING_LINE, _SUSPEND_LINE.format(n=42)],
    )
    post = """
if [ ! -f "$REPL_DONE1" ]; then echo "REPLACEMENT_NEVER_APPENDED"; fi
"""
    r = _run(_harness(victim, grace=6, post=post), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "REPLACEMENT_NEVER_APPENDED" not in r.stdout, r.stdout
    assert "suspended 11 in-flight game(s)" in r.stdout, r.stdout + r.stderr
    assert "Selfplay drained cleanly: 11 in-flight game(s) suspended" in r.stdout, (
        r.stdout
    )
    assert "suspended 53 in-flight game(s)" not in r.stdout, (
        f"11 and the replacement's 42 were summed:\n{r.stdout}"
    )
    assert "cleanly: 53" not in r.stdout, r.stdout
    assert "WARNING" not in r.stdout, r.stdout


# --- the remaining could-not-verify routes ---------------------------------


def test_a_failed_mktemp_makes_every_worker_could_not_verify() -> None:
    """`dstate=""`. Without its own state the block has no offsets at all, so
    NOTHING can be verified — including a drain that went perfectly. Degrading
    to either of the other two verdicts here is what the review rejected."""
    poison = """
mktemp() {
    if [ "$1" = "-d" ]; then return 1; fi
    command mktemp "$@"
}
"""
    r = _run(_harness(poison + _victim(1, action="bank_and_exit", suspend=9), grace=6),
             timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "COULD NOT VERIFY" in r.stdout, r.stdout + r.stderr
    assert "no snapshot dir (mktemp -d failed)" in r.stdout, r.stdout
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr


def test_a_log_that_vanishes_between_snapshot_and_verdict_is_could_not_verify() -> None:
    """Present when the offset was taken, gone when the evidence is read."""
    victim = _victim(1, action="ignore") + """
( sleep 2; rm -f "$WLOG1" ) >/dev/null 2>&1 </dev/null &
"""
    r = _run(_harness(victim, grace=6), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "COULD NOT VERIFY" in r.stdout, r.stdout + r.stderr
    assert "log disappeared during teardown" in r.stdout, r.stdout
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout


def test_the_snapshot_dir_is_not_leaked_when_stop_exits_early(tmp_path: Path) -> None:
    """The explicit `rm -rf` sits in the verdict block at the END of stop(), so
    anything that exits in between would leak the dir — on the filesystem with
    this repo's documented growth-leak history. A trap armed at creation covers
    it. Proven by aborting between the snapshot and the verdict.

    Run as a CHILD script, not inline: the leak is only observable after the
    shell that armed the trap has exited, and the block's own `trap ... EXIT`
    would displace any the harness tried to set for itself.
    """
    snaproot = tmp_path / "snaproot"
    snaproot.mkdir()
    poison = f"""
mktemp() {{
    if [ "$1" = "-d" ]; then command mktemp -d -p {snaproot}; return $?; fi
    command mktemp "$@"
}}
"""
    inner = tmp_path / "inner.sh"
    inner.write_text(_harness(
        poison + _victim(1, action="bank_and_exit", suspend=2),
        grace=6, post_drain='echo "FORCED_ABORT"; exit 7',
    ))
    r = _run(f"""
bash {inner}
echo "INNER_RC=$?"
if [ -z "$(ls -A {snaproot})" ]; then
    echo "SNAPROOT_EMPTY"
else
    echo "SNAPROOT_LEAKED: $(ls -A {snaproot})"
fi
""", timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "FORCED_ABORT" in r.stdout, r.stdout + r.stderr
    assert "INNER_RC=7" in r.stdout, r.stdout + r.stderr
    assert "drained cleanly" not in r.stdout, (
        f"the abort did not land between snapshot and verdict:\n{r.stdout}"
    )
    assert "SNAPROOT_LEAKED" not in r.stdout, (
        f"the snapshot dir outlived the aborted stop:\n{r.stdout}"
    )
    assert "SNAPROOT_EMPTY" in r.stdout, r.stdout + r.stderr


def test_the_set_e_hazard_in_the_snapshot_loop_cannot_abort_stop() -> None:
    """A failed offset capture must cost the EVIDENCE, never the teardown.

    `wc -c < "$logf" > "$dstate/$w.off"` was the last command of the snapshot
    loop body, unguarded, under this file's `set -e` — so ENOSPC or EROFS on
    /tmp exited the shell BEFORE `kill -TERM $wpids`, before the driver was
    signalled, before `ray stop`. The operator would see the "Draining" line,
    an errno and exit 1, with training still running.

    Reproduced by making the snapshot dir read-only, which is what a full or
    read-only /tmp looks like to these writes.
    """
    victim = _victim(1, action="bank_and_exit", suspend=4)
    # Wrap mktemp so the snapshot dir the block creates is unwritable.
    poison = """
mktemp() {
    if [ "$1" = "-d" ]; then
        local d
        d=$(command mktemp -d)
        chmod a-w "$d"
        echo "$d"
        return 0
    fi
    command mktemp "$@"
}
"""
    r = _run(_harness(poison + victim, grace=10), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    # The teardown completed: the workers were signalled and stop() ran on.
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 0, r.stdout + r.stderr
    # And the lost evidence is reported as lost evidence, not as a verdict.
    assert "COULD NOT VERIFY" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "drained cleanly" not in r.stdout, r.stdout
