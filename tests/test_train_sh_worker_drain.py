"""`train.sh stop` must SIGTERM the selfplay workers and wait for them, so their
handler can suspend in-flight games before ray is torn down.

These tests execute the drain block EXTRACTED FROM THE REAL FILE, not a copy of
it, so an edit to `scripts/train.sh` that breaks the contract fails here. The
block is run against real short-lived processes with `pgrep` stubbed to name
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
_ORPHAN_END = 'echo "Stopped"'


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


def _victim(n: int, *, ignore_term: bool = False, publish: bool = True) -> str:
    """A worker-shaped victim process, gated on readiness before it is published
    to the stub's pool.

    THE GATE IS THE POINT. The previous victims were
    `bash -c 'exec -a "<argv>" sleep 300' & echo $! >> "$POOL_FILE"`, which
    registers the pid BEFORE `execve` completes — and /proc/<pid>/cmdline is
    EMPTY during that window (measured 2026-07-31: 84/200 = 42% when sampled
    immediately after fork). An empty cmdline does not match the worker
    pattern, so the victim was invisible to `pgrep`, `wpids` came back empty
    and the drain drained nothing: `test_orphaned_workers_are_drained_after_ray
    _teardown` and `test_a_revived_worker_does_not_extend_the_wait` both passed
    VACUOUSLY, and flaked under load when the window widened.

    So the victim now touches a READY file of its own after startup, the
    harness waits for it, and only then appends the pid to `$POOL_FILE` — the
    pool is the stub's whole universe, so publishing last makes visibility
    atomic and removes the race entirely rather than papering over it. The pid
    is appended even if the gate times out, so the reaper still gets it, and
    `VICTIM_NEVER_READY` in stdout fails the test loudly instead.

    Not `exec -a`: that sets argv[0] to one string, so `--work-dir` and
    `--log-file` would not be separate cmdline elements, and the drain's
    `awk '/^--log-file$/{getline}'` parse could never find a value. A real
    worker's argv has them as distinct elements (verified 2026-07-31 with
    `pgrep -af` against the four live workers).

    ``ignore_term`` keeps the process alive through the grace period so the
    "still running" branch is the one under test. ``publish=False`` leaves the
    pid out of the pool so the caller can publish it later, which is how a
    mid-drain revival is simulated.
    """
    handler = (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)" if ignore_term else "pass"
    )
    publish_line = f'echo "$V{n}" >> "$POOL_FILE"' if publish else ""
    return f"""
PYF{n}=$(mktemp --suffix=.py)
READY{n}=$(mktemp -u)
cat > "$PYF{n}" <<'PYEOF'
import signal, sys, time
argv = sys.argv
READY = argv[argv.index("--ready") + 1]
{handler}
open(READY, "w").close()
time.sleep(300)
PYEOF
python3 "$PYF{n}" -m chess_anti_engine.worker --work-dir /tmp/fake_worker \
    --ready "$READY{n}" >/dev/null 2>&1 </dev/null &
V{n}=$!
for _ in $(seq 200); do
    if [ -f "$READY{n}" ]; then break; fi
    sleep 0.05
done
if [ ! -f "$READY{n}" ]; then echo "VICTIM_NEVER_READY {n}"; fi
{publish_line}
"""


def _harness(victim: str, *, grace: int, post: str = "") -> str:
    """Run the real drain block under the same `set -e` train.sh uses, against
    real processes.
    """
    return f"""
set -e
export CAE_STOP_GRACE_SECONDS={grace}
POOL_FILE=$(mktemp)
{_PGREP_STUB}
{victim}
drain() {{
{_drain_block()}
}}
drain
echo "REACHED_DRIVER_KILL"
{post}
# Reap the stubs. `wait` would block forever on a victim that survived the
# drain, which is the expected state in two of these tests. The stubs also
# redirect stdout, or a surviving grandchild would hold the pipe open and
# subprocess.run() would block on EOF long after bash exited.
for j in $(jobs -p); do kill -9 "$j" 2>/dev/null || true; done
for p in $(cat "$POOL_FILE"); do kill -9 "$p" 2>/dev/null || true; done
rm -f "$POOL_FILE"
"""


def test_the_pgrep_stub_honours_the_pattern_it_is_given() -> None:
    """A CONTROL ON THE CONTROL.

    Every process-executing test in this file selects its victims through the
    stub, so a stub that matches everything makes all of them assert nothing
    about the production pattern. This test drives the stub directly: the real
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
    assert "NOMATCH: \n" in r.stdout + "\n", f"a non-matching pattern found something:\n{r.stdout}"
    assert "NOPAT: rc=2" in r.stdout, f"a pattern-less call did not error:\n{r.stdout}"
    assert "PGREP_STUB_NO_PATTERN" in r.stderr, r.stderr


def test_workers_are_signalled_and_the_wait_ends_when_they_exit() -> None:
    victim = _victim(1) + _victim(2)
    r = _run(_harness(victim, grace=30))
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "Draining selfplay workers" in r.stdout, r.stdout + r.stderr
    assert "Workers drained after" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
    # THE POINT: a clean drain must not stop stop(). The block runs under
    # `set -e` and the only thing keeping its non-zero for-loop status from
    # aborting the function is that another command follows it — so this
    # assertion, not the drain output above, is what catches a rearrangement
    # that leaves training half-stopped with the driver still up.
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 0, r.stderr


def test_a_worker_that_ignores_sigterm_warns_instead_of_hanging() -> None:
    r = _run(_harness(_victim(1, ignore_term=True), grace=4))
    # If the victim were signalled before it installed SIG_IGN it would die and
    # this would silently become the clean-drain case, asserting nothing.
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "Workers drained after" not in r.stdout, r.stdout
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
    """
    victim = _victim(1) + _victim(2, publish=False) + """
( sleep 1; echo "$V2" >> "$POOL_FILE"; sleep 60 ) >/dev/null 2>&1 </dev/null &
"""
    post = """
if ! grep -qx "$V2" "$POOL_FILE"; then echo "REVIVAL_UNPUBLISHED"; fi
"""
    r = _run(_harness(victim, grace=30, post=post), timeout=40)
    assert "VICTIM_NEVER_READY" not in r.stdout, r.stdout
    assert "REVIVAL_UNPUBLISHED" not in r.stdout, r.stdout
    assert "Workers drained after" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
    # A pgrep-until-empty loop would have spent the full 30s and then warned.
    waited = int(r.stdout.split("Workers drained after ")[1].split("s")[0])
    assert waited <= 6, f"drain waited {waited}s; it should not track revivals"


def test_no_workers_is_a_no_op() -> None:
    r = _run(_harness("", grace=30))
    assert "Draining selfplay workers" not in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr


def _orphan_harness(victim: str, *, grace: int) -> str:
    """The SECOND pass, which runs after ray teardown. `wpat` is injected from
    the real file's single definition, not restated here."""
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
    r = _run(_orphan_harness(_victim(1, ignore_term=True), grace=4))
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


@pytest.mark.parametrize("marker", [_BEGIN, _END])
def test_extraction_markers_exist(marker: str) -> None:
    assert marker in TRAIN_SH.read_text()
