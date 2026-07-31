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


def _harness(victim: str, *, grace: int) -> str:
    """Run the real drain block under the same `set -e` train.sh uses, against
    real processes.

    `pgrep` is stubbed rather than real because the real one would match — and
    then SIGTERM — the four production workers running on this machine.

    The stub honours BOTH of the things the production call depends on:
    liveness and the PATTERN. Liveness, because a fixed list cannot tell a
    pid-capture drain from a pgrep-polling one (both converge), and it is what
    lets a test add a process partway through the way the driver's
    `revive_dead_selfplay_processes` does. The pattern, because ignoring it
    made the drain's ONLY production selector untested: replacing it with a
    name that matches nothing left all seven tests green while draining nothing
    at all. Victims therefore carry a realistic worker argv via `exec -a`, and
    the stub greps /proc/<pid>/cmdline with the pattern it was actually given.
    """
    return f"""
set -e
export CAE_STOP_GRACE_SECONDS={grace}
POOL_FILE=$(mktemp)
pgrep() {{
    local p pat=""
    for a in "$@"; do
        case "$a" in --) ;; -*) ;; *) pat="$a" ;; esac
    done
    while read -r p; do
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
            if tr '\\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE -- "$pat"; then
                echo "$p"
            fi
        fi
    done < "$POOL_FILE"
}}
{victim}
drain() {{
{_drain_block()}
}}
drain
echo "REACHED_DRIVER_KILL"
# Reap the stubs. `wait` would block forever on a victim that survived the
# drain, which is the expected state in two of these tests. The stubs also
# redirect stdout, or a surviving grandchild would hold the pipe open and
# subprocess.run() would block on EOF long after bash exited.
for j in $(jobs -p); do kill -9 "$j" 2>/dev/null || true; done
for p in $(cat "$POOL_FILE"); do kill -9 "$p" 2>/dev/null || true; done
rm -f "$POOL_FILE"
"""


def test_workers_are_signalled_and_the_wait_ends_when_they_exit() -> None:
    victim = """
bash -c 'exec -a "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker" sleep 300' >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
bash -c 'exec -a "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker" sleep 300' >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
"""
    r = _run(_harness(victim, grace=30))
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
    victim = """
READY=$(mktemp -u)
PYF=$(mktemp --suffix=.py)
cat > "$PYF" <<'EOF'
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], "w").close()   # readiness: SIG_IGN is installed
time.sleep(300)
EOF
bash -c 'exec -a "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker" python3 "$1" "$2"' _ "$PYF" "$READY" >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
for _ in $(seq 200); do
    if [ -f "$READY" ]; then break; fi
    sleep 0.05
done
if [ ! -f "$READY" ]; then echo "STUB_NEVER_READY"; fi
"""
    r = _run(_harness(victim, grace=4))
    # If the stub were signalled before it installed SIG_IGN it would die and
    # this would silently become the clean-drain case, asserting nothing.
    assert "STUB_NEVER_READY" not in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "Workers drained after" not in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr


def test_a_revived_worker_does_not_extend_the_wait() -> None:
    """The driver's `revive_dead_selfplay_processes` relaunches a worker that
    exits mid-iteration. The drain waits on the pids captured ONCE; if it
    re-ran `pgrep` instead it would keep finding the replacement and burn the
    whole grace period, then warn falsely."""
    victim = """
bash -c 'exec -a "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker" sleep 300' >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
# The revival: appears 1s in — after the drain captured its pid list (t~0) and
# before its first liveness check (t~2). It is never signalled and outlives the
# wait, exactly like a worker the driver relaunched mid-teardown. A revival
# landing after the first check would not discriminate the two loop shapes,
# because the captured list has already gone empty by then.
( sleep 1
  bash -c 'exec -a "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker" sleep 300' >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
  sleep 60 ) >/dev/null 2>&1 </dev/null &
"""
    r = _run(_harness(victim, grace=30), timeout=40)
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
pgrep() {{
    local p pat=""
    for a in "$@"; do
        case "$a" in --) ;; -*) ;; *) pat="$a" ;; esac
    done
    while read -r p; do
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
            if tr '\\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE -- "$pat"; then
                echo "$p"
            fi
        fi
    done < "$POOL_FILE"
}}
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


_FAKE_WORKER_ARGV = (
    "/usr/bin/python3 -m chess_anti_engine.worker --work-dir /tmp/fake_worker"
)


def test_orphaned_workers_are_drained_after_ray_teardown() -> None:
    """Nothing before this point kills a worker — verified 2026-07-31 against
    /proc/<pid>/cmdline, no worker matches `ray::` or `raylet`. So a worker the
    driver revived DURING the first pass survives ray teardown, reuses the dead
    worker's work_dir, and would claim-by-rename the selfplay_resume/ files the
    first pass just banked."""
    victim = f"""
bash -c 'exec -a "{_FAKE_WORKER_ARGV}" sleep 300' >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
"""
    r = _run(_orphan_harness(victim, grace=30))
    assert "Draining orphaned workers" in r.stdout, r.stdout + r.stderr
    assert "STILL_ALIVE" not in r.stdout, r.stdout
    assert "REACHED_END_OF_STOP" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 0, r.stderr


def test_an_orphan_that_ignores_sigterm_is_killed_not_left_running() -> None:
    """Unlike the first pass, this one must never leave a survivor: there is no
    later stage to catch it, and `train.sh start` would race it."""
    victim = f"""
READY=$(mktemp -u)
PYF=$(mktemp --suffix=.py)
cat > "$PYF" <<'EOF'
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], "w").close()
time.sleep(300)
EOF
bash -c 'exec -a "{_FAKE_WORKER_ARGV}" python3 "$1" "$2"' _ "$PYF" "$READY" >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
for _ in $(seq 200); do
    if [ -f "$READY" ]; then break; fi
    sleep 0.05
done
if [ ! -f "$READY" ]; then echo "STUB_NEVER_READY"; fi
"""
    r = _run(_orphan_harness(victim, grace=4))
    assert "STUB_NEVER_READY" not in r.stdout, r.stdout
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


# ---------------------------------------------------------------------------
# The verdict must come from the SUSPEND LINE, not from liveness
# ---------------------------------------------------------------------------
#
# 2026-07-31, first real teardown with the handler: workers 01/02/03 each
# logged "exiting after suspend" BEFORE the 90s deadline and banked 22-23
# in-flight games, and stop() named all three in
#     WARNING: workers still alive after 90s (...); their in-flight games
#     will be DISCARDED.
# because the liveness probe caught them tearing down their CUDA context. The
# message reported a success as data loss. These tests pin the replacement:
# only a worker that banked NOTHING is a loss.

_SUSPEND_LINE = "INFO chess_anti_engine.selfplay.resume selfplay resume: suspended games={n} records=7 skipped=0"


def _logging_victim(*, stale: int = 0, suspend: int = 0, ignore_term: bool = False) -> str:
    """A victim carrying a real `--log-file` argv, which is where the drain
    reads its evidence from.

    ``stale`` writes a suspend line BEFORE the drain starts; ``suspend``
    writes one in the SIGTERM handler. ``ignore_term`` keeps the process alive
    through the grace period so the "still running" branch is the one under
    test.
    """
    handler = (
        f'signal.signal(signal.SIGTERM, lambda *_: log("{_SUSPEND_LINE.format(n=suspend)}"))'
        if suspend
        else "signal.signal(signal.SIGTERM, signal.SIG_IGN)"
        if ignore_term
        else "pass"
    )
    exit_after = "" if ignore_term else "\n    sys.exit(0)"
    stale_line = (
        f'log("{_SUSPEND_LINE.format(n=stale)}")' if stale else "pass"
    )
    # NOT `exec -a`, unlike the other victims here. `exec -a "a b c"` sets
    # argv[0] to that ONE string, so `--log-file` would not be a separate
    # cmdline element and the drain's `awk /^--log-file$/{{getline}}` could
    # never find it — the first version of this test failed for exactly that
    # reason, against production code that is correct. A real worker's argv has
    # `--log-file` and its value as distinct elements (verified 2026-07-31 with
    # `pgrep -af` against the four live workers), so the victim is launched with
    # real separate arguments instead. The `wpat` match is unaffected: the
    # stub greps the whole space-joined cmdline, which still contains
    # `-m chess_anti_engine.worker `.
    return f"""
WLOG=$(mktemp)
READY=$(mktemp -u)
PYF=$(mktemp --suffix=.py)
cat > "$PYF" <<'EOF'
import signal, sys, time
argv = sys.argv
LOG = argv[argv.index("--log-file") + 1]
READY = argv[argv.index("--ready") + 1]
def log(msg):
    with open(LOG, "a") as fh:
        fh.write(msg + "\\n")
        fh.flush()
{stale_line}
{handler}
open(READY, "w").close()
time.sleep(300){exit_after}
EOF
python3 "$PYF" -m chess_anti_engine.worker --work-dir /tmp/fake_worker \
    --log-file "$WLOG" --ready "$READY" >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
for _ in $(seq 200); do
    if [ -f "$READY" ]; then break; fi
    sleep 0.05
done
if [ ! -f "$READY" ]; then echo "STUB_NEVER_READY"; fi
"""


def test_a_worker_that_banked_its_games_is_not_reported_as_discarded() -> None:
    """THE REGRESSION. Still running at the deadline, but it suspended 23
    games — that is a success, not a loss."""
    r = _run(_harness(_logging_victim(suspend=23, ignore_term=True), grace=6), timeout=40)
    assert "STUB_NEVER_READY" not in r.stdout, r.stdout
    assert "suspended 23 in-flight game(s)" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" not in r.stdout, r.stdout
    assert "WARNING" not in r.stdout, r.stdout
    assert "23 in-flight game(s) suspended" in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr


def test_a_worker_that_banked_nothing_still_warns() -> None:
    """The POSITIVE control for the warning. Without this, a change that simply
    deleted the warning would pass the test above."""
    r = _run(_harness(_logging_victim(ignore_term=True), grace=6), timeout=40)
    assert "STUB_NEVER_READY" not in r.stdout, r.stdout
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "NO suspend recorded" in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr


def test_a_suspend_line_from_BEFORE_the_drain_is_not_credited() -> None:
    """NEGATIVE CONTROL. `suspended games=` is emitted by every in-session reco
    restart too, so a whole-file grep would find an hours-old line and read it
    as proof THIS drain worked. The block records each log's byte offset before
    signalling; this test fails if that offset is dropped."""
    r = _run(_harness(_logging_victim(stale=99, ignore_term=True), grace=6), timeout=40)
    assert "STUB_NEVER_READY" not in r.stdout, r.stdout
    assert "99" not in r.stdout, f"stale pre-drain suspend line was credited:\n{r.stdout}"
    assert "WARNING" in r.stdout, r.stdout + r.stderr
    assert "DISCARDED" in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr
