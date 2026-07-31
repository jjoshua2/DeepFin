"""`train.sh stop` must SIGTERM the selfplay workers and wait for them, so their
handler can suspend in-flight games before ray is torn down.

These tests execute the drain block EXTRACTED FROM THE REAL FILE, not a copy of
it, so an edit to `scripts/train.sh` that breaks the contract fails here. The
block is run against real short-lived processes with `pgrep` stubbed to name
them.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

TRAIN_SH = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"

_BEGIN = "── Drain selfplay BEFORE anything is killed"
_END = 'echo "Stopping PID $pid ..."'


def _drain_block() -> str:
    lines = TRAIN_SH.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if _BEGIN in ln]
    ends = [i for i, ln in enumerate(lines) if _END in ln]
    assert len(starts) == 1, f"drain block marker not unique: {starts}"
    assert len(ends) == 1, f"end marker not unique: {ends}"
    assert starts[0] < ends[0]
    return "\n".join(lines[starts[0]:ends[0]])


def _run(script: str, *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def _harness(victim: str, *, grace: int) -> str:
    """Run the real drain block under the same `set -e` train.sh uses, against
    processes `pgrep` is stubbed to report."""
    return f"""
set -e
export CAE_STOP_GRACE_SECONDS={grace}
{victim}
pgrep() {{ echo "$VICTIM_PIDS" | tr ' ' '\\n' | sed '/^$/d'; }}
export -f pgrep 2>/dev/null || true
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
"""


def test_workers_are_signalled_and_the_wait_ends_when_they_exit() -> None:
    victim = """
sleep 300 >/dev/null 2>&1 </dev/null & A=$!
sleep 300 >/dev/null 2>&1 </dev/null & B=$!
VICTIM_PIDS="$A $B"
"""
    r = _run(_harness(victim, grace=30))
    assert "Draining selfplay workers" in r.stdout, r.stdout + r.stderr
    assert "Workers drained after" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
    # THE POINT: stop() must go on to kill the driver. Before the `if kill -0`
    # rewrite the last worker's exit made the for-loop's status non-zero, and
    # `set -e` aborted stop() right here — leaving training half-stopped with
    # the driver still up.
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr
    assert r.returncode == 0, r.stderr


def test_a_worker_that_ignores_sigterm_warns_instead_of_hanging() -> None:
    victim = """
bash -c 'trap "" TERM; sleep 300' >/dev/null 2>&1 </dev/null & A=$!
VICTIM_PIDS="$A"
"""
    r = _run(_harness(victim, grace=4))
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
sleep 300 >/dev/null 2>&1 </dev/null & A=$!
# The "revived" worker: never signalled, still alive when the wait ends.
bash -c 'trap "" TERM; sleep 300' >/dev/null 2>&1 </dev/null & R=$!
VICTIM_PIDS="$A"
"""
    r = _run(_harness(victim, grace=30), timeout=40)
    assert "Workers drained after" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
    # A pgrep-until-empty loop would have spent the full 30s and then warned.
    waited = int(r.stdout.split("Workers drained after ")[1].split("s")[0])
    assert waited <= 6, f"drain waited {waited}s; it should not track revivals"


def test_no_workers_is_a_no_op() -> None:
    r = _run(_harness('VICTIM_PIDS=""', grace=30))
    assert "Draining selfplay workers" not in r.stdout, r.stdout
    assert "REACHED_DRIVER_KILL" in r.stdout, r.stdout + r.stderr


def test_train_sh_still_parses() -> None:
    r = _run(f"bash -n {TRAIN_SH}")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("marker", [_BEGIN, _END])
def test_extraction_markers_exist(marker: str) -> None:
    assert marker in TRAIN_SH.read_text()
