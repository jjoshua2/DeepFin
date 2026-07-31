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
    real processes.

    `pgrep` is stubbed to report whichever members of $POOL_FILE are ALIVE
    RIGHT NOW, rather than a fixed list. That is what lets a test add a
    process partway through and have the stub start reporting it — the
    behaviour of the driver's `revive_dead_selfplay_processes`. A fixed-list
    stub cannot tell a pid-capture drain from a pgrep-polling one, because
    both converge.
    """
    return f"""
set -e
export CAE_STOP_GRACE_SECONDS={grace}
POOL_FILE=$(mktemp)
pgrep() {{
    local p
    while read -r p; do
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "$p"; fi
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
sleep 300 >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
sleep 300 >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
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
python3 -c 'import signal,time,sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], "w").close()
time.sleep(300)' "$READY" >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
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
sleep 300 >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
# The revival: appears 1s in — after the drain captured its pid list (t~0) and
# before its first liveness check (t~2). It is never signalled and outlives the
# wait, exactly like a worker the driver relaunched mid-teardown. A revival
# landing after the first check would not discriminate the two loop shapes,
# because the captured list has already gone empty by then.
( sleep 1
  sleep 300 >/dev/null 2>&1 </dev/null & echo $! >> "$POOL_FILE"
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


def test_train_sh_still_parses() -> None:
    r = _run(f"bash -n {TRAIN_SH}")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("marker", [_BEGIN, _END])
def test_extraction_markers_exist(marker: str) -> None:
    assert marker in TRAIN_SH.read_text()
