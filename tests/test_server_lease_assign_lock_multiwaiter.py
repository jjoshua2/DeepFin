"""#417: `_LeaseAssignLock` must still exclude when MANY waiters steal at once.

`tests/test_server_lease_assign_lock.py` passes on `main` and covers the
SINGLE-stealer case. The steal path's TOCTOU only shows up with N waiters: each
judged staleness from the file it had read and then unlinked whatever the name
resolved to, including a successor's brand-new lock, so all N won their own
`O_EXCL` create. Measured on `main`: peak 8 threads inside the critical section,
640/640 waiters admitted, zero busy refusals, and two workers assigned to one
trial while a second trial was starved to zero.

⚑ Threads are the production shape, not a convenience: `lease_trial` is a sync
`def`, so Starlette runs it in the threadpool and N worker polls are N threads
in ONE process over one `.assign.lock`.

Every test here is bounded by a wall clock and FAILS rather than hangs -- the
repo has no `pytest-timeout`, and a concurrency gate that wedges is a gate that
cannot fail. Nothing is killed by name; the "dead holder" pid comes from a child
this module spawns and reaps itself.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.server.app import _LeaseAssignBusy, _LeaseAssignLock

# Wall-clock ceiling for a whole multi-threaded round. Two orders of magnitude
# above the ~10ms a round actually needs.
_ROUND_DEADLINE_S = 60.0
_WAITERS = 16
_ROUNDS = 30
_SECTION_S = 0.003


def _dead_pid() -> int:
    """A pid that is certainly gone: spawn a child that exits, and reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=60)
    return proc.pid


def _plant(lock_path: Path, *, pid: int, token: str = "crashed-holder") -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({
            "pid": pid,
            "host": socket.gethostname(),
            "token": token,
            "created_at_unix": time.time(),
        }),
        encoding="utf-8",
    )


class _Census:
    """Peak simultaneous occupancy of the critical section."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.entries = 0
        self.busy = 0

    def enter(self) -> None:
        with self.lock:
            self.inside += 1
            self.entries += 1
            self.peak = max(self.peak, self.inside)

    def leave(self) -> None:
        with self.lock:
            self.inside -= 1

    def refused(self) -> None:
        with self.lock:
            self.busy += 1


def _hammer(lock_path: Path, census: _Census, *, waiters: int) -> None:
    """One round: `waiters` threads released together onto one planted lock."""
    start = threading.Barrier(waiters)

    def waiter() -> None:
        try:
            start.wait(timeout=_ROUND_DEADLINE_S)
        except threading.BrokenBarrierError:  # pragma: no cover - wedge guard
            return
        lk = _LeaseAssignLock(lock_path, timeout_s=1.0, stale_after_s=1e9)
        try:
            lk.__enter__()
        except _LeaseAssignBusy:
            census.refused()
            return
        census.enter()
        time.sleep(_SECTION_S)
        census.leave()
        lk.__exit__(None, None, None)

    threads = [threading.Thread(target=waiter) for _ in range(waiters)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + _ROUND_DEADLINE_S
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    stuck = [t for t in threads if t.is_alive()]
    if stuck:  # pragma: no cover - wedge guard
        pytest.fail(
            f"{len(stuck)} lease-lock waiters did not finish within "
            f"{_ROUND_DEADLINE_S}s -- treating a wedge as a failure, not a hang"
        )


def _run_rounds(
    lock_path: Path, *, pid_source, rounds: int = _ROUNDS, waiters: int = _WAITERS,
) -> _Census:
    census = _Census()
    for _ in range(rounds):
        _plant(lock_path, pid=pid_source())
        _hammer(lock_path, census, waiters=waiters)
        lock_path.unlink(missing_ok=True)
    return census


def _acquire_within(lock: _LeaseAssignLock, *, seconds: float) -> object:
    """Acquire on a worker thread and REQUIRE an answer inside `seconds`.

    Returns the lock on success or the raised exception; fails the test if the
    call never returns. A lock acquisition that can spin forever is the exact
    defect this module exists to pin, so the assertion about it must not be
    written in a way that hangs when the defect is present.
    """
    box: list[object] = []

    def attempt() -> None:
        try:
            box.append(lock.__enter__())
        except BaseException as exc:
            # Reported, never swallowed: the caller asserts on what came back.
            box.append(exc)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        pytest.fail(
            f"__enter__ did not return within {seconds}s -- the acquisition "
            "loop is spinning instead of honouring its deadline"
        )
    assert box, "the worker thread produced no outcome"
    return box[0]


def test_live_holder_admits_nobody(tmp_path: Path) -> None:
    """Control: nothing licenses a steal, so exclusion must hold trivially.

    Passes on `main` too -- included because it is what proves the failure
    below is the STEAL path and not the lock as a whole. The holder pid is this
    very process, so it is unambiguously alive and no child is left running.
    """
    census = _run_rounds(tmp_path / ".assign.lock", pid_source=os.getpid, rounds=3)

    assert census.entries == 0, "a live holder's lock must never be taken"
    assert census.busy == 3 * _WAITERS
    assert census.peak == 0


def test_many_waiters_stealing_one_dead_holders_lock_never_overlap(tmp_path: Path) -> None:
    """#417: FAILS on origin/main (peak 8 of 16 inside the section at once).

    With the holder pid gone every waiter judges the same file stale, so every
    waiter reaches the steal path simultaneously. Exactly one may end up holding
    the lock at any instant.
    """
    census = _run_rounds(tmp_path / ".assign.lock", pid_source=_dead_pid)

    assert census.peak == 1, (
        f"lease-assign lock admitted {census.peak} concurrent holders "
        f"({census.entries} entries, {census.busy} refusals over {_ROUNDS} rounds)"
    )


def test_stealing_still_makes_progress_and_does_not_starve_waiters(tmp_path: Path) -> None:
    """Exclusion must not be bought by failing closed.

    A fix that simply refused every steal would pass the test above and wedge
    the lease route after a crash -- the outage this class exists to prevent.
    Every waiter must still get in, one at a time.
    """
    census = _run_rounds(tmp_path / ".assign.lock", pid_source=_dead_pid, rounds=5)

    assert census.entries == 5 * _WAITERS, (
        f"only {census.entries} of {5 * _WAITERS} waiters were admitted; "
        f"{census.busy} were refused"
    )
    assert census.busy == 0
    assert census.peak == 1


def test_two_polls_never_assign_two_workers_to_one_trial(tmp_path: Path) -> None:
    """The lease-level wrong outcome from #417, end to end.

    On `main` both polls sit inside the section together, read the same lease
    state, and pick the same trial: `run_tA` gets two workers and `run_tB` gets
    zero, which the class's own docstring calls "not recoverable at all". A
    `_LeaseAssignBusy` for the second poll is a CORRECT outcome here -- it is a
    503 + Retry-After that the worker retries for free.
    """
    from chess_anti_engine.server.lease import assign_trial_lease

    trials = ["run_tA", "run_tB"]
    leases_root = tmp_path / "leases"
    lock_path = leases_root / ".assign.lock"
    _plant(lock_path, pid=_dead_pid())

    both_inside = threading.Barrier(2)
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def poll(worker_id: str) -> None:
        lk = _LeaseAssignLock(lock_path, timeout_s=1.0, stale_after_s=1e9)
        try:
            lk.__enter__()
        except _LeaseAssignBusy:
            with results_lock:
                results.append({"worker": worker_id, "trial": None})
            return
        try:
            # Times out when exclusion HOLDS -- that is the good path.
            with contextlib.suppress(threading.BrokenBarrierError):
                both_inside.wait(timeout=2.0)
            lease = assign_trial_lease(
                leases_root=leases_root,
                username="volunteer",
                worker_info={"worker_id": worker_id},
                available_trials=trials,
                manifest_loader=lambda _t: {"training_iteration": 10},
                requested_lease_id="",
                min_workers_per_trial=1,
            )
            with results_lock:
                results.append({"worker": worker_id, "trial": lease.get("trial_id")})
        finally:
            lk.__exit__(None, None, None)

    threads = [threading.Thread(target=poll, args=(w,)) for w in ("w1", "w2")]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + _ROUND_DEADLINE_S
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):  # pragma: no cover - wedge guard
        pytest.fail(f"lease polls did not finish within {_ROUND_DEADLINE_S}s")

    assert len(results) == 2
    per_trial = {t: sum(1 for r in results if r["trial"] == t) for t in trials}
    assert max(per_trial.values()) <= 1, (
        f"two workers assigned to one trial: {per_trial} from {results}"
    )


def test_a_fresh_claim_blocks_a_second_stealer(tmp_path: Path) -> None:
    """Pins the mechanism: the claim name is what makes the steal exclusive.

    A waiter that has already claimed the steal of this exact file owns the
    identity-derived name; a second waiter must find it taken and refuse rather
    than remove anything. Without this the test above could pass for the wrong
    reason (e.g. incidental timing) and no test would fail if the claim were
    quietly dropped.
    """
    lock_path = tmp_path / ".assign.lock"
    _plant(lock_path, pid=_dead_pid())

    probe = _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=1e9)
    identity = probe._snapshot().identity
    assert identity is not None
    claim = probe._claim_path(identity)
    claim.write_text("claimed by another waiter", encoding="utf-8")

    # ⚑ BOUNDED, because the failure mode here is a HANG, not a wrong answer.
    # `__enter__` must give up at its deadline; if the deadline check is ever
    # removed the steal path spins forever and a bare `pytest.raises` would
    # wedge the suite instead of failing it. Verified: with the check deleted,
    # this asserts instead of hanging.
    outcome = _acquire_within(probe, seconds=20.0)
    assert isinstance(outcome, _LeaseAssignBusy), (
        f"expected _LeaseAssignBusy once the claim is taken, got {outcome!r}"
    )
    assert lock_path.exists(), "the stale lock must NOT have been removed"


def test_an_abandoned_claim_does_not_wedge_stealing_forever(tmp_path: Path) -> None:
    """A stealer that crashed mid-claim must not disable the steal path.

    The claim name is derived from the judged file, so a grave left by a dead
    process would otherwise block every future steal of that lock -- a
    permanent lease outage after exactly the crash this class recovers from.
    The age sweep bounds it at `stale_after_s`.
    """
    lock_path = tmp_path / ".assign.lock"
    _plant(lock_path, pid=_dead_pid())

    probe = _LeaseAssignLock(lock_path, timeout_s=0.5, stale_after_s=1.0)
    identity = probe._snapshot().identity
    assert identity is not None
    claim = probe._claim_path(identity)
    claim.write_text("abandoned by a crashed stealer", encoding="utf-8")
    old = time.time() - 3600
    os.utime(claim, (old, old))

    with probe:
        assert probe._held
    assert not claim.exists(), "the abandoned claim should have been swept"


def test_steal_needs_only_o_excl_no_exotic_filesystem_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portability pin: the steal must not depend on `link` or `rename`.

    This ships to volunteer machines, not just the WSL2/ext4 host, and 9p and
    FAT-backed mounts lack hardlinks. An earlier draft claimed with `link(2)`
    and fell back to a rename where that failed; the fallback measured WEAKER
    than the claim (the multi-waiter census caught it admitting more than one
    holder), so the design now uses only `O_CREAT|O_EXCL` -- the same primitive
    the lock itself is built on. Breaking both syscalls proves that is true
    rather than merely intended.
    """
    def unavailable(*_a: object, **_k: object) -> None:
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(os, "link", unavailable)
    monkeypatch.setattr(os, "rename", unavailable)

    census = _run_rounds(tmp_path / ".assign.lock", pid_source=_dead_pid, rounds=5)

    assert census.peak == 1, (
        f"admitted {census.peak} concurrent holders without link/rename "
        f"({census.entries} entries, {census.busy} refusals)"
    )
    assert census.entries == 5 * _WAITERS, "waiters must still get in"
