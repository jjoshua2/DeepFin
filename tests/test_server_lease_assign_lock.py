"""Audit A17: the lease-assign file lock must actually exclude.

The pre-fix `_LeaseAssignLock.__enter__` was `while True` with no failure exit.
Past its deadline it unlinked the lock file every 50ms *without checking
whether the holder was alive*, so a merely SLOW holder had its lock deleted;
the waiter's next `O_EXCL` create then succeeded and both ran the critical
section. `test_a_slow_holder_is_not_stolen_from` is the negative control for
that — it is the banked repro from `scratchpad/audit_lease_lock_repro.py`
pointed at the shipped class, and it FAILS on the pre-fix code (observed:
"both holders inside", overlap set within ~0.6s).

The tests import the shipped `_LeaseAssignLock`; they do not re-transcribe it.
A rule in a docstring is not a control.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.server.app import _LeaseAssignBusy, _LeaseAssignLock

# The holder's critical section outlasts the waiter's timeout. Shipped
# timeout_s is 10.0; shortening it changes how long the holder must take, not
# the outcome. `stale_after_s` is pinned well above `hold_for` so the holder is
# unambiguously fresh: this test is about a LIVE holder, not an expiry policy.
_TIMEOUT_S = 0.5
_HOLD_FOR_S = 1.5
_STALE_AFTER_S = 30.0


def test_a_slow_holder_is_not_stolen_from(tmp_path: Path) -> None:
    """Two threads, holder slower than the waiter's deadline. Never both inside."""
    lock_path = tmp_path / ".assign.lock"
    inside: list[str] = []
    overlap: list[str] = []
    busy: list[str] = []
    guard = threading.Lock()

    def worker(name: str, delay: float) -> None:
        time.sleep(delay)
        try:
            lock = _LeaseAssignLock(
                lock_path, timeout_s=_TIMEOUT_S, stale_after_s=_STALE_AFTER_S
            )
            with lock:
                with guard:
                    inside.append(name)
                    if len(inside) > 1:
                        overlap.append(",".join(inside))
                time.sleep(_HOLD_FOR_S)
                with guard:
                    inside.remove(name)
        except _LeaseAssignBusy:
            with guard:
                busy.append(name)

    threads = [
        threading.Thread(target=worker, args=("A", 0.0)),
        threading.Thread(target=worker, args=("B", 0.1)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert not overlap, f"mutual exclusion broken: {overlap}"
    # Failing CLOSED is the fix, so exactly one of the two must have been told
    # the lock was busy. Asserting only "no overlap" would also pass if the
    # lock deadlocked both callers out.
    assert busy == ["B"], f"expected B to be refused, got busy={busy} inside={inside}"
    assert not lock_path.exists(), "the holder must release its own lock on exit"


def test_a_crashed_holder_is_stolen_from(tmp_path: Path) -> None:
    """A dead holder's lock must not wedge the server."""
    lock_path = tmp_path / ".assign.lock"
    # A real process that really exits, so the pid is really gone: `os.kill(pid,
    # 0)` on a fabricated pid could hit a live unrelated process.
    dead_pid = _spawn_and_reap()
    lock_path.write_text(
        json.dumps(
            {
                "pid": dead_pid,
                "host": os.uname().nodename,
                "token": "not-ours",
                "created_at_unix": time.time(),  # FRESH: only deadness justifies this steal
            }
        ),
        encoding="utf-8",
    )

    t0 = time.time()
    with _LeaseAssignLock(lock_path, timeout_s=5.0, stale_after_s=3600.0):
        held = json.loads(lock_path.read_text(encoding="utf-8"))
    elapsed = time.time() - t0

    assert held["pid"] == os.getpid()
    assert elapsed < 1.0, f"crash recovery should be immediate, took {elapsed:.2f}s"
    assert not lock_path.exists()


def test_an_expired_lock_is_stolen_from(tmp_path: Path) -> None:
    """An alive-but-ancient holder is stale too — the generous-age branch."""
    lock_path = tmp_path / ".assign.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # ALIVE — only age justifies this steal
                "host": os.uname().nodename,
                "token": "not-ours",
                "created_at_unix": time.time() - 100.0,
            }
        ),
        encoding="utf-8",
    )
    with _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=1.0):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


@pytest.mark.parametrize(
    ("created_at", "label"),
    [
        (None, "missing"),
        ("not-a-number", "non-numeric"),
        (float("nan"), "NaN"),
        (float("inf"), "infinite"),
        (time.time() + 86_400.0, "a day in the future"),
    ],
)
def test_an_unusable_timestamp_falls_back_to_file_age(
    tmp_path: Path, created_at: object, label: str
) -> None:
    """The age backstop must not be skippable (the #338 review's blocking find).

    A NON-empty holder whose `created_at_unix` is unusable, plus a live-looking
    pid, had no clock source at all in the first revision: `st_mtime` was
    consulted only when the holder dict was EMPTY. Such a lock was never
    stealable -- the reviewer aged one to 10,000s with `stale_after_s = 1.0`
    and it stayed BUSY permanently, a total lease-path outage on exactly the
    crashed-holder case this class promises to handle. Each blocked poll also
    pins a threadpool token for the full timeout.

    The pid here is OUR OWN and alive, so only the age test can justify the
    steal -- the same shape a recycled pid produces after a crash. The future
    timestamp is not hypothetical: WSL2 clock jumps produce it.
    """
    lock_path = tmp_path / ".assign.lock"
    holder: dict[str, Any] = {"pid": os.getpid(), "host": os.uname().nodename, "token": "theirs"}
    if created_at is not None:
        # json can't encode NaN/inf round-trippably by default, and the point
        # is that the FILE may contain anything, so write it permissively.
        holder["created_at_unix"] = created_at
    lock_path.write_text(json.dumps(holder), encoding="utf-8")
    aged_to = time.time() - 10_000.0
    os.utime(lock_path, (aged_to, aged_to))

    t0 = time.time()
    with _LeaseAssignLock(lock_path, timeout_s=2.0, stale_after_s=1.0):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] != "theirs"
    elapsed = time.time() - t0

    assert elapsed < 1.0, (
        f"created_at_unix={label} made a 10000s-old lock unstealable for "
        f"{elapsed:.2f}s; the mtime fallback is not universal"
    )


def test_an_unbounded_json_integer_does_not_500_the_lease_path(tmp_path: Path) -> None:
    """JSON integers have no upper bound; Python floats do.

    `10 ** 400` parses fine and then raises `OverflowError` inside `float()`.
    Unguarded that propagates out of the acquisition and the lease route
    answers 500 -- persistently, because the offending lock file is still
    there next poll. The corrupt-value case is precisely what the mtime
    fallback exists to route around, so it must reach it.
    """
    lock_path = tmp_path / ".assign.lock"
    # Hand-assembled: `json.dumps` would render the big int fine, but writing
    # it literally is what makes the fixture's point legible.
    holder = json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "token": "theirs"})
    lock_path.write_text(
        holder[:-1] + ', "created_at_unix": ' + "1" + "0" * 400 + "}", encoding="utf-8"
    )
    assert isinstance(
        json.loads(lock_path.read_text(encoding="utf-8"))["created_at_unix"], int
    ), "the fixture must reach float() as an int, not as inf"
    aged_to = time.time() - 10_000.0
    os.utime(lock_path, (aged_to, aged_to))

    # Reaches the mtime fallback rather than raising: the lock is stealable.
    with _LeaseAssignLock(lock_path, timeout_s=2.0, stale_after_s=1.0):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] != "theirs"


def test_a_future_mtime_reports_age_zero_not_a_negative(tmp_path: Path) -> None:
    """Pins the clamp itself, which no outcome-level test can reach.

    A negative age is still `< stale_after_s`, so dropping `max(0.0, ...)`
    leaves every steal/busy decision unchanged and the clamp survives
    mutation. Its value is in what it hands to the CALLER: the number is
    logged as the evidence for a steal, and a negative age would read as a
    lock taken in the future. Asserted on `_lock_age` directly, which is the
    only place the clamp is observable.

    Distinct from `test_a_clock_that_ran_backwards_reads_as_brand_new`, which
    pins the busy DECISION; this pins the number that decision is made from.
    """
    lock_path = tmp_path / ".assign.lock"
    lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    now = time.time()
    ahead = now + 10_000.0
    os.utime(lock_path, (ahead, ahead))

    aged = _LeaseAssignLock(lock_path)._lock_age({"pid": os.getpid()}, now)

    assert aged is not None
    assert aged[1] == "file mtime"
    assert aged[0] == 0.0, f"a future mtime must clamp to 0.0, got {aged[0]}"


def test_a_bool_is_not_a_timestamp(tmp_path: Path) -> None:
    """`True` is an `int` to `isinstance`, so a naive numeric check reads it as
    the epoch + 1s and invents an ancient lock. Here the FILE is fresh, so the
    correct answer is busy: an `isinstance(raw, (int, float))` that does not
    exclude `bool` steals from a live holder instead."""
    lock_path = tmp_path / ".assign.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "token": "theirs",
                "created_at_unix": True,
            }
        ),
        encoding="utf-8",
    )
    with (
        pytest.raises(_LeaseAssignBusy),
        _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=60.0),
    ):
        pytest.fail("a bool timestamp was read as an ancient lock")
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "theirs"


def test_a_clock_that_ran_backwards_reads_as_brand_new(tmp_path: Path) -> None:
    """Age is clamped non-negative, so a backwards clock cannot force a steal.

    The failure this guards is the mirror of the one above: an unclamped
    negative age is still a number, and any later comparison that flips sign
    turns "the future" into "ancient". Clamping keeps a jumped clock on the
    SAFE side -- the lock reads as brand new and the caller gets a busy answer
    it can retry, rather than a steal from a live holder.
    """
    lock_path = tmp_path / ".assign.lock"
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "token": "theirs"}),
        encoding="utf-8",
    )
    ahead = time.time() + 10_000.0
    os.utime(lock_path, (ahead, ahead))

    with (
        pytest.raises(_LeaseAssignBusy),
        _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=1.0),
    ):
        pytest.fail("stole a live holder's lock because the clock jumped forward")
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "theirs"


def test_a_legacy_bare_pid_lock_file_is_judged_on_age(tmp_path: Path) -> None:
    """`f"{pid}\n"` is valid JSON, so it decodes to an int, so it takes the
    age path -- which is why the `JSONDecodeError` legacy branch was dead code
    and is gone. Pins the behaviour the deleted branch's comment claimed."""
    lock_path = tmp_path / ".assign.lock"
    lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    fresh = time.time()
    os.utime(lock_path, (fresh, fresh))
    with (
        pytest.raises(_LeaseAssignBusy),
        _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=60.0),
    ):
        pytest.fail("stole a fresh legacy lock file")

    aged_to = time.time() - 10_000.0
    os.utime(lock_path, (aged_to, aged_to))
    with _LeaseAssignLock(lock_path, timeout_s=2.0, stale_after_s=1.0):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_release_after_being_stolen_from_leaves_the_successor_alone(
    tmp_path: Path,
) -> None:
    """The half that turns one stolen lock into a cascade.

    Pre-fix, `__exit__` unlinked whenever `_held`, so a stolen-from holder
    deleted its successor's lock and a third caller walked in.
    """
    lock_path = tmp_path / ".assign.lock"
    victim = _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=1.0)
    victim.__enter__()
    successor_payload = json.dumps(
        {"pid": os.getpid(), "host": os.uname().nodename, "token": "successor"}
    )
    lock_path.write_text(successor_payload, encoding="utf-8")

    victim.__exit__(None, None, None)

    assert lock_path.exists(), "the successor's lock was deleted by the old holder"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "successor"


def test_busy_is_raised_not_stolen_when_the_holder_is_fresh(tmp_path: Path) -> None:
    """The deadline path fails the acquisition; it does not force it."""
    lock_path = tmp_path / ".assign.lock"
    with _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=3600.0):
        with (
            pytest.raises(_LeaseAssignBusy),
            _LeaseAssignLock(lock_path, timeout_s=0.2, stale_after_s=3600.0),
        ):
            pytest.fail("acquired a lock held by a live, fresh holder")
        assert lock_path.exists(), "a refused acquisition must not unlink the lock"


def _spawn_and_reap() -> int:
    """Return the pid of a process that has certainly exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_the_lease_route_answers_503_not_500_when_the_lock_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal must reach the worker as retryable, not as a server fault.

    A 500 traceback would send a worker into its error path over a condition
    that clears on its own in milliseconds.
    """
    from fastapi.testclient import TestClient

    from chess_anti_engine.server import app as app_mod
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password("p")
    save_users(
        tmp_path / "users.json",
        {"u": UserRecord(username="u", salt_b64=salt, hash_b64=hsh, iterations=iters)},
    )
    publish = tmp_path / "trials" / "trial_a" / "publish"
    publish.mkdir(parents=True)
    (publish / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "server_version": "0.0.1",
                "min_worker_version": "0.0.0",
                "trial_id": "trial_a",
                "task": {"type": "selfplay"},
            }
        ),
        encoding="utf-8",
    )

    class _AlwaysBusy:
        def __init__(self, path: Path, **_kw: object) -> None:
            raise _LeaseAssignBusy(f"held by a live holder: {path}")

    monkeypatch.setattr(app_mod, "_LeaseAssignLock", _AlwaysBusy)
    client = TestClient(
        app_mod.create_app(server_root=str(tmp_path), users_db="users.json"),
        raise_server_exceptions=False,
    )
    resp = client.post(
        "/v1/lease_trial",
        json={},
        auth=("u", "p"),
        headers={"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"},
    )

    assert resp.status_code == 503, resp.text
    assert resp.headers.get("Retry-After") == "1"
