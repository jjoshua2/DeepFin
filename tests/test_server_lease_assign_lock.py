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
