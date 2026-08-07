"""`users.json` survives a read-modify-write from two PROCESSES at once.

⚑ THIS GAP WAS DORMANT AND SELF-REGISTRATION WOKE IT UP. It was recorded in the
#343 review as adjacent and out of scope, and it was: with the upload counters
moved to their own file, `manage_users` was the only writer of `users.json`, so
an unlocked read-modify-write could not lose anything. TOFU registration made
the SERVER a writer again, in a different process from the operator's CLI, which
`stats_write_lock` cannot see.

The dangerous direction is not the one people expect. Losing a just-registered
account is annoying; losing the OPERATOR'S DISABLE means a revoked worker keeps
uploading, and the operator has no signal — `manage_users disable` exited 0 and
said it worked.

Measured on the pre-fix code, with the disable landing inside the server's RMW:

    alice.disabled = False   (operator set it True)
    LOST UPDATE: the revocation was silently reverted
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from chess_anti_engine.server.auth import (
    UserRecord,
    UsersDbBusy,
    ensure_user,
    hash_password,
    load_users,
    save_users,
    users_db_lock,
    verify_password,
)

_REPO = Path(__file__).resolve().parents[1]


_WAITING_MARKER = "waiting for the users-db lock"


def _spawn_cli_disable(db: Path, username: str) -> subprocess.Popen[str]:
    """`manage_users disable` as the operator runs it: a separate PROCESS.

    A thread would prove nothing — the bug is precisely that a threading lock
    cannot see another process, so the test has to cross that boundary.
    """
    return subprocess.Popen(
        [sys.executable, "-m", "chess_anti_engine.server.manage_users",
         "--users-db", str(db), "disable", username],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
    )


def _await_blocked_on_the_lock(handle: subprocess.Popen[str], *, timeout_s: float = 120.0) -> None:
    """Block until the child is OBSERVABLY waiting on the users-db lock.

    ⚑ THE HANDSHAKE IS THE CONTENTION ITSELF, and this replaces a `time.sleep`
    that made the whole gate a coin flip. The sleep had to cover the child's
    cold start — interpreter plus imports, measured at 0.62-1.05s on this box
    against a 1.0s sleep — so when the child was slow its write landed AFTER
    the racing write instead of inside it, and the assertions passed for the
    wrong reason. The reviewer measured 40% escape on the exact defect this
    file names. Widening the sleep only moves the boundary to a slower machine.

    The child announces contention on its first failed acquisition, so seeing
    that line means it is past startup, past argparse, and parked on the lock —
    which is the state the test needs and the only one it accepts. If the line
    never arrives, that is a failure, not a reason to continue.
    """
    assert handle.stderr is not None
    deadline = time.monotonic() + timeout_s
    seen: list[str] = []
    while time.monotonic() < deadline:
        line = handle.stderr.readline()
        if not line:
            break
        seen.append(line)
        if _WAITING_MARKER in line:
            return
    handle.kill()
    raise AssertionError(
        "the CLI never reported waiting on the users-db lock, so the interleave "
        f"under test never happened. stderr so far: {seen!r}"
    )


def test_an_operator_disable_survives_a_concurrent_server_registration(
    tmp_path: Path,
) -> None:
    """⚑ THE REVOCATION MUST WIN, OR IT IS NOT A REVOCATION.

    The server's registration read-modify-write is reproduced here rather than
    driven through HTTP, so the interleave is deterministic instead of lucky:
    the CLI is launched while the server holds the lock, mid-RMW. Without the
    cross-process lock the CLI writes first and the server's stale copy
    overwrites it; with the lock the CLI blocks until the server is done and
    then applies the disable to the post-registration state.

    Both effects must survive. A fix that serialised by dropping one of the two
    writes would pass a test that only checked the disable.
    """
    db = tmp_path / "users.json"
    ensure_user(db, username="alice", password="alice-password")

    proc: list[subprocess.CompletedProcess[str]] = []

    def registration() -> None:
        salt_b64, hash_b64, iterations = hash_password("bob-password")
        with users_db_lock(db):
            users = load_users(db)
  # The operator acts inside this window. Launched from in here so it
  # cannot arrive before the lock is taken, and then WAITED FOR until it
  # is observably blocked on that lock -- no sleep, so a slow cold start
  # cannot let its write land after ours and pass this test for the wrong
  # reason. Not joined here: with the lock working it cannot finish until
  # the enclosing `with` releases.
            handle = _spawn_cli_disable(db, "alice")
            _await_blocked_on_the_lock(handle)
            users["bob"] = UserRecord(
                username="bob", salt_b64=salt_b64, hash_b64=hash_b64,
                iterations=iterations,
            )
            save_users(db, users)
        out, err = handle.communicate(timeout=120)
        proc.append(subprocess.CompletedProcess(handle.args, handle.returncode, out, err))

    thread = threading.Thread(target=registration)
    thread.start()
    thread.join(timeout=180)
    assert not thread.is_alive(), "the registration never completed"
    assert proc, "the CLI subprocess never reported"
    assert proc[0].returncode == 0, proc[0]

    final = load_users(db)
    assert final["alice"].disabled is True, (
        "the operator's revocation was overwritten by the server's stale copy: "
        "a disabled worker keeps uploading, and `manage_users disable` reported "
        "success"
    )
    assert "bob" in final, "the registration was lost instead"


def test_the_cli_and_the_server_cannot_hold_the_lock_at_once(tmp_path: Path) -> None:
    """The lock is real across processes, asserted by timing the CLI's wait.

    Without this, `users_db_lock` could be a no-op context manager and the test
    above would still pass whenever the interleave happened to be benign.
    """
    db = tmp_path / "users.json"
    ensure_user(db, username="alice", password="alice-password")

    with users_db_lock(db):
        handle = _spawn_cli_disable(db, "alice")
  # Reaching this line at all is the assertion: the child said it is waiting
  # on the lock, which it can only say because the lock is held here.
        _await_blocked_on_the_lock(handle)
        assert handle.poll() is None, (
            "the CLI completed while this process held the users-db lock — the "
            "lock is not being taken on the CLI side"
        )
    handle.communicate(timeout=120)
    assert handle.returncode == 0
    assert load_users(db)["alice"].disabled is True


def test_the_lock_refuses_rather_than_writing_unlocked(tmp_path: Path) -> None:
    """⚑ A LOCK THAT GIVES UP AND WRITES ANYWAY IS THE ORIGINAL BUG.

    Past the deadline the acquisition RAISES. The message names the holding pid,
    because "who is holding it" is the first thing an operator needs and the
    lock file is the only place that records it.
    """
    db = tmp_path / "users.json"
    ensure_user(db, username="alice", password="alice-password")

    with (
        users_db_lock(db),
        pytest.raises(UsersDbBusy) as exc,
        users_db_lock(db, timeout_s=0.2),
    ):
        pytest.fail("acquired a lock this process already holds exclusively")
    assert str(os.getpid()) in str(exc.value)


def test_the_lock_is_released_when_a_holder_dies(tmp_path: Path) -> None:
    """flock, not an O_EXCL lock file, and this is the difference.

    A17's lease lock needed a staleness test precisely because a crashed holder
    leaves its lock file behind and someone has to decide whether stealing it is
    safe. The kernel drops an flock when the process dies, so there is nothing
    to steal and no heuristic to get wrong. Pinned because switching to a
    lock-file scheme would silently reintroduce that whole class of problem.
    """
    db = tmp_path / "users.json"
    ensure_user(db, username="alice", password="alice-password")

    script = (
        "import sys, time;"
        f"sys.path.insert(0, {str(_REPO)!r});"
        "from chess_anti_engine.server.auth import users_db_lock;"
        f"ctx = users_db_lock({str(db)!r});"
        "ctx.__enter__();"
        "print('held', flush=True);"
        "time.sleep(60)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
        cwd=str(_REPO), env={**os.environ, "PYTHONPATH": str(_REPO)},
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(UsersDbBusy), users_db_lock(db, timeout_s=0.2):
            pass
        holder.kill()
        holder.wait(timeout=60)
  # No cleanup, no staleness test, no steal: the kernel already released it.
        with users_db_lock(db, timeout_s=5.0):
            pass
    finally:
        if holder.poll() is None:
            holder.kill()
    assert load_users(db)["alice"].disabled is False


def test_hashing_is_not_done_while_the_lock_is_held(tmp_path: Path) -> None:
    """PBKDF2 is ~50ms; the lock is cross-process and shared with the server.

    Hashing inside it would serialise the KDF between the operator's CLI and
    every registration for no benefit — only the load-mutate-save has to be
    atomic. Asserted by holding the lock and timing an `upsert_user`'s hash: it
    must reach the lock, not sail past it.
    """
    from chess_anti_engine.server import auth as auth_mod

    db = tmp_path / "users.json"
    ensure_user(db, username="alice", password="alice-password")

    hashed_while_locked: list[bool] = []
    real_hash = auth_mod.hash_password
    holding = threading.Event()

    def spy(password: str, **kw):
        hashed_while_locked.append(holding.is_set())
        return real_hash(password, **kw)

    auth_mod.hash_password = spy
    try:
        with users_db_lock(db):
            holding.set()
            with pytest.raises(UsersDbBusy):
                auth_mod.upsert_user(db, username="bob", password="bob-password")
            holding.clear()
    finally:
        auth_mod.hash_password = real_hash

    assert hashed_while_locked == [True], (
        "upsert_user did not hash before trying the lock: the KDF is inside the "
        "critical section, which serialises it across processes"
    )


def test_the_migration_does_not_widen_the_credential_file(tmp_path: Path) -> None:
    """⚑ A ONE-TIME MIGRATION THAT SILENTLY WIDENS PERMISSIONS.

    `save_users` writes `users.json` 0600 because a world-WRITABLE credential
    file is an auth bypass by hash replacement. `migrate_user_stats` is the only
    other writer of that file and it rewrites the whole of it — without a mode,
    so a correctly-0600 file came back from the migration at the umask default.

    `umask 0` deliberately: at the usual 022 the file lands 0644 either way and
    the test would pass without the fix.
    """
    from chess_anti_engine.server.auth import migrate_user_stats

    db = tmp_path / "users.json"
    salt, digest, iterations = hash_password("alice-password")
    legacy = {
        "alice": {
            "salt_b64": salt, "hash_b64": digest, "iterations": iterations,
            "disabled": False, "uploads": 7, "total_bytes": 12, "machines": {},
        },
    }
    saved = os.umask(0)
    try:
        db.write_text(json.dumps(legacy), encoding="utf-8")
        os.chmod(db, 0o600)
        carried = migrate_user_stats(db, tmp_path / "user_stats.json")
        mode = stat.S_IMODE(db.stat().st_mode)
    finally:
        os.umask(saved)

    assert carried == 1
    assert mode == 0o600, f"the migration left users.json {mode:04o}"
    assert verify_password("alice-password", load_users(db)["alice"])
