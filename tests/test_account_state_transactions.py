"""Account changes retain revocations across password and operator transactions."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from chess_anti_engine.server import auth, manage_users
from chess_anti_engine.server.access import BANS_FILENAME, AccessGuard, BanList, BannedIdentity

_REPO = Path(__file__).resolve().parents[1]


def _cli(monkeypatch: pytest.MonkeyPatch, db: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["manage_users", "--users-db", str(db), *args])
    manage_users.main()


def _operator(db: Path, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "chess_anti_engine.server.manage_users",
         "--users-db", str(db), *args],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO), "CAE_TEST_ACCOUNT_PASSWORD": "new-password"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _wait_for_contention_or_completion(child: subprocess.Popen[str]) -> None:
    """Force the stale-write interleave on baseline, or observe the real lock.

    The unfixed writer finishes before the parent's stale snapshot is saved.
    The fixed writer announces contention before the parent may release its
    lock. A scheduler delay alone can never satisfy either handshake.
    """
    assert child.stderr is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        ready, _, _ = select.select([child.stderr], [], [], 0.1)
        if not ready:
            continue
        line = child.stderr.readline()
        if "waiting for the users-db lock" in line:
            return
        if not line:
            assert child.wait(timeout=5) == 0
            return
    pytest.fail("the competing operator neither completed nor reported lock contention")


def _finish(child: subprocess.Popen[str]) -> None:
    out, err = child.communicate(timeout=5)
    assert child.returncode == 0, (out, err)


def _stop(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        child.kill()
    child.communicate(timeout=5)


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_rotation_preserves_enabled_or_disabled_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disabled: bool,
) -> None:
    db = tmp_path / "users.json"
    auth.ensure_user(db, username="synthetic", password="old-password", disabled=disabled)
    _cli(monkeypatch, db, "set-password", "synthetic", "--password", "new-password")
    record = auth.load_users(db)["synthetic"]
    assert record.disabled is disabled
    assert auth.verify_password("new-password", record)
    assert not auth.verify_password("old-password", record)


def test_rotation_preserves_a_disable_that_lands_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "users.json"
    auth.ensure_user(db, username="synthetic", password="old-password")
    real_hash = auth.hash_password

    def hash_after_disable(password: str) -> tuple[str, str, int]:
        auth.set_disabled(db, username="synthetic", disabled=True)
        return real_hash(password)

    monkeypatch.setattr(auth, "hash_password", hash_after_disable)
    _cli(monkeypatch, db, "set-password", "synthetic", "--password", "new-password")
    record = auth.load_users(db)["synthetic"]
    assert record.disabled
    assert auth.verify_password("new-password", record)


def test_cli_rotation_cannot_recreate_an_account_deleted_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "users.json"
    auth.ensure_user(db, username="synthetic", password="old-password")
    real_hash = auth.hash_password

    def hash_after_delete(password: str) -> tuple[str, str, int]:
        auth.save_users(db, {})
        return real_hash(password)

    monkeypatch.setattr(auth, "hash_password", hash_after_delete)
    with pytest.raises(SystemExit, match="not found"):
        _cli(monkeypatch, db, "set-password", "synthetic", "--password", "new-password")
    assert auth.load_users(db) == {}


def test_upsert_retains_creation_and_explicit_state_compatibility(tmp_path: Path) -> None:
    db = tmp_path / "users.json"
    auth.upsert_user(db, username="synthetic", password="first-password")
    assert not auth.load_users(db)["synthetic"].disabled
    auth.upsert_user(db, username="synthetic", password="second-password", disabled=True)
    assert auth.load_users(db)["synthetic"].disabled
    auth.upsert_user(db, username="synthetic", password="third-password", disabled=False)
    record = auth.load_users(db)["synthetic"]
    assert not record.disabled
    assert auth.verify_password("third-password", record)


@pytest.mark.parametrize("operation", ["disable", "set-password"])
def test_legacy_migration_retains_a_concurrent_operator_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    db = tmp_path / "users.json"
    stats_path = auth.user_stats_path_for(db)
    auth.ensure_user(db, username="synthetic", password="old-password")
    legacy = json.loads(db.read_text(encoding="utf-8"))
    legacy["synthetic"].update(uploads=7, total_positions=11)
    db.write_text(json.dumps(legacy), encoding="utf-8")
    real_save = auth.save_user_stats
    children: list[subprocess.Popen[str]] = []

    def save_with_operator(path: str | Path, stats: dict[str, auth.UserStats], *, durable: bool = False) -> None:
        real_save(path, stats, durable=durable)
        args = [operation, "synthetic"]
        if operation == "set-password":
            args.extend(["--password-env", "CAE_TEST_ACCOUNT_PASSWORD"])
        child = _operator(db, *args)
        children.append(child)
        _wait_for_contention_or_completion(child)

    monkeypatch.setattr(auth, "save_user_stats", save_with_operator)
    try:
        assert auth.migrate_user_stats(db, stats_path) == 1
        assert len(children) == 1
        _finish(children[0])
        record = auth.load_users(db)["synthetic"]
        if operation == "disable":
            assert record.disabled
            assert auth.verify_password("old-password", record)
        else:
            assert auth.verify_password("new-password", record)
            assert not auth.verify_password("old-password", record)
        stats = auth.load_user_stats(stats_path)["synthetic"]
        assert (stats.uploads, stats.total_positions) == (7, 11)
        assert "uploads" not in json.loads(db.read_text(encoding="utf-8"))["synthetic"]
        assert auth.migrate_user_stats(db, stats_path) == 0
    finally:
        for child in children:
            _stop(child)


@pytest.mark.parametrize("operation", ["ban", "unban"])
def test_concurrent_ban_updates_preserve_each_operators_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    db = tmp_path / "users.json"
    path = tmp_path / BANS_FILENAME
    BanList(path=path, usernames={"synthetic-old"}, notes={"synthetic-old": "old reason"}).save()
    real_load = BanList.load
    children: list[subprocess.Popen[str]] = []

    def load_with_competitor(path: str | Path, *, strict: bool = False) -> BanList:
        snapshot = real_load(path, strict=strict)
        if not children:
            child = _operator(db, "ban", "--username", "synthetic-other", "--reason", "other reason")
            children.append(child)
            _wait_for_contention_or_completion(child)
        return snapshot

    monkeypatch.setattr(BanList, "load", staticmethod(load_with_competitor))
    target = "synthetic-new" if operation == "ban" else "synthetic-old"
    try:
        _cli(monkeypatch, db, operation, "--username", target)
        assert len(children) == 1
        _finish(children[0])
        final = real_load(path)
        assert "synthetic-other" in final.usernames
        assert final.notes["synthetic-other"] == "other reason"
        assert ("synthetic-old" in final.usernames) is (operation == "ban")
        assert ("synthetic-new" in final.usernames) is (operation == "ban")
        if operation == "unban":
            assert "synthetic-old" not in final.notes
    finally:
        for child in children:
            _stop(child)


def test_auth_can_reload_bans_while_an_operator_holds_the_writer_lock(tmp_path: Path) -> None:
    path = tmp_path / BANS_FILENAME
    guard = AccessGuard(path)
    with auth.users_db_lock(path):
        BanList(path=path, usernames={"synthetic-banned"}).save()
        with pytest.raises(BannedIdentity):
            guard.check_not_banned(username="synthetic-banned", ip=None)
        guard.check_not_banned(username="synthetic-allowed", ip=None)
