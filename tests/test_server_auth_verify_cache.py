"""Audit A7: authentication must not recompute PBKDF2 on every request.

`_auth_user` re-read the users DB, re-parsed it, and recomputed PBKDF2-SHA256
at 200k iterations for EVERY authenticated request — 73.4 ms of CPU per call.
Workers poll, and on a cold start they all poll at once, so the herd pins the
threadpool and everything behind it queues.

The gate is `test_repeat_requests_compute_pbkdf2_once`: it counts calls to the
shipped `_pbkdf2`, so it measures the actual work, not a proxy for it. On
`origin/main` it reads N of N.

The rest of the module is the invalidation contract, and it is the part worth
reviewing: a credential cache that is merely fast is a security defect. Each
test below names the mutation it prevents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_anti_engine.server import auth as auth_mod
from chess_anti_engine.server.auth import (
    UserRecord,
    hash_password,
    save_users,
    set_disabled,
    upsert_user,
)

_HEADERS = {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


class _Pbkdf2Counter:
    """Counts the real thing. Delegates, so verification still verifies —
    a stub returning a constant would make every assertion below vacuous."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = auth_mod._pbkdf2

        def counted(password: str, *, salt: bytes, iterations: int) -> bytes:
            self.calls += 1
            return real(password, salt=salt, iterations=iterations)

        monkeypatch.setattr(auth_mod, "_pbkdf2", counted)

    def reset(self) -> int:
        was, self.calls = self.calls, 0
        return was


def _seed(server_root: Path, username: str = "u", password: str = "p") -> None:
    salt, hsh, iters = hash_password(password)
    save_users(
        server_root / "users.json",
        {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)},
    )


def _client(server_root: Path):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    return TestClient(create_app(server_root=str(server_root), users_db="users.json"))


def _ping(client, *, user: str = "u", password: str = "p"):
    # `/v1/report_bad_shard` is the cheapest route that actually DEPENDS on
    # `_auth_user`: the dependency runs before the handler, so any 2xx/4xx that
    # is not 401/403 means auth passed. Most GET routes here are unauthenticated
    # -- pinging one of those would have made every assertion in this module
    # vacuous, which is exactly the failure mode this file is about.
    return client.post(
        "/v1/report_bad_shard",
        json={"sha256": "0" * 64, "reason": "auth-probe"},
        auth=(user, password),
        headers=_HEADERS,
    )


def test_repeat_requests_compute_pbkdf2_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE COST GATE. Ten requests, one credential, one PBKDF2 (on main: ten)."""
    _seed(tmp_path)
    counter = _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)

    for _ in range(10):
        assert _ping(client).status_code not in (401, 403), "auth must still pass"

    assert counter.calls == 1, (
        f"10 requests with one credential ran PBKDF2 {counter.calls} times; "
        f"each one is ~73ms of CPU on a threadpool token"
    )


def test_a_changed_password_invalidates_the_cached_allow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevents the mutation that matters most: a cache that keeps accepting a
    password after it was changed. `upsert_user` writes a fresh salt and hash,
    so the material the cached verification was made against no longer
    matches and the entry cannot be used."""
    _seed(tmp_path)
    counter = _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)
    assert _ping(client).status_code not in (401, 403)
    counter.reset()

    upsert_user(tmp_path / "users.json", username="u", password="new-secret")

    assert _ping(client, password="p").status_code == 401, "the OLD password still works"
    assert _ping(client, password="new-secret").status_code not in (401, 403)
    assert counter.calls >= 1, "a changed credential must be re-verified for real"


def test_a_disabled_user_is_rejected_on_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Revocation has no grace period. `disabled` is read from the CURRENT
    record on every request and is never part of what the cache stores —
    folding it in is how a revoked user keeps working until a restart."""
    _seed(tmp_path)
    _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)
    assert _ping(client).status_code not in (401, 403)

    set_disabled(tmp_path / "users.json", username="u", disabled=True)

    assert _ping(client).status_code == 403


def test_a_deleted_user_is_rejected_on_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cached allow must not outlive the record it was made against."""
    _seed(tmp_path)
    _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)
    assert _ping(client).status_code not in (401, 403)

    (tmp_path / "users.json").write_text(json.dumps({}), encoding="utf-8")

    assert _ping(client).status_code == 401


def test_a_wrong_password_is_re_verified_every_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NO NEGATIVE CACHING, asserted as an exact count.

    Caching rejections would let an attacker learn "wrong" cheaply and would
    need bounded eviction to avoid being a memory sink. The expensive
    direction is the correct one here, so this pins the cost rather than
    bounding it: five wrong attempts must pay five PBKDF2 runs.
    """
    _seed(tmp_path)
    counter = _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)

    for _ in range(5):
        assert _ping(client, password="wrong").status_code == 401
    assert counter.calls == 5, "rejections must not be cached"

    # And a wrong password never populates the cache for the right one.
    counter.reset()
    assert _ping(client).status_code not in (401, 403)
    assert counter.calls == 1


def test_an_unknown_user_never_reaches_pbkdf2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged from main, pinned because the cache rewrote this branch: a
    username with no record is a 401 without a hash computation, so an
    attacker cannot spend the server's CPU with names that do not exist."""
    _seed(tmp_path)
    counter = _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)

    assert _ping(client, user="nobody", password="p").status_code == 401
    assert counter.calls == 0


def test_upload_stats_writes_do_not_flush_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the cache is not keyed on the file stamp alone.

    `record_upload` + `save_users` rewrites users.json on every upload, so a
    stamp-keyed cache would be flushed by ordinary traffic that cannot
    possibly have changed a password. Simulated here by rewriting the file
    with the counters bumped and the credential material untouched.
    """
    _seed(tmp_path)
    counter = _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)
    assert _ping(client).status_code not in (401, 403)
    counter.reset()

    for i in range(3):
        raw = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
        raw["u"]["uploads"] = i + 1
        raw["u"]["total_bytes"] = (i + 1) * 1000
        (tmp_path / "users.json").write_text(json.dumps(raw), encoding="utf-8")
        assert _ping(client).status_code not in (401, 403)

    assert counter.calls == 0, (
        f"{counter.calls} re-verifications caused by stats writes; the cache is "
        f"keyed on the file version rather than on the credential material"
    )


def test_the_db_is_re_read_only_when_the_file_changes(tmp_path: Path) -> None:
    """The second half of the cost: the JSON parse per request.

    Asserted on the cache object directly — a request-level probe cannot see
    the difference between a cached parse and a fast one.
    """
    _seed(tmp_path)
    cache = auth_mod.VerifiedCredentialCache(tmp_path / "users.json")

    for _ in range(5):
        assert "u" in cache.users()
    assert cache.db_reads == 1

    upsert_user(tmp_path / "users.json", username="u", password="other")
    assert "u" in cache.users()
    assert cache.db_reads == 2


def test_a_missing_users_db_is_empty_not_an_error(tmp_path: Path) -> None:
    """`load_users` returns {} for an absent file and the cache must agree —
    including caching that absence, then noticing the file appearing."""
    cache = auth_mod.VerifiedCredentialCache(tmp_path / "users.json")
    assert cache.users() == {}
    assert cache.verify("u", "p") is None

    _seed(tmp_path)
    assert cache.verify("u", "p") is not None
