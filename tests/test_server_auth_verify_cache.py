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
    load_users,
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


def test_a_disabled_user_is_rejected_on_the_next_request(tmp_path: Path) -> None:
    """Revocation has no grace period. `disabled` is read from the CURRENT
    record on every request and is never part of what the cache stores —
    folding it in is how a revoked user keeps working until a restart.

    The `_Pbkdf2Counter` this used to build was never asserted on, which made
    it look like the test pinned that the rejection happened on a cache HIT
    when it pinned nothing of the sort. That claim is made properly by
    `test_a_revoked_user_is_rejected_on_the_cached_path`; carrying a fixture
    that reads as evidence and is not is worse than not having it.
    """
    _seed(tmp_path)
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
    """Why the cache is keyed on the credential MATERIAL, not on the file stamp.

    Historically this was load-bearing for cost: `record_upload` + `save_users`
    rewrote users.json on every upload, so a stamp-keyed cache would have been
    flushed by ordinary traffic that cannot possibly have changed a password.
    Since the counter split, uploads do not touch users.json at all — see
    `test_an_upload_does_not_touch_the_credential_file`.

    The test stays, and is worth more than its original framing: it pins that
    ANY rewrite leaving the material identical is a cache hit. That is the leg
    that survives if some future caller starts rewriting users.json for an
    unrelated reason, which is exactly the assumption that just changed once.
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


# ── #343 review, finding A: the anti-poisoning guards were unobserved ────────
#
# The reviewer's mutation run showed both legs SURVIVING: dropping `st_ino`
# from the stamp, and removing the re-stat entirely (read-then-cache). The
# code was right; nothing would have noticed it becoming wrong. For an auth
# cache the regression these prevent is old content pinned under a current
# stamp — a revoked user or a changed password working indefinitely, with no
# self-correction. Both tests below are the mechanisms the reviewer specified.


def test_a_write_during_the_read_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-stat leg. Force a write to land WHILE `load_users` is running.

    Read-then-stat would pair the OLD content with the NEW stamp and every
    later request would agree the stamp is current — poisoned permanently.
    Stat-read-re-stat leaves the stamps unequal, so the result is returned
    without being cached and the next call re-reads.
    """
    _seed(tmp_path)
    users_path = tmp_path / "users.json"
    cache = auth_mod.VerifiedCredentialCache(users_path)

    real_load = auth_mod.load_users
    fired = {"n": 0}

    def racing_load(path):  # type: ignore[no-untyped-def]
        out = real_load(path)
        if fired["n"] == 0:
            fired["n"] = 1
            # The write the reader is about to lose the race to.
            set_disabled(users_path, username="u", disabled=True)
        return out

    monkeypatch.setattr(auth_mod, "load_users", racing_load)
    users = cache.users()

    assert fired["n"] == 1, "the racing write never fired; the test proves nothing"
    # Returned data is the pre-write snapshot -- that is fine and expected.
    assert users["u"].disabled is False
    # What must NOT happen is that snapshot being cached under the new stamp.
    assert cache._stamp is None, "stale content was cached under a current stamp"

    monkeypatch.setattr(auth_mod, "load_users", real_load)
    assert cache.users()["u"].disabled is True, "the cache never self-corrected"


def test_an_identical_rewrite_still_changes_the_stamp(tmp_path: Path) -> None:
    """The `st_ino` leg.

    Every write goes through `atomic_write_text` -> `os.replace`, so the inode
    is always new. Without `st_ino` in the stamp, a rewrite with identical
    length landing in the same mtime nanosecond would be invisible, and the
    cache would keep serving pre-write credentials. Drop `inode` from
    `_DbStamp` and this fails.
    """
    _seed(tmp_path)
    users_path = tmp_path / "users.json"
    before = auth_mod._DbStamp.of(users_path)

    raw = users_path.read_text(encoding="utf-8")
    save_users(users_path, load_users(users_path))
    assert users_path.read_text(encoding="utf-8") == raw, "content must be identical"

    after = auth_mod._DbStamp.of(users_path)
    assert after != before, "an identical rewrite left the stamp unchanged"
    assert after.inode != before.inode
    assert after.size == before.size, "size cannot be what distinguished them"


def test_a_revoked_user_is_rejected_on_the_cached_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What `test_a_disabled_user_is_rejected_on_the_next_request` looked like
    it pinned and did not: that the rejection happens WITHOUT re-running
    PBKDF2, i.e. on a genuine cache hit rather than because the cache happened
    to miss. A change that made every request a miss would leave that test
    green while testing the wrong path."""
    _seed(tmp_path)
    counter = _Pbkdf2Counter(monkeypatch)
    client = _client(tmp_path)
    assert _ping(client).status_code not in (401, 403)
    assert counter.reset() == 1

    set_disabled(tmp_path / "users.json", username="u", disabled=True)

    assert _ping(client).status_code == 403
    assert counter.calls == 0, (
        f"the revoked request spent {counter.calls} PBKDF2 run(s); it was a cache "
        f"MISS, so this does not show revocation working on the cached path"
    )


def test_the_retained_verification_digest_is_keyed(tmp_path: Path) -> None:
    """#343 finding D. What the cache retains must be useless outside this
    process.

    An unsalted sha256(password) is an offline-crackable digest of a LIVE
    credential that outlives the plaintext — the plaintext is freed with the
    request, this lived for the process lifetime, so a memory dump taken hours
    later yielded a crackable digest for every active credential. Keyed with a
    per-process random key it yields nothing without the key, which dies with
    the process.
    """
    import hashlib

    _seed(tmp_path)
    cache = auth_mod.VerifiedCredentialCache(tmp_path / "users.json")
    assert cache.verify("u", "p") is not None
    stored = cache._verified["u"][0]

    assert stored != hashlib.sha256(b"p").digest(), "retained digest is unsalted"
    assert len(stored) == 32, "still a fixed-length constant-time comparison"

    # Two caches in the same process must not agree either, or the key is not
    # actually per-instance random.
    other = auth_mod.VerifiedCredentialCache(tmp_path / "users.json")
    assert other.verify("u", "p") is not None
    assert other._verified["u"][0] != stored

    # And the behaviour it exists for is unchanged: same secret still hits.
    before = cache.pbkdf2_verifications
    assert cache.verify("u", "p") is not None
    assert cache.pbkdf2_verifications == before, "keying broke the cache hit"
    assert cache.verify("u", "wrong") is None


def test_users_returns_a_defensive_copy(tmp_path: Path) -> None:
    """#343 finding C. `users()` handed out the cached dict by reference.

    Nothing mutated it at the time, but `record_upload(cache.users(), ...)` was
    one keystroke away and would have poisoned the cache without touching the
    file — which the stamp design is structurally blind to: no write, no new
    inode, no re-read, wrong data forever.

    ⚑ BOTH RETURN PATHS, because they are separate copies and each leaks on its
    own. The CACHED-HIT path is what finding C was about — it used to
    `return self._users` by reference — and guarding only the miss let
    `if self._stamp == stamp: return dict(self._users)` be reverted with the
    whole suite still green. The MISS path leaks too, which is less obvious: it
    stores `self._users = users` and then returns that same object, so handing
    it back uncopied is the identical hazard one call earlier.

    Measured, not assumed — each copy reverted alone fails this test:
      * hit-path only  -> FAILED (the reviewer's MU-A, which used to survive)
      * miss-path only -> FAILED
    """
    _seed(tmp_path)
    cache = auth_mod.VerifiedCredentialCache(tmp_path / "users.json")

    # Leg 1 — the MISS path (first call, nothing cached yet).
    first = cache.users()
    first.pop("u")
    first["intruder"] = UserRecord(
        username="intruder", salt_b64="x", hash_b64="y", iterations=1,
    )

    second = cache.users()
    assert "u" in second, "a caller emptied the cached DB through the returned dict"
    assert "intruder" not in second, "a caller injected a credential into the cache"

    # Leg 2 — the CACHED-HIT path. `second` above was served from the cache
    # (the file has not changed), so mutating it is the exact hazard.
    reads_before = cache.db_reads
    second.pop("u")
    second["intruder"] = UserRecord(
        username="intruder", salt_b64="x", hash_b64="y", iterations=1,
    )

    third = cache.users()
    assert cache.db_reads == reads_before, "not a cache hit; this leg proves nothing"
    assert "u" in third, "a caller emptied the cache through a HIT-path result"
    assert "intruder" not in third, "a caller injected a credential through a HIT"
    assert cache.verify("u", "p") is not None
