"""Per-upload counters live in their own file, not in the credential DB.

#344's review (finding B) is the reason: #344 exempted the per-upload
`save_users` call from fsync, but `users.json` was ONE file, so that write
rewrote the credentials too. After it, the newest on-disk credentials were
exactly as unflushed as the counters — an unclean shutdown could still cost a
just-created password, which is the outcome the durable default was chosen to
prevent. The exemption bought ~50 ms per iteration, under 0.01% of wall clock.

Splitting the files removes the trade instead of picking a side, and it also
shrinks two pre-existing problems: the users.json read-modify-write on the
upload path (which could lose a concurrent `set_disabled`) is gone, and the
auth cache's invalidation story stops depending on upload traffic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from chess_anti_engine.server.auth import (
    UserRecord,
    UserStats,
    ensure_user,
    hash_password,
    load_user_stats,
    load_users,
    migrate_user_stats,
    record_upload,
    save_user_stats,
    save_users,
    set_disabled,
    upsert_user,
    user_stats_path_for,
)

_LOGGER = "chess_anti_engine.server"


def _rec(username: str = "u", password: str = "p") -> UserRecord:
    salt, hsh, iters = hash_password(password)
    return UserRecord(
        username=username, salt_b64=salt, hash_b64=hsh, iterations=iters,
    )


def _legacy_users_json(path: Path, *, uploads: int = 7, positions: int = 1234) -> None:
    """A pre-split users.json: credentials and counters in one object."""
    salt, hsh, iters = hash_password("p")
    path.write_text(json.dumps({
        "u": {
            "salt_b64": salt, "hash_b64": hsh, "iterations": iters, "disabled": False,
            "uploads": uploads, "total_bytes": 5000, "total_positions": positions,
            "last_upload_at_unix": 1700000000,
            "machines": {"boxA": {"uploads": uploads, "positions": positions,
                                  "last_upload_at_unix": 1700000000}},
        },
    }), encoding="utf-8")


# ── migration ────────────────────────────────────────────────────────────────


def test_migration_carries_counters_and_strips_them_from_the_credential_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    _legacy_users_json(users_path)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert migrate_user_stats(users_path, stats_path) == 1

    stats = load_user_stats(stats_path)
    assert stats["u"].uploads == 7
    assert stats["u"].total_positions == 1234
    assert stats["u"].total_bytes == 5000
    assert stats["u"].last_upload_at_unix == 1700000000
    assert stats["u"].machines["boxA"]["positions"] == 1234

    # Counters are GONE from the credential file, so nothing can re-migrate
    # them and no upload write can carry them along.
    raw = json.loads(users_path.read_text(encoding="utf-8"))
    assert set(raw["u"]) == {"salt_b64", "hash_b64", "iterations", "disabled"}, raw["u"]
    # Credentials survived the rewrite intact.
    assert load_users(users_path)["u"].salt_b64 == raw["u"]["salt_b64"]
    assert any("migrated upload counters" in r.getMessage() for r in caplog.records)


def test_migration_does_not_double_count(tmp_path: Path) -> None:
    """The requirement that makes this safe to run at every startup.

    Two mechanisms make it safe and either alone would suffice: the stats file
    short-circuits a repeat, and the carry is an ASSIGNMENT rather than an
    addition. What this test pins is their CONJUNCTION — the totals are
    unchanged after repeat migrations, both with the short-circuit in play and
    with it defeated.

    ⚑ It does NOT separate them, and no test can. The `stats_p.exists()`
    short-circuit means migration only ever runs against a non-existent stats
    file, so deleting that file to defeat the short-circuit also deletes
    anything there was to add to: assignment and addition are indistinguishable
    in the only scenario migration reaches. Concretely, rewriting the carry to
    ADD to existing stats SURVIVES this suite (reviewer's MU-D). That is not a
    gap, because the redundancy is the point — but the earlier wording here
    claimed a decomposition the test does not deliver, and on a migration that
    runs at every startup the comment should not over-promise.
    """
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    _legacy_users_json(users_path, uploads=7)

    assert migrate_user_stats(users_path, stats_path) == 1
    for _ in range(5):
        assert migrate_user_stats(users_path, stats_path) == 0  # short-circuit
    assert load_user_stats(stats_path)["u"].uploads == 7

    # Second leg: restore the legacy source and delete the stats file, so the
    # short-circuit cannot help and the migration genuinely re-runs.
    _legacy_users_json(users_path, uploads=7)
    stats_path.unlink()
    assert migrate_user_stats(users_path, stats_path) == 1
    assert load_user_stats(stats_path)["u"].uploads == 7, "totals must not inflate"


def test_migration_survives_a_crash_between_its_two_writes(tmp_path: Path) -> None:
    """Stats file first, strip second — so the failure mode is counters in both
    places (resolved in favour of the stats file), never counters nowhere."""
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    _legacy_users_json(users_path, uploads=9)

    # Simulate: the stats write landed, the strip did not.
    save_user_stats(stats_path, {"u": UserStats(uploads=9)}, durable=True)
    assert json.loads(users_path.read_text(encoding="utf-8"))["u"]["uploads"] == 9

    assert migrate_user_stats(users_path, stats_path) == 0  # will not re-run
    assert load_user_stats(stats_path)["u"].uploads == 9    # not doubled
    # And the stale legacy keys still parse, rather than breaking auth.
    assert load_users(users_path)["u"].disabled is False


def test_a_fresh_install_migrates_nothing(tmp_path: Path) -> None:
    """Negative control: with no legacy counters there is nothing to carry, and
    in particular no stats file is created to short-circuit a later real one."""
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    save_users(users_path, {"u": _rec()})
    assert migrate_user_stats(users_path, stats_path) == 0
    assert not stats_path.exists()
    assert migrate_user_stats(users_path, stats_path) == 0


def test_legacy_counters_do_not_break_loading_credentials(tmp_path: Path) -> None:
    """`UserRecord` no longer has these fields; an un-migrated file must still
    parse or every worker on that server fails to authenticate."""
    users_path = tmp_path / "users.json"
    _legacy_users_json(users_path)
    users = load_users(users_path)
    assert set(users) == {"u"}
    assert not hasattr(users["u"], "uploads")


def test_a_genuinely_unknown_key_still_raises(tmp_path: Path) -> None:
    """Negative control for the filter above: only the KNOWN legacy counter
    keys are dropped. A typo'd or unexpected field must not be swallowed —
    silently ignoring input is the defect class this audit exists for."""
    users_path = tmp_path / "users.json"
    salt, hsh, iters = hash_password("p")
    users_path.write_text(json.dumps({
        "u": {"salt_b64": salt, "hash_b64": hsh, "iterations": iters,
              "disbaled": True},  # typo
    }), encoding="utf-8")
    with pytest.raises(TypeError):
        load_users(users_path)


# ── the split itself ─────────────────────────────────────────────────────────


def test_an_upload_does_not_touch_the_credential_file(tmp_path: Path) -> None:
    """The headline. Recording an upload must leave users.json byte-identical
    AND at the same inode — the auth cache keys on the inode, so even a
    same-content rewrite would be a needless invalidation."""
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    save_users(users_path, {"u": _rec()})
    before_bytes = users_path.read_bytes()
    before_ino = users_path.stat().st_ino

    stats = load_user_stats(stats_path)
    record_upload(stats, username="u", bytes_uploaded=100, positions=10, machine_id="m")
    save_user_stats(stats_path, stats)

    assert users_path.read_bytes() == before_bytes
    assert users_path.stat().st_ino == before_ino
    assert load_user_stats(stats_path)["u"].uploads == 1


def test_record_upload_accumulates(tmp_path: Path) -> None:
    """Positive control: the counters must still count. A split that quietly
    stopped recording would pass every test above."""
    stats_path = tmp_path / "users.stats.json"
    for i in range(3):
        stats = load_user_stats(stats_path)
        record_upload(
            stats, username="u", bytes_uploaded=100, positions=10, machine_id="m",
        )
        save_user_stats(stats_path, stats)
        assert load_user_stats(stats_path)["u"].uploads == i + 1

    final = load_user_stats(stats_path)["u"]
    assert final.total_bytes == 300
    assert final.total_positions == 30
    assert final.machines["m"]["uploads"] == 3
    assert final.last_upload_at_unix is not None


def test_a_password_change_no_longer_has_to_carry_counters(tmp_path: Path) -> None:
    """`upsert_user` used to copy five counter fields across the rewrite.
    Forgetting one silently zeroed a contributor's history; now it cannot,
    because the counters are not in the file it writes."""
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    ensure_user(users_path, username="u", password="p")
    save_user_stats(stats_path, {"u": UserStats(uploads=42, total_positions=999)})

    upsert_user(users_path, username="u", password="new")
    set_disabled(users_path, username="u", disabled=True)

    kept = load_user_stats(stats_path)["u"]
    assert (kept.uploads, kept.total_positions) == (42, 999)


def test_an_upload_cannot_revert_a_concurrent_disable(tmp_path: Path) -> None:
    """The lost-update shape the reviewer recorded on #343, shrunk.

    The upload path used to do its own load->mutate->save on users.json while
    holding a lock that `set_disabled` (a separate PROCESS, via manage_users)
    does not take. An upload that read the DB before a disable wrote back the
    pre-disable record and silently re-enabled a revoked user. The upload path
    no longer opens that file, so this interleaving cannot revert anything.
    """
    users_path = tmp_path / "users.json"
    stats_path = user_stats_path_for(users_path)
    ensure_user(users_path, username="u", password="p")

    # Upload begins: reads the counter file (NOT the credential file).
    stats = load_user_stats(stats_path)
    # Admin revokes, in another process, mid-upload.
    set_disabled(users_path, username="u", disabled=True)
    # Upload finishes and writes back.
    record_upload(stats, username="u", bytes_uploaded=1, positions=1)
    save_user_stats(stats_path, stats)

    assert load_users(users_path)["u"].disabled is True, "revocation was reverted"


def test_the_stats_path_is_derived_from_the_db_name(tmp_path: Path) -> None:
    """Two servers sharing a directory with different --users-db names must not
    fight over one stats file."""
    a = user_stats_path_for(tmp_path / "users.json")
    b = user_stats_path_for(tmp_path / "other.json")
    assert a != b
    assert a.name == "users.stats.json"
    assert a.parent == tmp_path


def test_corrupt_stats_file_is_survivable_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Counters must never be able to fail an upload — but the degradation is
    logged, so 'the numbers reset' is explicable rather than mysterious."""
    stats_path = tmp_path / "users.stats.json"
    stats_path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert load_user_stats(stats_path) == {}
    assert any("unreadable" in r.getMessage() for r in caplog.records)


# ── end to end through the real upload route ─────────────────────────────────


def test_uploading_through_the_app_writes_only_the_stats_file(tmp_path: Path) -> None:
    """The integration leg: whatever the helpers do in isolation, the served
    route is what matters."""
    from tests.test_server_upload_security import (
        _build_client,
        _build_valid_zarr_tar,
        _default_headers,
        _sample,
        _seed_user,
    )

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    users_path = server_root / "users.json"
    before_ino = users_path.stat().st_ino
    before_bytes = users_path.read_bytes()

    client = _build_client(server_root)
    tar_bytes = _build_valid_zarr_tar(tmp_path, samples=[_sample(i) for i in range(3)])
    r = client.post(
        "/v1/upload_shard", auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers={**_default_headers(), "X-CAE-Machine-ID": "boxA"},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True, r.text

    assert users_path.read_bytes() == before_bytes, "an upload rewrote credentials"
    assert users_path.stat().st_ino == before_ino

    stats = load_user_stats(user_stats_path_for(users_path))
    assert stats["u"].uploads == 1
    assert stats["u"].total_positions == 3
    assert stats["u"].machines["boxA"]["uploads"] == 1


def test_the_app_migrates_a_legacy_db_at_startup(tmp_path: Path) -> None:
    from tests.test_server_upload_security import _build_client

    server_root = tmp_path / "server"
    server_root.mkdir()
    _legacy_users_json(server_root / "users.json", uploads=11, positions=222)

    _build_client(server_root)  # create_app runs the migration

    stats = load_user_stats(user_stats_path_for(server_root / "users.json"))
    assert stats["u"].uploads == 11
    assert stats["u"].total_positions == 222
