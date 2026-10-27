"""Malformed ban containers follow the documented visible fail-open policy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from chess_anti_engine.server.access import AccessGuard, BanList, BannedIdentity

_LOGGER = "chess_anti_engine.server.access"
_BAD_CONTAINERS = [None, False, 7, 2.5, "synthetic-worker", {"synthetic-worker": True}]


def _policy(field_name: str, value: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "usernames": ["synthetic-worker"],
        "ips": ["203.0.113.8"],
        "notes": {"synthetic-worker": "synthetic reason"},
    }
    policy[field_name] = value
    return policy


def _assert_warning(caplog: pytest.LogCaptureFixture, path: Path, field_name: str) -> None:
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert str(path) in message
    assert field_name in message
    assert "NOBODY BANNED" in message


@pytest.mark.parametrize("field_name", ["usernames", "ips"])
@pytest.mark.parametrize("value", _BAD_CONTAINERS)
def test_non_array_field_invalidates_the_entire_ban_policy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, field_name: str, value: object,
) -> None:
    path = tmp_path / "bans.json"
    path.write_text(json.dumps(_policy(field_name, value)), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        bans = BanList.load(path)
    assert bans.usernames == set()
    assert bans.ips == set()
    assert bans.notes == {}
    assert not bans.is_banned(username="s", ip="2")
    _assert_warning(caplog, path, field_name)


@pytest.mark.parametrize("field_name", ["usernames", "ips"])
@pytest.mark.parametrize("value", [None, "synthetic-worker"])
def test_access_guard_starts_with_malformed_policy_without_partial_bans(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, field_name: str, value: object,
) -> None:
    path = tmp_path / "bans.json"
    path.write_text(json.dumps(_policy(field_name, value)), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        guard = AccessGuard(path)
        guard.check_not_banned(username="synthetic-worker", ip="203.0.113.8")
        guard.check_not_banned(username="s", ip="2")
    _assert_warning(caplog, path, field_name)


@pytest.mark.parametrize("field_name", ["usernames", "ips"])
@pytest.mark.parametrize("value", [None, "synthetic-worker"])
def test_auth_reload_fails_open_visibly_then_recovers_after_policy_repair(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, field_name: str, value: object,
) -> None:
    path = tmp_path / "bans.json"
    BanList(path=path, usernames={"synthetic-worker"}, ips={"203.0.113.8"}).save()
    guard = AccessGuard(path)
    with pytest.raises(BannedIdentity):
        guard.check_not_banned(username="synthetic-worker", ip="203.0.113.8")

    # Replace the inode so the real cache stamp changes without timing assumptions.
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(_policy(field_name, value)), encoding="utf-8")
    replacement.replace(path)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        guard.check_not_banned(username="synthetic-worker", ip="203.0.113.8")
        guard.check_not_banned(username="s", ip="2")
    _assert_warning(caplog, path, field_name)

    caplog.clear()
    BanList(path=path, usernames={"synthetic-repaired"}).save()
    with pytest.raises(BannedIdentity):
        guard.check_not_banned(username="synthetic-repaired", ip=None)
    guard.check_not_banned(username="synthetic-worker", ip="203.0.113.8")
    assert caplog.records == []


@pytest.mark.parametrize(("payload", "usernames", "ips", "notes"), [
    ({}, set(), set(), {}),
    ({"usernames": ["synthetic-worker"], "notes": {"synthetic-worker": "reason"}},
     {"synthetic-worker"}, set(), {"synthetic-worker": "reason"}),
    ({"ips": ["203.0.113.8"]}, set(), {"203.0.113.8"}, {}),
    ({"usernames": ["synthetic-worker", "synthetic-worker", " ", " spaced "]},
     {"synthetic-worker", " spaced "}, set(), {}),
    # Preserve legacy element conversion; this change validates containers only.
    ({"usernames": [7, None, True], "ips": [42], "notes": ["ignored"]},
     {"7", "None", "True"}, {"42"}, {}),
])
def test_supported_policy_values_round_trip_without_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    payload: dict[str, object], usernames: set[str], ips: set[str], notes: dict[str, str],
) -> None:
    path = tmp_path / "bans.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        first = BanList.load(path)
        assert (first.usernames, first.ips, first.notes) == (usernames, ips, notes)
        first.save()
        second = BanList.load(path)
    assert (second.usernames, second.ips, second.notes) == (usernames, ips, notes)
    assert caplog.records == []


def test_absent_ban_policy_remains_empty_and_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "absent.json"
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        bans = BanList.load(path)
    assert (bans.usernames, bans.ips, bans.notes) == (set(), set(), {})
    assert caplog.records == []
    assert not path.exists()
