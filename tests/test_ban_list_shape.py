"""Malformed ban containers follow the documented visible fail-open policy."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from chess_anti_engine.server import manage_users
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


def _cli(monkeypatch: pytest.MonkeyPatch, db: Path, command: str, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["manage_users", "--users-db", str(db), command, *args])
    manage_users.main()


@pytest.mark.parametrize("command", ["ban", "unban"])
@pytest.mark.parametrize("field_name", ["usernames", "ips"])
@pytest.mark.parametrize("value", _BAD_CONTAINERS)
def test_cli_refuses_malformed_container_without_losing_other_bans_or_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    command: str, field_name: str, value: object,
) -> None:
    path = tmp_path / "bans.json"
    # ips=null plus a valid username reproduces the cloud qualifier's data loss.
    original = (json.dumps(_policy(field_name, value), indent=2) + "\n").encode()
    path.write_bytes(original)
    with pytest.raises(SystemExit, match=f"refusing to modify ban list:.*{field_name}") as exc:
        _cli(monkeypatch, tmp_path / "users.json", command,
             "--username", "synthetic-new-banned", "--ip", "203.0.113.9")
    assert exc.value.code != 0
    assert path.read_bytes() == original
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "users.json").exists()


@pytest.mark.parametrize("command", ["ban", "unban"])
@pytest.mark.parametrize("original", [b"{not json", b'["synthetic-worker"]', b"\xff"])
def test_cli_refuses_corrupt_policy_without_rewriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, original: bytes,
) -> None:
    path = tmp_path / "bans.json"
    path.write_bytes(original)
    with pytest.raises(SystemExit, match="refusing to modify ban list"):
        _cli(monkeypatch, tmp_path / "users.json", command, "--username", "synthetic-worker")
    assert path.read_bytes() == original


@pytest.mark.parametrize("command", ["ban", "unban"])
def test_cli_refuses_unreadable_policy_path_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    path = tmp_path / "bans.json"
    path.mkdir()
    marker = path / "synthetic-marker"
    marker.write_bytes(b"preserve")
    with pytest.raises(SystemExit, match="refusing to modify ban list"):
        _cli(monkeypatch, tmp_path / "users.json", command, "--username", "synthetic-worker")
    assert path.is_dir()
    assert marker.read_bytes() == b"preserve"


@pytest.mark.parametrize("command", ["ban", "unban"])
@pytest.mark.parametrize(("option", "bucket", "target"), [
    ("--username", "usernames", "synthetic-worker"),
    ("--ip", "ips", "203.0.113.8"),
])
def test_cli_valid_updates_preserve_unrelated_bans_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    command: str, option: str, bucket: str, target: str,
) -> None:
    path = tmp_path / "bans.json"
    notes = {
        "synthetic-worker": "worker reason", "203.0.113.8": "ip reason",
        "synthetic-other": "other reason",
    }
    payload = _policy("notes", notes)
    payload["usernames"] = ["synthetic-worker", "synthetic-other"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    _cli(monkeypatch, tmp_path / "users.json", command, option, target)
    saved = BanList.load(path, strict=True)
    expected_users = {"synthetic-worker", "synthetic-other"}
    expected_ips = {"203.0.113.8"}
    if command == "unban":
        (expected_users if bucket == "usernames" else expected_ips).discard(target)
        notes.pop(target)
    assert saved.usernames == expected_users
    assert saved.ips == expected_ips
    assert saved.notes == notes


@pytest.mark.parametrize("command", ["ban", "unban"])
def test_cli_missing_policy_is_a_valid_empty_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    _cli(monkeypatch, tmp_path / "users.json", command, "--username", "synthetic-worker")
    saved = BanList.load(tmp_path / "bans.json", strict=True)
    assert saved.usernames == ({"synthetic-worker"} if command == "ban" else set())
    assert saved.ips == set()
    assert saved.notes == {}


@pytest.mark.parametrize("field_name", ["usernames", "ips"])
def test_list_bans_retains_read_fallback_without_modifying_malformed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture, field_name: str,
) -> None:
    path = tmp_path / "bans.json"
    original = json.dumps(_policy(field_name, None)).encode()
    path.write_bytes(original)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _cli(monkeypatch, tmp_path / "users.json", "list-bans")
    assert "no bans" in capsys.readouterr().out
    assert path.read_bytes() == original
    _assert_warning(caplog, path, field_name)
