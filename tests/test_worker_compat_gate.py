"""A11/A12: the worker-compat gate must be decidable, visible, and terminal
only when it means to be.

A11: `_check_worker_compat` returned `(True, "")` whenever `_load_manifest`
returned None, and `_load_manifest` returned None for a MISSING manifest and
for an unreadable one alike, swallowing the error. So a corrupt manifest turned
`min_worker_version` / `protocol_version` enforcement off for every
worker-facing route, with no log line anywhere -- a day spent admitting
unversioned workers read exactly like a day spent enforcing correctly. The
concrete cost is a stale worker uploading v1 146-plane shards into a
v2_threats replay buffer.

A12: compat rejections ride an HTTP 200. That turns out to be REQUIRED, not an
oversight, and `test_a12_*` below pins why using the worker's own response
handlers.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import pytest

_LOGGER = "chess_anti_engine.server"


def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password(password)
    users = {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)}
    save_users(server_root / "users.json", users)


def _client(server_root: Path):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    return TestClient(create_app(server_root=str(server_root), users_db="users.json"))


def _server(tmp_path: Path) -> Path:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    return server_root


def _publish(server_root: Path, manifest: Any) -> Path:
    pub = server_root / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    mf = pub / "manifest.json"
    mf.write_text(
        manifest if isinstance(manifest, str) else json.dumps(manifest), encoding="utf-8",
    )
    return mf


def _headers(version: str = "9.9.9", protocol: str = "1") -> dict[str, str]:
    return {"X-CAE-Worker-Version": version, "X-CAE-Protocol-Version": protocol}


def _tar_bytes() -> bytes:
    """A tarball the compat gate rejects before anything inspects it. Its
    contents are irrelevant -- the gate runs first, which is the point."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"not a real shard"
        info = tarfile.TarInfo("s.zarr/x")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _upload(client, **kwargs):
    return client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("s.zarr.tar", _tar_bytes(), "application/x-tar")},
        **kwargs,
    )


def _msgs(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _LOGGER and r.levelno >= level
    ]


_GOOD_MANIFEST = {"min_worker_version": "1.0.0", "protocol_version": 1}
_CORRUPT = "{ this is not json"


# ── A11: unreadable manifest must not silently open the gate ─────────────────


def test_corrupt_manifest_does_not_admit_an_upload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The headline defect. On the old code this upload was ADMITTED: the JSON
    error was swallowed, the gate returned (True, ""), and the shard was
    accepted with no version check and no log line."""
    server_root = _server(tmp_path)
    _publish(server_root, _CORRUPT)
    client = _client(server_root)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        r = _upload(client, headers=_headers())

    assert r.status_code == 503, r.text
    body = r.json()
    assert body.get("stored") is not True, body
    warnings = _msgs(caplog, logging.WARNING)
    assert any("manifest UNREADABLE" in m for m in warnings), warnings
    # The counters ride the warning: the finding was unfalsifiability, so the
    # line has to carry enough to answer "how long has this been happening".
    assert any("unreadable=1" in m for m in warnings), warnings


def test_corrupt_manifest_poll_is_retryable_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """503, never 426.

    This is the "do not strand a healthy fleet" constraint, and it is not a
    style preference: the worker's `_poll_manifest` sends 426 to
    `_handle_upgrade_required`, which raises SystemExit. Answering a transient
    read error with 426 would kill every worker in the fleet. 503 lands in the
    `status_code != 200` branch -- sleep and retry.
    """
    server_root = _server(tmp_path)
    _publish(server_root, _CORRUPT)
    client = _client(server_root)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        r = client.get("/v1/manifest", headers=_headers())

    assert r.status_code == 503, r.text
    assert r.status_code != 426
    assert any("manifest UNREADABLE" in m for m in _msgs(caplog, logging.WARNING))


def test_corrupt_manifest_blocks_arena_upload_too(tmp_path: Path) -> None:
    """The gate guards four routes; a fix on only the shard route would leave
    the same admission hole open next door."""
    server_root = _server(tmp_path)
    _publish(server_root, _CORRUPT)
    client = _client(server_root)
    r = client.post(
        "/v1/upload_arena_result", auth=("u", "p"), json={"x": 1}, headers=_headers(),
    )
    assert r.status_code == 503, r.text


def test_unreadable_warning_does_not_flood(tmp_path: Path,
                                           caplog: pytest.LogCaptureFixture) -> None:
    """A busy fleet polls many times a second. One line per distinct fault, not
    one per request -- otherwise the useful line is buried by its own copies."""
    server_root = _server(tmp_path)
    _publish(server_root, _CORRUPT)
    client = _client(server_root)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(8):
            client.get("/v1/manifest", headers=_headers())

    unreadable = [m for m in _msgs(caplog, logging.WARNING) if "manifest UNREADABLE" in m]
    assert len(unreadable) == 1, unreadable


# ── A11: the legitimate open case stays open, and says so ────────────────────


def test_absent_manifest_still_admits_and_announces_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Fail-open before the first publish is deliberate: a worker cannot be
    judged against requirements that were never published. What was missing is
    any record that the gate was open, so this asserts BOTH."""
    server_root = _server(tmp_path)
    client = _client(server_root)  # nothing published

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        r = _upload(client, headers=_headers(version="0.0.0"))

    # Admitted past the gate: it fails later on shard contents, not on compat.
    assert r.json().get("reason_code") is None, r.text
    infos = _msgs(caplog, logging.INFO)
    assert any("manifest not published yet" in m for m in infos), infos
    assert any("enforcement is OPEN" in m for m in infos), infos


def test_absent_announcement_is_once_per_trial(tmp_path: Path,
                                               caplog: pytest.LogCaptureFixture) -> None:
    """Legitimate and continuous until the first publish, so it must not be a
    per-request line."""
    server_root = _server(tmp_path)
    client = _client(server_root)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(5):
            client.get("/v1/manifest", headers=_headers())
    absent = [m for m in _msgs(caplog, logging.INFO) if "manifest not published yet" in m]
    assert len(absent) <= 1, absent


def test_healthy_manifest_still_enforces_and_still_admits(tmp_path: Path) -> None:
    """Positive + negative control for the whole gate: with a READABLE manifest
    an old worker is rejected and a current one is admitted. Without this, a
    gate stuck permanently closed would pass every test above."""
    server_root = _server(tmp_path)
    _publish(server_root, _GOOD_MANIFEST)
    client = _client(server_root)

    old = _upload(client, headers=_headers(version="0.0.1"))
    assert old.status_code == 200
    assert old.json().get("rejected") is True, old.text

    current = _upload(client, headers=_headers(version="9.9.9"))
    assert current.json().get("reason_code") is None, current.text


# ── A12: the 200 is load-bearing ─────────────────────────────────────────────


def test_a12_rejection_body_carries_a_machine_readable_code(tmp_path: Path) -> None:
    server_root = _server(tmp_path)
    _publish(server_root, _GOOD_MANIFEST)
    client = _client(server_root)

    r = _upload(client, headers=_headers(version="0.0.1"))
    body = r.json()
    assert body["reason_code"] == "worker_too_old", body
    assert body["rejected"] is True
    assert isinstance(body.get("reason"), str), body
    assert body["reason"], body

    r2 = _upload(client, headers=_headers(protocol="7"))
    assert r2.json()["reason_code"] == "protocol_mismatch", r2.text


def test_a12_rejection_is_logged_server_side(tmp_path: Path,
                                             caplog: pytest.LogCaptureFixture) -> None:
    """A rejection the operator cannot see is the A11 defect wearing a
    different hat: the shard is destroyed worker-side and nothing on the server
    says why."""
    server_root = _server(tmp_path)
    _publish(server_root, _GOOD_MANIFEST)
    client = _client(server_root)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _upload(client, headers=_headers(version="0.0.1"))
    warnings = _msgs(caplog, logging.WARNING)
    assert any("compat REJECT" in m and "worker_too_old" in m for m in warnings), warnings


def test_a12_worker_treats_a_real_rejection_as_terminal(tmp_path: Path) -> None:
    """WHY the 200 stays. Fed to the worker's OWN handlers, a genuine compat
    rejection must resolve to "quarantine, stop retrying".

    Change the status to 4xx and `_upload_response_rejection_reason` returns
    None (it requires 200), so the shard is never quarantined, while
    `_upload_response_allows_pending_delete` also returns False -- the worker
    would resend an incompatible shard forever. That is why A12 is closed by
    adding `reason_code`, not by changing the status.
    """
    from chess_anti_engine.worker import (
        _upload_response_allows_pending_delete,
        _upload_response_rejection_reason,
    )

    server_root = _server(tmp_path)
    _publish(server_root, _GOOD_MANIFEST)
    client = _client(server_root)
    r = _upload(client, headers=_headers(version="0.0.1"))

    assert _upload_response_rejection_reason(r) is not None, r.text
    assert _upload_response_allows_pending_delete(r) is False


def test_a12_worker_keeps_the_shard_when_the_gate_is_undecidable(tmp_path: Path) -> None:
    """The mirror image, and the reason the unreadable case must NOT reuse the
    200 rejection body: a transient server-side fault must never terminally
    destroy a healthy worker's data.

    503 makes both worker helpers decline -- no quarantine, no local delete --
    which is `keep it and retry`.
    """
    from chess_anti_engine.worker import (
        _upload_response_allows_pending_delete,
        _upload_response_rejection_reason,
    )

    server_root = _server(tmp_path)
    _publish(server_root, _CORRUPT)
    client = _client(server_root)
    r = _upload(client, headers=_headers())

    assert r.status_code == 503
    assert _upload_response_rejection_reason(r) is None, "shard would be quarantined"
    assert _upload_response_allows_pending_delete(r) is False, "shard would be deleted"


# ── the recovery path ────────────────────────────────────────────────────────


def test_gate_recovers_when_the_manifest_is_repaired(tmp_path: Path) -> None:
    """Fail-closed must be a pause, not a latch: once the manifest parses
    again the same worker is admitted without a server restart."""
    server_root = _server(tmp_path)
    _publish(server_root, _CORRUPT)
    client = _client(server_root)
    assert _upload(client, headers=_headers()).status_code == 503

    _publish(server_root, _GOOD_MANIFEST)
    ok = _upload(client, headers=_headers(version="9.9.9"))
    assert ok.status_code == 200, ok.text
    assert ok.json().get("reason_code") is None, ok.text
