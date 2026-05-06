"""Adversarial upload path tests: tar-escape + compaction-write-failure."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)


def _sample(i: int = 0) -> ReplaySample:
    p = np.zeros(4672, dtype=np.float32)
    p[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=p,
        wdl_target=1,
    )


def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password(password)
    users = {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)}
    save_users(server_root / "users.json", users)


def _build_client(server_root: Path):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    app = create_app(server_root=str(server_root), users_db="users.json")
    return TestClient(app)


def _default_headers() -> dict[str, str]:
    return {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


def _build_valid_zarr_tar(tmp_path: Path, *, samples: list[ReplaySample]) -> bytes:
    """Write a valid zarr shard and tar it using the production packer so
    this test actually exercises the worker→server wire format."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    zp = tmp_path / "valid.zarr"
    arrs = samples_to_arrays(samples)
    meta = ShardMeta(username="u", games=1, positions=len(samples), model_sha256="abc1234567", model_step=0)
    save_local_shard_arrays(zp, arrs=arrs, meta=meta)
    _, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


def _malicious_tar_with_symlink() -> bytes:
    """Build a tar whose first member is a symlink inside the extract dir that
    points to /tmp/escape, followed by a regular file written through the link.

    If extraction honored the symlink, the second member would land outside the
    extract sandbox at /tmp/escape/data. A hardened extractor must reject this.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        link = tarfile.TarInfo(name="shard.zarr/escape_link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp"
        tf.addfile(link)
        # File written through the symlink would escape to /tmp/pwn
        payload = tarfile.TarInfo(name="shard.zarr/escape_link/pwn")
        payload_bytes = b"pwned\n"
        payload.size = len(payload_bytes)
        tf.addfile(payload, io.BytesIO(payload_bytes))
    return buf.getvalue()


def _malicious_tar_with_traversal() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        m = tarfile.TarInfo(name="../escape_via_dotdot")
        m.size = 4
        tf.addfile(m, io.BytesIO(b"pwn\n"))
    return buf.getvalue()


def test_zarr_tar_upload_rejects_symlink_member(tmp_path) -> None:
    """A tar containing a symlink must be quarantined, not extracted."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)
    # Canary file that an escape attack could target.
    canary = Path("/tmp") / "pwn"
    canary.unlink(missing_ok=True)

    tar_bytes = _malicious_tar_with_symlink()
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("stored") is False
    assert body.get("rejected") is True
    assert "link" in body.get("reason", "").lower()
    # No file escaped the sandbox.
    assert not canary.exists(), "tar symlink was followed — extraction sandbox breached"


def test_zarr_tar_upload_rejects_path_traversal(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _malicious_tar_with_traversal(), "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("rejected") is True
    assert "traversal" in body.get("reason", "").lower() or "escape" in body.get("reason", "").lower()


def test_zarr_tar_upload_happy_path(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)
    tar_bytes = _build_valid_zarr_tar(tmp_path, samples=[_sample(i) for i in range(3)])

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("stored") is True
    assert body.get("positions") == 3


def test_compaction_flush_failure_preserves_accumulator(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If compaction write fails, the in-memory buffer must be retained so the
    next upload can retry the flush — otherwise replay samples that dedup has
    already marked as 'seen' are silently dropped."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Intercept save_local_shard_arrays at the module the server imports from;
    # first compaction write fails, second succeeds. This is the actual
    # failure mode (disk-full / rename race) that Codex flagged.
    from chess_anti_engine.server import app as app_module

    original = app_module.save_local_shard_arrays
    calls = {"n": 0}

    def flaky_save(path, *, arrs, meta):  # type: ignore[no-untyped-def]
        # Only flaky for the server compactor's writes (other test setup calls
        # save_local_shard_arrays too, via _build_valid_zarr_tar).
        if str(meta.username) == "server_compactor":
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(28, "No space left on device (simulated)")
        return original(path, arrs=arrs, meta=meta)

    monkeypatch.setattr(app_module, "save_local_shard_arrays", flaky_save)

    # Tiny compaction threshold so the flush triggers on the first upload.
    app = app_module.create_app(
        server_root=str(server_root),
        users_db="users.json",
        upload_compact_shard_size=1,
    )

    from fastapi.testclient import TestClient

    client = TestClient(app)

    tar_bytes_1 = _build_valid_zarr_tar(tmp_path / "u1", samples=[_sample(0), _sample(1)])
    r1 = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard1.zarr.tar", tar_bytes_1, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1.get("stored") is True
    assert body1.get("positions") == 2
    assert calls["n"] == 1, "expected one flush attempt during first upload"
    # No compacted shard yet — flush failed.
    compacted_dir = server_root / "inbox" / "_compacted"
    assert not compacted_dir.exists() or not list(compacted_dir.glob("*.zarr"))

    # Second upload triggers another flush attempt; this time it succeeds and
    # should persist ALL three samples (2 from the failed attempt + 1 new).
    tar_bytes_2 = _build_valid_zarr_tar(tmp_path / "u2", samples=[_sample(2)])
    r2 = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard2.zarr.tar", tar_bytes_2, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r2.status_code == 200, r2.text
    assert calls["n"] == 2, "expected a second (succeeding) flush attempt"

    from chess_anti_engine.replay.shard import load_shard_arrays

    compacted = list(compacted_dir.glob("*.zarr"))
    assert len(compacted) == 1, f"expected 1 compacted shard, found {compacted}"
    arrs, meta = load_shard_arrays(compacted[0])
    # The critical assertion: no replay data was lost due to the first failure.
    assert arrs["x"].shape[0] == 3
    assert int(meta["positions"]) == 3


def test_manifest_artifact_filename_cannot_escape_publish_root(tmp_path) -> None:
    from chess_anti_engine.server.app import resolve_publish_artifact_path

    publish = tmp_path / "server" / "publish"
    publish.mkdir(parents=True)

    assert resolve_publish_artifact_path(publish, "../../outside_stockfish") is None
    assert resolve_publish_artifact_path(publish, str(tmp_path / "outside_stockfish")) is None


def test_manifest_artifact_filename_serves_publish_child(tmp_path) -> None:
    from chess_anti_engine.server.app import resolve_publish_artifact_path

    publish = tmp_path / "server" / "publish"
    publish.mkdir(parents=True)

    assert resolve_publish_artifact_path(publish, "stockfish") == publish / "stockfish"
