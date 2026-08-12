"""Adversarial upload path tests: tar-escape + compaction-write-failure."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    extract_uploaded_shard_tar,
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


def _build_client(server_root: Path, **kwargs):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    app = create_app(server_root=str(server_root), users_db="users.json", **kwargs)
    return TestClient(app)


def _default_headers() -> dict[str, str]:
    return {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


def _build_valid_zarr_tar(
    tmp_path: Path,
    *,
    samples: list[ReplaySample],
    input_history_encoding: str | None = None,
) -> bytes:
    """Write a valid zarr shard and tar it using the production packer so
    this test actually exercises the worker→server wire format."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    zp = tmp_path / "valid.zarr"
    arrs = samples_to_arrays(samples)
    meta = ShardMeta(
        username="u",
        games=1,
        positions=len(samples),
        model_sha256="abc1234567",
        model_step=0,
        input_history_encoding=input_history_encoding,
    )
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


def _tar_with_large_declared_payload() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        root = tarfile.TarInfo(name="shard.zarr")
        root.type = tarfile.DIRTYPE
        tf.addfile(root)
        payload = tarfile.TarInfo(name="shard.zarr/too_large")
        payload_bytes = b"x" * 11
        payload.size = len(payload_bytes)
        tf.addfile(payload, io.BytesIO(payload_bytes))
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


def test_zarr_tar_upload_rejects_declared_payload_cap_before_extract(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, max_upload_uncompressed_bytes=10)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_with_large_declared_payload(), "application/x-tar")},
        headers=_default_headers(),
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("stored") is False
    assert body.get("rejected") is True
    assert "payload too large" in body.get("reason", "")
    assert not any((server_root / "trials" / "default" / "inbox" / "u").glob("tmp_*.zarr"))


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


def test_zarr_tar_upload_keeps_missing_and_tagged_history_accumulators_separate(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, upload_compact_shard_size=100)
    legacy_tar = _build_valid_zarr_tar(tmp_path / "legacy", samples=[_sample(0)])
    root_tar = _build_valid_zarr_tar(
        tmp_path / "root",
        samples=[_sample(1)],
        input_history_encoding="lc0_root",
    )

    first = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("legacy.zarr.tar", legacy_tar, "application/x-tar")},
        headers=_default_headers(),
    )
    second = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("root.zarr.tar", root_tar, "application/x-tar")},
        headers=_default_headers(),
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json().get("stored") is True
    assert second.json().get("stored") is True


def test_zarr_tar_extract_falls_back_without_filter_kw(tmp_path, monkeypatch) -> None:
    tar_bytes = _build_valid_zarr_tar(tmp_path / "src", samples=[_sample(i) for i in range(2)])
    tar_path = tmp_path / "shard.zarr.tar"
    tar_path.write_bytes(tar_bytes)
    original = tarfile.TarFile.extractall

    def legacy_extractall(self, path=".", members=None, *, numeric_owner=False, filter=None):
        if filter is not None:
            raise TypeError("extractall() got an unexpected keyword argument 'filter'")
        return original(self, path=path, members=members, numeric_owner=numeric_owner)

    monkeypatch.setattr(tarfile.TarFile, "extractall", legacy_extractall)

    extracted = extract_uploaded_shard_tar(tar_path, tmp_path / "extract")

    assert (extracted / ".zgroup").exists()


def test_zarr_tar_upload_rejects_position_cap_before_ingest(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, max_upload_positions=2)
    tar_bytes = _build_valid_zarr_tar(tmp_path, samples=[_sample(i) for i in range(3)])

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
    assert "too many positions" in body.get("reason", "")


def test_bad_shard_report_is_persisted(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    r = client.post(
        "/v1/trials/trial_a/report_bad_shard",
        auth=("u", "p"),
        json={"shard_name": "bad.zarr", "reason": "local invalid shard"},
        headers={**_default_headers(), "X-CAE-Machine-ID": "machine-a"},
    )

    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True
    reports = list((server_root / "trials" / "trial_a" / "quarantine" / "client_reports").glob("*.json"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "bad.zarr" in text
    assert "machine-a" in text


def test_bad_shard_report_persists_only_bounded_payload_fields(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    r = client.post(
        "/v1/trials/trial_a/report_bad_shard",
        auth=("u", "p"),
        json={
            "shard_name": "bad.zarr",
            "reason": "x" * 10_000,
            "unused_blob": "y" * 10_000,
        },
        headers={**_default_headers(), "X-CAE-Machine-ID": "machine-a"},
    )

    assert r.status_code == 200, r.text
    reports = list((server_root / "trials" / "trial_a" / "quarantine" / "client_reports").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert set(report["payload"]) == {"shard_name", "reason"}
    assert report["payload"]["shard_name"] == "bad.zarr"
    assert len(report["payload"]["reason"]) < 600
    assert "unused_blob" not in reports[0].read_text(encoding="utf-8")


def test_outcome_stats_merge_caps_distinct_keys() -> None:
    from chess_anti_engine.server.app import _MAX_OUTCOME_STAT_KEYS, _merge_outcome_stats

    dst: dict[str, int] = {}
    _merge_outcome_stats(dst, {f"outcome_{i}": 1 for i in range(_MAX_OUTCOME_STAT_KEYS + 50)})

    assert len(dst) == _MAX_OUTCOME_STAT_KEYS


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

    def flaky_save(path, *, arrs, meta):
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


def test_arena_username_cannot_escape_arena_inbox(tmp_path) -> None:
    from chess_anti_engine.server.app import resolve_arena_user_dir

    arena = tmp_path / "server" / "arena_inbox"
    arena.mkdir(parents=True)

    assert resolve_arena_user_dir(arena, "../outside") is None
    assert resolve_arena_user_dir(arena, str(tmp_path / "outside")) is None
    assert resolve_arena_user_dir(arena, r"..\outside") is None


def test_arena_username_uses_single_child_dir(tmp_path) -> None:
    from chess_anti_engine.server.app import resolve_arena_user_dir

    arena = tmp_path / "server" / "arena_inbox"
    arena.mkdir(parents=True)

    assert resolve_arena_user_dir(arena, "worker_1") == arena / "worker_1"


def test_shard_run_id_must_match_upload_trial() -> None:
    from chess_anti_engine.server.app import shard_run_id_matches_upload_trial

    assert shard_run_id_matches_upload_trial("trial_a", "trial_a")
    assert shard_run_id_matches_upload_trial("trial_a", "")
    assert not shard_run_id_matches_upload_trial("trial_b", "trial_a")
    assert not shard_run_id_matches_upload_trial(None, "trial_a")


def _tar_with_many_empty_members(count: int) -> bytes:
    """Zero declared payload bytes, `count` members.

    This passes every existing cap -- compressed size, declared uncompressed
    bytes, and position count -- because none of them is a member COUNT.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        root = tarfile.TarInfo(name="shard.zarr")
        root.type = tarfile.DIRTYPE
        tf.addfile(root)
        for i in range(count):
            m = tarfile.TarInfo(name=f"shard.zarr/{i}")
            m.size = 0
            tf.addfile(m, io.BytesIO(b""))
    return buf.getvalue()


def test_extract_rejects_member_count_above_cap(tmp_path) -> None:
    """Member count is an independent axis from declared bytes."""
    tar_path = tmp_path / "many.tar"
    tar_path.write_bytes(_tar_with_many_empty_members(50))

    # Sanity: the byte cap cannot see this archive at all -- zero payload bytes.
    with pytest.raises(ValueError, match="too many members"):
        extract_uploaded_shard_tar(
            tar_path, tmp_path / "out", max_extract_bytes=1_000_000, max_members=10,
        )


def test_extract_allows_member_count_at_cap(tmp_path) -> None:
    """The cap is a ceiling, not an off-by-one rejection of the legal case."""
    tar_path = tmp_path / "few.tar"
    # 10 empty members + 1 dir member = 11 total.
    tar_path.write_bytes(_tar_with_many_empty_members(10))

    out = extract_uploaded_shard_tar(
        tar_path, tmp_path / "out", max_extract_bytes=1_000_000, max_members=11,
    )
    assert out.exists()


def test_extract_rejects_over_long_member_name(tmp_path) -> None:
    from chess_anti_engine.replay.shard import MAX_UPLOAD_TAR_NAME_LEN

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        m = tarfile.TarInfo(name="shard.zarr/" + ("a" * (MAX_UPLOAD_TAR_NAME_LEN + 10)))
        m.size = 0
        tf.addfile(m, io.BytesIO(b""))
    tar_path = tmp_path / "longname.tar"
    tar_path.write_bytes(buf.getvalue())

    with pytest.raises(ValueError, match="over-long member name"):
        extract_uploaded_shard_tar(tar_path, tmp_path / "out")


# ⚑ EVERY upload route, ENUMERATED FROM THE APP -- not a hand-written list.
#
# `_upload_shard_impl` is an inner function that the route wrappers call by
# KEYWORD, so a header declared only on the impl never receives a value: it
# keeps its `Header(...)` FieldInfo default, whose `str()` is a non-empty
# "annotation=nonetype required=false alias=..." blob that equals no digest and
# therefore 422s EVERY upload through that wrapper. That is not hypothetical --
# it is what the first version of this change did.
#
# A literal list cannot cover a wrapper added later, which is the whole failure
# mode. Discovering the routes from `create_app` means a third wrapper is
# covered the moment it exists.
def _discover_upload_routes() -> list[str]:
    from fastapi.routing import APIRoute

    from chess_anti_engine.server.app import create_app

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        app = create_app(server_root=td, users_db="users.json")
        paths = sorted(
            r.path for r in app.routes
            if isinstance(r, APIRoute) and r.path.endswith("/upload_shard")
        )
    # Substitute a concrete id for the path parameter so the route is callable.
    return [p.replace("{trial_id}", "t1") for p in paths]


UPLOAD_ROUTES = _discover_upload_routes()


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_rejects_content_sha256_mismatch(tmp_path, route) -> None:
    """A declared digest that disagrees with the received bytes is a corrupted
    transfer, and must NOT be stored."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    tar_bytes = _build_valid_zarr_tar(tmp_path / "shards", samples=[_sample()])

    r = client.post(
        route,
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers={**_default_headers(), "X-CAE-Content-SHA256": "00" * 32},
    )
    # ⚑ NOT 200-with-rejected: that shape tells the worker the shard is
    # terminally bad and makes it quarantine a shard of real selfplay. A
    # corrupted transfer must stay retryable, which is what a non-200 means
    # to `_upload_response_allows_pending_delete`.
    assert r.status_code == 422, r.text
    assert "sha256 mismatch" in r.text
    inbox = server_root / "inbox"
    stored = list(inbox.rglob("*.zarr")) if inbox.exists() else []
    assert stored == [], f"corrupted upload was stored anyway: {stored}"


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_accepts_correct_content_sha256(tmp_path, route) -> None:
    import hashlib

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    tar_bytes = _build_valid_zarr_tar(tmp_path / "shards", samples=[_sample()])
    good = hashlib.sha256(tar_bytes).hexdigest()

    r = client.post(
        route,
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers={**_default_headers(), "X-CAE-Content-SHA256": good.upper()},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("rejected") is not True, r.text


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_without_digest_header_still_accepted(tmp_path, route) -> None:
    """Verify-if-present. A worker predating the header is a normal fleet state
    during a rolling upgrade; requiring the header would take selfplay down for
    the length of the rollout."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    tar_bytes = _build_valid_zarr_tar(tmp_path / "shards", samples=[_sample()])
    r = client.post(
        route,
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("rejected") is not True, r.text


def test_production_default_member_cap_is_in_force(tmp_path) -> None:
    """⚑ Pins the DEFAULT, not a value the test passes in.

    The server calls `extract_uploaded_shard_tar` without `max_members`
    (`app.py`), so the only thing standing between production and an uncapped
    walk is the signature default. Both cap tests above pass an explicit
    `max_members=`, so setting the default to None left them green -- the
    production value was observed by nothing.
    """
    from chess_anti_engine.replay.shard import MAX_UPLOAD_TAR_MEMBERS

    assert MAX_UPLOAD_TAR_MEMBERS == 20000

    import inspect

    default = inspect.signature(extract_uploaded_shard_tar).parameters["max_members"].default
    assert default == MAX_UPLOAD_TAR_MEMBERS, (
        "the server passes no max_members, so a None/absent default silently "
        "uncaps the production extraction walk"
    )

    # And the default actually bites when nothing is passed.
    tar_path = tmp_path / "over.tar"
    tar_path.write_bytes(_tar_with_many_empty_members(MAX_UPLOAD_TAR_MEMBERS + 1))
    with pytest.raises(ValueError, match="too many members"):
        extract_uploaded_shard_tar(tar_path, tmp_path / "out")


def test_real_production_shard_is_far_under_the_member_cap(tmp_path) -> None:
    """The cap must sit above the largest legitimate upload, or it silently
    rejects good data. Derived from the real shard shape, not asserted."""
    from chess_anti_engine.replay.shard import MAX_UPLOAD_TAR_MEMBERS

    zp = tmp_path / "real.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(64)]),
        meta=ShardMeta(username="u", games=1, positions=64, model_sha256="abc", model_step=0),
    )
    _, buf = pack_shard_for_upload(zp)
    with tarfile.open(fileobj=io.BytesIO(buf.getvalue()), mode="r:") as tf:
        members = len(tf.getmembers())

    assert members < MAX_UPLOAD_TAR_MEMBERS, (
        f"a legitimate shard has {members} members vs a cap of {MAX_UPLOAD_TAR_MEMBERS}"
    )
    # Chunk count scales with positions, array count does not; the headroom the
    # comment claims is ~2.35x at the server's 50k-position ceiling.
    assert members < MAX_UPLOAD_TAR_MEMBERS // 10
