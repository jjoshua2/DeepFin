"""F002: server-side upload durability regression tests.

Before this change ``_upload_shard_impl`` materialized the uploaded shard
into an in-memory accumulator and deleted both the upload tar and the
extracted zarr. If the server crashed between the upload acknowledgement
and the next compaction-flush, replay samples were lost even though the
worker had been told ``stored: true``.

These tests pin the new on-disk durability contract:

1. Upload below compaction threshold → a ``_pending/<file>.zarr`` exists.
2. A subsequent ``create_app`` over the same ``server_root`` re-seeds the
   accumulator and the next forced flush produces a compacted shard with
   all samples.
3. A successful flush deletes the contributing pending zarr.
4. Two uploads from different ``(trial, model_sha)`` keys land in
   independent pending shards; flushing one keeps the other intact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    ShardMeta,
    load_shard_arrays,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)


def _sample(i: int = 0) -> ReplaySample:
    pol = np.zeros(4672, dtype=np.float32)
    pol[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=pol,
        wdl_target=1,
    )


def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password(password)
    users = {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)}
    save_users(server_root / "users.json", users)


def _build_app(server_root: Path, *, upload_compact_shard_size: int = 2000):
    from chess_anti_engine.server.app import create_app

    return create_app(
        server_root=str(server_root),
        users_db="users.json",
        upload_compact_shard_size=upload_compact_shard_size,
    )


def _build_client(server_root: Path, *, upload_compact_shard_size: int = 2000):
    from fastapi.testclient import TestClient

    return TestClient(_build_app(server_root, upload_compact_shard_size=upload_compact_shard_size))


def _default_headers() -> dict[str, str]:
    return {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


def _build_zarr_tar(
    tmp_path: Path,
    *,
    samples: list[ReplaySample],
    model_sha256: str,
) -> bytes:
    tmp_path.mkdir(parents=True, exist_ok=True)
    zp = tmp_path / "valid.zarr"
    arrs = samples_to_arrays(samples)
    meta = ShardMeta(
        username="u",
        games=1,
        positions=len(samples),
        model_sha256=model_sha256,
        model_step=0,
    )
    save_local_shard_arrays(zp, arrs=arrs, meta=meta)
    _, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


def _pending_dir(server_root: Path) -> Path:
    return server_root / "inbox" / "_pending"


def _compacted_dir(server_root: Path) -> Path:
    return server_root / "inbox" / "_compacted"


def test_upload_below_threshold_persists_pending_zarr(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Compaction threshold high so a 2-sample upload stays in the buffer.
    client = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1",
        samples=[_sample(0), _sample(1)],
        model_sha256="aaaa1111",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("stored") is True
    assert body.get("positions") == 2

    pending = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending) == 1, f"expected one pending shard, found {pending}"

    # Compacted shard not yet written — flush has not run.
    compacted = _compacted_dir(server_root)
    assert not compacted.exists() or not list(compacted.glob(f"*{LOCAL_SHARD_SUFFIX}"))

    # Sanity-check the pending zarr is the actual shard payload, not a tar.
    arrs, meta = load_shard_arrays(pending[0])
    assert arrs["x"].shape[0] == 2
    assert int(meta["positions"]) == 2
    assert str(meta.get("model_sha256")) == "aaaa1111"


def test_restart_replays_pending_uploads_into_compacted_shard(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # First app: accept an upload, leave it pending (high compaction threshold).
    client_a = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1",
        samples=[_sample(i) for i in range(3)],
        model_sha256="bbbb2222",
    )
    r = client_a.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True

    pending_before = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending_before) == 1
    # Drop the first app without flushing — this simulates a process crash.
    del client_a

    # Second app over the same server_root: recovery must re-seed accumulators
    # from the pending shard so a forced flush persists the samples.
    from chess_anti_engine.server import app as app_module

    app_b = _build_app(server_root, upload_compact_shard_size=2000)
    # Reach into the app's state via the lifespan-attached helpers indirectly:
    # we exercise the public flush via a small-threshold app that triggers on
    # the next upload, OR by importing the helper. The simplest deterministic
    # check is to issue a second upload of a different shard that pushes the
    # same accumulator past the compaction threshold and observe both samples
    # land in the compacted shard.
    from fastapi.testclient import TestClient

    # Rebuild the app with a small threshold that forces a flush on the next
    # upload. The pending shard from the first app should already have
    # re-seeded the accumulator with 3 samples, so any further upload tips
    # the accumulator over the threshold.
    del app_b
    app_c = _build_app(server_root, upload_compact_shard_size=4)
    client_b = TestClient(app_c)

    tar_bytes_2 = _build_zarr_tar(
        tmp_path / "u2",
        samples=[_sample(10), _sample(11)],
        model_sha256="bbbb2222",
    )
    r2 = client_b.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard2.zarr.tar", tar_bytes_2, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r2.status_code == 200, r2.text
    assert r2.json().get("stored") is True

    compacted = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, f"expected one compacted shard, found {compacted}"
    arrs, meta = load_shard_arrays(compacted[0])
    # 3 from the recovered first upload + 2 from the trigger upload.
    assert arrs["x"].shape[0] == 5
    assert int(meta["positions"]) == 5

    # Successful flush must have deleted both contributing pending shards.
    pending_after = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert pending_after == [], f"pending shards not cleaned up: {pending_after}"

    # Sanity: the unused module import keeps lint happy and documents intent.
    assert hasattr(app_module, "_PENDING_DIR_NAME")


def test_successful_flush_deletes_pending_shards(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Tiny threshold — the very first upload triggers a flush.
    client = _build_client(server_root, upload_compact_shard_size=1)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1",
        samples=[_sample(0), _sample(1)],
        model_sha256="cccc3333",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True

    compacted = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, f"expected one compacted shard, found {compacted}"

    pending = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert pending == [], f"pending shard not cleaned up after flush: {pending}"


def test_two_distinct_keys_yield_two_pending_files_and_independent_flush(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # High threshold so neither upload flushes on its own.
    client = _build_client(server_root, upload_compact_shard_size=2000)

    # Two uploads that share the trial (default, None) but have different
    # model_sha256 — distinct accumulator keys.
    tar_a = _build_zarr_tar(
        tmp_path / "uA",
        samples=[_sample(0), _sample(1)],
        model_sha256="aaaaaaaa",
    )
    tar_b = _build_zarr_tar(
        tmp_path / "uB",
        samples=[_sample(2), _sample(3)],
        model_sha256="bbbbbbbb",
    )
    rA = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("a.zarr.tar", tar_a, "application/x-tar")},
        headers=_default_headers(),
    )
    rB = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("b.zarr.tar", tar_b, "application/x-tar")},
        headers=_default_headers(),
    )
    assert rA.status_code == 200 and rA.json().get("stored") is True
    assert rB.status_code == 200 and rB.json().get("stored") is True

    pending = sorted(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending) == 2, f"expected two pending shards, found {pending}"

    # Now force-flush only the model_sha256=aaaaaaaa accumulator by reaching
    # into the app via a fresh recovery call. Easier path: rebuild the app
    # with a small threshold and issue one extra tiny upload to model A; only
    # A should compact, B's pending should remain.
    from fastapi.testclient import TestClient

    app2 = _build_app(server_root, upload_compact_shard_size=3)
    client2 = TestClient(app2)
    tar_a_extra = _build_zarr_tar(
        tmp_path / "uA2",
        samples=[_sample(4)],
        model_sha256="aaaaaaaa",
    )
    r_extra = client2.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("a_extra.zarr.tar", tar_a_extra, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r_extra.status_code == 200 and r_extra.json().get("stored") is True

    compacted = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, f"expected one compacted shard, found {compacted}"
    arrs, meta = load_shard_arrays(compacted[0])
    # 2 (recovered A) + 1 (A extra) = 3 samples, all from key A.
    assert arrs["x"].shape[0] == 3
    assert str(meta.get("model_sha256")) == "aaaaaaaa"

    # B's pending shard must still be there.
    pending_after = sorted(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending_after) == 1, f"expected B's pending intact, found {pending_after}"
    arrs_b, meta_b = load_shard_arrays(pending_after[0])
    assert arrs_b["x"].shape[0] == 2
    assert str(meta_b.get("model_sha256")) == "bbbbbbbb"
