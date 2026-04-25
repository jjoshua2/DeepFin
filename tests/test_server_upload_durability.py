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


def _in_flight_dir(server_root: Path) -> Path:
    return server_root / "inbox" / "_in_flight"


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


def test_recovery_drops_in_flight_when_compacted_token_match_exists(tmp_path) -> None:
    """Crash mid-cleanup: a flush wrote the compacted shard but never deleted
    the ``_in_flight/<token>/`` staging dir. Recovery must token-match against
    ``_compacted/`` and delete the leftover staging dir WITHOUT re-seeding —
    otherwise the same samples land in the buffer twice.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # First app: tiny threshold flushes immediately, then we manually simulate
    # the crash-mid-cleanup state by re-staging the contributing shard.
    client = _build_client(server_root, upload_compact_shard_size=1)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1",
        samples=[_sample(0), _sample(1)],
        model_sha256="dddd4444",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200 and r.json().get("stored") is True

    compacted_paths = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted_paths) == 1
    compacted_name = compacted_paths[0].name
    # Pull the flush_token out of the compacted filename: format is
    # ``<int>_<sha8>_<g>g_<p>p_<token>.zarr``.
    token = compacted_name[: -len(LOCAL_SHARD_SUFFIX)].split("_")[-1]
    assert len(token) == 16

    # Reconstruct an in-flight shard with the same flush_token, simulating a
    # crash that wrote the compacted shard but didn't delete the staging dir.
    staging = _in_flight_dir(server_root) / token
    staging.mkdir(parents=True, exist_ok=True)
    leftover_zarr = staging / "leftover_pending.zarr"
    save_local_shard_arrays(
        leftover_zarr,
        arrs=samples_to_arrays([_sample(0), _sample(1)]),
        meta=ShardMeta(username="u", games=1, positions=2, model_sha256="dddd4444"),
    )
    del client

    # Restart: recovery should delete the in-flight dir (token matches an
    # existing compacted shard) and NOT re-seed the orphaned shard.
    app2 = _build_app(server_root, upload_compact_shard_size=1)
    from fastapi.testclient import TestClient
    client2 = TestClient(app2)

    # In-flight dir gone, compacted unchanged, no new pending re-seed.
    assert not staging.exists(), "in-flight staging dir should have been deleted"
    assert len(list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))) == 1
    assert list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}")) == []

    # Sanity: a fresh upload still goes through cleanly post-recovery.
    tar_extra = _build_zarr_tar(
        tmp_path / "u2",
        samples=[_sample(5)],
        model_sha256="dddd4444",
    )
    r2 = client2.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("extra.zarr.tar", tar_extra, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r2.status_code == 200 and r2.json().get("stored") is True
    # Two compacted shards now: original + the new one. The leftover in-flight
    # samples were NOT replayed.
    assert len(list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))) == 2


def test_recovery_dedups_worker_retry_against_recovered_pending(tmp_path) -> None:
    """A worker retries an upload after the server crashes mid-accept. The
    pending zarr left on disk has the full sha in its filename so recovery
    backfills ``recent_upload_shas`` — the live dedup path then drops the
    retry, preventing the recovered samples from being doubled.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # First app: accept upload, leave pending.
    client_a = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1",
        samples=[_sample(i) for i in range(3)],
        model_sha256="eeee5555",
    )
    r = client_a.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("retry.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200 and r.json().get("stored") is True
    assert len(list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))) == 1
    del client_a

    # Restart, then "worker retries" by re-POSTing the SAME tar bytes (same
    # upload sha). Live dedup must treat it as already seen.
    from fastapi.testclient import TestClient
    app_b = _build_app(server_root, upload_compact_shard_size=2000)
    client_b = TestClient(app_b)
    r_retry = client_b.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("retry.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r_retry.status_code == 200, r_retry.text
    # Dedup fired: stored is False (it was already accumulated by the recovery).
    assert r_retry.json().get("stored") is False, r_retry.json()
    # Exactly one pending shard — the recovered one. The retry's pending was
    # staged then deleted by the duplicate-upload branch.
    pending_now = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending_now) == 1, f"unexpected pending count: {pending_now}"

    # Force a flush by tightening the threshold via a fresh upload.
    app_c = _build_app(server_root, upload_compact_shard_size=4)
    client_c = TestClient(app_c)
    tar_more = _build_zarr_tar(
        tmp_path / "u2",
        samples=[_sample(20)],
        model_sha256="eeee5555",
    )
    r_more = client_c.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("more.zarr.tar", tar_more, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r_more.status_code == 200 and r_more.json().get("stored") is True
    compacted = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, compacted
    arrs, _meta = load_shard_arrays(compacted[0])
    # 3 (recovered original) + 1 (trigger) = 4. The retry was deduped — its
    # 3 samples MUST NOT appear here.
    assert arrs["x"].shape[0] == 4, f"retry was double-counted: {arrs['x'].shape[0]}"


def test_recovery_drops_orphan_duplicate_pending_with_same_sha(tmp_path) -> None:
    """Two pending zarrs with the same upload sha can co-exist if the
    duplicate-upload branch's ``delete_shard_path`` silently failed before a
    crash. Recovery must re-seed only the first and delete the rest;
    re-seeding both would double-count those samples in replay.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Build a real pending shard via the live path.
    client = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1",
        samples=[_sample(i) for i in range(2)],
        model_sha256="ffff6666",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("orig.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200 and r.json().get("stored") is True

    pending = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending) == 1
    original = pending[0]
    # Same sha (parts[1]) but different timestamp + token to model an
    # orphaned duplicate that the dedup-rmtree silently failed to clean up.
    stem = original.stem
    int_now, sha_full, _token = stem.split("_", 2)
    orphan_name = f"{int(int_now) + 1}_{sha_full}_deadbeefdeadbeef{LOCAL_SHARD_SUFFIX}"
    orphan_path = original.parent / orphan_name
    save_local_shard_arrays(
        orphan_path,
        arrs=samples_to_arrays([_sample(i) for i in range(2)]),
        meta=ShardMeta(username="u", games=1, positions=2, model_sha256="ffff6666"),
    )
    del client

    # Restart. Recovery scans pending in sorted order; the original (smaller
    # int_now) wins, the orphan is dropped.
    from fastapi.testclient import TestClient
    app2 = _build_app(server_root, upload_compact_shard_size=4)
    client2 = TestClient(app2)
    pending_after = sorted(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending_after) == 1, f"orphan not dropped: {pending_after}"
    assert pending_after[0].name == original.name, "wrong pending kept"

    # Force a flush; only the original 2 samples should appear in compacted.
    tar_more = _build_zarr_tar(
        tmp_path / "u2",
        samples=[_sample(50), _sample(51), _sample(52)],
        model_sha256="ffff6666",
    )
    r_more = client2.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("more.zarr.tar", tar_more, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r_more.status_code == 200 and r_more.json().get("stored") is True
    compacted = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, compacted
    arrs, _meta = load_shard_arrays(compacted[0])
    # 2 (recovered original) + 3 (trigger upload). The orphan's 2 samples
    # must not have been re-seeded.
    assert arrs["x"].shape[0] == 5, f"orphan was re-seeded: {arrs['x'].shape[0]}"
