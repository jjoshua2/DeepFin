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

import hashlib
import json
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
    assert rA.status_code == 200
    assert rA.json().get("stored") is True
    assert rB.status_code == 200
    assert rB.json().get("stored") is True

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
    assert r_extra.status_code == 200
    assert r_extra.json().get("stored") is True

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
    upload_sha = hashlib.sha256(tar_bytes).hexdigest()
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200
    assert r.json().get("stored") is True

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
    leftover_zarr = staging / f"123_{upload_sha}_leftoverpending{LOCAL_SHARD_SUFFIX}"
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
    retry = client2.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("retry.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert retry.status_code == 200, retry.text
    assert retry.json().get("stored") is False, retry.json()

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
    assert r2.status_code == 200
    assert r2.json().get("stored") is True
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
    assert r.status_code == 200
    assert r.json().get("stored") is True
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
    assert r_more.status_code == 200
    assert r_more.json().get("stored") is True
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
    assert r.status_code == 200
    assert r.json().get("stored") is True

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
    assert r_more.status_code == 200
    assert r_more.json().get("stored") is True
    compacted = list(_compacted_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(compacted) == 1, compacted
    arrs, _meta = load_shard_arrays(compacted[0])
    # 2 (recovered original) + 3 (trigger upload). The orphan's 2 samples
    # must not have been re-seeded.
    assert arrs["x"].shape[0] == 5, f"orphan was re-seeded: {arrs['x'].shape[0]}"


def test_startup_recovery_partial_restore_preserves_in_flight(tmp_path) -> None:
    """Review fix: startup recovery of an orphaned in-flight group must not
    delete the token dir when any shard's restore rename to ``_pending``
    fails — that shard only exists inside the token dir. Trigger the failure
    with a same-named non-empty ``_pending`` entry: zarr shards are
    directories, so the rename refuses to overwrite it."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    token = "ab" * 8
    staging = _in_flight_dir(server_root) / token
    staging.mkdir(parents=True)
    name_conflict = f"100_{'a' * 64}_{'1' * 16}{LOCAL_SHARD_SUFFIX}"
    name_ok = f"101_{'b' * 64}_{'2' * 16}{LOCAL_SHARD_SUFFIX}"
    for name, sample_idx in ((name_conflict, 0), (name_ok, 1)):
        save_local_shard_arrays(
            staging / name,
            arrs=samples_to_arrays([_sample(sample_idx)]),
            meta=ShardMeta(username="u", games=1, positions=1, model_sha256="abcd1234"),
        )
    pending_dir = _pending_dir(server_root)
    pending_dir.mkdir(parents=True)
    save_local_shard_arrays(
        pending_dir / name_conflict,
        arrs=samples_to_arrays([_sample(2)]),
        meta=ShardMeta(username="u", games=1, positions=1, model_sha256="abcd1234"),
    )

    # Startup recovery runs inside create_app. No compacted shard matches the
    # token, so it tries to move both staged shards back to _pending; the
    # conflicting one fails.
    _build_app(server_root, upload_compact_shard_size=2000)

    # The shard whose restore failed must survive in the token dir (old code
    # rmtree'd the whole dir here); the other one moved to _pending.
    assert (staging / name_conflict).is_dir(), "un-restored in-flight shard was deleted"
    assert (pending_dir / name_ok).is_dir()
    assert not (staging / name_ok).exists()


def test_failed_compaction_with_failed_restore_preserves_in_flight(tmp_path, monkeypatch) -> None:
    """Audit fix: when the compaction write fails AND a restore rename back to
    _pending also fails, the in-flight dir must be LEFT IN PLACE (startup
    recovery re-seeds it) — deleting it would destroy the samples."""
    import chess_anti_engine.server.app as app_mod

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Threshold 4: the first 2-sample upload stays pending; the second
    # crosses it and triggers the flush inline.
    client = _build_client(server_root, upload_compact_shard_size=4)
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
    assert r.status_code == 200, r.text
    pending_before = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending_before) == 1

    # Force the next flush to hit the worst case: the compaction write blows
    # up AND the restore rename back to _pending fails (e.g. dir vanished).
    def _boom(**_kwargs):
        raise RuntimeError("simulated compaction failure")

    monkeypatch.setattr(app_mod, "_flush_buffered_upload_to_inbox", _boom)
    real_replace = Path.replace

    def _failing_replace(self, target):
        # Only the RESTORE direction (in-flight -> pending) fails; uploads
        # staging into _pending and the pending -> in-flight move both work.
        if "_in_flight" in str(self) and "_pending" in str(target):
            raise OSError("simulated restore failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _failing_replace)
    try:
        # The second upload crosses the per-key sample threshold and
        # triggers the flush inline; the endpoint logs the failure and keeps
        # the accumulator, so the request itself succeeds.
        client.post(
            "/v1/upload_shard",
            auth=("u", "p"),
            files={"file": ("shard2.zarr.tar", _build_zarr_tar(
                tmp_path / "u2",
                samples=[_sample(2), _sample(3), _sample(4)],
                model_sha256="dddd4444",
            ), "application/x-tar")},
            headers=_default_headers(),
        )
    finally:
        monkeypatch.setattr(Path, "replace", real_replace)

    # The staged samples must still exist SOMEWHERE on disk: either restored
    # to _pending or preserved in an in-flight group — never deleted.
    in_flight_root = server_root / "inbox" / "_in_flight"
    surviving = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    if in_flight_root.is_dir():
        surviving += [
            p for d in in_flight_root.iterdir() if d.is_dir()
            for p in d.glob(f"*{LOCAL_SHARD_SUFFIX}")
        ]
    assert surviving, "staged shard was deleted on failed flush + failed restore"


def _quarantine_unloadable_dir(server_root: Path) -> Path:
    return server_root / "quarantine" / "unloadable"


def test_corrupt_pending_shard_is_quarantined_not_retried_forever(tmp_path) -> None:
    """A pending shard whose zarr metadata is truncated can never be loaded, so
    recovery must move it aside. Skipping in place left 7 corrupt shards being
    retried on every startup for up to 5 days (2026-07-24)."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    client = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1", samples=[_sample(i) for i in range(3)], model_sha256="cccc3333",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    pending = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending) == 1
    del client

    # Corrupt it exactly the way the live shards were: truncate the zarr
    # metadata so json parsing fails inside load_shard_arrays.
    shard = pending[0]
    for meta in list(shard.rglob(".zarray")) + list(shard.rglob(".zgroup")):
        meta.write_text("")

    _build_app(server_root, upload_compact_shard_size=2000)  # startup recovery runs

    assert not list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}")), (
        "corrupt shard must not remain in _pending to be retried again"
    )
    quarantined = list(_quarantine_unloadable_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(quarantined) == 1, "corrupt shard must be preserved for post-mortem"
    assert quarantined[0].name == shard.name


def test_quarantine_collision_does_not_lose_a_shard(tmp_path) -> None:
    """Two corrupt shards with the same filename must both survive quarantine."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    pending = _pending_dir(server_root)
    pending.mkdir(parents=True, exist_ok=True)
    qdir = _quarantine_unloadable_dir(server_root)
    qdir.mkdir(parents=True, exist_ok=True)

    name = f"1784488027_{'a' * 64}_deadbeefdeadbeef{LOCAL_SHARD_SUFFIX}"
    (qdir / name).mkdir()  # an earlier quarantine already claimed this name
    corrupt = pending / name
    corrupt.mkdir()
    (corrupt / ".zgroup").write_text("")

    _build_app(server_root, upload_compact_shard_size=2000)

    assert not list(pending.glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(list(qdir.glob(f"*{LOCAL_SHARD_SUFFIX}"))) == 2, "must not overwrite"


def test_transient_oserror_is_skipped_not_quarantined(tmp_path, monkeypatch) -> None:
    """OSError says nothing about the shard's bytes — quarantining on fd
    exhaustion or EIO would move aside a perfectly good shard. Retry instead."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    client = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path / "u1", samples=[_sample(i) for i in range(3)], model_sha256="dddd4444",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert len(list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))) == 1
    del client

    # create_app imports load_shard_arrays from the replay module at call
    # time, so the source module is what must be patched.
    from chess_anti_engine.replay import shard as shard_module

    def _boom(*_a, **_k):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(shard_module, "load_shard_arrays", _boom)
    _build_app(server_root, upload_compact_shard_size=2000)

    assert len(list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))) == 1, (
        "a transient load error must leave the shard in _pending for a later retry"
    )
    assert not _quarantine_unloadable_dir(server_root).exists()


def test_corrupt_pending_shard_under_a_trial_lands_in_that_trials_quarantine(tmp_path) -> None:
    """Guards the trial-key plumbing: a mis-keyed quarantine root would write
    one trial's corrupt shards into another trial's directory."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    tid = "abcde_00000"
    pending = server_root / "trials" / tid / "inbox" / "_pending"
    pending.mkdir(parents=True)
    name = f"1784488027_{'b' * 64}_feedfacefeedface{LOCAL_SHARD_SUFFIX}"
    corrupt = pending / name
    corrupt.mkdir()
    (corrupt / ".zgroup").write_text("")

    _build_app(server_root, upload_compact_shard_size=2000)

    assert not list(pending.glob(f"*{LOCAL_SHARD_SUFFIX}"))
    trial_q = server_root / "trials" / tid / "quarantine" / "unloadable"
    assert [p.name for p in trial_q.glob(f"*{LOCAL_SHARD_SUFFIX}")] == [name]
    assert not (server_root / "quarantine" / "unloadable").exists(), (
        "must not land in the default (no-trial) quarantine root"
    )


def test_quarantined_shard_records_why(tmp_path) -> None:
    """These shards are found days later — the reason must survive log rotation."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    pending = _pending_dir(server_root)
    pending.mkdir(parents=True)
    corrupt = pending / f"1784488027_{'c' * 64}_0badc0de0badc0de{LOCAL_SHARD_SUFFIX}"
    corrupt.mkdir()
    (corrupt / ".zgroup").write_text("")

    _build_app(server_root, upload_compact_shard_size=2000)

    reasons = list(_quarantine_unloadable_dir(server_root).glob("*.reason.txt"))
    assert len(reasons) == 1
    assert reasons[0].read_text().strip(), "reason sidecar must not be empty"


# ---------------------------------------------------------------------------
# #406: `shard_meta_violations` ran on the live upload path only, so recovery
# re-admitted stored metadata without it.
# ---------------------------------------------------------------------------


def _upload_one_pending_shard(server_root: Path, tmp_path: Path) -> Path:
    """Drive a real upload through the real route and return its pending zarr.

    ⚑ The pending shard is BUILT BY PRODUCTION, not hand-written. A test that
    assembled the on-disk state itself would still pass against a build whose
    upload path never produced that shape.
    """
    client = _build_client(server_root, upload_compact_shard_size=2000)
    tar_bytes = _build_zarr_tar(
        tmp_path, samples=[_sample(i) for i in range(2)], model_sha256="dddd4444",
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True, r.json()
    pending = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert len(pending) == 1, pending
    return pending[0]


def _corrupt_counters(zarr_root: Path, **counters: int) -> None:
    """Flip counter digits in a stored shard's `.zattrs`, in place.

    This is the threat model exactly: `.zattrs` is plain uncompressed JSON with
    no checksum, so a bit flip (or an edit) after the upload-time check leaves
    valid JSON carrying wrong numbers. The arrays are Blosc/zstd and would fail
    to decompress loudly; the counter channel has no such protection.
    """
    attrs_path = zarr_root / ".zattrs"
    raw = json.loads(attrs_path.read_text(encoding="utf-8"))
    raw.update(counters)
    attrs_path.write_text(json.dumps(raw, indent=4, sort_keys=True), encoding="utf-8")


def test_recovery_quarantines_a_pending_shard_with_corrupt_counters(tmp_path) -> None:
    """⚑⚑ REGRESSION (#406, P2). `shard_meta_violations` guarded the live
    upload path only. A pending shard sits on disk across a crash and a
    restart, and `_scan_pending_dir` handed its stored metadata straight to
    `acc.add_upload` — so post-validation corruption of the least-protected
    channel was admitted on exactly the path the check exists to close.
    `wins`/`draws`/`losses` become the PID's curriculum winrate.

    Sanction is QUARANTINE, not rejection: the bytes stay on disk under
    `quarantine/unloadable` with the failing predicate in a sidecar, so a false
    positive costs one shard and is recoverable by hand. That is strictly
    weaker than the upload route's terminal `rejected`, which makes the worker
    drop the shard forever.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    pending_shard = _upload_one_pending_shard(server_root, tmp_path / "u1")
    name = pending_shard.name
    # Accepted at upload with games=1; now the stored counters say 5 decisive
    # results out of 1 game.
    _corrupt_counters(pending_shard, games=1, wins=5, draws=0, losses=0)

    # Restart: `create_app` runs `_recover_pending_uploads`.
    _build_app(server_root, upload_compact_shard_size=2000)

    assert not list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}")), (
        "a shard whose counters contradict themselves was re-seeded into the "
        "accumulator by startup recovery"
    )
    qdir = _quarantine_unloadable_dir(server_root)
    assert [p.name for p in qdir.glob(f"*{LOCAL_SHARD_SUFFIX}")] == [name]
    reason = (qdir / f"{name}.reason.txt").read_text(encoding="utf-8")
    assert "wins+draws+losses" in reason, (
        f"sidecar must name the failing predicate, got: {reason!r}"
    )


def test_corrupt_array_data_is_refused_at_upload_not_deferred_to_recovery(
    tmp_path,
) -> None:
    """⚑ CLAIM-PINNING, not a regression test — and it pins a SECURITY claim.

    `_quarantine_unloadable_pending` argues that sink is harder to reach than
    the budgeted ones, and cites this measurement. The vector it has to
    survive: a shard whose `.zattrs` is well-formed (so the metadata invariants
    pass) but whose COMPRESSED ARRAY DATA is corrupt — accepted at upload,
    promoted to `_pending`, failing only at the startup load, which is the one
    thing that writes to `quarantine/unloadable`. An uploader rotating trial
    ids could then seed that sink under arbitrarily many invented trials.

    Measured here: it is refused at upload with a blosc decompression error and
    lands in `quarantine/invalid` (which IS budgeted), because `_finish_upload`
    does a NON-LAZY `load_shard_arrays` BEFORE it promotes. If anyone makes that
    load lazy, this test fails and the docstring's cited measurement is caught
    going stale instead of quietly becoming false.

    ⚑ THIS TEST USED TO CITE TWO GATING CALLS AND THERE IS ONLY ONE (#407
    review). `arrays_to_samples` runs AFTER the promote (`load_shard_arrays`
    :3457/:3463 → promote :3610 → `arrays_to_samples` :3672), so it gates
    nothing: a shard that loads and then fails `arrays_to_samples` 500s the
    upload and is LEFT IN `_pending` under an attacker-chosen trial id. What
    this test actually pins is the first call, which is the one that runs early
    enough to matter.

    ⚑ No input with that shape was found, so this is a narrowed claim rather
    than a new vector, and `_scan_pending_dir` is BOOT-ONLY — a real rate limit
    on the sink that the original argument never mentioned.

    ⚑ This disproves ONE construction. It is not a proof that the sink is
    unreachable, and the docstring does not claim it is.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    zp = tmp_path / "shard.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(4)]),
        meta=ShardMeta(
            username="u", games=1, positions=4, model_sha256="eeee5555", model_step=0,
        ),
    )
    # Metadata untouched; only the compressed chunks of `x` are destroyed.
    chunks = [p for p in (zp / "x").iterdir() if not p.name.startswith(".")]
    assert chunks, "no chunk files to corrupt — zarr layout changed"
    for c in chunks:
        c.write_bytes(b"\x00" * max(16, c.stat().st_size))
    _, buf = pack_shard_for_upload(zp)

    client = _build_client(server_root, upload_compact_shard_size=100_000)
    r = client.post(
        "/v1/trials/attacker_1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", buf.getvalue(), "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("rejected") is True, r.json()

    trial_root = server_root / "trials" / "attacker_1"
    assert not list((trial_root / "inbox" / "_pending").glob(f"*{LOCAL_SHARD_SUFFIX}")), (
        "a shard with unreadable array data was promoted to _pending; it would "
        "reach quarantine/unloadable at the next restart, under a trial id the "
        "uploader chose"
    )
    # It went to the sink that IS budgeted.
    assert list((trial_root / "quarantine" / "invalid").glob("*"))

    # And a restart finds nothing to quarantine.
    _build_app(server_root, upload_compact_shard_size=100_000)
    assert not (trial_root / "quarantine" / "unloadable").exists()


def test_recovery_still_re_seeds_a_consistent_pending_shard(tmp_path) -> None:
    """⚑⚑ FALSE-POSITIVE CONTROL, and the load-bearing half of this pair.

    `shard_meta_violations` is an outage-class function: an earlier revision
    summed children that do not partition their parent (a single adjudicated
    draw increments `selfplay_games`, `selfplay_adjudicated_games` AND
    `selfplay_draw_games`) and would have rejected ordinary shards — ingest to
    zero. Wiring it onto a NEW path is exactly where that class of mistake
    lands, and on the recovery path a false positive is silent data loss rather
    than a loud rejection. So: an untouched shard from the real upload route
    must survive the restart untouched.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    pending_shard = _upload_one_pending_shard(server_root, tmp_path / "u1")

    _build_app(server_root, upload_compact_shard_size=2000)

    surviving = list(_pending_dir(server_root).glob(f"*{LOCAL_SHARD_SUFFIX}"))
    assert [p.name for p in surviving] == [pending_shard.name], (
        "recovery quarantined a shard the upload route itself produced"
    )
    assert not _quarantine_unloadable_dir(server_root).exists(), (
        "a consistent shard reached quarantine/unloadable"
    )
