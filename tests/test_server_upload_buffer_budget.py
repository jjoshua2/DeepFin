"""Opt-in server compactor pressure must preserve retry and raw-source durability."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.server import app as app_module
from tests.test_server_upload_durability import (
    _build_zarr_tar, _default_headers, _sample, _seed_user,
)


def _app(root: Path, *, groups: int = 2):
    return app_module.create_app(
        server_root=root, upload_compact_shard_size=2000,
        upload_compact_buffer_bytes=1024 * 1024,
        upload_compact_buffer_uploads=groups,
        max_upload_uncompressed_bytes=1024 * 1024,
    )


def _post(client: TestClient, payload: bytes):
    return client.post(
        "/v1/upload_shard", auth=("u", "p"),
        files={"file": ("shard.zarr.tar", payload, "application/x-tar")},
        headers=_default_headers(),
    )


def _payload(tmp_path: Path, index: int) -> bytes:
    return _build_zarr_tar(
        tmp_path / f"source-{index}", samples=[_sample(index)], model_sha256=f"model-{index}",
    )


def _committed_rows(root: Path) -> int:
    return sum(
        int(load_shard_arrays(path)[0]["x"].shape[0])
        for path in (root / "inbox" / "_compacted").glob("*.zarr")
    )


def test_group_pressure_flushes_early_and_restart_dedupes(tmp_path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _seed_user(root)
    payloads = [_payload(tmp_path, i) for i in range(5)]
    client = TestClient(_app(root))
    for payload in payloads:
        response = _post(client, payload)
        assert response.status_code == 200, response.text
        assert response.json()["stored"] is True
        assert len(list((root / "inbox" / "_pending").glob("*.zarr"))) <= 2
    assert _committed_rows(root) == 3
    # Construct without graceful shutdown to exercise real pending recovery.
    restarted = TestClient(_app(root))
    for payload in payloads:
        response = _post(restarted, payload)
        assert response.status_code == 200, response.text
        assert response.json()["stored"] is False
    assert _committed_rows(root) == 3
    assert len(list((root / "inbox" / "_pending").glob("*.zarr"))) == 2


def test_flush_failure_preserves_accepted_bytes_and_retry(tmp_path, monkeypatch) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _seed_user(root)
    client = TestClient(_app(root, groups=1))
    first, second = _payload(tmp_path, 1), _payload(tmp_path, 2)
    assert _post(client, first).json()["stored"] is True
    pending = next((root / "inbox" / "_pending").glob("*.zarr"))
    original = {p.relative_to(pending): p.read_bytes() for p in pending.rglob("*") if p.is_file()}
    writer = app_module._flush_buffered_upload_to_inbox

    def fail(**_kwargs):
        raise OSError("fixture disk unavailable")

    monkeypatch.setattr(app_module, "_flush_buffered_upload_to_inbox", fail)
    response = _post(client, second)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert {p.relative_to(pending): p.read_bytes() for p in pending.rglob("*") if p.is_file()} == original
    assert len(list((root / "inbox" / "_pending").glob("*.zarr"))) == 1
    monkeypatch.setattr(app_module, "_flush_buffered_upload_to_inbox", writer)
    assert _post(client, second).json()["stored"] is True
    assert _committed_rows(root) == 1
    assert _post(client, second).json()["stored"] is False


@pytest.mark.parametrize("budget", [-1, 1])
def test_budget_must_cover_current_per_upload_limit(tmp_path, budget) -> None:
    with pytest.raises(ValueError, match="compaction buffer"):
        app_module.create_app(server_root=tmp_path, upload_compact_buffer_bytes=budget)


def test_deferred_recovery_resumes_after_writer_recovers(tmp_path, monkeypatch) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _seed_user(root)
    payloads = [_payload(tmp_path, i) for i in range(3)]
    legacy = TestClient(app_module.create_app(server_root=root))
    for payload in payloads:
        assert _post(legacy, payload).json()["stored"] is True
    writer = app_module._flush_buffered_upload_to_inbox

    def fail(**_kwargs):
        raise OSError("fixture writer unavailable during startup")

    monkeypatch.setattr(app_module, "_flush_buffered_upload_to_inbox", fail)
    recovering = TestClient(_app(root, groups=1))
    assert len(list((root / "inbox" / "_pending").glob("*.zarr"))) == 3
    monkeypatch.setattr(app_module, "_flush_buffered_upload_to_inbox", writer)
    # Flush/recovery runs before manifest lookup; no published model is needed.
    recovering.get("/v1/manifest", headers=_default_headers())
    assert _committed_rows(root) == 2
    assert len(list((root / "inbox" / "_pending").glob("*.zarr"))) == 1
    for payload in payloads:
        assert _post(recovering, payload).json()["stored"] is False


def test_live_recovery_waits_for_upload_promotion_and_admission(tmp_path, monkeypatch) -> None:
    """A manifest rescan cannot own a promoted but unacknowledged request.

    Observe the actual upload lock, without replacing its exclusion, so the
    assertion waits for the competing acquisition rather than for a guessed
    scheduler delay. Both requests still use the real HTTP/Zarr path.
    """
    import inspect
    import threading
    from concurrent.futures import ThreadPoolExecutor

    root = tmp_path / "server"
    root.mkdir()
    _seed_user(root)
    payloads = [_payload(tmp_path, i) for i in range(4)]
    legacy = TestClient(app_module.create_app(server_root=root))
    for payload in payloads[:3]:
        assert _post(legacy, payload).json()["stored"] is True
    writer = app_module._flush_buffered_upload_to_inbox

    def fail(**_kwargs):
        raise OSError("fixture writer unavailable during startup")

    monkeypatch.setattr(app_module, "_flush_buffered_upload_to_inbox", fail)
    app = _app(root, groups=1)
    monkeypatch.setattr(app_module, "_flush_buffered_upload_to_inbox", writer)
    client = TestClient(app)
    pending = root / "inbox" / "_pending"
    assert len(list(pending.glob("*.zarr"))) == 3

    endpoint = next(getattr(route, "endpoint") for route in app.routes
                    if getattr(route, "path", None) == "/v1/manifest")
    serve = inspect.getclosurevars(endpoint).nonlocals["_serve_manifest"]
    impl = inspect.getclosurevars(serve).nonlocals["_get_manifest_impl"]
    flush = inspect.getclosurevars(impl).nonlocals["_flush_ready_upload_accumulators"]
    lock_cell = dict(zip(flush.__code__.co_freevars, flush.__closure__, strict=True))["upload_lock"]
    real_lock = lock_cell.cell_contents
    promoted = threading.Event()
    competing_acquire = threading.Event()
    release = threading.Event()

    class ObservedLock:
        def __enter__(self):
            if promoted.is_set() and not release.is_set():
                competing_acquire.set()
            real_lock.acquire()
            return self

        def __exit__(self, *_exc):
            real_lock.release()

    lock_cell.cell_contents = ObservedLock()
    replace = Path.replace

    def pause_after_promotion(path, target):
        result = replace(path, target)
        if Path(target).parent == pending and not promoted.is_set():
            promoted.set()
            if not release.wait(30):
                raise TimeoutError("fixture promotion barrier was not released")
        return result

    monkeypatch.setattr(Path, "replace", pause_after_promotion)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            upload = executor.submit(_post, client, payloads[3])
            try:
                assert promoted.wait(30), "upload never reached its staging barrier"
                assert real_lock.locked(), "promotion occurred outside the admission/recovery lock"
                manifest = executor.submit(client.get, "/v1/manifest", headers=_default_headers())
                assert competing_acquire.wait(30), "manifest never attempted the actual upload lock"
                assert not manifest.done(), "manifest ran past the held upload/admission lock"
            finally:
                release.set()
            response = upload.result(timeout=30)
            assert response.status_code == 200, response.text
            assert response.json()["stored"] is True
            assert manifest.result(timeout=30).status_code == 404
    finally:
        release.set()
        lock_cell.cell_contents = real_lock

    assert _committed_rows(root) == 3
    pending_paths = list(pending.glob("*.zarr"))
    assert len(pending_paths) == 1
    assert int(load_shard_arrays(pending_paths[0])[0]["x"].shape[0]) == 1
    policy_ids = [
        int(index)
        for path in [*(root / "inbox" / "_compacted").glob("*.zarr"), *pending_paths]
        for index in load_shard_arrays(path)[0]["policy_target"].argmax(axis=1)
    ]
    assert sorted(policy_ids) == [0, 1, 2, 3]
    # The new upload and all recovered sources retain their dedupe witnesses.
    for payload in payloads:
        response = _post(client, payload)
        assert response.status_code == 200, response.text
        assert response.json()["stored"] is False
