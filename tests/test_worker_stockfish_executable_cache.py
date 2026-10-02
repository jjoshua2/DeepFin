from __future__ import annotations

import hashlib
import logging
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import chess_anti_engine.worker as worker_mod
import chess_anti_engine.worker_assets as worker_assets
from chess_anti_engine.worker import WorkerSession


def _stockfish_session(cache_dir: Path, *, last_sha: str | None) -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.args = SimpleNamespace(
        stockfish_path=None,
        stockfish_from_server=True,
        sf_workers=1,
        sf_nice=0,
    )
    session.cache_dir = cache_dir
    session.last_sf_sha = last_sha
    session.log = logging.getLogger("test.worker_stockfish_executable_cache")
    session.server = "https://worker.invalid"
    session.sf = None
    session.sf_multipv_active = None
    session.sf_hash_mb_active = None
    session.sf_syzygy_path_active = None
    session.sf_path_active = None
    return session


def _manifest(payload: bytes) -> tuple[dict, str, Path]:
    sha = hashlib.sha256(payload).hexdigest()
    return (
        {"stockfish": {"endpoint": "/v1/stockfish", "sha256": sha, "filename": "stockfish"}},
        sha,
        Path(f"stockfish_{sha}_stockfish"),
    )


@pytest.mark.parametrize("last_sha", [None, "matching"])
def test_sync_stockfish_repairs_cached_binary_mode(tmp_path, monkeypatch, last_sha) -> None:
    payload = b"fake stockfish executable"
    manifest, sha, relative_path = _manifest(payload)
    cached = tmp_path / relative_path
    cached.write_bytes(payload)
    cached.chmod(0o600)
    session = _stockfish_session(
        tmp_path,
        last_sha=sha if last_sha == "matching" else None,
    )
    constructed: list[str] = []

    class FakeStockfish:
        def __init__(self, path: str, **_kwargs) -> None:
            mode = stat.S_IMODE(os.stat(path).st_mode)
            assert mode & 0o111 == 0o111, f"cached Stockfish mode is {mode:o}"
            constructed.append(path)

    monkeypatch.setattr(worker_mod, "StockfishUCI", FakeStockfish)
    monkeypatch.setattr(worker_mod, "_worker_headers", dict)
    monkeypatch.setattr(
        worker_mod,
        "_download_and_verify_shared",
        lambda *_args, **_kwargs: pytest.fail("valid cached binary should not redownload"),
    )

    result = WorkerSession._sync_stockfish(session, manifest, 100, 1, 16)

    assert result == str(cached)
    assert constructed == [str(cached)]
    assert stat.S_IMODE(cached.stat().st_mode) & 0o111 == 0o111


def test_concurrent_cache_consumer_never_sees_published_nonexecutable_binary(
    tmp_path, monkeypatch,
) -> None:
    payload = b"fake stockfish executable"
    manifest, sha, relative_path = _manifest(payload)
    cached = tmp_path / relative_path
    published = threading.Event()
    allow_publish_return = threading.Event()
    publisher_errors: list[BaseException] = []
    consumer_errors: list[BaseException] = []
    constructed: list[str] = []
    published_modes: list[int] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size: int):
            del chunk_size
            yield payload

    class FakeStockfish:
        def __init__(self, path: str, **_kwargs) -> None:
            mode = stat.S_IMODE(os.stat(path).st_mode)
            assert mode & 0o111 == 0o111, f"cached Stockfish mode is {mode:o}"
            constructed.append(path)

    real_replace = os.replace

    def pause_after_publish(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        real_replace(src, dst)
        if Path(dst) == cached:
            published_modes.append(stat.S_IMODE(os.stat(dst).st_mode))
            published.set()
            assert allow_publish_return.wait(timeout=5)

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(worker_mod, "_worker_headers", dict)
    monkeypatch.setattr(worker_mod, "StockfishUCI", FakeStockfish)
    monkeypatch.setattr(worker_assets.os, "replace", pause_after_publish)

    publisher = _stockfish_session(tmp_path, last_sha=None)
    consumer = _stockfish_session(tmp_path, last_sha=sha)

    def sync(session: WorkerSession, errors: list[BaseException]) -> None:
        try:
            WorkerSession._sync_stockfish(session, manifest, 100, 1, 16)
        except BaseException as exc:
            errors.append(exc)

    publisher_thread = threading.Thread(target=sync, args=(publisher, publisher_errors))
    publisher_thread.start()
    assert published.wait(timeout=5)
    consumer_thread = threading.Thread(target=sync, args=(consumer, consumer_errors))
    consumer_thread.start()
    try:
        consumer_thread.join(timeout=1)
    finally:
        allow_publish_return.set()
        publisher_thread.join(timeout=5)
        consumer_thread.join(timeout=5)

    assert not publisher_thread.is_alive()
    assert not consumer_thread.is_alive()
    assert not publisher_errors
    assert not consumer_errors
    assert constructed == [str(cached), str(cached)]
    assert len(published_modes) == 1
    assert published_modes[0] & 0o111 == 0o111
    assert stat.S_IMODE(cached.stat().st_mode) & 0o111 == 0o111
