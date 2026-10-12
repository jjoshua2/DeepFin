from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from chess_anti_engine.worker_assets import (
    _cached_sha_asset_needs_refresh,
    _download_opening_book,
    _safe_manifest_filename,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_cached_sha_asset_needs_refresh_when_same_sha_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "book.zip"

    assert _cached_sha_asset_needs_refresh(
        path=missing,
        sha256="abc123",
        last_sha256="abc123",
    )


def test_cached_sha_asset_reuses_existing_file_when_same_sha_repeats(tmp_path: Path) -> None:
    cached = tmp_path / "book.zip"
    cached.write_bytes(b"cached-book")

    assert not _cached_sha_asset_needs_refresh(
        path=cached,
        sha256="abc123",
        last_sha256="abc123",
    )


def test_download_opening_book_redownloads_when_same_sha_uses_new_filename(
    tmp_path: Path, monkeypatch
) -> None:
    book_bytes = b"opening-book"
    sha = _sha256_bytes(book_bytes)
    old_path = tmp_path / f"opening_{sha}_old.zip"
    old_path.write_bytes(book_bytes)

    downloads: list[Path] = []

    def _fake_download_and_verify_shared(url: str, *, out_path: Path, expected_sha256: str, headers: dict) -> None:  # mock matches real signature
        del headers
        assert url == "http://server/v1/opening_book"
        assert expected_sha256 == sha
        downloads.append(out_path)
        out_path.write_bytes(book_bytes)

    monkeypatch.setattr("chess_anti_engine.worker_assets._download_and_verify_shared", _fake_download_and_verify_shared)

    path, returned_sha = _download_opening_book(
        {
            "opening_book": {
                "filename": "new.zip",
                "sha256": sha,
                "endpoint": "/v1/opening_book",
            }
        },
        "opening_book",
        tmp_path,
        cache_prefix="opening",
        default_endpoint="/v1/opening_book",
        server_url_fn=lambda endpoint: f"http://server{endpoint}",
        headers={"Authorization": "Bearer test"},
        log=logging.getLogger("test"),
        last_sha=sha,
    )

    expected_path = tmp_path / f"opening_{sha}_new.zip"
    assert downloads == [expected_path]
    assert path == str(expected_path)
    assert returned_sha == sha
    assert expected_path.read_bytes() == book_bytes


def test_manifest_asset_filename_is_reduced_to_basename() -> None:
    assert _safe_manifest_filename("../../evil.bin", default="stockfish") == "evil.bin"
    assert _safe_manifest_filename("nested/book.bin", default="book") == "book.bin"
    assert _safe_manifest_filename("", default="book") == "book"


def test_download_opening_book_does_not_trust_manifest_path_components(
    tmp_path: Path, monkeypatch
) -> None:
    book_bytes = b"opening-book"
    sha = _sha256_bytes(book_bytes)
    downloads: list[Path] = []

    def _fake_download_and_verify_shared(url: str, *, out_path: Path, expected_sha256: str, headers: dict) -> None:
        del url, expected_sha256, headers
        downloads.append(out_path)
        out_path.write_bytes(book_bytes)

    monkeypatch.setattr("chess_anti_engine.worker_assets._download_and_verify_shared", _fake_download_and_verify_shared)

    path, returned_sha = _download_opening_book(
        {
            "opening_book": {
                "filename": "../../outside.bin",
                "sha256": sha,
                "endpoint": "/v1/opening_book",
            }
        },
        "opening_book",
        tmp_path,
        cache_prefix="opening",
        default_endpoint="/v1/opening_book",
        server_url_fn=lambda endpoint: f"http://server{endpoint}",
        headers={},
        log=logging.getLogger("test"),
        last_sha=None,
    )

    expected_path = tmp_path / f"opening_{sha}_outside.bin"
    assert downloads == [expected_path]
    assert path == str(expected_path)
    assert returned_sha == sha
    assert expected_path.exists()
    assert not (tmp_path.parent / "outside.bin").exists()


def test_concurrent_legacy_downloads_publish_complete_files_atomically(tmp_path, monkeypatch):
    import threading

    import requests

    from chess_anti_engine.worker_assets import _download

    out_path = tmp_path / "opening.bin"
    first_chunks_written = threading.Barrier(2)
    published = threading.Event()
    errors = []

    class Response:
        def __init__(self, is_second):
            self.is_second = is_second

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            del chunk_size
            yield b"pay"
            first_chunks_written.wait(timeout=5)
            if self.is_second:
                assert published.wait(timeout=5)
            yield b"load"

    monkeypatch.setattr(
        requests, "get",
        lambda *_args, **_kwargs: Response(threading.current_thread().name == "second"),
    )

    def download(is_second):
        try:
            _download("https://example.invalid/book", out_path=out_path)
            if not is_second:
                published.set()
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=download, args=(False,), name="first")
    second = threading.Thread(target=download, args=(True,), name="second")
    first.start()
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert out_path.read_bytes() == b"payload"


def test_checksum_mismatch_never_replaces_cached_asset(tmp_path, monkeypatch):
    import hashlib

    import pytest
    import requests

    from chess_anti_engine.worker_assets import _download_and_verify

    out_path = tmp_path / "model.pt"
    out_path.write_bytes(b"previous")
    expected = hashlib.sha256(b"correct").hexdigest()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            del chunk_size
            yield b"wrong"

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _download_and_verify(
            "https://example.invalid/model",
            out_path=out_path,
            expected_sha256=expected,
        )

    assert out_path.read_bytes() == b"previous"
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []
