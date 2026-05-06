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

    def _fake_download_and_verify_shared(url: str, *, out_path: Path, expected_sha256: str, headers: dict) -> None:  # pylint: disable=unused-argument  # mock matches real signature
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

    def _fake_download_and_verify_shared(url: str, *, out_path: Path, expected_sha256: str, headers: dict) -> None:  # pylint: disable=unused-argument
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
