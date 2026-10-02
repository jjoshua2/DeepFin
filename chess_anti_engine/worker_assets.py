from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from chess_anti_engine.utils import sha256_file as _sha256_file
import contextlib


def _safe_manifest_filename(value: object, *, default: str) -> str:
    name = Path(str(value or default)).name
    return name or str(default)


def _ensure_executable(path: Path) -> None:
    """Best-effort chmod +x for POSIX systems."""
    try:
        if os.name != "nt":
            st = os.stat(path)
            os.chmod(path, st.st_mode | 0o111)
    except OSError:
        pass  # stat/chmod refused by filesystem — downstream will fail loud if exec matters


def _download(
    url: str,
    *,
    out_path: Path,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    expected_sha256: str | None = None,
) -> None:
    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise RuntimeError("worker requires requests; install with pip install -e '.[worker]' ") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            fd = -1
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if expected_sha256:
            got = _sha256_file(tmp)
            if got != str(expected_sha256):
                raise RuntimeError(
                    f"sha256 mismatch for {out_path.name}: got={got} "
                    f"expected={expected_sha256}",
                )
        os.replace(tmp, out_path)
    except Exception:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        tmp.unlink(missing_ok=True)
        raise


def _download_and_verify(
    url: str,
    *,
    out_path: Path,
    expected_sha256: str | None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> None:
    """Download to a private temp file, verify it, then atomically publish.

    If verification fails, retry once. Invalid bytes are never visible at
    out_path and concurrent downloads cannot share a temporary file.
    """
    exp = str(expected_sha256 or "")

    def _once() -> None:
        _download(
            url, out_path=out_path, timeout=timeout, headers=headers,
            expected_sha256=exp or None,
        )

    try:
        _once()
    except Exception:
        _once()


def _download_and_verify_shared(
    url: str,
    *,
    out_path: Path,
    expected_sha256: str | None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    lock_timeout_s: float = 600.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if not expected_sha256 or _sha256_file(out_path) == str(expected_sha256):
            return
        out_path.unlink(missing_ok=True)

    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    deadline = time.time() + float(lock_timeout_s)
    have_lock = False

    while not have_lock:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            have_lock = True
        except FileExistsError:
            if out_path.exists():
                if not expected_sha256 or _sha256_file(out_path) == str(expected_sha256):
                    return
                out_path.unlink(missing_ok=True)
            if time.time() >= deadline:
                with contextlib.suppress(Exception):
                    lock_path.unlink(missing_ok=True)
            time.sleep(0.25)

    try:
        if out_path.exists():
            if not expected_sha256 or _sha256_file(out_path) == str(expected_sha256):
                return
            out_path.unlink(missing_ok=True)
        _download_and_verify(
            url,
            out_path=out_path,
            expected_sha256=expected_sha256,
            timeout=timeout,
            headers=headers,
        )
    finally:
        with contextlib.suppress(Exception):
            lock_path.unlink(missing_ok=True)


def _prune_cached_models(*, cache_dir: Path, keep_shas: set[str]) -> None:
    """Delete cached model checkpoints not in keep_shas.

    Files:
    - model_<sha>.pt (downloaded from /v1/model)
    - best_<sha>.pt (downloaded from /v1/best_model)

    This keeps worker disk usage bounded as best/latest advance over time.
    """
    keep = {str(s) for s in keep_shas if str(s)}

    for p in cache_dir.glob("model_*.pt"):
        name = p.name
        if not name.startswith("model_") or not name.endswith(".pt"):
            continue
        sha = name[len("model_") : -len(".pt")]
        if sha and sha not in keep:
            p.unlink(missing_ok=True)

    for p in cache_dir.glob("best_*.pt"):
        name = p.name
        if not name.startswith("best_") or not name.endswith(".pt"):
            continue
        sha = name[len("best_") : -len(".pt")]
        if sha and sha not in keep:
            p.unlink(missing_ok=True)


def _cached_sha_asset_needs_refresh(*, path: Path, sha256: str, last_sha256: str | None = None) -> bool:
    """Return True when a cached SHA-addressed asset must be refreshed.

    Repeated SHAs can reuse an already-validated file, but only if the expected
    cache path still exists. This covers local cache eviction and manifest
    filename changes without re-hashing large assets on every poll.
    """
    if str(sha256) == str(last_sha256 or ""):
        return not path.exists()
    if not path.exists():
        return True
    return _sha256_file(path) != str(sha256)


def _download_opening_book(
    manifest: dict,
    key: str,
    cache_dir: Path,
    *,
    cache_prefix: str,
    default_endpoint: str,
    server_url_fn,
    headers: dict,
    log,
    last_sha: str | None = None,
) -> tuple[str | None, str | None]:
    """Download an opening book asset from the manifest.

    Returns (local_path, manifest_sha).  Skips I/O when *last_sha* matches
    the manifest SHA (the file was already verified on a prior iteration).
    """
    if key not in manifest:
        return None, None
    ob = manifest.get(key) or {}
    filename = _safe_manifest_filename(ob.get("filename"), default=key)
    sha = str(ob.get("sha256") or "")
    endpoint = str(ob.get("endpoint") or default_endpoint)

    if sha:
        ob_path = cache_dir / f"{cache_prefix}_{sha}_{filename}"
        if _cached_sha_asset_needs_refresh(path=ob_path, sha256=sha, last_sha256=last_sha):
            log.info("downloading %s sha=%s filename=%s", key, sha, filename)
            _download_and_verify_shared(
                server_url_fn(endpoint),
                out_path=ob_path,
                expected_sha256=sha,
                headers=headers,
            )
    else:
        ob_path = cache_dir / f"{cache_prefix}_{filename}"
        if not ob_path.exists():
            _download(
                server_url_fn(endpoint),
                out_path=ob_path,
                headers=headers,
            )
    return str(ob_path), sha or None
