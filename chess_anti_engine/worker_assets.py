from __future__ import annotations

import os
import time
from pathlib import Path

from chess_anti_engine.utils import sha256_file as _sha256_file


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
) -> None:
    try:
        import requests  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("worker requires requests; install with pip install -e '.[worker]' ") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    tmp.replace(out_path)


def _download_and_verify(
    url: str,
    *,
    out_path: Path,
    expected_sha256: str | None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> None:
    """Download a file and verify its sha256 if provided.

    If verification fails, we delete and retry once.
    """
    exp = str(expected_sha256 or "")

    def _once() -> None:
        _download(url, out_path=out_path, timeout=timeout, headers=headers)
        if exp:
            got = _sha256_file(out_path)
            if got != exp:
                raise RuntimeError(f"sha256 mismatch for {out_path.name}: got={got} expected={exp}")

    try:
        _once()
    except Exception:
        out_path.unlink(missing_ok=True)
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
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
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
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


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
    filename = str(ob.get("filename") or key)
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
