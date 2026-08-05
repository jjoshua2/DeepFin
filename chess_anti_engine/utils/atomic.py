"""Atomic file-write helpers.

All functions write to a tmp file in the destination directory, then rename
it into place. Parent directories are created if needed. Tmp names include
the pid and a uuid to prevent collisions across concurrent writers, and are
cleaned up on failure.

Two distinct guarantees, which are easy to conflate:

* **Atomicity** — a concurrent reader sees either the old file or the new one,
  never a half-written mixture. Delivered by the rename, always, at no cost.
* **Durability** — after the call returns, the new contents survive an unclean
  shutdown (power loss, hard reset, VM kill). Delivered by ``durable=True``
  (the default), which fsyncs the tmp file before the rename and the containing
  directory after it. Without it, ext4 ``data=ordered`` usually saves the
  contents by a well-known heuristic, but that is a filesystem courtesy and not
  a guarantee — and this project has had unclean reboots.

Durability is not free: measured on this project's ext4-on-WSL2 root, a 2.5 KB
JSON write costs ~0.22 ms plain and ~11 ms durable (median), with a tail past
1.8 s. Pass ``durable=False`` for files that are pure telemetry — content that
a crash may discard because it is re-derived, not recovered from.
"""
from __future__ import annotations

import errno
import os
import shutil
import uuid
from pathlib import Path
from collections.abc import Callable

# Directory fsync is unsupported on some filesystems. These errnos mean exactly
# "this fs does not implement it", which is not a write failure; anything else
# (EIO, ENOSPC, EBADF) is real and must reach the caller rather than leaving a
# write that only LOOKS durable.
_DIR_FSYNC_UNSUPPORTED = frozenset({
    errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.EACCES,
})


def _fsync_file(path: Path) -> None:
    """Flush ``path``'s contents to stable storage."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Flush ``path``'s directory ENTRIES, so the rename itself is durable.

    fsyncing the file only persists its bytes; without this the rename can be
    lost and the destination reverts to the old inode (or to nothing).
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        if exc.errno in _DIR_FSYNC_UNSUPPORTED:
            return
        raise
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _DIR_FSYNC_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def _tmp_path_for(path: Path, *, preserve_suffix: bool = False) -> Path:
    if preserve_suffix:
  # e.g. foo.npz -> foo.tmp.<pid>.<uuid>.npz
        return path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")
  # e.g. foo.json -> foo.json.tmp.<pid>.<uuid>  (does not match "*.json" glob)
    return path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def _cleanup_tmp(tmp: Path, *, sweep: bool) -> None:
    """Remove ``tmp``, and on the failure path anything named ``<tmp>*`` too.

    A writer that appends its own extension (``numpy.savez`` adds ``.npz`` when
    the path does not already end in it) leaves ``<tmp>.npz`` on disk, which the
    old cleanup never touched: it unlinked ``tmp``, a file that in that case was
    never created. Every such file is orphaned forever, and this repo already
    tracks disk-growth leaks. The tmp name carries a uuid, so a prefix sweep of
    the parent directory cannot match anyone else's file.

    ``sweep`` is False once the rename succeeded — the common case — because the
    sweep is a directory scan, and some of these directories (shard inboxes)
    hold thousands of entries. Nothing needs cleaning there anyway: the rename
    consumed ``tmp``.
    """
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass  # best-effort temp cleanup; os.replace succeeded or original error surfaced
    if not sweep:
        return
    try:
        siblings = list(tmp.parent.glob(tmp.name + "*"))
    except OSError:
        return
    for extra in siblings:
        try:
            if extra.is_dir():
                shutil.rmtree(extra, ignore_errors=True)
            else:
                extra.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write(
    path: Path,
    writer: Callable[[Path], object],
    *,
    preserve_suffix: bool = False,
    durable: bool = True,
    mode: int | None = None,
) -> None:
    """Invoke ``writer(tmp)`` then atomically rename tmp to ``path``.

    Use this when the content source requires a path (``torch.save``,
    ``save_npz``, ``shutil.copy2``). For in-memory bytes or text, prefer
    :func:`atomic_write_bytes` or :func:`atomic_write_text`.

    By default the tmp name is ``<name>.tmp.<pid>.<uuid>`` so globs like
    ``*.json`` / ``*.npz`` in the same directory do not match it. Pass
    ``preserve_suffix=True`` only when the writer dispatches on the file
    extension — e.g. ``numpy.savez`` auto-appends ``.npz`` when the path
    does not already end in it. (``torch.save`` does NOT dispatch on
    extension, so it does not need this flag.)

    ``durable=True`` (default) fsyncs the tmp file before the rename and the
    containing directory after it, so the result survives an unclean shutdown.
    Pass ``durable=False`` only for content a crash may simply discard — see
    the module docstring for the measured cost.

    ``mode`` is the permission the file is CREATED with: the tmp is opened
    ``O_CREAT | O_EXCL`` at ``mode`` *before* the writer runs, so the secret is
    never on disk at the umask default for any window at all. chmod-ing the tmp
    after filling it -- the first version of this parameter -- still left the
    content readable for the whole write-and-fsync, which for a users.json big
    enough to fsync is not a theoretical window. Use ``mode`` for anything an
    unprivileged local user must not read or, worse, WRITE: a world-writable
    ``users.json`` is an auth bypass by hash replacement, not a disclosure.

    A writer that REPLACES the tmp rather than filling it (``shutil.copy2``
    copies the source's bits, ``numpy.savez`` creates its own file) brings its
    own permissions, so the mode is re-applied after the writer returns as well.
    That second application is a backstop for those writers, not the mechanism.

    Raises ``FileNotFoundError`` with an actionable message when the writer did
    not produce ``tmp``, which in practice means it appended an extension and
    ``preserve_suffix=True`` was missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path, preserve_suffix=preserve_suffix)
    replaced = False
    try:
        if mode is not None:
  # ⚑ CREATED at `mode`, not chmod'ed afterwards. The tmp holds the same
  # bytes the destination will, so a tmp at the umask default is the same
  # exposure as a destination at the umask default -- just briefer. O_EXCL
  # so a name collision is an error rather than a silent takeover of a file
  # this call did not create.
            os.close(os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode))
        writer(tmp)
        if not tmp.exists():
            # os.replace would raise FileNotFoundError naming a path the caller
            # never chose, which reads as a disappearing-file mystery rather
            # than as the flag error it is.
            raise FileNotFoundError(
                f"atomic_write: writer did not create {tmp.name!r} while writing "
                f"{path.name!r}. A writer that appends its own extension needs "
                f"preserve_suffix=True.",
            )
        if mode is not None:
  # Backstop for writers that replace the tmp instead of filling it (see the
  # docstring). For the fill-in-place writers this is a no-op, and the file
  # was already correct for the whole of the write.
            os.chmod(str(tmp), mode)
        if durable:
            _fsync_file(tmp)
        os.replace(str(tmp), str(path))
        replaced = True
        if durable:
            _fsync_dir(path.parent)
    finally:
        _cleanup_tmp(tmp, sweep=not replaced)


def atomic_write_bytes(path: Path, data: bytes, *, durable: bool = True) -> None:
    atomic_write(path, lambda p: p.write_bytes(data), durable=durable)


def atomic_write_text(
    path: Path, text: str, *, encoding: str = "utf-8", durable: bool = True,
    mode: int | None = None,
) -> None:
    atomic_write(
        path, lambda p: p.write_text(text, encoding=encoding),
        durable=durable, mode=mode,
    )


def atomic_copy2(src: Path, dst: Path, *, durable: bool = True) -> None:
    """``shutil.copy2`` into place atomically (preserves file metadata)."""
    atomic_write(dst, lambda tmp: shutil.copy2(src, tmp), durable=durable)
