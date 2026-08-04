"""A13: `atomic_write` is atomic; these pin that it is also DURABLE.

Atomicity (a reader sees old or new, never a mixture) was already delivered by
the rename. Durability (the new contents survive an unclean shutdown) was not:
there was no `fsync` anywhere, so the guarantee rested on ext4's `data=ordered`
rename heuristic — a filesystem courtesy, not a contract, and this project has
had unclean reboots.

The fsync assertions below observe the REAL `os.fsync` calls and resolve each
file descriptor back to an inode, rather than patching the module's own
helpers, which would only prove the helpers call themselves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.utils.atomic import (
    atomic_copy2,
    atomic_write,
    atomic_write_bytes,
    atomic_write_text,
)


class _FsyncSpy:
    """Record the inode of every fd passed to `os.fsync`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.inodes: list[tuple[int, int]] = []
        real = os.fsync

        def _spy(fd: int) -> None:
            st = os.fstat(fd)
            self.inodes.append((st.st_dev, st.st_ino))
            real(fd)

        monkeypatch.setattr(os, "fsync", _spy)

    def saw(self, path: Path) -> bool:
        st = path.stat()
        return (st.st_dev, st.st_ino) in self.inodes

    def __len__(self) -> int:
        return len(self.inodes)


# ── durability ───────────────────────────────────────────────────────────────


def test_durable_write_fsyncs_the_file_and_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both syncs are needed and neither substitutes for the other: syncing the
    file persists its bytes, syncing the directory persists the RENAME. Without
    the second, a crash can leave the destination pointing at the old inode
    even though the new bytes are safely on disk."""
    spy = _FsyncSpy(monkeypatch)
    dest = tmp_path / "state.json"
    atomic_write_text(dest, '{"iteration": 7}')

    assert dest.read_text(encoding="utf-8") == '{"iteration": 7}'
    # The tmp file is gone by now, so its inode is matched via the destination:
    # `os.replace` moves the directory entry, it does not copy the inode.
    assert spy.saw(dest), "tmp file was never fsynced before the rename"
    assert spy.saw(tmp_path), "containing directory was never fsynced after the rename"
    assert len(spy) == 2, spy.inodes


def test_durable_is_the_default_for_every_public_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A helper that quietly skipped the sync would be exactly the "value
    accepted and then silently ignored" shape this audit hunts."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    cases = [
        ("bytes", lambda p: atomic_write_bytes(p, b"x")),
        ("text", lambda p: atomic_write_text(p, "x")),
        ("copy2", lambda p: atomic_copy2(src, p)),
        ("write", lambda p: atomic_write(p, lambda t: t.write_bytes(b"x"))),
    ]
    for label, fn in cases:
        spy = _FsyncSpy(monkeypatch)
        dest = tmp_path / f"out_{label}"
        fn(dest)
        assert spy.saw(dest), f"{label}: file not fsynced"
        assert spy.saw(tmp_path), f"{label}: directory not fsynced"


def test_durable_false_skips_both_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the three tests above. If it failed, they would be
    passing on syncs someone else issued rather than on the ones under test --
    and the per-upload telemetry exemption in `server/app.py` would be paying
    the ~11 ms it was written to avoid."""
    spy = _FsyncSpy(monkeypatch)
    dest = tmp_path / "telemetry.json"
    atomic_write_text(dest, '{"games_per_s": 1.0}', durable=False)

    assert dest.read_text(encoding="utf-8") == '{"games_per_s": 1.0}'
    assert len(spy) == 0, spy.inodes


def test_durable_false_is_still_atomic(tmp_path: Path) -> None:
    """Dropping durability must not drop the guarantee callers actually rely
    on: no reader ever observes a partial file, and no tmp survives."""
    dest = tmp_path / "t.json"
    dest.write_text("old", encoding="utf-8")
    atomic_write_text(dest, "new" * 10_000, durable=False)
    assert dest.read_text(encoding="utf-8") == "new" * 10_000
    assert [p.name for p in tmp_path.iterdir()] == ["t.json"]


# ── the orphaned-tmp disk leak ───────────────────────────────────────────────


def _savez_writer(tmp: Path) -> None:
    """`numpy.savez` appends `.npz` when the path does not end in it -- so it
    creates `<tmp>.npz` and NOT `<tmp>`. This is the exact writer the module
    docstring warns about."""
    np.savez(str(tmp), a=np.arange(4))


def test_missing_preserve_suffix_leaves_no_orphan(tmp_path: Path) -> None:
    """The old `finally` unlinked `tmp` -- a file this writer never created --
    while `<tmp>.npz`, the file it DID create, was orphaned forever. The repo
    already tracks disk-growth leaks; this was one of the sources."""
    dest = tmp_path / "shard.npz"
    with pytest.raises(FileNotFoundError) as excinfo:
        atomic_write(dest, _savez_writer)  # preserve_suffix omitted -- the bug

    # The error now names the flag instead of reading as a vanishing file.
    assert "preserve_suffix" in str(excinfo.value), str(excinfo.value)
    # Nothing left behind: not the tmp, not the extension-mangled sibling.
    assert list(tmp_path.iterdir()) == [], sorted(p.name for p in tmp_path.iterdir())


def test_preserve_suffix_true_is_the_working_path(tmp_path: Path) -> None:
    """Positive control for the test above: the same writer succeeds with the
    flag, so the failure there is attributable to the flag and not to numpy."""
    dest = tmp_path / "shard.npz"
    atomic_write(dest, _savez_writer, preserve_suffix=True)
    with np.load(dest) as z:
        assert list(z["a"]) == [0, 1, 2, 3]
    assert [p.name for p in tmp_path.iterdir()] == ["shard.npz"]


def test_writer_exception_leaves_no_orphan(tmp_path: Path) -> None:
    """A writer that creates its file and THEN fails must not leak either."""
    def _boom(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        (tmp.parent / (tmp.name + ".sidecar")).write_bytes(b"junk")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_write(tmp_path / "dest.bin", _boom)
    assert list(tmp_path.iterdir()) == [], sorted(p.name for p in tmp_path.iterdir())


def test_cleanup_sweep_does_not_touch_other_files(tmp_path: Path) -> None:
    """The sweep matches on the tmp name, which carries a uuid -- it must never
    reach a neighbouring file, least of all the destination itself."""
    keep = tmp_path / "important.json"
    keep.write_text("keep me", encoding="utf-8")
    (tmp_path / "shard.npz").write_text("previous", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        atomic_write(tmp_path / "shard.npz", _savez_writer)

    assert keep.read_text(encoding="utf-8") == "keep me"
    # A failed write leaves the PREVIOUS destination intact -- that is the
    # point of writing to a tmp first.
    assert (tmp_path / "shard.npz").read_text(encoding="utf-8") == "previous"


# ── directory fsync must not swallow a real error ────────────────────────────


def test_directory_fsync_reraises_real_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_fsync_dir` tolerates "this filesystem does not implement it", which is
    not a write failure. EIO is a write failure, and swallowing it would hand
    back a write that only LOOKS durable."""
    real = os.fsync
    dest = tmp_path / "x.json"

    def _eio(fd: int) -> None:
        st = os.fstat(fd)
        if st.st_ino == tmp_path.stat().st_ino:
            raise OSError(5, "Input/output error")  # EIO
        real(fd)

    monkeypatch.setattr(os, "fsync", _eio)
    with pytest.raises(OSError, match="Input/output error"):
        atomic_write_text(dest, "x")


def test_directory_fsync_tolerates_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """...but an EINVAL from a filesystem without directory fsync must not
    fail the write, or this change would break those filesystems outright."""
    import errno

    real = os.fsync
    dest = tmp_path / "x.json"

    def _einval(fd: int) -> None:
        st = os.fstat(fd)
        if st.st_ino == tmp_path.stat().st_ino:
            raise OSError(errno.EINVAL, "Invalid argument")
        real(fd)

    monkeypatch.setattr(os, "fsync", _einval)
    atomic_write_text(dest, "x")
    assert dest.read_text(encoding="utf-8") == "x"


# ── the production exemptions are the ones the PR claims ─────────────────────


def test_only_the_per_upload_telemetry_writers_are_exempted() -> None:
    """`durable=False` is a deliberate, documented exemption for the three
    per-upload counter writers. If a fourth appears without a decision, this
    fails and forces the decision to be made rather than inherited.

    Kept as a source scan on purpose: the alternative is to trust a comment.
    """
    import ast

    import chess_anti_engine.utils.atomic as atomic_mod

    root = Path(atomic_mod.__file__).resolve().parents[1]
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "atomic.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Parsed, not grepped: comments and docstrings ABOUT the exemption
            # are not exemptions, and a line-based scan counts them.
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "durable" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value is False:
                    key = str(path.relative_to(root))
                    found[key] = found.get(key, 0) + 1
    # Exactly the three per-upload counter writers: `_record_gpu_throughput`,
    # `_record_trial_throughput`, and the `record_upload`/`save_users` pair in
    # `_upload_shard_impl`. Counted by file rather than by line so the test does
    # not go red on unrelated edits above them.
    assert found == {"server/app.py": 3}, found


def test_credential_writes_stay_durable_while_upload_counters_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`users.json` carries BOTH credentials and per-upload counters, and only
    one of them is disposable.

    This exemption was found by counting real fsyncs on an upload request, not
    by reading the call graph -- `save_users` runs on every accepted shard via
    `record_upload`, which is not obvious from either end. Losing a counter
    increment to a crash costs nothing; losing a just-created password locks a
    worker out of the fleet.
    """
    from chess_anti_engine.server.auth import (
        UserRecord,
        ensure_user,
        hash_password,
        save_users,
    )

    users_path = tmp_path / "users.json"
    salt, hsh, iters = hash_password("p")
    rec = UserRecord(username="u", salt_b64=salt, hash_b64=hsh, iterations=iters)

    spy = _FsyncSpy(monkeypatch)
    save_users(users_path, {"u": rec}, durable=False)
    assert len(spy) == 0, spy.inodes

    spy = _FsyncSpy(monkeypatch)
    save_users(users_path, {"u": rec})  # default
    assert spy.saw(users_path), "default save_users must be durable"

    # The admin path must not silently inherit the exemption.
    spy = _FsyncSpy(monkeypatch)
    ensure_user(users_path, username="v", password="pw")
    assert spy.saw(users_path), "ensure_user must be durable"


def test_atomic_module_docstring_states_the_durability_contract() -> None:
    """The finding was partly that "atomic" invites a reader to infer
    durability. A docstring that says only "atomic" is what caused it."""
    import chess_anti_engine.utils.atomic as atomic_mod

    doc = (atomic_mod.__doc__ or "").lower()
    assert "durab" in doc, "module docstring never mentions durability"
    assert "fsync" in doc
    assert (atomic_mod.atomic_write.__doc__ or "").count("durable") >= 1


def test_public_helpers_accept_durable_kwarg() -> None:
    """Signature guard: `atomic_write_text(..., durable=False)` is what the two
    exempted call sites pass. A helper that dropped the kwarg would raise, but
    one that ACCEPTED and ignored it would not -- that case is covered by
    `test_durable_false_skips_both_syncs`."""
    import inspect

    for fn in (atomic_write, atomic_write_bytes, atomic_write_text, atomic_copy2):
        params = inspect.signature(fn).parameters
        assert "durable" in params, fn.__name__
        default: Any = params["durable"].default
        assert default is True, (fn.__name__, default)
