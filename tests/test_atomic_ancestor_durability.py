"""Ancestor barrier topology, not a physical power-loss qualification."""
from __future__ import annotations

import errno
import os
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.utils import atomic


class _Trace:
    """Forward real fsync only inside this test's owned directory tree."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root.resolve()
        self.events: list[tuple[str, Path | None]] = []
        real = os.fsync

        def sync(fd: int) -> None:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            # Resolve directory identities portably, including ancestors outside
            # the fixture. Outside identities are observed, not actually flushed.
            candidates = [self.root, *self.root.parents, *self.root.rglob("*")]
            matched: Path | None = None
            for path in candidates:
                try:
                    current = path.stat()
                except FileNotFoundError:
                    continue
                if (current.st_dev, current.st_ino) == identity:
                    matched = path.resolve()
                    break
            kind = "dir" if stat.S_ISDIR(info.st_mode) else "file"
            self.events.append((kind, matched))
            if matched is not None and matched.is_relative_to(self.root):
                real(fd)

        monkeypatch.setattr(os, "fsync", sync)

    def dirs(self) -> list[Path]:
        return [p for kind, p in self.events if kind == "dir" and p is not None]


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("preexisting", [False, True])
def test_complete_directory_chain_precedes_writer_even_when_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int, preexisting: bool,
) -> None:
    parent = tmp_path.joinpath(*[f"level{i}" for i in range(depth)])
    if preexisting:
        parent.mkdir(parents=True, exist_ok=True)
    dest = parent / "state.json"
    trace = _Trace(tmp_path, monkeypatch)
    before_writer: list[Path] = []

    def write(path: Path) -> None:
        before_writer.extend(trace.dirs())
        path.write_text("complete")

    atomic.atomic_write(dest, write)
    expected = {parent, *parent.parents}
    assert set(before_writer) == expected
    assert len(before_writer) == len(expected)
    assert trace.events[-2][0] == "file"
    assert trace.events[-1] == ("dir", parent)
    assert dest.read_text() == "complete"
    assert list(parent.iterdir()) == [dest]


@pytest.mark.parametrize("kind", ["text", "bytes", "copy2", "worker_config", "lease"])
def test_real_public_helpers_and_persisted_consumers_reach_ancestor_barriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    trace = _Trace(tmp_path, monkeypatch)
    parent = tmp_path / "run" / "state"
    dest = parent / "state.json"
    if kind == "text":
        atomic.atomic_write_text(dest, "value")
    elif kind == "bytes":
        atomic.atomic_write_bytes(dest, b"value")
    elif kind == "copy2":
        src = tmp_path / "source.bin"
        src.write_bytes(b"value")
        atomic.atomic_copy2(src, dest)
    elif kind == "worker_config":
        from chess_anti_engine.worker_config import load_worker_config, save_worker_config
        save_worker_config(dest, {"password": "inert test value"})
        assert load_worker_config(dest) == {"password": "inert test value"}
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    else:
        from chess_anti_engine.server.lease import load_lease, save_lease
        dest = parent / "fixture.json"
        save_lease(leases_root=parent, lease={"lease_id": "fixture"})
        assert load_lease(leases_root=parent, lease_id="fixture") == {"lease_id": "fixture"}
    assert dest.is_file()
    assert {parent, *parent.parents} <= set(trace.dirs())


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("backtrack", [False, True])
def test_symlink_targets_and_lexical_prefixes_are_both_synchronized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool, backtrack: bool,
) -> None:
    physical_parent = tmp_path / "physical"
    target = physical_parent / "target"
    target.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    parent = alias / ".." / "state" if backtrack else alias / "state"
    dest = parent / "data"
    if relative:
        monkeypatch.chdir(tmp_path)
        dest = dest.relative_to(tmp_path)
    trace = _Trace(tmp_path, monkeypatch)
    observed: list[Path] = []

    def write(path: Path) -> None:
        observed.extend(trace.dirs())
        path.write_bytes(b"done")

    atomic.atomic_write(dest, write)
    assert {tmp_path, physical_parent, target} <= set(observed)
    assert len(observed) == len(set(observed)), "resolved aliases must be deduplicated"
    assert dest.read_bytes() == b"done"
    assert alias.is_symlink()


@pytest.mark.parametrize("relative_target", [False, True])
def test_intermediate_symlink_target_namespaces_are_synchronized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_target: bool,
) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    physical = tmp_path / "physical"
    target = physical / "target"
    target.mkdir(parents=True)
    second = bridge / "second"
    second.symlink_to(Path("../physical/target") if relative_target else target,
                      target_is_directory=True)
    first = tmp_path / "first"
    first.symlink_to(bridge / "second", target_is_directory=True)
    trace = _Trace(tmp_path, monkeypatch)
    observed: list[Path] = []

    def write(path: Path) -> None:
        observed.extend(trace.dirs())
        path.write_text("value")

    atomic.atomic_write(first / "state", write)
    assert {tmp_path, bridge, physical, target} <= set(observed)
    assert len(observed) == len(set(observed))
    assert (target / "state").read_text() == "value"
    assert first.is_symlink()
    assert second.is_symlink()


def test_directory_alias_to_own_parent_has_finite_deduplicated_barriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(Path("."), target_is_directory=True)
    trace = _Trace(tmp_path, monkeypatch)
    dest = alias / "alias" / "alias" / "state"
    atomic.atomic_write_text(dest, "value")
    assert trace.dirs().count(tmp_path) == 2, "one preflight plus one commit barrier"
    assert len(trace.dirs()) == len(tmp_path.parents) + 2
    assert (tmp_path / "state").read_text() == "value"


@pytest.mark.parametrize("code", [errno.EIO, errno.ENOSPC, errno.EBADF])
@pytest.mark.parametrize("preexisting_file", [False, True])
def test_ancestor_error_preserves_file_and_retry_repeats_failed_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, preexisting_file: bool,
) -> None:
    dest = tmp_path / "run" / "state" / "data"
    if preexisting_file:
        dest.parent.mkdir(parents=True)
        dest.write_text("old")
    trace = _Trace(tmp_path, monkeypatch)
    real = atomic._fsync_dir
    pending = True
    writer_calls = 0

    def sync(path: Path) -> None:
        nonlocal pending
        if path == tmp_path and pending:
            pending = False
            raise OSError(code, "ancestor barrier failed")
        real(path)

    def write(path: Path) -> None:
        nonlocal writer_calls
        writer_calls += 1
        path.write_text("new")

    monkeypatch.setattr(atomic, "_fsync_dir", sync)
    with pytest.raises(OSError, match="ancestor barrier failed") as caught:
        atomic.atomic_write(dest, write)
    assert caught.value.errno == code
    assert writer_calls == 0
    assert dest.parent.is_dir()
    assert (dest.read_text() if dest.exists() else None) == ("old" if preexisting_file else None)
    trace.events.clear()
    atomic.atomic_write(dest, write)
    assert writer_calls == 1
    assert dest.read_text() == "new"
    assert {dest.parent, *dest.parent.parents} <= set(trace.dirs())
    assert list(dest.parent.iterdir()) == [dest]


@pytest.mark.parametrize("stage", ["mkdir", "resolve", "writer", "file_fsync", "replace"])
def test_precommit_failures_preserve_previous_file_and_remove_only_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    dest = tmp_path / "run" / "state" / "data"
    dest.parent.mkdir(parents=True)
    dest.write_text("old")
    sibling = dest.parent / "unrelated"
    sibling.write_text("keep")
    _Trace(tmp_path, monkeypatch)
    writer_calls = 0
    if stage == "mkdir":
        monkeypatch.setattr(Path, "mkdir", lambda *_a, **_kw: _fail())
    elif stage == "resolve":
        original = Path.resolve

        def resolve(path: Path, **kwargs: Any) -> Path:
            if path == dest.parent:
                _fail()
            return original(path, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve)
    elif stage == "file_fsync":
        monkeypatch.setattr(atomic, "_fsync_file", lambda _p: _fail())
    elif stage == "replace":
        monkeypatch.setattr(os, "replace", lambda *_a: _fail())

    def write(path: Path) -> None:
        nonlocal writer_calls
        writer_calls += 1
        path.write_text("new")
        if stage == "writer":
            _fail()

    with pytest.raises(OSError, match="injected failure"):
        atomic.atomic_write(dest, write)
    assert dest.read_text() == "old"
    assert sibling.read_text() == "keep"
    assert sorted(p.name for p in dest.parent.iterdir()) == ["data", "unrelated"]
    if stage in {"mkdir", "resolve"}:
        assert writer_calls == 0


def _fail() -> None:
    raise OSError(errno.EIO, "injected failure")


def test_postreplace_sync_error_is_visible_commit_uncertainty_not_old_file_promise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "run" / "data"
    dest.parent.mkdir()
    dest.write_text("old")
    trace = _Trace(tmp_path, monkeypatch)
    real = atomic._fsync_dir
    leaf_calls = 0

    def sync(path: Path) -> None:
        nonlocal leaf_calls
        if path == dest.parent:
            leaf_calls += 1
            if leaf_calls == 2:
                _fail()
        real(path)

    monkeypatch.setattr(atomic, "_fsync_dir", sync)
    with pytest.raises(OSError, match="injected failure"):
        atomic.atomic_write_text(dest, "new")
    assert dest.read_text() == "new"
    assert list(dest.parent.iterdir()) == [dest]
    trace.events.clear()
    atomic.atomic_write_text(dest, "retry")
    assert dest.read_text() == "retry"
    assert {dest.parent, *dest.parent.parents} <= set(trace.dirs())


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP, errno.EPERM, errno.EACCES])
def test_existing_unsupported_directory_sync_policy_is_not_broadened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int,
) -> None:
    dest = tmp_path / "state" / "data"
    trace = _Trace(tmp_path, monkeypatch)
    real = os.fsync

    def sync(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode) and info.st_ino == tmp_path.stat().st_ino:
            raise OSError(code, "existing tolerated directory sync error")
        real(fd)

    monkeypatch.setattr(os, "fsync", sync)
    atomic.atomic_write_text(dest, "new")
    assert dest.read_text() == "new"
    assert dest.parent in trace.dirs()


def test_non_durable_does_not_resolve_or_sync_ancestor_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "resolve", lambda *_a, **_kw: pytest.fail("unexpected resolution"))
    monkeypatch.setattr(os, "fsync", lambda _fd: pytest.fail("unexpected fsync"))
    dest = tmp_path / "nested" / "state"
    atomic.atomic_write_text(dest, "telemetry", durable=False)
    assert dest.read_text() == "telemetry"
    assert list(dest.parent.iterdir()) == [dest]


def test_unrelated_concurrent_directory_contents_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _Trace(tmp_path, monkeypatch)
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    (unrelated / "old").write_text("old")
    start = threading.Event()
    done = threading.Event()

    def other_writer() -> None:
        if start.wait(5):
            (unrelated / "new").write_text("concurrent")
            done.set()

    thread = threading.Thread(target=other_writer)
    thread.start()
    real = atomic._fsync_dir

    def sync(path: Path) -> None:
        if path == tmp_path:
            start.set()
            assert done.wait(5)
        real(path)

    monkeypatch.setattr(atomic, "_fsync_dir", sync)
    try:
        atomic.atomic_write_text(tmp_path / "run" / "state", "target")
    finally:
        start.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert (unrelated / "old").read_text() == "old"
    assert (unrelated / "new").read_text() == "concurrent"
    assert tmp_path in trace.dirs()


def test_symlink_cycle_fails_without_unbounded_ancestor_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Trace(tmp_path, monkeypatch)
    loop = tmp_path / "loop"
    loop.symlink_to(loop, target_is_directory=True)
    with pytest.raises(FileExistsError, match="File exists"):
        atomic.atomic_write_text(loop / "state", "never written")
    assert list(tmp_path.iterdir()) == [loop]


def test_concurrent_writers_creating_shared_ancestors_keep_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "run" / "state"
    _Trace(tmp_path, monkeypatch)
    original_mkdir = Path.mkdir
    rendezvous = threading.Barrier(2)
    errors: list[Exception] = []

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == parent and not path.parent.exists():
            rendezvous.wait(timeout=5)
        original_mkdir(path, *args, **kwargs)

    def write(name: str) -> None:
        try:
            atomic.atomic_write_text(parent / name, name)
        except Exception as exc:  # propagate failures out of the worker thread
            errors.append(exc)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    threads = [threading.Thread(target=write, args=(name,)) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not errors
    assert (parent / "first").read_text() == "first"
    assert (parent / "second").read_text() == "second"
    assert sorted(p.name for p in parent.iterdir()) == ["first", "second"]


def test_chain_resolver_rejects_symlink_cycle_before_any_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = tmp_path / "loop"
    loop.symlink_to(loop, target_is_directory=True)
    monkeypatch.setattr(atomic, "_fsync_dir", lambda _p: pytest.fail("unexpected barrier"))
    with pytest.raises(OSError, match="symbolic links") as caught:
        atomic._fsync_directory_chain(loop)
    assert caught.value.errno == errno.ELOOP
