"""Writer-lock syscalls are spies: no live locks, buffer, arrays, or threads."""
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.replay import disk_buffer as disk


@pytest.fixture
def lock_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    owner = SimpleNamespace(_shard_dir=Path("synthetic-replay"), _writer_lock_fd=None)
    state: dict[str, Any] = {"named_mode": stat.S_IFREG, "fd_mode": stat.S_IFREG,
                             "named_ino": 7, "fd_ino": 7, "events": []}
    lock = owner._shard_dir / ".writer.lock"
    original_lstat = Path.lstat

    def lstat(path: Path) -> Any:
        if path != lock:
            return original_lstat(path)
        if state.get("missing"):
            state["missing"] = False
            raise FileNotFoundError(path)
        return SimpleNamespace(st_mode=state["named_mode"], st_dev=1, st_ino=state["named_ino"])

    def open_spy(path: str, flags: int, mode: int) -> int:
        state["events"].append(("open", path, flags, mode))
        if state.get("open_error"):
            raise OSError("open failed")
        if "after_open_mode" in state:
            state["named_mode"] = state["after_open_mode"]
        return 77

    def fstat(_fd: int) -> Any:
        if state.get("fstat_error"):
            raise OSError("fstat failed")
        return SimpleNamespace(st_mode=state["fd_mode"], st_dev=1, st_ino=state["fd_ino"])

    def flock(fd: int, operation: int) -> None:
        state["events"].append(("flock", fd, operation))
        if state.get("contended") and operation != disk.fcntl.LOCK_UN:
            raise BlockingIOError("held")

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(disk.os, "open", open_spy)
    monkeypatch.setattr(disk.os, "fstat", fstat)
    monkeypatch.setattr(disk.fcntl, "flock", flock)
    monkeypatch.setattr(disk.os, "close", lambda fd: state["events"].append(("close", fd)))
    monkeypatch.setattr(disk.os, "pread", lambda *_args: b"456\n")
    monkeypatch.setattr(disk.os, "getpid", lambda: 123)
    monkeypatch.setattr(disk.os, "ftruncate", lambda *args: state["events"].append(("truncate", *args)))
    monkeypatch.setattr(disk.os, "pwrite", lambda *args: state["events"].append(("write", *args)))
    return owner, state


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFDIR, stat.S_IFCHR])
def test_existing_nonregular_lock_refuses_before_open(lock_spies: Any, mode: int) -> None:
    owner, state = lock_spies
    state["named_mode"] = mode
    with pytest.raises(RuntimeError, match="not a regular file"):
        disk.DiskReplayBuffer._claim_writer(owner)
    assert state["events"] == []
    assert owner._writer_lock_fd is None


@pytest.mark.parametrize("change", ["fd_mode", "named_ino", "after_open_mode"])
def test_opened_inode_must_be_regular_and_match_name(lock_spies: Any, change: str) -> None:
    owner, state = lock_spies
    state[change] = stat.S_IFIFO if change != "named_ino" else 9
    with pytest.raises(RuntimeError, match="inode changed or is not a regular"):
        disk.DiskReplayBuffer._claim_writer(owner)
    assert [event[0] for event in state["events"]] == ["open", "close"]
    assert owner._writer_lock_fd is None


@pytest.mark.parametrize("failure", ["open_error", "fstat_error"])
def test_open_and_validation_failures_leave_no_owned_fd(lock_spies: Any, failure: str) -> None:
    owner, state = lock_spies
    state[failure] = True
    with pytest.raises(RuntimeError, match="cannot claim replay writer lock"):
        disk.DiskReplayBuffer._claim_writer(owner)
    expected = ["open"] if failure == "open_error" else ["open", "close"]
    assert [event[0] for event in state["events"]] == expected
    assert owner._writer_lock_fd is None


def test_regular_contention_closes_without_pid_stamping(lock_spies: Any) -> None:
    owner, state = lock_spies
    state["contended"] = True
    with pytest.raises(RuntimeError, match="pid 456"):
        disk.DiskReplayBuffer._claim_writer(owner)
    assert [event[0] for event in state["events"]] == ["open", "flock", "close"]
    assert owner._writer_lock_fd is None


@pytest.mark.parametrize("missing", [False, True])
def test_regular_lock_claim_reuse_and_release(lock_spies: Any, missing: bool) -> None:
    owner, state = lock_spies
    state["missing"] = missing
    disk.DiskReplayBuffer._claim_writer(owner)
    assert owner._writer_lock_fd == 77
    assert [event[0] for event in state["events"]] == ["open", "flock", "truncate", "write"]
    assert state["events"][-1] == ("write", 77, b"123\n", 0)
    flags = state["events"][0][2]
    assert flags & disk.os.O_RDWR
    if hasattr(disk.os, "O_NOFOLLOW"):
        assert flags & disk.os.O_NOFOLLOW
    disk.DiskReplayBuffer._claim_writer(owner)
    assert len(state["events"]) == 4
    disk.DiskReplayBuffer._release_writer(owner)
    assert owner._writer_lock_fd is None
    assert state["events"][-2:] == [("flock", 77, disk.fcntl.LOCK_UN), ("close", 77)]
    disk.DiskReplayBuffer._release_writer(owner)
    assert len(state["events"]) == 6


def test_missing_nofollow_flag_still_checks_named_inode(lock_spies: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    owner, state = lock_spies
    monkeypatch.delattr(disk.os, "O_NOFOLLOW", raising=False)
    state["after_open_mode"] = stat.S_IFLNK
    with pytest.raises(RuntimeError, match="inode changed or is not a regular"):
        disk.DiskReplayBuffer._claim_writer(owner)
    assert [event[0] for event in state["events"]] == ["open", "close"]
