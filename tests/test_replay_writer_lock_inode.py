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


# Real filesystem, no spies: the layouts production can actually produce.


def _real_owner(shard_dir: Path) -> Any:
    return SimpleNamespace(_shard_dir=shard_dir, _writer_lock_fd=None)


@pytest.mark.parametrize("pre_existing", [False, True])
def test_real_lock_in_symlinked_shard_dir_is_claimed(tmp_path: Path, pre_existing: bool) -> None:
    """A symlinked replay DIRECTORY must not be refused.

    O_NOFOLLOW and the lstat checks apply to the lock's final component only,
    so a run whose replay dir is a link (e.g. relocated bulk storage) keeps
    ingesting. MUTATION: resolve/lstat the parent too -- this refuses.
    """
    real = tmp_path / "real_shards"
    real.mkdir()
    link = tmp_path / "replay_shards"
    link.symlink_to(real, target_is_directory=True)
    if pre_existing:
        (real / ".writer.lock").write_text("999\n")
    owner = _real_owner(link)
    disk.DiskReplayBuffer._claim_writer(owner)
    try:
        assert owner._writer_lock_fd is not None
        assert (real / ".writer.lock").read_text() == f"{disk.os.getpid()}\n"
    finally:
        disk.DiskReplayBuffer._release_writer(owner)
    assert owner._writer_lock_fd is None


def test_real_symlinked_lock_file_is_refused_and_target_untouched(tmp_path: Path) -> None:
    """A lock NAME that is a symlink is refused before anything is written."""
    shards = tmp_path / "replay_shards"
    shards.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep\n")
    (shards / ".writer.lock").symlink_to(victim)
    owner = _real_owner(shards)
    with pytest.raises(RuntimeError, match="not a regular file"):
        disk.DiskReplayBuffer._claim_writer(owner)
    assert owner._writer_lock_fd is None
    assert victim.read_text() == "keep\n"


def test_real_second_writer_is_refused(tmp_path: Path) -> None:
    shards = tmp_path / "replay_shards"
    shards.mkdir()
    first, second = _real_owner(shards), _real_owner(shards)
    disk.DiskReplayBuffer._claim_writer(first)
    try:
        with pytest.raises(RuntimeError, match="already has an active writer"):
            disk.DiskReplayBuffer._claim_writer(second)
        assert second._writer_lock_fd is None
    finally:
        disk.DiskReplayBuffer._release_writer(first)


def test_real_lock_swapped_between_open_and_flock_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name is replaced AFTER the pre-lock inode check but BEFORE flock.

    Deterministic: the swap runs inside the first LOCK_EX call. Without the
    re-check under the lock the claim would succeed on the orphaned inode while
    a second writer could lock the new file. MUTATION: drop the post-flock
    ``_require_lock_inode`` -- the claim succeeds and stamps the orphan.
    """
    shards = tmp_path / "replay_shards"
    shards.mkdir()
    lock = shards / ".writer.lock"
    real_flock = disk.fcntl.flock
    swapped: list[bool] = []

    def swap_then_flock(fd: int, operation: int) -> None:
        if operation & disk.fcntl.LOCK_EX and not swapped:
            swapped.append(True)
            lock.unlink()
            lock.write_text("")  # a fresh regular file under the same name
        real_flock(fd, operation)

    monkeypatch.setattr(disk.fcntl, "flock", swap_then_flock)
    owner = _real_owner(shards)
    with pytest.raises(RuntimeError, match="replaced while being claimed"):
        disk.DiskReplayBuffer._claim_writer(owner)
    assert swapped == [True]
    assert owner._writer_lock_fd is None
    assert lock.read_text() == ""  # nothing stamped into the new file
    monkeypatch.undo()
    # The directory is claimable once nobody is swapping it.
    retry = _real_owner(shards)
    disk.DiskReplayBuffer._claim_writer(retry)
    try:
        assert retry._writer_lock_fd is not None
    finally:
        disk.DiskReplayBuffer._release_writer(retry)
