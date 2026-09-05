"""A missing eviction target must not ratchet the checkpoint index backwards.

Ray names checkpoint directories purely from
`StorageContext.current_checkpoint_index`. That counter lives in two places
meant to stay in lockstep -- the actor's copy (`Trainable.save`) and the
driver's copy (`Trial.on_checkpoint`) -- but only the driver's is persisted,
and only the driver's is skipped when `register_checkpoint` raises. So one
failed eviction leaves the persisted index behind the directories that exist,
and the NEXT restart seeds the actor from it and overwrites live checkpoints.

Observed live 2026-07-25: the trial restored from `checkpoint_000250` and then
wrote 246, 247, 248, 249, 250, 251 -- five existing checkpoints silently
replaced, including the restore source. The manager then held two entries with
the same path, so evicting the stale one deleted a live checkpoint, which
produced the next missing-path raise. Each restart made it worse.

Two independent fixes, tested separately here:
  * `_make_checkpoint_eviction_idempotent` removes the cause.
  * `_guard_checkpoint_index` removes the consequence, and is what repairs an
    index that has already drifted.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.tune.harness import _make_checkpoint_eviction_idempotent
from chess_anti_engine.tune.trainable_report import (
    _existing_checkpoint_dirs,
    _guard_checkpoint_index,
    _prune_trial_checkpoints,
)


@pytest.fixture
def unpatched_delete_fs_path():
    """Restore the module binding so a patch can't leak between tests."""
    from ray.train._internal import checkpoint_manager as ckpt_mgr

    original = ckpt_mgr._delete_fs_path
    yield ckpt_mgr
    ckpt_mgr._delete_fs_path = original


def _make_checkpoint(path: Path):
    from ray.tune import Checkpoint

    path.mkdir(parents=True, exist_ok=True)
    (path / "state.pt").write_bytes(b"weights")
    return Checkpoint.from_directory(str(path))


def _training_result(path: Path):
    from ray.train._internal.session import _TrainingResult

    return _TrainingResult(checkpoint=_make_checkpoint(path), metrics={})


# ---------------------------------------------------------------------------
# The cause: eviction of an already-absent path
# ---------------------------------------------------------------------------


def test_ray_still_raises_on_a_missing_path_without_the_patch(tmp_path: Path) -> None:
    """Pins the upstream behaviour the patch exists for.

    If a Ray upgrade ever makes `_delete_fs_path` tolerant on its own, this
    fails and the patch can be deleted rather than carried forever.
    """
    import pyarrow.fs

    from ray.train._internal.storage import _delete_fs_path

    with pytest.raises(FileNotFoundError):
        _delete_fs_path(fs=pyarrow.fs.LocalFileSystem(), fs_path=str(tmp_path / "gone"))


def test_patched_eviction_treats_an_absent_target_as_deleted(
    tmp_path: Path, unpatched_delete_fs_path, capsys: pytest.CaptureFixture[str]
) -> None:
    import pyarrow.fs

    _make_checkpoint_eviction_idempotent()

    unpatched_delete_fs_path._delete_fs_path(
        fs=pyarrow.fs.LocalFileSystem(), fs_path=str(tmp_path / "gone")
    )

    assert "already absent" in capsys.readouterr().out


def test_patched_eviction_still_deletes_a_path_that_exists(
    tmp_path: Path, unpatched_delete_fs_path
) -> None:
    """The patch must not turn eviction into a no-op -- that would leak disk."""
    import pyarrow.fs

    victim = tmp_path / "checkpoint_000001"
    victim.mkdir()
    (victim / "state.pt").write_bytes(b"weights")

    _make_checkpoint_eviction_idempotent()
    unpatched_delete_fs_path._delete_fs_path(
        fs=pyarrow.fs.LocalFileSystem(), fs_path=str(victim)
    )

    assert not victim.exists()


def test_patching_twice_does_not_nest_wrappers(unpatched_delete_fs_path) -> None:
    original = unpatched_delete_fs_path._delete_fs_path

    _make_checkpoint_eviction_idempotent()
    once = unpatched_delete_fs_path._delete_fs_path
    assert once is not original, "the first call must actually replace the binding"

    _make_checkpoint_eviction_idempotent()

    assert unpatched_delete_fs_path._delete_fs_path is once


def test_a_ray_layout_change_cannot_stop_training_from_starting(
    unpatched_delete_fs_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Best-effort, like the scheduler hotpatch it sits beside."""
    del unpatched_delete_fs_path._delete_fs_path

    _make_checkpoint_eviction_idempotent()

    assert "non-fatal" in capsys.readouterr().out


@pytest.mark.usefixtures("unpatched_delete_fs_path")
def test_register_checkpoint_survives_an_evicted_dir_deleted_behind_its_back(
    tmp_path: Path,
) -> None:
    """The end-to-end case, against a real `_CheckpointManager`.

    This is the live scenario: our own `_prune_trial_checkpoints` removes a
    directory the driver's manager still tracks, and the manager then tries to
    evict it. Without the patch that raise escapes `register_checkpoint` and
    aborts the caller's remaining bookkeeping.
    """
  # Both imported from the manager's own module: that binding is by definition
  # the config class this Ray version's manager accepts.
    from ray.train._internal.checkpoint_manager import (
        CheckpointConfig,
        _CheckpointManager,
    )

    manager = _CheckpointManager(checkpoint_config=CheckpointConfig(num_to_keep=1))
    stale = tmp_path / "checkpoint_000000"
    manager.register_checkpoint(_training_result(stale))

  # The second deleter strikes: gone from disk, still in the manager's list.
    for child in stale.iterdir():
        child.unlink()
    stale.rmdir()

    with pytest.raises(FileNotFoundError):
        manager.register_checkpoint(_training_result(tmp_path / "checkpoint_000001"))

    manager = _CheckpointManager(checkpoint_config=CheckpointConfig(num_to_keep=1))
    manager.register_checkpoint(_training_result(stale))
    for child in stale.iterdir():
        child.unlink()
    stale.rmdir()

    _make_checkpoint_eviction_idempotent()
    manager.register_checkpoint(_training_result(tmp_path / "checkpoint_000002"))

    tracked = [
        Path(r.checkpoint.path).name
        for r in manager._checkpoint_results
        if r.checkpoint is not None
    ]
    assert tracked == ["checkpoint_000002"], (
        "the new checkpoint must be registered and the stale entry dropped"
    )


# ---------------------------------------------------------------------------
# The consequence: an index that has already drifted
# ---------------------------------------------------------------------------


def _install_fake_session(monkeypatch: pytest.MonkeyPatch, index: int) -> Any:
    """Stand in for `ray.train._internal.session.get_session()`."""
    import ray.train._internal.session as session_mod

    storage = SimpleNamespace(current_checkpoint_index=index)
    monkeypatch.setattr(
        session_mod, "get_session", lambda: SimpleNamespace(storage=storage)
    )
    return storage


def _make_checkpoint_dirs(trial_dir: Path, indices: list[int]) -> None:
    for i in indices:
        (trial_dir / f"checkpoint_{i:06d}").mkdir(parents=True)


def test_a_drifted_index_is_advanced_past_every_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live case: index says 245, checkpoint_000250 is sitting on disk."""
    _make_checkpoint_dirs(tmp_path, [246, 247, 248, 249, 250])
    storage = _install_fake_session(monkeypatch, 245)

    assert _guard_checkpoint_index(trial_dir=tmp_path) == 250

    assert storage.current_checkpoint_index == 250, (
        "the next write is index+1, so landing exactly on the highest existing "
        "index is what makes the next checkpoint 251"
    )
    assert "overwrite" in capsys.readouterr().out


def test_an_index_already_past_the_newest_checkpoint_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_checkpoint_dirs(tmp_path, [249, 250])
    storage = _install_fake_session(monkeypatch, 250)

    assert _guard_checkpoint_index(trial_dir=tmp_path) is None

    assert storage.current_checkpoint_index == 250
    assert capsys.readouterr().out == ""


def test_a_fresh_trial_dir_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No checkpoints yet -- index -1 is correct and must not be touched."""
    storage = _install_fake_session(monkeypatch, -1)

    assert _guard_checkpoint_index(trial_dir=tmp_path) is None

    assert storage.current_checkpoint_index == -1


def test_unparseable_checkpoint_dirs_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`checkpoint_tmp_*` and friends must not be read as an index."""
    _make_checkpoint_dirs(tmp_path, [3])
    (tmp_path / "checkpoint_tmp_abc123").mkdir()
    (tmp_path / "checkpoint_000009").write_text("a file, not a checkpoint dir")
    storage = _install_fake_session(monkeypatch, 1)

    assert _guard_checkpoint_index(trial_dir=tmp_path) == 3

    assert storage.current_checkpoint_index == 3


def test_an_unavailable_session_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Outside a Tune session `get_session()` returns None; don't crash the trial."""
    import ray.train._internal.session as session_mod

    _make_checkpoint_dirs(tmp_path, [5])
    monkeypatch.setattr(session_mod, "get_session", lambda: None)

    with caplog.at_level(logging.WARNING):
        assert _guard_checkpoint_index(trial_dir=tmp_path) is None

    assert "checkpoint index" in caplog.text


def test_the_index_really_is_what_names_the_directory() -> None:
    """Pins the Ray surface the guard depends on.

    The fakes above would keep passing against a Ray that derived checkpoint
    names some other way, or that advanced the index by something other than
    one. Both assumptions are load-bearing: the guard sets the index to the
    highest existing one precisely because the next write is index + 1.
    """
    from ray.train._internal.storage import StorageContext

    storage = StorageContext(storage_path="/tmp", experiment_dir_name="exp")
    storage.current_checkpoint_index = 250
    assert storage.checkpoint_dir_name == "checkpoint_000250"

    storage._update_checkpoint_index({})
    assert storage.current_checkpoint_index == 251
    assert storage.checkpoint_dir_name == "checkpoint_000251"


# ---------------------------------------------------------------------------
# The second deleter
# ---------------------------------------------------------------------------


def test_checkpoints_are_ordered_by_index_not_by_name(tmp_path: Path) -> None:
    """Name-sorting breaks the moment the run crosses six digits.

    Ray zero-pads to six, so `checkpoint_1000000` sorts BEFORE
    `checkpoint_999999` as a string -- and the pruner would then delete the
    newest checkpoints instead of the oldest.
    """
    _make_checkpoint_dirs(tmp_path, [999998, 999999, 1000000, 1000001])

    assert [i for i, _ in _existing_checkpoint_dirs(tmp_path)] == [
        999998, 999999, 1000000, 1000001,
    ]


def test_pruning_keeps_the_newest_and_says_what_it_deleted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unlogged deleter of ~650MB directories cannot be attributed later."""
    _make_checkpoint_dirs(tmp_path, [10, 11, 12, 13, 14])

    with caplog.at_level(logging.INFO):
        _prune_trial_checkpoints(trial_dir=tmp_path, keep_last=2)

    survivors = sorted(p.name for p in tmp_path.glob("checkpoint_*"))
    assert survivors == ["checkpoint_000013", "checkpoint_000014"]
    assert "checkpoint_000010" in caplog.text
    assert "keep_last=2" in caplog.text


def test_pruning_below_the_keep_count_deletes_nothing_and_stays_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _make_checkpoint_dirs(tmp_path, [1, 2])

    with caplog.at_level(logging.INFO):
        _prune_trial_checkpoints(trial_dir=tmp_path, keep_last=6)

    assert len(list(tmp_path.glob("checkpoint_*"))) == 2
    assert caplog.text == ""


def test_a_checkpoint_that_survives_deletion_is_reported_as_a_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ignore_errors=True` hides the failure, so the log must not repeat it.

    Claiming a directory was pruned when it is still on disk is the same
    class of defect this whole module exists to remove: a confident number
    that nothing checked against reality.
    """
    _make_checkpoint_dirs(tmp_path, [20, 21, 22, 23])

  # Stand in for the real causes (EACCES, EBUSY, a read-only mount): rmtree
  # swallows the error and the directory is still there afterwards.
    monkeypatch.setattr(
        "chess_anti_engine.tune.trainable_report.shutil.rmtree",
        lambda path, ignore_errors=False: None,
    )

    with caplog.at_level(logging.INFO):
        _prune_trial_checkpoints(trial_dir=tmp_path, keep_last=2)

    assert len(list(tmp_path.glob("checkpoint_*"))) == 4
    assert "failed to prune 2" in caplog.text
    assert "checkpoint_000020" in caplog.text
    assert "checkpoint_000021" in caplog.text
    assert "retention is NOT being realized" in caplog.text
    assert "pruned" not in caplog.text.replace("failed to prune", "")


def test_a_partial_failure_reports_both_halves(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stuck directory must not suppress the report of the ones that went."""
    _make_checkpoint_dirs(tmp_path, [30, 31, 32, 33])
    stuck = tmp_path / "checkpoint_000030"
    real_rmtree = shutil.rmtree

    def flaky(path, ignore_errors: bool = False) -> None:
        if Path(path) == stuck:
            return
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(
        "chess_anti_engine.tune.trainable_report.shutil.rmtree", flaky
    )

    with caplog.at_level(logging.INFO):
        _prune_trial_checkpoints(trial_dir=tmp_path, keep_last=2)

    assert sorted(p.name for p in tmp_path.glob("checkpoint_*")) == [
        "checkpoint_000030", "checkpoint_000032", "checkpoint_000033",
    ]
    assert "pruned 1 trial checkpoint(s)" in caplog.text
    assert "checkpoint_000031" in caplog.text
    assert "failed to prune 1" in caplog.text
    assert "checkpoint_000030" in caplog.text
