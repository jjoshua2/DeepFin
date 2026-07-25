"""Per-trial state a restart has to read back must not live in a staging dir.

`ray.train.get_context().get_trial_dir()` returns the per-Ray-session driver
staging directory. Ray syncs it UP to persistent storage and never syncs it
back DOWN, and every restart opens a new Ray session -- so it is EMPTY at
process start. Every startup read against it therefore resolves to "missing",
silently, because "sidecar absent" is also the legitimate first-run case.

Measured live 2026-07-25 on trial `4c17c`: 68 per-session staging dirs, each
holding a `best.json` containing only that session's own minimum
(5.259 -> 5.237 -> 5.139 -> 5.212 -> 5.092 -> 5.172 ... -- wandering, not
monotone), and all 33 post-restart rows in `result.json` setting `best_loss`
exactly equal to that row's `train_loss`. `best/best_model.pt` was therefore
"best since the last restart", never "best ever".

`_resolve_pause_marker_paths` already documents this exact hazard for pause
markers. It was never carried across to the state the trial reloads.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_anti_engine.tune.trainable import _durable_trial_dir


class _NotLocal:
    """Stand-in for S3FileSystem etc. -- anything not openable as a path."""


def _fake_session(monkeypatch: pytest.MonkeyPatch, storage) -> None:
    import ray.train._internal.session as session_mod

    monkeypatch.setattr(
        session_mod, "get_session", lambda: SimpleNamespace(storage=storage)
    )


def test_the_durable_dir_is_ray_persistent_trial_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyarrow.fs import LocalFileSystem

    staging = tmp_path / "session_x" / "driver_artifacts" / "train_trial_4c17c"
    staging.mkdir(parents=True)
    persistent = tmp_path / "runs" / "tune" / "train_trial_4c17c"

    _fake_session(monkeypatch, SimpleNamespace(
        storage_filesystem=LocalFileSystem(), trial_fs_path=str(persistent),
    ))

    assert _durable_trial_dir(staging) == persistent
    assert persistent.is_dir(), "the durable dir must be usable immediately"


def test_state_written_there_is_still_there_for_the_next_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: two processes, two staging dirs, one record.

    Under the old behaviour the second process reads its own empty staging dir
    and starts from scratch -- which is what re-seeded `best_loss` 33 times.
    """
    from pyarrow.fs import LocalFileSystem

    persistent = tmp_path / "runs" / "tune" / "train_trial_4c17c"
    _fake_session(monkeypatch, SimpleNamespace(
        storage_filesystem=LocalFileSystem(), trial_fs_path=str(persistent),
    ))

    first_staging = tmp_path / "session_1" / "train_trial_4c17c"
    first_staging.mkdir(parents=True)
    (_durable_trial_dir(first_staging) / "best.json").write_text('{"best_loss": 4.59}')

  # A restart: new Ray session, new -- empty -- staging dir.
    second_staging = tmp_path / "session_2" / "train_trial_4c17c"
    second_staging.mkdir(parents=True)
    assert not any(second_staging.iterdir())

    carried = _durable_trial_dir(second_staging) / "best.json"
    assert carried.exists()
    assert carried.read_text() == '{"best_loss": 4.59}'


def test_non_local_storage_falls_back_to_the_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`trial_fs_path` on remote storage is not a path we can open."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _fake_session(monkeypatch, SimpleNamespace(
        storage_filesystem=_NotLocal(), trial_fs_path="s3://bucket/exp/trial",
    ))

    assert _durable_trial_dir(staging) == staging
    out = capsys.readouterr().out
    assert "NOT survive a restart" in out
    assert "_NotLocal" in out


def test_an_unavailable_session_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Ray internals change must degrade to today's behaviour, not crash
    the trial at startup."""
    import ray.train._internal.session as session_mod

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(session_mod, "get_session", lambda: None)

    assert _durable_trial_dir(staging) == staging
    assert "could not resolve the durable trial dir" in capsys.readouterr().out


def test_a_missing_trial_fs_path_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pyarrow.fs import LocalFileSystem

    staging = tmp_path / "staging"
    staging.mkdir()
    _fake_session(monkeypatch, SimpleNamespace(storage_filesystem=LocalFileSystem()))

    assert _durable_trial_dir(staging) == staging
    assert "could not resolve the durable trial dir" in capsys.readouterr().out


# --- Surface pins on the Ray API this depends on --------------------------


def test_ray_still_separates_the_staging_path_from_the_persistent_one() -> None:
    """If these ever became the same path the fix would be a silent no-op."""
    from ray.train._internal.storage import StorageContext

    assert hasattr(StorageContext, "trial_fs_path")
    assert hasattr(StorageContext, "trial_driver_staging_path")


def test_rays_artifact_upload_does_not_delete_at_the_destination() -> None:
    """Load-bearing: we now write files into `trial_fs_path`, which is also
    where Ray uploads the staging dir. A mirroring upload that deleted extras
    would silently remove `best.json` between iterations."""
    import inspect

    from ray.train._internal.storage import _upload_to_fs_path

    src = inspect.getsource(_upload_to_fs_path)
    assert "_pyarrow_fs_copy_files" in src
    assert "delete" not in src


def test_ray_syncs_artifacts_up_only() -> None:
    """The fix relies on there being no download half: nothing restores the
    staging dir from storage, which is why the startup reads always miss."""
    import inspect

    from ray.train._internal.storage import StorageContext

    src = inspect.getsource(StorageContext.persist_artifacts)
    assert "sync_up" in src
    assert "sync_down" not in src


# --- Checkpoint dirs live in persistent storage, not the staging dir ------
#
# `checkpoint_NNNNNN/` is written by Ray to `trial_fs_path`. Both consumers
# that scan for those directories by disk glob -- `_guard_checkpoint_index`
# and `_prune_trial_checkpoints` (PR #241) -- therefore have to be pointed at
# the durable dir. Against the staging dir they glob an empty directory and
# silently do nothing, which looks exactly like "nothing needed doing".


def test_the_index_guard_finds_checkpoints_in_the_durable_dir(tmp_path: Path) -> None:
    from chess_anti_engine.tune.trainable_report import _existing_checkpoint_dirs

    staging = tmp_path / "session_1" / "train_trial_4c17c"
    staging.mkdir(parents=True)
    durable = tmp_path / "runs" / "tune" / "train_trial_4c17c"
    durable.mkdir(parents=True)
    for idx in (271, 272, 273):
        (durable / f"checkpoint_{idx:06d}").mkdir()

    assert _existing_checkpoint_dirs(staging) == [], (
        "the staging dir is where the guard used to look"
    )
    assert [i for i, _ in _existing_checkpoint_dirs(durable)] == [271, 272, 273]
