"""progress.csv must never be appended to under a header that no longer fits.

Ray's CSVLoggerCallback fixes the header on the first row and, on resume,
appends without re-heading while taking its fieldnames from the current result
dict. A report-schema change therefore writes rows positionally against a stale
header. `_make_csv_logger_schema_safe` rotates the old file instead.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from chess_anti_engine.tune.harness import _make_csv_logger_schema_safe

ray_csv = pytest.importorskip("ray.tune.logger.csv")


class _FakeStorage:
    def __init__(self, trial_fs_path: Path) -> None:
        self.trial_fs_path = str(trial_fs_path)
        self.storage_filesystem = None  # local storage: plain paths


class _FakeTrial:
    """The surface CSVLoggerCallback actually touches on a trial.

    `local_path` is the per-Ray-session STAGING dir and `storage.trial_fs_path`
    is the durable one; keeping them distinct here is the whole point, because
    conflating them is the defect this module exists to catch.
    """

    def __init__(self, local_path: Path, durable: Path | None = None) -> None:
        self.local_path = str(local_path)
        self.storage = _FakeStorage(durable if durable is not None else local_path)
        self.checkpoint = object()  # truthy => Ray attempts _restore_from_remote

    def init_local_path(self) -> None:
        Path(self.local_path).mkdir(parents=True, exist_ok=True)

    def __hash__(self) -> int:
        return hash(self.local_path)


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", newline="") as fh:
        return list(csv.reader(fh))


def _make_callback(monkeypatch: pytest.MonkeyPatch):
    """A callback whose `_restore_from_remote` behaves like Ray's.

    Ray copies the DURABLE progress.csv down into the staging dir on every
    `_setup_trial`, swallowing FileNotFoundError. Stubbing this to a no-op --
    which the first version of this file did -- removes precisely the
    mechanism that can defeat the rotation, so the tests passed against code
    that did not work in production.
    """
    cb = ray_csv.CSVLoggerCallback()

    def restore(self, file_name: str, trial) -> None:
        del self
        if not getattr(trial, "checkpoint", None):
            return
        src = Path(trial.storage.trial_fs_path) / file_name
        dst = Path(trial.local_path) / file_name
        if src.resolve() == dst.resolve():
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        except FileNotFoundError:
            pass  # Ray logs a warning and continues

    monkeypatch.setattr(type(cb), "_restore_from_remote", restore)
    return cb


@pytest.fixture(autouse=True)
def _restore_ray_logger():
    """Undo the class-level patch so it cannot leak into other test modules."""
    original = ray_csv.CSVLoggerCallback.log_trial_result
    yield
    ray_csv.CSVLoggerCallback.log_trial_result = original


def _unpatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore Ray's own log_trial_result, whatever ran before this test."""
    fn = ray_csv.CSVLoggerCallback.log_trial_result
    inner = getattr(fn, "_cae_inner", None)
    if inner is not None:
        monkeypatch.setattr(ray_csv.CSVLoggerCallback, "log_trial_result", inner)


OLD = {"training_iteration": 1, "wdl_loss": 0.75, "timestamp": 1000.0}
NEW = {
    "training_iteration": 2,
    "wdl_loss": 0.79,
    "wdl_onehot_loss": 0.75,
    "timestamp": 2000.0,
}


def test_unpatched_ray_misaligns_the_row_after_a_schema_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterize the defect, so the fix below is measured against real behaviour."""
    _unpatched(monkeypatch)
    trial = _FakeTrial(tmp_path)
    progress = tmp_path / "progress.csv"

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(0, trial, dict(OLD))
    cb.log_trial_end(trial)

  # A resume: fresh callback, same file, one extra reported key.
    cb = _make_callback(monkeypatch)
    cb.log_trial_result(1, trial, dict(NEW))
    cb.log_trial_end(trial)

    rows = _rows(progress)
    header, appended = rows[0], rows[-1]
    assert len(header) == 3
    assert len(appended) == 4, "expected the raw Ray behaviour to over-fill the row"
  # The damage that matters: `timestamp` no longer reads a timestamp.
    assert dict(zip(header, appended))["timestamp"] == "0.75"


def test_patch_rotates_and_keeps_every_column_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_csv_logger_schema_safe()

    trial = _FakeTrial(tmp_path)
    progress = tmp_path / "progress.csv"

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(0, trial, dict(OLD))
    cb.log_trial_end(trial)

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(1, trial, dict(NEW))
    cb.log_trial_end(trial)

    rows = _rows(progress)
    assert rows[0] == list(NEW), "the new file must be headed by the new schema"
    assert len(rows) == 2, "the rotated file must not carry the old rows forward"
    row = dict(zip(rows[0], rows[1]))
    assert row["timestamp"] == "2000.0"
    assert row["wdl_loss"] == "0.79"
    assert row["wdl_onehot_loss"] == "0.75"

  # The old rows survive beside the new file rather than being destroyed.
    rotated = list(tmp_path.glob("progress.*.csv"))
    assert len(rotated) == 1
    old_rows = _rows(rotated[0])
    assert old_rows[0] == list(OLD)
    assert dict(zip(old_rows[0], old_rows[1]))["timestamp"] == "1000.0"


def test_patch_is_inert_when_the_schema_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No rotation, no lost history, and resumed rows keep appending."""
    _make_csv_logger_schema_safe()

    trial = _FakeTrial(tmp_path)
    progress = tmp_path / "progress.csv"

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(0, trial, dict(OLD))
    cb.log_trial_end(trial)

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(1, trial, {**OLD, "training_iteration": 2})
    cb.log_trial_end(trial)

    rows = _rows(progress)
    assert rows[0] == list(OLD)
    assert len(rows) == 3, "both rows must remain in the same file"
    assert not list(tmp_path.glob("progress.*.csv"))


def test_rotation_survives_the_durable_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production layout: staging and durable are DIFFERENT directories.

    This is the case the first version of this patch got wrong. Rotating only
    `trial.local_path` leaves the durable file intact, `_setup_trial`'s
    `_restore_from_remote` copies it straight back down with the OLD header,
    `_trial_continue` returns True, and Ray appends the misaligned row anyway
    -- while the patch prints a success message.
    """
    _make_csv_logger_schema_safe()

    staging = tmp_path / "session" / "driver_artifacts" / "trial"
    durable = tmp_path / "runs" / "tune" / "trial"
    staging.mkdir(parents=True)
    durable.mkdir(parents=True)
    trial = _FakeTrial(staging, durable=durable)

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(0, trial, dict(OLD))
    cb.log_trial_end(trial)
  # Ray syncs staging up to durable; emulate that so the durable copy exists.
    shutil.copyfile(staging / "progress.csv", durable / "progress.csv")

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(1, trial, dict(NEW))
    cb.log_trial_end(trial)

    rows = _rows(staging / "progress.csv")
    assert rows[0] == list(NEW), (
        "the durable restore put the stale header back and Ray appended under it"
    )
    assert dict(zip(rows[0], rows[1]))["timestamp"] == "2000.0"

  # History must be preserved where the consumers actually read -- the durable
  # dir -- not only in a session tmpdir that gets discarded.
    durable_rotated = list(durable.glob("progress.*.csv"))
    assert durable_rotated, "the durable rows were not preserved"
    assert _rows(durable_rotated[0])[0] == list(OLD)
    assert not (durable / "progress.csv").exists(), (
        "the durable file must be moved aside, or the next restore resurrects it"
    )


def test_a_failed_rotation_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rotation that cannot complete must not take the training loop with it.

    The docstring promises best-effort behaviour, and this runs inside Ray's
    result loop: `CallbackList.on_trial_result` and
    `TuneController._process_trial_save` have no try/except, so an exception
    here propagates into the driver.
    """
    _make_csv_logger_schema_safe()

    trial = _FakeTrial(tmp_path)
    progress = tmp_path / "progress.csv"

    cb = _make_callback(monkeypatch)
    cb.log_trial_result(0, trial, dict(OLD))
    cb.log_trial_end(trial)

    def boom(_self: Path, _target: Path) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "rename", boom)

    cb = _make_callback(monkeypatch)
  # Must not raise, and must still write the row somewhere valid.
    cb.log_trial_result(1, trial, dict(NEW))
    cb.log_trial_end(trial)

    rows = _rows(progress)
    assert rows[0] == list(OLD), "the original header must survive a failed rotation"
    assert len(rows) == 3, "the row must still be appended, not lost"
    assert not list(tmp_path.glob("progress.*.csv"))


def test_patch_is_idempotent() -> None:
    """run_tune may be called more than once in a process; wrappers must not stack."""
    _make_csv_logger_schema_safe()
    once = ray_csv.CSVLoggerCallback.log_trial_result
    _make_csv_logger_schema_safe()
    assert ray_csv.CSVLoggerCallback.log_trial_result is once
