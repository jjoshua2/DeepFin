"""progress.csv must never be appended to under a header that no longer fits.

Ray's CSVLoggerCallback fixes the header on the first row and, on resume,
appends without re-heading while taking its fieldnames from the current result
dict. A report-schema change therefore writes rows positionally against a stale
header. `_make_csv_logger_schema_safe` rotates the old file instead.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from chess_anti_engine.tune.harness import _make_csv_logger_schema_safe

ray_csv = pytest.importorskip("ray.tune.logger.csv")


class _FakeTrial:
    """The surface CSVLoggerCallback actually touches on a trial."""

    def __init__(self, local_path: Path) -> None:
        self.local_path = str(local_path)

    def init_local_path(self) -> None:
        Path(self.local_path).mkdir(parents=True, exist_ok=True)

    def __hash__(self) -> int:
        return hash(self.local_path)


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", newline="") as fh:
        return list(csv.reader(fh))


def _make_callback(monkeypatch: pytest.MonkeyPatch):
    cb = ray_csv.CSVLoggerCallback()
  # _restore_from_remote pulls the file from cloud storage; there is none here.
    monkeypatch.setattr(
        type(cb), "_restore_from_remote", lambda self, name, trial: None
    )
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
