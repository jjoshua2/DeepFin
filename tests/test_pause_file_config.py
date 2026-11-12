"""Custom pause paths are validated before a trial reaches pause handling."""
import os
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.tune.trainable_config_ops import _resolve_pause_marker_paths
from chess_anti_engine.tune.trial_config import TrialConfig


@pytest.mark.parametrize("value", [True, False, 0, 1, 1.5, [], {}, b"pause.txt"])
def test_invalid_pause_file_is_rejected_at_config_boundary(value: Any) -> None:
    with pytest.raises(ValueError, match="pause_file must be a text path or None"):
        TrialConfig.from_dict({"pause_file": value})


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_pause_file_preserves_default_markers(value: Any, tmp_path: Path) -> None:
    trial = tmp_path / "session" / "trial"
    work = tmp_path / "persistent"
    tc = TrialConfig.from_dict({"pause_file": value, "work_dir": str(work)})
    assert _resolve_pause_marker_paths(tc=tc, trial_dir=trial) == [
        trial / "pause.txt", trial.parent / "pause.txt", work / "tune" / "pause.txt",
    ]


def test_normaliser_strips_and_blank_is_none_on_its_own() -> None:
    """The normaliser's own strip, observed on ``tc.pause_file`` itself.

    The resolver strips again, so a resolver-level assertion cannot see a
    normaliser that forgot to (mutant ``return path or None`` passed every
    other test here).
    """
    assert TrialConfig.from_dict({"pause_file": "  /x  "}).pause_file == "/x"
    assert TrialConfig.from_dict({"pause_file": "   "}).pause_file is None


@pytest.mark.parametrize("raw", ["~/p.txt", "  ~/p.txt  "])
def test_tilde_is_expanded(raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``~`` used to reach the resolver literally and become
    ``<ephemeral tune root>/~/p.txt`` -- a path no operator would create."""
    monkeypatch.setenv("HOME", str(tmp_path))
    tc = TrialConfig.from_dict({"pause_file": raw})
    assert tc.pause_file == str(tmp_path / "p.txt")
    trial = tmp_path / "session" / "trial"
    assert _resolve_pause_marker_paths(tc=tc, trial_dir=trial)[0] == tmp_path / "p.txt"


@pytest.mark.parametrize("raw", ["custom.txt", "sub/pause.txt", "./pause.txt", Path("pause.txt")])
def test_relative_pause_file_is_refused_loudly(raw: Any) -> None:
    """A relative path has no anchor the trial and the operator share: the
    trial's only candidate is the per-session Ray staging dir."""
    with pytest.raises(ValueError, match="pause_file must be an absolute path"):
        TrialConfig.from_dict({"pause_file": raw})


@pytest.mark.parametrize("as_path", [False, True])
def test_custom_pause_file_reaches_resolver(as_path: bool, tmp_path: Path) -> None:
    trial = tmp_path / "session" / "trial"
    marker = tmp_path / "custom.txt"
    value = marker if as_path else f"  {marker}  "
    tc = TrialConfig.from_dict({"pause_file": value})
    assert tc.pause_file == os.fspath(marker)
    assert _resolve_pause_marker_paths(tc=tc, trial_dir=trial) == [
        marker, trial / "pause.txt", trial.parent / "pause.txt",
    ]


def test_custom_default_marker_is_deduplicated(tmp_path: Path) -> None:
    trial = tmp_path / "session" / "trial"
    tc = TrialConfig.from_dict({"pause_file": trial.parent / "pause.txt"})
    assert _resolve_pause_marker_paths(tc=tc, trial_dir=trial) == [
        trial.parent / "pause.txt", trial / "pause.txt",
    ]


def test_yaml_boolean_is_rejected_before_pause_resolution() -> None:
    import yaml

    config = yaml.safe_load("pause_file: true\n")
    with pytest.raises(ValueError, match="pause_file must be a text path or None"):
        TrialConfig.from_dict(config)


@pytest.mark.parametrize("raw", ["~", "~/", "  ~  ", "{tmp}/", "{tmp}", "{tmp}/sub/", "{tmp}/nosuch/"])
def test_directory_pause_file_is_refused(
    raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any existing path is a marker, so a directory (``~`` is the home dir)
    would park the trial at its first iteration forever."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError, match="pause_file must be a FILE path"):
        TrialConfig.from_dict({"pause_file": raw.format(tmp=tmp_path)})


def test_file_inside_home_is_still_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert TrialConfig.from_dict({"pause_file": "~/pause.txt"}).pause_file == str(tmp_path / "pause.txt")
