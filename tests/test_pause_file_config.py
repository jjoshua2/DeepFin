"""Custom pause paths are validated before a trial reaches pause handling."""
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


@pytest.mark.parametrize("as_path", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_custom_pause_file_reaches_resolver(
    as_path: bool, absolute: bool, tmp_path: Path,
) -> None:
    trial = tmp_path / "session" / "trial"
    marker = tmp_path / "custom.txt" if absolute else Path("custom.txt")
    value = marker if as_path else f"  {marker}  "
    tc = TrialConfig.from_dict({"pause_file": value})
    expected = marker if absolute else trial.parent / marker
    assert _resolve_pause_marker_paths(tc=tc, trial_dir=trial) == [
        expected, trial / "pause.txt", trial.parent / "pause.txt",
    ]


def test_custom_default_marker_is_deduplicated(tmp_path: Path) -> None:
    trial = tmp_path / "session" / "trial"
    tc = TrialConfig.from_dict({"pause_file": Path("pause.txt")})
    assert _resolve_pause_marker_paths(tc=tc, trial_dir=trial) == [
        trial.parent / "pause.txt", trial / "pause.txt",
    ]


def test_yaml_boolean_is_rejected_before_pause_resolution() -> None:
    import yaml

    config = yaml.safe_load("pause_file: true\n")
    with pytest.raises(ValueError, match="pause_file must be a text path or None"):
        TrialConfig.from_dict(config)
