"""The source checkout retains Ceres' fixed control directory without running jobs."""

from unittest.mock import patch

import pytest

from scripts import ceres_materialize as tool


@pytest.mark.parametrize("profile", sorted(tool.PROFILES))
def test_checkout_has_default_control_directory(profile):
    assert (tool.BASE / "README.md").is_file()
    before = set(tool.BASE.iterdir())
    affinity, lock = tool.execution_settings({"profile": profile})
    assert affinity == [0, 1]
    assert lock == tool.LOCK
    profile_lock = tool.BASE / f"{profile}.preparation.lock"
    _, selected = tool.execution_settings({
        "profile": profile, "preparation_lock": str(profile_lock),
    })
    assert selected == profile_lock
    assert set(tool.BASE.iterdir()) == before


def test_missing_control_directory_still_fails_closed(tmp_path):
    missing = tmp_path / "missing"
    with (
        patch.object(tool, "BASE", missing),
        patch.object(tool, "LOCK", missing / "preparation.lock"),
        pytest.raises(ValueError, match="preparation lock"),
    ):
        tool.execution_settings({"profile": "Ceres100"})
    assert not missing.exists()
