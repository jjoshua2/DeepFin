"""Auto-retirement streak logic + safe yaml repoint (scripts/blindspot_retire_step.py)."""
from __future__ import annotations

from scripts.blindspot_retire_step import update_streaks


def test_retire_requires_two_consecutive_aware() -> None:
    keys = ["a", "b", "c"]
    state: dict[str, int] = {}
    # read 1: a & b AWARE (net_q <= -0.4), c not -> streaks 1/1/0, nobody retires yet
    retire, resolved = update_streaks(keys, [-0.6, -0.5, +0.2], state,
                                      resolved_below=-0.4, min_consecutive=2)
    assert resolved == 2
    assert retire == set()
    assert state == {"a": 1, "b": 1, "c": 0}
    # read 2: a stays AWARE (streak 2 -> retire), b REGRESSES (reset to 0), c AWARE (1)
    retire, resolved = update_streaks(keys, [-0.7, +0.1, -0.9], state,
                                      resolved_below=-0.4, min_consecutive=2)
    assert retire == {"a"}          # two in a row
    assert state["b"] == 0          # a single-read regression un-arms it
    assert state["c"] == 1


def test_regression_resets_streak_no_flukes() -> None:
    state: dict[str, int] = {}
    # AWARE, AWARE, regress, AWARE -> never two-in-a-row-until-the-final-pair
    for q, expect_retire in [(-0.6, False), (-0.6, True), (+0.3, False), (-0.6, False)]:
        retire, _ = update_streaks(["x"], [q], state, resolved_below=-0.4, min_consecutive=2)
        assert ("x" in retire) == expect_retire
