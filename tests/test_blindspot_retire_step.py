"""Unit tests for blind-spot retire / probation re-feed decision logic.

Pure functions only — no GPU, no Stockfish, no filesystem side effects.
Covers: consecutive-AWARE retirement, deep-read guard, probation re-feed band,
min-pool suppression, state-file load/dump backward compatibility, and verbatim
re-feed of original seed lines (including ``# weight=N`` markers).
"""
from __future__ import annotations

from scripts.blindspot_retire_step import (
    DEEP_BELOW,
    REFEED_ABOVE,
    build_active_list,
    dump_retire_state,
    load_retire_state,
    refeed_retired,
    update_streaks,
)


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


def test_deep_read_guard_blocks_shallow_only_streak() -> None:
    """Two consecutive shallow AWARE reads (-0.5, -0.4] must NOT retire when
    --retire-require-deep-read is set (hair-width / shell-noise filter)."""
    state: dict[str, int] = {}
    deep: dict[str, bool] = {}
    # two shallow AWARE reads at -0.42
    for q in (-0.42, -0.41):
        retire, _ = update_streaks(
            ["s"], [q], state, resolved_below=-0.4, min_consecutive=2,
            deep_seen=deep, require_deep_read=True, deep_below=DEEP_BELOW)
    assert state["s"] == 2
    assert deep.get("s", False) is False
    assert retire == set()  # blocked: no deep read in the streak


def test_deep_read_guard_allows_when_one_read_is_deep() -> None:
    """One deep (<= -0.5) among the consecutive AWARE reads is enough."""
    state: dict[str, int] = {}
    deep: dict[str, bool] = {}
    # shallow then deep
    update_streaks(["d"], [-0.42], state, resolved_below=-0.4, min_consecutive=2,
                   deep_seen=deep, require_deep_read=True, deep_below=DEEP_BELOW)
    retire, _ = update_streaks(
        ["d"], [-0.55], state, resolved_below=-0.4, min_consecutive=2,
        deep_seen=deep, require_deep_read=True, deep_below=DEEP_BELOW)
    assert deep["d"] is True
    assert retire == {"d"}

    # deep then shallow also counts (deep_seen sticky within the streak)
    state2: dict[str, int] = {}
    deep2: dict[str, bool] = {}
    update_streaks(["d"], [-0.60], state2, resolved_below=-0.4, min_consecutive=2,
                   deep_seen=deep2, require_deep_read=True, deep_below=DEEP_BELOW)
    retire2, _ = update_streaks(
        ["d"], [-0.41], state2, resolved_below=-0.4, min_consecutive=2,
        deep_seen=deep2, require_deep_read=True, deep_below=DEEP_BELOW)
    assert retire2 == {"d"}


def test_deep_seen_resets_with_streak() -> None:
    state: dict[str, int] = {}
    deep: dict[str, bool] = {}
    update_streaks(["x"], [-0.70], state, resolved_below=-0.4, min_consecutive=2,
                   deep_seen=deep, require_deep_read=True)
    assert deep["x"] is True
    # regression clears both streak and deep flag
    update_streaks(["x"], [+0.1], state, resolved_below=-0.4, min_consecutive=2,
                   deep_seen=deep, require_deep_read=True)
    assert state["x"] == 0
    assert deep["x"] is False


def test_require_deep_read_off_ignores_deep_flag() -> None:
    """Default (flag off): two shallow AWARE reads still retire — pre-probation behaviour."""
    state: dict[str, int] = {}
    deep: dict[str, bool] = {}
    update_streaks(["s"], [-0.42], state, resolved_below=-0.4, min_consecutive=2,
                   deep_seen=deep, require_deep_read=False)
    retire, _ = update_streaks(
        ["s"], [-0.41], state, resolved_below=-0.4, min_consecutive=2,
        deep_seen=deep, require_deep_read=False)
    assert retire == {"s"}


def test_refeed_above_band() -> None:
    """Re-feed only when net_q is strictly above -0.2 (not at the bar, not below)."""
    keys = ["still_aware", "border", "blind", "deep_ok"]
    qs = [-0.30, -0.20, +0.10, -0.55]
    got = refeed_retired(keys, qs, refeed_above=REFEED_ABOVE)
    assert got == {"blind"}  # +0.10 > -0.2; -0.20 is NOT >
    # empty / all-still-retired
    assert refeed_retired(["a"], [-0.35]) == set()
    assert refeed_retired([], []) == set()


def test_build_active_list_retire_and_refeed_verbatim() -> None:
    active_lines = [
        "fenA w - - 0 1",
        "fenB w - - 0 1 # weight=3",
        "fenC w - - 0 1",
    ]
    active_keys = ["A", "B", "C"]
    # B was previously retired with its weight marker preserved in the store
    retired_lines = {
        "R1": "fenR1 w - - 0 1 # weight=5",
        "R2": "fenR2 w - - 0 1",
    }
    # Retire A; re-feed R1 (not R2)
    keep, applied_retire, applied_refeed = build_active_list(
        active_lines, active_keys, retire_keys={"A"},
        retired_lines=retired_lines, refeed_keys={"R1"},
        min_pool=2,
    )
    assert applied_retire == {"A"}
    assert applied_refeed == {"R1"}
    assert keep == [
        "fenB w - - 0 1 # weight=3",
        "fenC w - - 0 1",
        "fenR1 w - - 0 1 # weight=5",  # verbatim incl. weight marker
    ]


def test_build_active_list_min_pool_suppresses_retire() -> None:
    lines = ["a", "b", "c"]
    keys = ["A", "B", "C"]
    keep, applied_retire, applied_refeed = build_active_list(
        lines, keys, retire_keys={"A", "B"},
        retired_lines={}, refeed_keys=set(),
        min_pool=2,  # keep would be 1 < 2
    )
    assert applied_retire == set()
    assert applied_refeed == set()
    assert keep == lines  # unchanged


def test_build_active_list_refeed_only() -> None:
    lines = ["a", "b"]
    keys = ["A", "B"]
    retired = {"R": "r-line # weight=2"}
    keep, applied_retire, applied_refeed = build_active_list(
        lines, keys, retire_keys=set(),
        retired_lines=retired, refeed_keys={"R"},
        min_pool=1,
    )
    assert applied_retire == set()
    assert applied_refeed == {"R"}
    assert keep == ["a", "b", "r-line # weight=2"]


def test_state_load_old_flat_format() -> None:
    """Pre-probation state files are a flat {placement: streak_int} map."""
    raw = {"place/a": 1, "place/b": 0, "place/c": 2}
    streaks, retired, deep = load_retire_state(raw)
    assert streaks == {"place/a": 1, "place/b": 0, "place/c": 2}
    assert retired == {}
    assert deep == {}


def test_state_roundtrip_with_meta() -> None:
    streaks = {"a": 1, "b": 0}
    retired = {"r": "fenR w - - 0 1 # weight=4"}
    deep = {"a": True, "b": False}
    dumped = dump_retire_state(streaks, retired, deep)
    # False deep flags omitted; streaks stay top-level ints
    assert dumped["a"] == 1
    assert dumped["b"] == 0
    assert dumped["__retired__"] == retired
    assert dumped["__deep__"] == {"a": True}
    assert "b" not in dumped["__deep__"]

    s2, r2, d2 = load_retire_state(dumped)
    assert s2 == streaks
    assert r2 == retired
    assert d2 == {"a": True}


def test_state_meta_keys_not_treated_as_streaks() -> None:
    raw = {
        "real": 2,
        "__retired__": {"gone": "fen # weight=1"},
        "__deep__": {"real": True},
    }
    streaks, retired, deep = load_retire_state(raw)
    assert streaks == {"real": 2}
    assert "__retired__" not in streaks
    assert retired == {"gone": "fen # weight=1"}
    assert deep == {"real": True}


def test_probation_decision_end_to_end() -> None:
    """Streak retire + probation re-feed compose into one list update."""
    # Active: three seeds. x has streak 1 already; this read makes x retire.
    # Retired store has r_blind (should re-feed) and r_ok (stays retired).
    streaks = {"x": 1, "y": 0, "z": 0}
    deep: dict[str, bool] = {}
    active_keys = ["x", "y", "z"]
    active_q = [-0.6, +0.1, -0.3]  # x AWARE, y blind, z near-blind
    retire, resolved = update_streaks(
        active_keys, active_q, streaks,
        resolved_below=-0.4, min_consecutive=2,
        deep_seen=deep, require_deep_read=False)
    assert resolved == 1
    assert retire == {"x"}

    retired_lines = {
        "r_blind": "blind-fen w - - 0 1 # weight=7",
        "r_ok": "ok-fen w - - 0 1",
    }
    refeed = refeed_retired(
        ["r_blind", "r_ok"], [+0.47, -0.31], refeed_above=REFEED_ABOVE)
    assert refeed == {"r_blind"}

    keep, applied_retire, applied_refeed = build_active_list(
        ["line-x", "line-y", "line-z"], active_keys, retire,
        retired_lines, refeed, min_pool=1)
    assert applied_retire == {"x"}
    assert applied_refeed == {"r_blind"}
    assert keep == ["line-y", "line-z", "blind-fen w - - 0 1 # weight=7"]
