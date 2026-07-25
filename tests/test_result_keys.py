from __future__ import annotations

from chess_anti_engine.tune.result_keys import row_counter, row_counter_opt


def test_reads_the_canonical_key() -> None:
    assert row_counter({"matching_games": 445}, "matching_games") == 445
    assert row_counter({"matching_positions": 9810}, "matching_positions") == 9810


def test_falls_back_to_the_pre_rename_key() -> None:
    """result.json files predating 2026-07-24 are still on disk and still read.

    replay_exchange sizes cross-trial sharing from OTHER trials' historical
    rows, so losing this fallback would silently share nothing — the exact
    class of silent degradation the rename exists to remove.
    """
    assert row_counter({"games_generated": 445}, "matching_games") == 445
    assert row_counter({"positions_added": 9810}, "matching_positions") == 9810


def test_canonical_key_wins_when_both_are_present() -> None:
    row = {"matching_games": 445, "games_generated": 1}
    assert row_counter(row, "matching_games") == 445


def test_missing_key_returns_the_default() -> None:
    assert row_counter({}, "matching_games") == 0
    assert row_counter({}, "matching_games", default=-1) == -1


def test_unparseable_value_returns_the_default_rather_than_raising() -> None:
    """A torn/garbage row must not crash a monitoring scan."""
    assert row_counter({"matching_games": "n/a"}, "matching_games", default=-1) == -1


def test_a_real_zero_is_not_confused_with_absent() -> None:
    """distributed_stale_games=0 is meaningful; a missing metric is not.

    Reporting a missing metric as agreement is how the views bug stayed
    invisible, so callers that must distinguish get row_counter_opt.
    """
    assert row_counter_opt({"matching_games": 0}, "matching_games") == 0
    assert row_counter_opt({}, "matching_games") is None


def test_opt_also_honors_the_legacy_name() -> None:
    assert row_counter_opt({"games_generated": 7}, "matching_games") == 7


def test_a_key_with_no_legacy_alias_does_not_invent_one() -> None:
    assert row_counter_opt({"replay_positions_ingested": 5}, "replay_positions_ingested") == 5
    assert row_counter_opt({}, "replay_positions_ingested") is None
