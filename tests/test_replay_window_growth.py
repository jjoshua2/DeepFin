from __future__ import annotations

from chess_anti_engine.tune.trainable_phases import _next_replay_window


def test_next_replay_window_uses_ingested_position_fraction() -> None:
    next_window, growth, frac = _next_replay_window(
        current_window=400_000,
        replay_window_max=1_000_000,
        replay_window_growth=5_000,
        replay_window_growth_frac=0.75,
        replay_positions_ingested=20_000,
    )

    assert next_window == 415_000
    assert growth == 15_000
    assert frac == 0.75


def test_next_replay_window_falls_back_to_fixed_growth() -> None:
    next_window, growth, frac = _next_replay_window(
        current_window=400_000,
        replay_window_max=1_000_000,
        replay_window_growth=5_000,
        replay_window_growth_frac=None,
        replay_positions_ingested=20_000,
    )

    assert next_window == 405_000
    assert growth == 5_000
    assert frac == 0.0


def test_next_replay_window_clamps_to_max() -> None:
    next_window, growth, frac = _next_replay_window(
        current_window=990_000,
        replay_window_max=1_000_000,
        replay_window_growth=5_000,
        replay_window_growth_frac=1.0,
        replay_positions_ingested=20_000,
    )

    assert next_window == 1_000_000
    assert growth == 10_000
    assert frac == 1.0
