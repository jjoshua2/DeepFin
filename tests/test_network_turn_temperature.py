from __future__ import annotations

import numpy as np

from chess_anti_engine.selfplay.config import SearchConfig
from chess_anti_engine.selfplay.network_turn import (
    _resample_actions_with_temperature,
    _scheduled_gumbel_scale,
)


def test_gumbel_temp_zero_preserves_search_survivor() -> None:
    probs = np.zeros((4672,), dtype=np.float32)
    probs[3] = 0.9
    probs[7] = 0.1
    actions = [7]

    _resample_actions_with_temperature(
        [probs],
        actions,
        [0.0],
        np.random.default_rng(1),
        c_temp_resample=None,
        use_c_resample=False,
        preserve_zero_temperature_actions=True,
    )

    assert actions == [7]


def test_puct_temp_zero_uses_policy_argmax() -> None:
    probs = np.zeros((4672,), dtype=np.float32)
    probs[3] = 0.9
    probs[7] = 0.1
    actions = [7]

    _resample_actions_with_temperature(
        [probs],
        actions,
        [0.0],
        np.random.default_rng(1),
        c_temp_resample=None,
        use_c_resample=False,
        preserve_zero_temperature_actions=False,
    )

    assert actions == [3]


def test_gumbel_positive_temperature_resamples_from_policy() -> None:
    probs = np.zeros((4672,), dtype=np.float32)
    probs[3] = 1.0
    probs[7] = 0.0
    actions = [7]

    _resample_actions_with_temperature(
        [probs],
        actions,
        [1.0],
        np.random.default_rng(1),
        c_temp_resample=None,
        use_c_resample=False,
        preserve_zero_temperature_actions=True,
    )

    assert actions == [3]


def test_low_positive_temperature_is_numerically_stable() -> None:
    probs = np.zeros((4672,), dtype=np.float32)
    probs[3] = 0.49
    probs[7] = 0.51
    actions = [3]

    _resample_actions_with_temperature(
        [probs],
        actions,
        [0.001],
        np.random.default_rng(1),
        c_temp_resample=None,
        use_c_resample=False,
        preserve_zero_temperature_actions=True,
    )

    assert actions == [7]


def test_gumbel_low_positive_temperature_falls_back_to_survivor_when_degenerate() -> None:
    probs = np.zeros((4672,), dtype=np.float32)
    actions = [7]

    _resample_actions_with_temperature(
        [probs],
        actions,
        [0.001],
        np.random.default_rng(1),
        c_temp_resample=None,
        use_c_resample=False,
        preserve_zero_temperature_actions=True,
    )

    assert actions == [7]


def test_scheduled_gumbel_scale_decays_by_full_move_number() -> None:
    search = SearchConfig(
        gumbel_scale=0.5,
        gumbel_scale_after=0.0,
        gumbel_scale_decay_start_move=15,
        gumbel_scale_decay_moves=15,
    )

    assert _scheduled_gumbel_scale(search, move_number=14) == 0.5
    assert _scheduled_gumbel_scale(search, move_number=15) == 0.5
    assert np.isclose(_scheduled_gumbel_scale(search, move_number=22), 0.2666666666666667)
    assert _scheduled_gumbel_scale(search, move_number=30) == 0.0
    assert _scheduled_gumbel_scale(search, move_number=31) == 0.0
