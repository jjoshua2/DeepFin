from __future__ import annotations

import numpy as np

from chess_anti_engine.selfplay.config import SearchConfig
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE, policy_batch_to_encoding
from chess_anti_engine.selfplay.network_turn import (
    _expand_policy_logits_for_ply,
    _refresh_gumbel_action_diag,
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


def test_compact_policy_logits_expand_before_ply_processing() -> None:
    full = np.full((1, POLICY_SIZE), -7.0, dtype=np.float32)
    full[0, COMPACT_TO_FULL_POLICY] = np.arange(COMPACT_TO_FULL_POLICY.shape[0], dtype=np.float32)
    compact = policy_batch_to_encoding(full, policy_encoding="lc0_1858")

    expanded = _expand_policy_logits_for_ply(compact, policy_encoding="lc0_1858")

    assert expanded.shape == (1, POLICY_SIZE)
    np.testing.assert_array_equal(expanded[0, COMPACT_TO_FULL_POLICY], compact[0])
    invalid = np.ones((POLICY_SIZE,), dtype=np.bool_)
    invalid[COMPACT_TO_FULL_POLICY] = False
    assert float(expanded[0, invalid].max()) == -1e9


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


def test_gumbel_diagnostics_refresh_after_temperature_resample() -> None:
    probs = np.zeros((4672,), dtype=np.float32)
    probs[3] = 0.8
    probs[7] = 0.2
    diag = {
        "action_prob": 0.2,
        "argmax_is_action": 0.0,
        "top_prob": 0.8,
    }

    out = _refresh_gumbel_action_diag(diag, probs, 3)

    assert out is not None
    assert out is not diag
    assert np.isclose(out["action_prob"], 0.8)
    assert out["argmax_is_action"] == 1.0
    assert out["top_prob"] == 0.8


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


def test_curriculum_gumbel_scale_has_independent_schedule() -> None:
    search = SearchConfig(
        gumbel_scale=0.5,
        gumbel_scale_after=0.0,
        gumbel_scale_decay_start_move=15,
        gumbel_scale_decay_moves=15,
        curriculum_gumbel_scale=0.25,
        curriculum_gumbel_scale_after=0.0,
        curriculum_gumbel_scale_decay_start_move=15,
        curriculum_gumbel_scale_decay_moves=15,
    )

    assert _scheduled_gumbel_scale(search, move_number=14, curriculum=True) == 0.25
    assert _scheduled_gumbel_scale(search, move_number=15, curriculum=True) == 0.25
    assert np.isclose(_scheduled_gumbel_scale(search, move_number=22, curriculum=True), 0.13333333333333336)
    assert _scheduled_gumbel_scale(search, move_number=30, curriculum=True) == 0.0
    assert _scheduled_gumbel_scale(search, move_number=31, curriculum=True) == 0.0
