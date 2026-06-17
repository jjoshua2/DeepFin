from __future__ import annotations

import numpy as np

from chess_anti_engine.train.targets import categorical_target_value, hlgauss_target


def test_blend_frac_zero_is_byte_identical_to_ternary_outcome() -> None:
    """Default (off) must return the raw ternary outcome, ignoring any SF eval."""
    for scalar_v in (1.0, 0.0, -1.0):
        assert categorical_target_value(scalar_v, None, blend_frac=0.0) == scalar_v
        # Even with an SF eval present, frac=0 must not touch the target.
        sf = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        assert categorical_target_value(scalar_v, sf, blend_frac=0.0) == scalar_v


def test_blend_mixes_outcome_with_sf_expected_score() -> None:
    """value = (1-f)*outcome + f*(W - L), same side-to-move POV.

    WDL arrays are chosen exactly representable in float32 so the blend is exact.
    """
    sf_win = np.array([0.75, 0.0, 0.25], dtype=np.float32)  # W-L = 0.5
    assert categorical_target_value(1.0, sf_win, blend_frac=0.5) == 0.75
    sf_loss = np.array([0.25, 0.0, 0.75], dtype=np.float32)  # W-L = -0.5
    assert categorical_target_value(-1.0, sf_loss, blend_frac=0.5) == -0.75
    # A drawn game where SF still sees an edge becomes a non-zero continuous value.
    assert categorical_target_value(0.0, sf_win, blend_frac=0.5) == 0.25


def test_blend_falls_back_to_outcome_without_sf_eval() -> None:
    assert categorical_target_value(1.0, None, blend_frac=0.5) == 1.0
    bad_shape = np.array([0.5, 0.5], dtype=np.float32)
    assert categorical_target_value(-1.0, bad_shape, blend_frac=0.5) == -1.0
    # All-zero row (no usable eval) must not divide-by-zero.
    assert categorical_target_value(1.0, np.zeros(3, dtype=np.float32), blend_frac=0.5) == 1.0


def test_blend_is_scale_invariant_normalized_vs_permille() -> None:
    """sf_wdl may be normalized probs (cp-logistic) or SF's raw permille
    (native / after rebuild_sf_targets); (W-L)/(W+D+L) must give the same
    blended value either way, so the offline rebuild matches finalize even when
    rebuild_sf_targets repopulated sf_wdl in permille."""
    norm = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    permille = np.array([600.0, 300.0, 100.0], dtype=np.float32)
    for scalar_v in (1.0, 0.0, -1.0):
        a = categorical_target_value(scalar_v, norm, blend_frac=0.5)
        b = categorical_target_value(scalar_v, permille, blend_frac=0.5)
        assert abs(a - b) < 1e-6


def test_three_way_blend_mirrors_wdl_head() -> None:
    """search_blend_frac adds the net's own-search WDL as a 3rd source, like the
    main value head: game_frac*v + sf_frac*E[sf] + search_frac*E[search]."""
    win = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # E = +1
    loss = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # E = -1
    # 3-way: outcome win, SF says loss, search says win, 0.25 each.
    # 0.5*(+1) + 0.25*(-1) + 0.25*(+1) = 0.5
    assert categorical_target_value(
        1.0, loss, blend_frac=0.25, search_wdl=win, search_blend_frac=0.25,
    ) == 0.5
    # search-only blend (sf weight 0): -1 outcome pulled halfway to search win.
    assert categorical_target_value(
        -1.0, None, blend_frac=0.0, search_wdl=win, search_blend_frac=0.5,
    ) == 0.0


def test_three_way_blend_renormalizes_when_sum_exceeds_one() -> None:
    """sf_frac + search_frac > 1 renormalizes (game_frac -> 0), same rule as the
    WDL head's blend_sum clamp."""
    win = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    loss = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # 0.75 + 0.75 -> 0.5/0.5, game_frac 0: 0.5*(-1) + 0.5*(+1) = 0.
    assert categorical_target_value(
        1.0, loss, blend_frac=0.75, search_wdl=win, search_blend_frac=0.75,
    ) == 0.0


def test_missing_search_row_drops_only_that_component() -> None:
    """A missing/empty search row drops the search component without
    redistributing its weight — the blend leans on SF + outcome."""
    loss = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # E = -1
    # search requested but absent -> behaves as sf-only at frac 0.5: 0.5*1 + 0.5*(-1) = 0
    assert categorical_target_value(
        1.0, loss, blend_frac=0.5, search_wdl=None, search_blend_frac=0.5,
    ) == 0.0
    # both fracs 0 stays byte-identical to the ternary outcome.
    assert categorical_target_value(1.0, loss, blend_frac=0.0, search_blend_frac=0.0) == 1.0


def test_blended_target_uses_interior_bins_unlike_ternary() -> None:
    """The point of the bet: a continuous value lands on interior HL-Gauss bins
    instead of spiking at the {-1,0,+1} extremes."""
    outcome = hlgauss_target(1.0, num_bins=32, sigma=0.04)
    blended = hlgauss_target(0.75, num_bins=32, sigma=0.04)
    # Pure win peaks at the top bin; the blended value peaks on an interior bin.
    assert int(np.argmax(outcome)) == 31
    assert 0 < int(np.argmax(blended)) < 31
    # Both remain valid distributions.
    assert abs(float(outcome.sum()) - 1.0) < 1e-5
    assert abs(float(blended.sum()) - 1.0) < 1e-5
