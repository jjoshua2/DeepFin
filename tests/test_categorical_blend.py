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
