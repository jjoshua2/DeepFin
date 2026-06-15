from __future__ import annotations

import numpy as np

from chess_anti_engine.train.target_builder import (
    CategoricalTargetParams,
    rebuild_categorical_target_in_arrays,
)
from chess_anti_engine.train.targets import categorical_target_value, hlgauss_target

_BINS = 32
_SIGMA = 0.04


def _ternary_batch() -> dict[str, np.ndarray]:
    """3 rows: win+SF-edge, draw+SF-loss-lean, loss with no SF eval.

    categorical_target is stored as the ternary HL-Gauss (what finalize bakes
    at blend_frac=0), so the no-SF row must come back unchanged under a blend.
    """
    wdl = np.array([0, 1, 2], dtype=np.int8)  # W, D, L
    sf_wdl = np.array(
        [[0.75, 0.0, 0.25], [0.25, 0.0, 0.75], [0.0, 0.0, 0.0]], dtype=np.float32,
    )
    has_sf_wdl = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    cat = np.stack([hlgauss_target(v, num_bins=_BINS, sigma=_SIGMA) for v in (1.0, 0.0, -1.0)])
    return {
        "wdl_target": wdl,
        "sf_wdl": sf_wdl,
        "has_sf_wdl": has_sf_wdl,
        "categorical_target": cat.astype(np.float32),
    }


def test_rebuild_matches_finalize_construction() -> None:
    """Rebuilt rows must equal hlgauss_target(categorical_target_value(...)) —
    the exact live finalize path — per row."""
    arrs = _ternary_batch()
    out = rebuild_categorical_target_in_arrays(
        arrs, params=CategoricalTargetParams(blend_frac=0.5, num_bins=_BINS, sigma=_SIGMA),
    )
    sf = out["sf_wdl"]
    has = out["has_sf_wdl"]
    for i, scalar_v in enumerate((1.0, 0.0, -1.0)):
        row = sf[i] if bool(has[i]) else None
        expected = hlgauss_target(
            categorical_target_value(scalar_v, row, blend_frac=0.5),
            num_bins=_BINS, sigma=_SIGMA,
        )
        np.testing.assert_allclose(out["categorical_target"][i], expected, atol=1e-6)


def test_rebuild_no_sf_row_falls_back_to_stored_ternary() -> None:
    """Row 2 has no SF eval, so a blend leaves it at the ternary target."""
    arrs = _ternary_batch()
    stored_row2 = arrs["categorical_target"][2].copy()
    out = rebuild_categorical_target_in_arrays(
        arrs, params=CategoricalTargetParams(blend_frac=0.5, num_bins=_BINS, sigma=_SIGMA),
    )
    np.testing.assert_allclose(out["categorical_target"][2], stored_row2, atol=1e-6)
    # The SF-backed rows did move off the ternary target.
    assert not np.allclose(out["categorical_target"][0], hlgauss_target(1.0, num_bins=_BINS, sigma=_SIGMA))


def test_rebuild_blend_frac_zero_is_noop() -> None:
    arrs = _ternary_batch()
    before = arrs["categorical_target"].copy()
    out = rebuild_categorical_target_in_arrays(
        arrs, params=CategoricalTargetParams(blend_frac=0.0, num_bins=_BINS, sigma=_SIGMA),
    )
    np.testing.assert_array_equal(out["categorical_target"], before)


def test_rebuild_missing_fields_is_noop() -> None:
    # No categorical_target / wdl_target → nothing to do, no crash.
    arrs: dict[str, np.ndarray] = {"x": np.zeros((2, 1), dtype=np.float32)}
    out = rebuild_categorical_target_in_arrays(
        arrs, params=CategoricalTargetParams(blend_frac=0.5),
    )
    assert "categorical_target" not in out


def test_rebuild_flag_flattens_from_train_section() -> None:
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flat = flatten_run_config_defaults({"train": {"rebuild_categorical_target": True}})
    assert flat["rebuild_categorical_target"] is True
