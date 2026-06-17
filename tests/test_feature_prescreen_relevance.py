"""Synthetic checks for feature_prescreen relevance label-masking (FIX 2).

Rows without a Stockfish label (``has_sf_wdl == 0``) still carry an all-zero
``sf_wdl`` and therefore an all-zero ``sf_wdl_regret`` (CE vs an all-zero
target). The relevance screen must mask those rows out before correlating the
candidate feature with regret, else unlabeled selfplay-only rows look like
zero-regret and bias the correlations/R^2.
"""
from __future__ import annotations

import numpy as np

from scripts.feature_prescreen import (
    _MIN_LABELED_ROWS,
    _label_mask,
    _select_relevance_regret,
)


def test_label_mask_defaults_all_true_when_field_missing() -> None:
    mask = _label_mask({}, "has_sf_wdl", 5)
    assert mask.dtype == bool
    assert mask.tolist() == [True] * 5


def test_label_mask_reads_has_flag() -> None:
    arrs = {"has_sf_wdl": np.array([1, 0, 1, 0, 1], dtype=np.float32)}
    mask = _label_mask(arrs, "has_sf_wdl", 5)
    assert mask.tolist() == [True, False, True, False, True]


def test_masking_changes_relevance_target_rows() -> None:
    # n rows, only the first `labeled` carry an SF label. The relevance target
    # must restrict to those rows (not all n), which changes which regret values
    # feed the correlation.
    n = 200
    labeled = 120
    has = np.zeros((n,), dtype=np.float32)
    has[:labeled] = 1.0
    arrs = {"has_sf_wdl": has}
    loss_keys = {"sf_wdl_regret", "policy_ce", "wdl_ce"}

    key, mask = _select_relevance_regret(loss_keys, arrs, n)
    assert key == "sf_wdl_regret"
    # Masking changes the row set: it is NOT all-true and selects exactly the
    # labeled rows.
    assert int(mask.sum()) == labeled
    assert mask.sum() != n
    assert np.array_equal(np.flatnonzero(mask), np.arange(labeled))

    # Applying the mask to a synthetic regret vector drops the unlabeled (zero)
    # rows, so the masked mean differs from the unmasked mean.
    rng = np.random.default_rng(0)
    regret = np.zeros((n,), dtype=np.float64)
    regret[:labeled] = rng.uniform(0.5, 1.5, size=labeled)  # labeled rows nonzero
    assert regret[mask].mean() != regret.mean()
    assert np.count_nonzero(regret[mask]) == labeled


def test_falls_back_to_label_free_target_when_too_few_labeled() -> None:
    # Fewer than the minimum labeled rows => sf_wdl_regret is skipped and the
    # screen falls back to a label-free regret (policy_ce) over all rows.
    n = 256
    labeled = _MIN_LABELED_ROWS - 1
    has = np.zeros((n,), dtype=np.float32)
    has[:labeled] = 1.0
    arrs = {"has_sf_wdl": has}
    loss_keys = {"sf_wdl_regret", "policy_ce", "wdl_ce"}

    key, mask = _select_relevance_regret(loss_keys, arrs, n)
    assert key == "policy_ce"
    # label-free target uses every row
    assert int(mask.sum()) == n


def test_full_label_coverage_keeps_sf_target_without_note() -> None:
    # All rows labeled => sf_wdl_regret is used and the mask keeps every row
    # (no masking note path that would change the row set).
    n = 100
    arrs = {"has_sf_wdl": np.ones((n,), dtype=np.float32)}
    key, mask = _select_relevance_regret({"sf_wdl_regret", "policy_ce"}, arrs, n)
    assert key == "sf_wdl_regret"
    assert int(mask.sum()) == n


def test_missing_has_flag_treats_all_rows_labeled() -> None:
    # Older shards without a stored has_sf_wdl array => _label_mask defaults to
    # all-true, so sf_wdl_regret is used over every row (backward compatible).
    n = 100
    key, mask = _select_relevance_regret({"sf_wdl_regret", "policy_ce"}, {}, n)
    assert key == "sf_wdl_regret"
    assert int(mask.sum()) == n
