"""Probability validity at the optional blunder ruler boundary."""
from __future__ import annotations

import numpy as np

from chess_anti_engine.eval.audit import expected_blunder_rates


def _reject(probs: list[float], message: str) -> None:
    try:
        expected_blunder_rates(
            np.array(probs), np.array([200.0] * len(probs)), (100.0,),
        )
    except ValueError as exc:
        error_message = str(exc)
    else:
        raise AssertionError("invalid probabilities were accepted")
    assert message in error_message


def test_nonfinite_entries_rejected_even_outside_blunder_subset() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        _reject([bad, 1.0], "probabilities must be finite")
        try:
            expected_blunder_rates(
                np.array([bad, 1.0]), np.array([0.0, 200.0]), (100.0,),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid non-blunder probability was accepted")


def test_nonfinite_total_from_finite_entries_rejected() -> None:
    _reject([1e308, 1e308], "total must be finite")
    _reject([-1e308, -1e308], "total must be finite")
    _reject([1e308, 1e308, -1e308, -1e308] * 2, "total must be finite")


def test_normalization_overflow_rejected() -> None:
    # Keep the existing signed-input policy, but never accept non-finite arithmetic.
    _reject([1.0, -1.0, float.fromhex("0x0.0000000000001p-1022")],
            "normalized blunder probabilities must be finite")


def test_subset_overflow_rejected_before_clamp() -> None:
    probs = np.array([1e308, -1e308, 1e308, -1e308, 1.0])
    regrets = np.array([200.0, 0.0, 200.0, 0.0, 0.0])
    try:
        expected_blunder_rates(probs, regrets, (100.0,))
    except ValueError as exc:
        error_message = str(exc)
    else:
        raise AssertionError("non-finite subset mass was clamped")
    assert "subset masses must be finite" in error_message


def test_finite_distribution_controls() -> None:
    regrets = np.array([0.0, 60.0, 150.0, 500.0])
    assert expected_blunder_rates(
        np.array([5.0, 2.0, 2.0, 1.0]), regrets, (200.0, 50.0, 100.0),
    ) == (0.1, 0.5, 0.30000000000000004)
    for probs in ([0.0, 0.0], [-1.0, -2.0], [1.0, -1.0]):
        assert expected_blunder_rates(
            np.array(probs), np.array([0.0, 200.0]), (100.0,),
        ) == (0.5,)
    # Finite signed probabilities retain their existing arithmetic and clamp.
    assert expected_blunder_rates(
        np.array([-1.0, 2.0]), np.array([0.0, 200.0]), (100.0,),
    ) == (1.0,)


def test_finite_edges_strict_threshold_and_infinite_regret() -> None:
    tiny = float.fromhex("0x0.0000000000001p-1022")
    regrets = np.array([0.0, 100.0])
    for probs in ([tiny, tiny], [5e307, 5e307]):
        assert expected_blunder_rates(
            np.array(probs), regrets, (99.0, 100.0),
        ) == (0.5, 0.0)
    assert expected_blunder_rates(np.array([1e308, tiny]), regrets, (99.0,)) == (0.0,)
    assert expected_blunder_rates(
        np.array([0.75, 0.25]), np.array([0.0, np.inf]), (100.0,),
    ) == (0.25,)


def test_empty_thresholds_do_not_validate_or_change_default_path() -> None:
    assert expected_blunder_rates(
        np.array([np.nan, np.inf]), np.array([0.0, 200.0]), (),
    ) == ()
