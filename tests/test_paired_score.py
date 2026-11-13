"""Independent small-sample arithmetic and refusal cases for pair means."""
from __future__ import annotations

import json
from math import isfinite

import pytest

from chess_anti_engine.paired_score import paired_score_interval


def test_five_equally_spaced_pairs_have_hand_calculated_unbiased_variance() -> None:
    result = paired_score_interval([0, 0.25, 0.5, 0.75, 1], expected_pairs=5)
    # Squared deviations total5/8, so unbiased variance is5/32.
    assert result['mean'] == 0.5
    assert result['variance'] == 5 / 32
    assert result['count'] == 5
    assert result['method'] == 'paired_normal_approx_95'
    assert result['lower'] == pytest.approx(0.15351767721859172)
    assert result['upper'] == pytest.approx(0.8464823227814083)
    assert result['fallback_reason'] is None


def test_two_opposite_pairs_use_sample_variance_and_clamp_normal_bounds() -> None:
    result = paired_score_interval([0, 1], expected_pairs=2)
    assert result['mean'] == 0.5
    assert result['variance'] == 0.5
    assert result['radius'] == 0.98
    assert result['lower'] == 0
    assert result['upper'] == 1
    assert result['method'] == 'paired_normal_approx_95'


def test_single_pair_has_undefined_variance_and_bounded_fallback() -> None:
    result = paired_score_interval([0.75], expected_pairs=1)
    assert result['variance'] is None
    assert result['mean'] == 0.75
    assert result['fallback_reason'] == 'single_pair'
    assert result['method'] == 'hoeffding_95_fallback'
    assert result['lower'] == 0
    assert result['upper'] == 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize('score', [0, 0.25, 0.5, 0.75, 1])
def test_constant_pairs_keep_uncertainty_even_at_extreme_scores(score: float) -> None:
    result = paired_score_interval([score] * 500, expected_pairs=500)
    assert result['mean'] == score
    assert result['variance'] == 0
    assert result['fallback_reason'] == 'zero_variance'
    assert result['method'] == 'hoeffding_95_fallback'
    assert result['radius'] == pytest.approx(0.060736146190830516)
    assert result['lower'] < result['upper']
    assert 0 <= result['lower'] <= score <= result['upper'] <= 1
    assert isfinite(result['lower'])
    assert isfinite(result['upper'])


@pytest.mark.parametrize('count', [True, False, 0, -1, 1.0, '1', None])
def test_invalid_declared_pair_count_refuses(count: object) -> None:
    with pytest.raises(ValueError, match='positive nonboolean integer'):
        paired_score_interval([0.5], expected_pairs=count)


def test_missing_pair_and_empty_results_refuse_instead_of_shrinking_endpoint() -> None:
    with pytest.raises(ValueError, match='complete nonempty pair count'):
        paired_score_interval([0.5] * 127, expected_pairs=128)
    with pytest.raises(ValueError, match='complete nonempty pair count'):
        paired_score_interval([], expected_pairs=128)


@pytest.mark.parametrize('value', [True, False, float('nan'), float('inf'), float('-inf'),
                                 0.1, -0.25, 1.25, 1.5, 2, '0.5', None, 10**1000])
def test_nonfinite_boolean_nonquarter_and_pair_totals_refuse(value: object) -> None:
    with pytest.raises(ValueError, match='pair means must be finite'):
        paired_score_interval([value], expected_pairs=1)
