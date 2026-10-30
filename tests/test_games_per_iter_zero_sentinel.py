"""TI4: the documented zero-start sentinel disables game-count ramping."""
import pytest

from chess_anti_engine.tune.distributed_runtime import _first_iteration_games_need
from chess_anti_engine.tune.trainable_metrics import _games_per_iter_for_iteration
from chess_anti_engine.tune.trial_config import TrialConfig


@pytest.mark.parametrize('iteration', [0, 1, 5, 9, 10, 11])
@pytest.mark.parametrize('target', [0, 440])
def test_zero_start_uses_target_throughout_ramp(iteration, target):
    tc = TrialConfig.from_dict({'games_per_iter': target, 'games_per_iter_start': 0,
                               'games_per_iter_ramp_iters': 10})
    assert _games_per_iter_for_iteration(tc, iteration) == max(1, target)


@pytest.mark.parametrize(('iteration', 'expected'), [(1, 400), (5, 416), (9, 432), (10, 440)])
def test_explicit_positive_start_keeps_existing_ramp(iteration, expected):
    tc = TrialConfig.from_dict({'games_per_iter': 440, 'games_per_iter_start': 400,
                               'games_per_iter_ramp_iters': 10})
    assert _games_per_iter_for_iteration(tc, iteration) == expected


@pytest.mark.parametrize('ramp_iters', [0, 1, 10])
@pytest.mark.parametrize('include_start', [False, True])
def test_first_iteration_ingest_need_matches_zero_or_omitted_start(ramp_iters, include_start):
    config = {'games_per_iter': 440, 'games_per_iter_ramp_iters': ramp_iters}
    if include_start:
        config['games_per_iter_start'] = 0
    assert _first_iteration_games_need(config) == 440
