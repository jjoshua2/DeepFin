import copy

import numpy as np
import pytest

from scripts.bootstrap_factorial_targets import mixed_targets, validate_base


def test_preserves_pretempered_bt4_and_calibrated_thirds():
    # BT4's stored 80/20 must remain 80/20, not become 94/6 by double sharpening.
    legal = np.array([[1, 1, 0]], dtype=np.uint8)
    p, w = mixed_targets(np.array([[.8, .2, 0]]), np.array([[.75, .25, 0]]),
                         legal, np.array([[0., 0., 100.]]),
                         np.zeros((1, 3)), np.zeros((1, 3)))
    np.testing.assert_allclose(p, [[.65, .35, 0]], atol=.0003)
    np.testing.assert_allclose(w, [[.5 + 1/9, 1/6 + 1/9, 1/9]], atol=.0003)


def test_illegal_ceres_mass_does_not_leak_and_value_heads_are_distinct():
    p, w = mixed_targets(np.array([[1., 0., 0.]]), np.array([[0., 0., 1.]]),
                         np.array([[1, 0, 1]]), np.array([[0., 1000., 0.]]),
                         np.array([[100., 0., 0.]]), np.array([[0., 100., 0.]]))
    np.testing.assert_allclose(p, [[.75, 0., .25]], atol=.0003)
    np.testing.assert_allclose(w, [[.2, .4/3, 2/3]], atol=.0003)


def test_rejects_non_v50_or_wrong_policy_temperature():
    s = {'policy_target_postprocess': {'kind': 'global', 'alpha': 1., 'bt4_temperature': .5},
         'value_target_postprocess': {'status': 'COMPLETE', 'kind': 'bt4_value_rewrite',
             'bt4_weight': .5, 'sf_weight': .5, 'mutated_arrays': ['search_wdl']}}
    validate_base(s)
    for section, key, bad in [('policy_target_postprocess', 'bt4_temperature', 1.),
                              ('value_target_postprocess', 'bt4_weight', 1.)]:
        other = copy.deepcopy(s)
        other[section][key] = bad
        with pytest.raises(ValueError):
            validate_base(other)
