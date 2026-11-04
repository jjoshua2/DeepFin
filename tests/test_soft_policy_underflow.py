"""Finite positive temperature preserves nondegenerate mass under default NumPy mode."""
import unittest

import numpy as np
import pytest

from chess_anti_engine.selfplay.temperature import apply_policy_temperature


class SoftPolicyUnderflow(unittest.TestCase):
    def test_tiny_temperature_retains_a_maximum_and_zero_support(self):
        p = np.zeros(1858, dtype=np.float32)
        p[:3] = [.4, .3, .3]
        q = apply_policy_temperature(p, .001)
        assert q.dtype == np.float32
        assert q[0] == 1.0
        assert np.count_nonzero(q[1:]) == 0
        assert q.sum() == 1.0

    def test_tiny_temperature_preserves_tied_maxima(self):
        p = np.array([.4, .4, .2, 0.], dtype=np.float32)
        np.testing.assert_array_equal(apply_policy_temperature(p, .001), [.5, .5, 0., 0.])

    def test_subnormal_unnormalized_powers_preserve_ratios(self):
        p = np.array([.5, .4999], dtype=np.float64)
        t = .00094
        expected = (p / p.max()) ** (1 / t)
        expected /= expected.sum()
        np.testing.assert_allclose(apply_policy_temperature(p, t), expected, rtol=1e-6)

    def test_normal_range_keeps_existing_arithmetic(self):
        p = np.array([.4, .3, .3], dtype=np.float32)
        for t in (.01, .5, 2., 3.):
            with self.subTest(t=t):
                expected = p.astype(np.float64)
                expected /= expected.sum()
                expected = expected ** (1 / t)
                expected /= expected.sum()
                np.testing.assert_array_equal(
                    apply_policy_temperature(p, t), expected.astype(np.float32),
                )

    def test_existing_identity_and_degenerate_input(self):
        p = np.array([.2, .8], dtype=np.float32)
        assert apply_policy_temperature(p, 1.) is p
        z = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(apply_policy_temperature(z, 2.), z)

    def test_zero_temperature_keeps_existing_exception(self):
        with pytest.raises(ZeroDivisionError):
            apply_policy_temperature(np.array([.4, .6], dtype=np.float32), 0.)
