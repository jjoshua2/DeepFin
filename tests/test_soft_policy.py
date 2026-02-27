import numpy as np

from chess_anti_engine.selfplay.manager import _apply_temperature


def test_apply_temperature_identity():
    p = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    q = _apply_temperature(p, 1.0)
    assert np.allclose(p, q)


def test_apply_temperature_softens():
    p = np.array([0.01, 0.01, 0.98], dtype=np.float32)
    q = _apply_temperature(p, 2.0)
    assert q[2] < p[2]
    assert np.isclose(float(q.sum()), 1.0, atol=1e-6)
