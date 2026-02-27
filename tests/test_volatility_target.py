import numpy as np


def test_volatility_abs_diff():
    w0 = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    w6 = np.array([0.3, 0.4, 0.3], dtype=np.float32)
    vol = np.abs(w6 - w0)
    assert np.allclose(vol, np.array([0.1, 0.1, 0.0], dtype=np.float32))
