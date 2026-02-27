import numpy as np

from chess_anti_engine.train.targets import hlgauss_target


def test_hlgauss_sums_to_one():
    p = hlgauss_target(0.25, num_bins=32, sigma=0.04)
    assert p.shape == (32,)
    assert np.isclose(float(p.sum()), 1.0, atol=1e-5)
    assert np.all(p >= 0)


def test_hlgauss_moves_mass_with_value():
    p1 = hlgauss_target(-1.0, num_bins=32, sigma=0.04)
    p2 = hlgauss_target(1.0, num_bins=32, sigma=0.04)
    assert int(np.argmax(p1)) < int(np.argmax(p2))
