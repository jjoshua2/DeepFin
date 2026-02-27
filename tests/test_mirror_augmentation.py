import numpy as np

from chess_anti_engine.moves.encode import (
    POLICY_SIZE,
    MIRROR_POLICY_MAP,
    mirror_policy_index,
)
from chess_anti_engine.replay.augment import mirror_sample
from chess_anti_engine.replay.buffer import ReplaySample


def test_mirror_policy_map_is_permutation_and_involution():
    m = np.asarray(MIRROR_POLICY_MAP)
    assert m.shape == (POLICY_SIZE,)

    # Permutation check
    assert len(set(map(int, m.tolist()))) == POLICY_SIZE

    # Involution check
    for i in [0, 1, 2, 123, 4096, POLICY_SIZE - 1]:
        assert mirror_policy_index(mirror_policy_index(i)) == i

    # Spot-check all indices in a vectorized way
    mm = m[m]
    assert np.array_equal(mm, np.arange(POLICY_SIZE, dtype=mm.dtype))


def test_mirror_sample_is_involution():
    rng = np.random.default_rng(0)

    x = rng.normal(size=(18, 8, 8)).astype(np.float32)

    p = rng.random(size=(POLICY_SIZE,)).astype(np.float32)
    p /= float(p.sum())

    ps = rng.random(size=(POLICY_SIZE,)).astype(np.float32)
    ps /= float(ps.sum())

    fp = rng.random(size=(POLICY_SIZE,)).astype(np.float32)
    fp /= float(fp.sum())

    s = ReplaySample(
        x=x,
        policy_target=p,
        wdl_target=2,
        priority=1.7,
        has_policy=True,
        sf_wdl=np.array([0.2, 0.3, 0.5], dtype=np.float32),
        sf_move_index=int(rng.integers(0, POLICY_SIZE)),
        moves_left=0.25,
        is_network_turn=True,
        categorical_target=rng.random(size=(32,)).astype(np.float32),
        policy_soft_target=ps,
        future_policy_target=fp,
        has_future=True,
        volatility_target=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        has_volatility=True,
        sf_volatility_target=np.array([0.04, 0.05, 0.06], dtype=np.float32),
        has_sf_volatility=True,
    )

    s2 = mirror_sample(mirror_sample(s))

    assert np.array_equal(s2.x, s.x)
    assert np.allclose(s2.policy_target, s.policy_target)
    assert s2.wdl_target == s.wdl_target
    assert float(s2.priority) == float(s.priority)
    assert bool(s2.has_policy) == bool(s.has_policy)

    assert np.allclose(s2.sf_wdl, s.sf_wdl)
    assert int(s2.sf_move_index) == int(s.sf_move_index)
    assert float(s2.moves_left) == float(s.moves_left)
    assert bool(s2.is_network_turn) == bool(s.is_network_turn)

    assert np.allclose(s2.categorical_target, s.categorical_target)
    assert np.allclose(s2.policy_soft_target, s.policy_soft_target)
    assert np.allclose(s2.future_policy_target, s.future_policy_target)
    assert bool(s2.has_future) == bool(s.has_future)

    assert np.allclose(s2.volatility_target, s.volatility_target)
    assert bool(s2.has_volatility) == bool(s.has_volatility)

    assert np.allclose(s2.sf_volatility_target, s.sf_volatility_target)
    assert bool(s2.has_sf_volatility) == bool(s.has_sf_volatility)
