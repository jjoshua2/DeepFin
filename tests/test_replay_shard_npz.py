import numpy as np

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import ShardMeta, load_npz, save_npz


def _sample(policy_size: int = 4672) -> ReplaySample:
    x = np.zeros((146, 8, 8), dtype=np.float32)
    pol = np.zeros((policy_size,), dtype=np.float32)
    pol[0] = 1.0
    s = ReplaySample(x=x, policy_target=pol, wdl_target=1)
    s.priority = 2.0
    s.has_policy = True
    s.sf_wdl = np.array([0.2, 0.7, 0.1], dtype=np.float32)
    s.sf_move_index = 123
    s.moves_left = 0.5
    s.is_network_turn = True
    s.categorical_target = np.ones((32,), dtype=np.float32) / 32.0
    s.policy_soft_target = pol.copy()
    s.future_policy_target = pol.copy()
    s.volatility_target = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    s.sf_volatility_target = np.array([0.2, 0.1, 0.0], dtype=np.float32)
    s.has_future = True
    s.has_volatility = True
    s.has_sf_volatility = True
    return s


def test_shard_roundtrip(tmp_path):
    samples = [_sample(), _sample()]
    meta = ShardMeta(username="alice", run_id="r1", positions=len(samples))

    p = tmp_path / "shard.npz"
    save_npz(p, samples=samples, meta=meta)

    out, meta_out = load_npz(p)
    assert len(out) == len(samples)
    assert meta_out["username"] == "alice"

    s0 = out[0]
    assert s0.policy_target.shape[0] == 4672
    assert s0.sf_wdl is not None
    assert s0.sf_move_index is not None
    assert s0.moves_left is not None
    assert s0.is_network_turn is not None
    assert s0.categorical_target is not None
    assert s0.policy_soft_target is not None
    assert s0.future_policy_target is not None
    assert s0.volatility_target is not None
    assert s0.sf_volatility_target is not None


def test_shard_rejects_empty(tmp_path):
    p = tmp_path / "empty.npz"
    try:
        save_npz(p, samples=[], meta=None)
        assert False, "expected ValueError"
    except ValueError:
        pass
