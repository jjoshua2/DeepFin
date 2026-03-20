import numpy as np

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    ShardMeta,
    iter_shard_paths,
    load_npz,
    load_npz_arrays,
    load_shard_arrays,
    local_shard_path,
    save_local_shard_arrays,
    save_npz,
    save_npz_arrays,
)


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


def test_save_npz_arrays_roundtrip(tmp_path):
    samples = [_sample(), _sample()]
    src = tmp_path / "src.npz"
    save_npz(src, samples=samples, meta=ShardMeta(username="alice", run_id="r1", positions=len(samples)))

    arrs, meta = load_npz_arrays(src)
    dst = tmp_path / "dst.npz"
    save_npz_arrays(dst, arrs=arrs, meta=meta)

    out, meta_out = load_npz(dst)
    assert len(out) == len(samples)
    assert meta_out["username"] == "alice"


def test_save_npz_arrays_uncompressed_roundtrip(tmp_path):
    samples = [_sample(), _sample()]
    src = tmp_path / "src_uncompressed.npz"
    save_npz(src, samples=samples, meta=ShardMeta(username="alice", run_id="r1", positions=len(samples)), compress=False)

    arrs, meta = load_npz_arrays(src)
    assert arrs["x"].shape[0] == len(samples)
    assert meta["username"] == "alice"


def test_local_zarr_shard_roundtrip(tmp_path):
    samples = [_sample(), _sample()]
    arrs = load_npz_arrays(save_npz(tmp_path / "seed.npz", samples=samples, meta={"positions": 2}))[0]
    path = local_shard_path(tmp_path / "replay", 3)
    out_path = save_local_shard_arrays(path, arrs=arrs, meta={"positions": 2, "username": "alice"})

    assert out_path.suffix == LOCAL_SHARD_SUFFIX
    assert out_path.exists()
    listed = iter_shard_paths(tmp_path / "replay")
    assert listed == [out_path]

    lazy_arrs, lazy_meta = load_shard_arrays(out_path, lazy=True)
    assert int(lazy_arrs["x"].shape[0]) == 2
    assert lazy_meta["username"] == "alice"

    eager_arrs, eager_meta = load_shard_arrays(out_path, lazy=False)
    assert eager_meta["positions"] == 2
    assert eager_arrs["policy_target"].shape == (2, 4672)


def test_save_prunes_unset_optional_arrays(tmp_path):
    policy = np.zeros((4672,), dtype=np.float32)
    policy[0] = 1.0
    sample = ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        priority=1.0,
        has_policy=True,
    )

    path = tmp_path / "minimal.npz"
    save_npz(path, samples=[sample], meta={"positions": 1})

    with np.load(path, allow_pickle=False) as z:
        assert "x" in z.files
        assert "policy_target" in z.files
        assert "wdl_target" in z.files
        assert "sf_wdl" not in z.files
        assert "has_sf_wdl" not in z.files
        assert "future_policy_target" not in z.files
        assert "has_future" not in z.files

    out, meta = load_npz(path)
    assert meta["positions"] == 1
    assert len(out) == 1
    assert out[0].sf_wdl is None
    assert out[0].future_policy_target is None
