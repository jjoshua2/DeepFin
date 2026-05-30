from __future__ import annotations

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths, local_shard_path, save_local_shard_arrays
from chess_anti_engine.tune.trainable_init import _seed_replay_from_shared_shards
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig


def _arrays(policy_size: int) -> dict[str, np.ndarray]:
    policy = np.zeros((2, policy_size), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((2,), dtype=np.int8),
        "priority": np.ones((2,), dtype=np.float32),
        "has_policy": np.ones((2,), dtype=np.uint8),
    }


def test_seed_replay_from_shared_shards_skips_policy_width_mismatch(tmp_path) -> None:
    shared = tmp_path / "shared"
    replay = tmp_path / "trial" / "replay"
    save_local_shard_arrays(local_shard_path(shared, 0), arrs=_arrays(4672), meta={"positions": 2})
    save_local_shard_arrays(local_shard_path(shared, 1), arrs=_arrays(1858), meta={"positions": 2})
    tc = TrialConfig(shared_shards_dir=str(shared), policy_encoding="lc0_1858")

    copied = _seed_replay_from_shared_shards(
        tc=tc,
        restore=RestoreResult(),
        replay_shard_dir=replay,
    )

    paths = iter_shard_paths(replay)
    assert copied == 1
    assert [path.name for path in paths] == [local_shard_path(shared, 1).name]
