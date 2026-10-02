from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.moves import policy_size_for_encoding
from chess_anti_engine.replay.shard import iter_shard_paths, local_shard_path, save_local_shard_arrays
from chess_anti_engine.tune.trainable_init import (
    _init_replay_buffers,
    _restore_checkpoint_or_salvage,
    _seed_replay_from_shared_shards,
)
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig


def _arrays(policy_size: int, *, x_planes: int = 146) -> dict[str, np.ndarray]:
    policy = np.zeros((2, policy_size), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((2, x_planes, 8, 8), dtype=np.float32),
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


class _LocalCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path

    def to_directory(self) -> str:
        return str(self.path)


class _TrainerLoadProbe:
    def __init__(self) -> None:
        self.loaded: Path | None = None

    def load(self, checkpoint: Path) -> None:
        self.loaded = checkpoint


def test_checkpoint_restore_reaches_replay_buffer_without_advancing_rng_or_pruning(tmp_path) -> None:
    """Exercise learner sidecar restore through the real disk-buffer constructor."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer.pt").write_bytes(b"synthetic trainer checkpoint")
    saved_rng = np.random.default_rng(41)
    saved_rng_state = saved_rng.bit_generator.state
    (checkpoint_dir / "rng_state.json").write_text(json.dumps(saved_rng_state), encoding="utf-8")
    (checkpoint_dir / "trial_meta.json").write_text(
        json.dumps({"owner_trial_id": "trial-a", "optimizer": "aurora"}),
        encoding="utf-8",
    )

    trial_dir = tmp_path / "trial-a"
    work_dir = tmp_path / "work"
    trial_dir.mkdir()
    work_dir.mkdir()
    trainer = _TrainerLoadProbe()
    rng = np.random.default_rng(999)
    restore, rng = _restore_checkpoint_or_salvage(
        config={"optimizer": "aurora"},
        trainer=trainer,
        device="cpu",
        trial_id="trial-a",
        trial_dir=trial_dir,
        base_seed=123,
        active_seed=999,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    assert trainer.loaded == checkpoint_dir / "trainer.pt"
    assert restore.startup_source == "checkpoint"
    assert not restore.cross_trial_restore
    assert restore.restored_window == 0
    assert restore.sampling_rng_restored
    assert rng.bit_generator.state == saved_rng_state

    tc = TrialConfig(
        replay_window_start=2,
        replay_window_max=20,
        shard_size=2,
        policy_encoding="lc0_1858",
    )
    replay_dir = trial_dir / "replay_shards"
    arrays = _arrays(
        policy_size_for_encoding(tc.policy_encoding),
        x_planes=input_plane_count(tc.input_extra_features),
    )
    for index in range(2):
        save_local_shard_arrays(
            local_shard_path(replay_dir, index),
            arrs=arrays,
            meta={"positions": 2},
        )

    buf, _, current_window, _, _ = _init_replay_buffers(
        tc=tc,
        config={"optimizer": "aurora"},
        restore=restore,
        trial_dir=trial_dir,
        work_dir=work_dir,
        rng=rng,
        ckpt=_LocalCheckpoint(checkpoint_dir),
    )
    try:
        assert len(iter_shard_paths(replay_dir)) == 2
        assert len(buf) == 4
        assert current_window == buf.capacity == 4
        assert buf.capacity <= tc.replay_window_max
        assert rng.bit_generator.state == saved_rng_state
    finally:
        buf.close()
