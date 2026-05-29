from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    ShardMeta,
    save_local_shard_arrays,
    samples_to_arrays,
)
from scripts import diagnose, diagnose_arch
from scripts.diagnose_wdl_by_age import _concat_batches as _concat_wdl_age_batches
from scripts.diagnose_wdl_by_age import _take_rows as _take_wdl_age_rows
from scripts.diagnose_replay import sample_replay_arrays


def _touch_shard(path: Path) -> None:
    path.mkdir(parents=True)


def _sample(i: int = 0) -> ReplaySample:
    policy = np.zeros((4672,), dtype=np.float32)
    policy[i] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
    )


def test_diagnose_replay_dir_prefers_trainer_replay_shards(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"
    replay_dir = trial_dir / "replay_shards"
    legacy_dir = trial_dir / "selfplay_shards"
    _touch_shard(replay_dir / "shard_000001.zarr")
    _touch_shard(legacy_dir / "shard_000001.zarr")

    resolved = diagnose._resolve_replay_dir(
        Namespace(replay_dir=None),
        cfg={},
        trial_dir=trial_dir,
    )

    assert resolved == replay_dir.resolve()


def test_diagnose_replay_dir_falls_back_to_selfplay_exports(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"
    legacy_dir = trial_dir / "selfplay_shards"
    _touch_shard(legacy_dir / "shard_000001.zarr")

    resolved = diagnose_arch._resolve_replay_dir(
        Namespace(replay_dir=None),
        cfg={},
        trial_dir=trial_dir,
    )

    assert resolved == legacy_dir.resolve()


def test_diagnose_replay_dir_errors_with_checked_paths(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"

    with pytest.raises(SystemExit) as exc:
        diagnose._resolve_replay_dir(
            Namespace(replay_dir=None),
            cfg={},
            trial_dir=trial_dir,
        )

    msg = str(exc.value)
    assert "No replay shards found" in msg
    assert "replay_shards" in msg
    assert "selfplay_shards" in msg


def test_diagnose_replay_dir_accepts_explicit_replay_dir(tmp_path: Path) -> None:
    explicit = tmp_path / "custom"
    explicit.mkdir()

    resolved = diagnose._resolve_replay_dir(
        Namespace(replay_dir=str(explicit)),
        cfg={},
        trial_dir=tmp_path / "ignored_trial",
    )

    assert resolved == explicit.resolve()


def test_sample_replay_arrays_broadcasts_scalar_history_metadata(tmp_path: Path) -> None:
    shard_dir = tmp_path / "replay"
    shard_dir.mkdir()
    save_local_shard_arrays(
        shard_dir / "shard_000001.zarr",
        arrs=samples_to_arrays([_sample(0), _sample(1)]),
        meta=ShardMeta(input_history_encoding="lc0_root", positions=2),
    )

    arrs, total, _shards = sample_replay_arrays(
        shard_dir,
        2,
        rng=np.random.default_rng(1),
        fields=("x", INPUT_HISTORY_ENCODING_ARRAY_KEY),
    )

    assert total == 2
    assert arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (2,)
    assert set(arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].astype(str).tolist()) == {"lc0_root"}


def test_sample_replay_arrays_supplies_blank_missing_history_metadata(tmp_path: Path) -> None:
    shard_dir = tmp_path / "replay"
    shard_dir.mkdir()
    save_local_shard_arrays(
        shard_dir / "shard_000001.zarr",
        arrs=samples_to_arrays([_sample(0)]),
        meta=ShardMeta(positions=1),
    )

    arrs, total, _shards = sample_replay_arrays(
        shard_dir,
        1,
        rng=np.random.default_rng(1),
        fields=("x", INPUT_HISTORY_ENCODING_ARRAY_KEY),
    )

    assert total == 1
    assert arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (1,)
    assert arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].item() == ""


def test_wdl_age_sampler_broadcasts_history_metadata_per_shard() -> None:
    first = _take_wdl_age_rows(
        {
            "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
            INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray("lc0_root"),
        },
        np.asarray([0], dtype=np.int64),
    )
    second = _take_wdl_age_rows(
        {
            "x": np.ones((2, 146, 8, 8), dtype=np.float32),
            INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray(""),
        },
        np.asarray([1], dtype=np.int64),
    )

    out = _concat_wdl_age_batches([first, second])

    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (2,)
    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].astype(str).tolist() == ["lc0_root", ""]
