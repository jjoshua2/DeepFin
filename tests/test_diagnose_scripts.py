from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import diagnose, diagnose_arch


def _touch_shard(path: Path) -> None:
    path.mkdir(parents=True)


def test_diagnose_replay_dir_prefers_trainer_replay_shards(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"
    replay_dir = trial_dir / "replay_shards"
    legacy_dir = trial_dir / "selfplay_shards"
    _touch_shard(replay_dir / "shard_000001.zarr")
    _touch_shard(legacy_dir / "shard_000001.zarr")

    resolved = diagnose._resolve_replay_dir(
        SimpleNamespace(replay_dir=None),
        cfg={},
        trial_dir=trial_dir,
    )

    assert resolved == replay_dir.resolve()


def test_diagnose_replay_dir_falls_back_to_selfplay_exports(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"
    legacy_dir = trial_dir / "selfplay_shards"
    _touch_shard(legacy_dir / "shard_000001.zarr")

    resolved = diagnose_arch._resolve_replay_dir(
        SimpleNamespace(replay_dir=None),
        cfg={},
        trial_dir=trial_dir,
    )

    assert resolved == legacy_dir.resolve()


def test_diagnose_replay_dir_errors_with_checked_paths(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"

    with pytest.raises(SystemExit) as exc:
        diagnose._resolve_replay_dir(
            SimpleNamespace(replay_dir=None),
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
        SimpleNamespace(replay_dir=str(explicit)),
        cfg={},
        trial_dir=tmp_path / "ignored_trial",
    )

    assert resolved == explicit.resolve()
