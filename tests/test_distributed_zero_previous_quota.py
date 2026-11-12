from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, save_local_shard_arrays
from chess_anti_engine.tune import distributed_runtime as runtime


def _write_shard(path: Path, *, model_sha: str, games: int):
    policy = np.zeros((games, POLICY_SIZE), dtype=np.float32)
    policy[:, 0] = 1
    arrs = {
        "x": np.zeros((games, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.ones((games,), dtype=np.int8),
        "priority": np.ones((games,), dtype=np.float32),
        "has_policy": np.ones((games,), dtype=np.uint8),
    }
    meta = {"model_sha256": model_sha, "games": games, "positions": games, "wins": games}
    save_local_shard_arrays(path, arrs=arrs, meta=meta)
    return arrs, meta


def _ingest(tmp_path: Path, *, accepted: set[str], previous: str | None, fraction: float, **kwargs):
    buf = DiskReplayBuffer(
        256, shard_dir=tmp_path / "replay", rng=np.random.default_rng(0),
        shuffle_cap=64, shard_size=128, read_only=False,
    )
    summary = runtime._ingest_distributed_selfplay(
        buf=buf,
        holdout_buf=ArrayReplayBuffer(32, rng=np.random.default_rng(1)),
        holdout_frac=0.0, holdout_frozen=False,
        inbox_dir=tmp_path / "inbox", processed_dir=tmp_path / "processed",
        target_games=64, accepted_model_shas=accepted,
        prev_model_sha=previous, prev_model_max_fraction=fraction,
        wait_timeout_s=5.0, poll_seconds=0.001, min_games_fraction=1.0,
        rng=np.random.default_rng(2), on_poll=None, **kwargs,
    )
    return summary, buf


@pytest.mark.parametrize("delivery", ["prefetch", "poll"])
def test_zero_previous_quota_waits_for_64_current_games(tmp_path, monkeypatch, delivery):
    """One late old game plus63 new games cannot finish a fresh64-game quota."""
    inbox = tmp_path / "inbox" / "worker_00"
    previous = inbox / f"00_previous{LOCAL_SHARD_SUFFIX}"
    current63 = inbox / f"01_current63{LOCAL_SHARD_SUFFIX}"
    current64 = inbox / f"02_current64{LOCAL_SHARD_SUFFIX}"
    old_arrays, old_meta = _write_shard(previous, model_sha="previous", games=1)
    new_arrays, new_meta = _write_shard(current63, model_sha="current", games=63)
    real_scan = runtime._iter_shard_paths_nested
    scans = []

    def scan(path):
        scans.append(path)
        if delivery == "prefetch" or len(scans) == 2:
            # Both initial shards must have been ingested before the last
            # current-model game arrives. The old row stays useful replay.
            assert not previous.exists()
            assert not current63.exists()
            _write_shard(current64, model_sha="current", games=1)
        return real_scan(path)

    class Prefetcher:
        def drain(self):
            return [(previous, old_arrays, old_meta), (current63, new_arrays, new_meta)]

    monkeypatch.setattr(runtime, "_iter_shard_paths_nested", scan)
    accepted = {"current", "previous"}
    summary, buf = _ingest(
        tmp_path, accepted=accepted, previous="previous", fraction=0.0,
        prefetcher=Prefetcher() if delivery == "prefetch" else None,
    )
    assert len(scans) == (1 if delivery == "prefetch" else 2)
    assert (tmp_path / "processed" / "worker_00" / current64.name).exists()
    assert summary["matching_games"] == 64
    assert summary["gate_cur_w"] == 64
    assert summary["gate_prev_w"] == 0
    assert summary["stale_games"] == 1
    assert summary["positions_replay_added"] == 65
    assert len(buf) == 65
    assert accepted == {"current", "previous"}


@pytest.mark.parametrize("previous", [None, "current"])
def test_zero_quota_keeps_single_current_sha_at_startup(tmp_path, previous):
    _write_shard(tmp_path / "inbox" / "worker_00" / f"current{LOCAL_SHARD_SUFFIX}", model_sha="current", games=64)
    accepted = {"current"}
    summary, buf = _ingest(tmp_path, accepted=accepted, previous=previous, fraction=0.0)
    assert summary["matching_games"] == 64
    assert summary["stale_games"] == 0
    assert len(buf) == 64
    assert accepted == {"current"}


@pytest.mark.parametrize(("fraction", "previous_games"), [(0.01, 2), (0.1, 8)])
def test_positive_previous_quota_preserves_whole_shard_overshoot(tmp_path, fraction, previous_games):
    inbox = tmp_path / "inbox" / "worker_00"
    # The first previous shard exceeds the positive cap but remains accepted
    # in full. Later previous data is stale and still enters replay.
    _write_shard(inbox / f"00_previous{LOCAL_SHARD_SUFFIX}", model_sha="previous", games=previous_games)
    _write_shard(inbox / f"01_previous_late{LOCAL_SHARD_SUFFIX}", model_sha="previous", games=1)
    _write_shard(inbox / f"02_current{LOCAL_SHARD_SUFFIX}", model_sha="current", games=64 - previous_games)
    accepted = {"current", "previous"}
    summary, buf = _ingest(tmp_path, accepted=accepted, previous="previous", fraction=fraction)
    assert summary["matching_games"] == 64
    assert summary["gate_prev_w"] == previous_games
    assert summary["gate_cur_w"] == 64 - previous_games
    assert summary["stale_games"] == 1
    assert len(buf) == 65
    assert accepted == {"current", "previous"}
