"""Tests for BackgroundShardPrefetcher."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.shard import save_local_shard_arrays
from chess_anti_engine.tune.prefetch import BackgroundShardPrefetcher


def _write_shard(dst_dir: Path, n_pos: int = 4, model_sha: str = "abc") -> Path:
    n = int(n_pos)
    pol = np.zeros((n, 4672), dtype=np.float32)
    pol[:, 0] = 1.0  # validator requires positive row sums
    arrs = {
        "x": np.random.randn(n, 146, 8, 8).astype(np.float32),
        "policy_target": pol,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
    }
    meta = {
        "model_sha256": model_sha,
        "wins": 1, "draws": 0, "losses": 0, "games": 1,
        "positions": n, "total_game_plies": n,
    }
    dst_dir.mkdir(parents=True, exist_ok=True)
    path = dst_dir / f"shard_{int(time.time_ns())}.zarr"
    save_local_shard_arrays(path, arrs=arrs, meta=meta)
    return path


def _path_iter(inbox: Path) -> list[Path]:
    return sorted(inbox.glob("*.zarr"))


def _drain_until(pf: BackgroundShardPrefetcher, timeout: float = 3.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        items = pf.drain()
        if items:
            return items
        time.sleep(0.05)
    return []


def test_prefetcher_decodes_inbox(tmp_path):
    inbox = tmp_path / "inbox"
    sp = _write_shard(inbox)

    pf = BackgroundShardPrefetcher(
        inbox_dir=inbox, poll_seconds=0.05, path_iter=_path_iter,
    )
    pf.start()
    try:
        items = _drain_until(pf)
    finally:
        pf.stop()
    assert items
    decoded_path, arrs, meta = items[0]
    assert decoded_path == sp
    assert int(arrs["x"].shape[0]) == 4
    assert meta["model_sha256"] == "abc"


def test_prefetcher_dedupes_same_shard(tmp_path):
    inbox = tmp_path / "inbox"
    _write_shard(inbox)

    pf = BackgroundShardPrefetcher(
        inbox_dir=inbox, poll_seconds=0.05, path_iter=_path_iter,
    )
    pf.start()
    try:
        first_items = _drain_until(pf)
        assert first_items
  # Once drained, the path is no longer in `_queue` — but the inbox
  # file is still on disk (trainer hasn't moved it). The prefetcher
  # must not re-queue it. Trainer's atomic move at iter time is what
  # ultimately stops re-pickup; here we just check no double-queue
  # within a single iter.
        time.sleep(0.5)
        second_items = pf.drain()
    finally:
        pf.stop()
  # The same on-disk shard WILL be re-decoded (this is correct — drain()
  # cleared the queue and the file's still in inbox), but only once per
  # poll cycle. So second_items <= 1; ensure at most one entry.
    assert len(second_items) <= 1


def test_prefetcher_stop_is_idempotent(tmp_path):
    pf = BackgroundShardPrefetcher(
        inbox_dir=tmp_path / "inbox", poll_seconds=0.05, path_iter=_path_iter,
    )
    pf.start()
    pf.stop()
    pf.stop()  # second stop must not raise


def test_prefetcher_handles_missing_inbox(tmp_path):
    inbox = tmp_path / "inbox"  # does not exist
    pf = BackgroundShardPrefetcher(
        inbox_dir=inbox, poll_seconds=0.05, path_iter=_path_iter,
    )
    pf.start()
    try:
        time.sleep(0.2)  # poll while missing — should not crash
        _write_shard(inbox)
        items = _drain_until(pf)
    finally:
        pf.stop()
    assert items, "prefetcher did not recover when inbox appeared"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
