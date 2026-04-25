from __future__ import annotations

from pathlib import Path

import numpy as np

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import arrays_to_samples, load_shard_arrays
from chess_anti_engine.server.app import (
    _buffered_upload_ready,
    _BufferedUploadAccumulator,
    _flush_buffered_upload_to_inbox,
)


def _load_compacted(path: Path) -> tuple[list[ReplaySample], dict]:
    arrs, meta = load_shard_arrays(path)
    return arrays_to_samples(arrs), meta


def _sample(policy_size: int = 4672) -> ReplaySample:
    x = np.zeros((146, 8, 8), dtype=np.float32)
    pol = np.zeros((policy_size,), dtype=np.float32)
    pol[0] = 1.0
    return ReplaySample(x=x, policy_target=pol, wdl_target=1)


def test_server_compacts_multiple_small_uploads_into_one_inbox_shard(tmp_path) -> None:
    inbox_root = tmp_path / "inbox"
    acc = _BufferedUploadAccumulator(
        trial_id="trial_0001",
        model_sha256="deadbeef",
        created_at_unix=100.0,
        last_update_unix=100.0,
    )

    acc.add_upload(
        samples=[_sample(), _sample()],
        meta={"games": 1, "positions": 2, "wins": 1, "curriculum_games": 1, "model_step": 11},
        now_unix=100.0,
    )
    assert not _buffered_upload_ready(
        acc=acc,
        now_unix=100.0,
        target_positions=4,
        max_age_s=90.0,
    )

    acc.add_upload(
        samples=[_sample(), _sample()],
        meta={"games": 1, "positions": 2, "wins": 1, "curriculum_games": 1, "model_step": 11},
        now_unix=101.0,
    )
    assert _buffered_upload_ready(
        acc=acc,
        now_unix=101.0,
        target_positions=4,
        max_age_s=90.0,
    )

    out = _flush_buffered_upload_to_inbox(
        inbox_root=inbox_root,
        acc=acc,
        now_unix=101.0,
        flush_token="abcd0123abcd0123",
    )
    assert out is not None
    assert out.parent == inbox_root / "_compacted"
    samples, meta = _load_compacted(out)
    assert len(samples) == 4
    assert meta["games"] == 2
    assert meta["positions"] == 4
    assert meta["wins"] == 2
    assert meta["curriculum_games"] == 2
    assert meta["model_sha256"] == "deadbeef"


def test_server_compactor_age_flushes_partial_buffer() -> None:
    acc = _BufferedUploadAccumulator(
        trial_id="trial_0002",
        model_sha256="cafebabe",
        created_at_unix=10.0,
        last_update_unix=10.0,
    )
    acc.add_upload(
        samples=[_sample()],
        meta={"games": 1, "positions": 1, "draws": 1, "total_draw_games": 1},
        now_unix=10.0,
    )
    assert _buffered_upload_ready(
        acc=acc,
        now_unix=101.0,
        target_positions=2000,
        max_age_s=90.0,
    )


def test_server_compactor_atomically_replaces_temp_path(tmp_path) -> None:
    inbox_root = tmp_path / "inbox"
    acc = _BufferedUploadAccumulator(
        trial_id="trial_0003",
        model_sha256="fadefeed",
        created_at_unix=50.0,
        last_update_unix=50.0,
    )
    acc.add_upload(
        samples=[_sample(), _sample()],
        meta={"games": 2, "positions": 2, "wins": 1, "losses": 1},
        now_unix=51.0,
    )

    out = _flush_buffered_upload_to_inbox(
        inbox_root=inbox_root,
        acc=acc,
        now_unix=52.0,
        flush_token="cafebabecafebabe",
    )

    assert out is not None
    assert out.exists()
    # save_local_shard_arrays uses "._<pid>_" prefix for its atomic tmp dir.
    assert not list((inbox_root / "_compacted").glob("._*"))
    samples, meta = _load_compacted(out)
    assert len(samples) == 2
    assert meta["positions"] == 2


# Removed test_server_compactor_falls_back_when_temp_rename_path_is_missing:
# it monkey-patched save_npz + os.replace to simulate a rename race in the old
# npz compaction path. The new path writes zarr via save_local_shard_arrays,
# which has its own internal atomic rename and surfaces rename failures for the
# server-level flush loop to retry on the next tick. The specific retry knob
# the old test asserted on no longer exists.
