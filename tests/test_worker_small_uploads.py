from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import arrays_to_samples, load_shard_arrays
from chess_anti_engine.worker_buffer import (
    _buffer_add_completed_game,
    _buffer_should_flush,
    _BufferedUpload,
    _flush_upload_buffer_to_pending,
    _maybe_flush_upload_buffer,
)


def _sample(policy_size: int = 4672) -> ReplaySample:
    x = np.zeros((146, 8, 8), dtype=np.float32)
    pol = np.zeros((policy_size,), dtype=np.float32)
    pol[0] = 1.0
    return ReplaySample(x=x, policy_target=pol, wdl_target=1)


def _game_batch(positions: int, *, input_history_encoding: str = "legacy") -> SimpleNamespace:
    return SimpleNamespace(
        samples=[_sample() for _ in range(positions)],
        input_history_encoding=input_history_encoding,
        games=1,
        positions=positions,
        w=1,
        d=0,
        l=0,
        total_game_plies=24,
        adjudicated_games=0,
        total_draw_games=0,
        selfplay_games=0,
        selfplay_adjudicated_games=0,
        selfplay_draw_games=0,
        curriculum_games=1,
        curriculum_adjudicated_games=0,
        curriculum_draw_games=0,
    )


def test_worker_buffer_flushes_on_position_target(tmp_path) -> None:
    buf = _BufferedUpload()
    first = _game_batch(2)
    first.outcome_stats = {
        "opening_book2_games": 1,
        "curriculum_book2_net_white_w": 1,
    }
    _buffer_add_completed_game(
        buf=buf,
        game_batch=first,
        now_s=100.0,
        model_sha="abc123",
        model_step=7,
    )
    assert not _buffer_should_flush(
        buf=buf,
        now_s=101.0,
        last_send_s=100.0,
        target_positions=4,
        flush_seconds=60.0,
    )

    second = _game_batch(2)
    second.outcome_stats = {
        "opening_book2_games": 1,
        "curriculum_book2_net_black_w": 1,
    }
    _buffer_add_completed_game(
        buf=buf,
        game_batch=second,
        now_s=102.0,
        model_sha="abc123",
        model_step=7,
    )
    assert _buffer_should_flush(
        buf=buf,
        now_s=102.0,
        last_send_s=100.0,
        target_positions=4,
        flush_seconds=60.0,
    )

    shard_path, elapsed_s = _flush_upload_buffer_to_pending(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=102.0,
    )

    assert shard_path is not None
    assert elapsed_s == 2.0
    _arrs, meta = load_shard_arrays(shard_path)
    samples = arrays_to_samples(_arrs)
    assert len(samples) == 4
    assert meta.get("run_id") is None
    assert meta["games"] == 2
    assert meta["positions"] == 4
    assert meta["wins"] == 2
    assert meta["curriculum_games"] == 2
    assert meta["outcome_stats"] == {
        "opening_book2_games": 2,
        "curriculum_book2_net_white_w": 1,
        "curriculum_book2_net_black_w": 1,
    }
    assert meta["diff_focus_records"] == 0
    assert meta["diff_focus_priority_min"] == 0.0
    assert meta["diff_focus_priority_max"] == 0.0
    assert not buf.samples
    assert buf.positions == 0


def test_worker_buffer_tags_pending_shard_with_input_history_encoding(tmp_path) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2, input_history_encoding="lc0_root_legacy_meta"),
        now_s=100.0,
        model_sha="abc123",
        model_step=7,
    )

    shard_path, _elapsed_s = _flush_upload_buffer_to_pending(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=101.0,
    )

    assert shard_path is not None
    _arrs, meta = load_shard_arrays(shard_path)
    assert meta["input_history_encoding"] == "lc0_root_legacy_meta"
    assert meta["positions"] == 2


def test_worker_buffer_preserves_diff_focus_metadata(tmp_path) -> None:
    buf = _BufferedUpload()
    first = _game_batch(2)
    first.diff_focus_records = 3
    first.diff_focus_kept = 2
    first.diff_focus_keep_prob_sum = 1.5
    first.diff_focus_keep_limited = 2
    first.diff_focus_sample_weight_sum = 2.5
    first.diff_focus_sample_weight_limited = 1
    first.diff_focus_priority_sum = 9.0
    first.diff_focus_priority_sq_sum = 35.0
    first.diff_focus_priority_min = 1.0
    first.diff_focus_priority_max = 5.0
    second = _game_batch(1)
    second.diff_focus_records = 2
    second.diff_focus_kept = 1
    second.diff_focus_keep_prob_sum = 1.0
    second.diff_focus_priority_sum = 8.0
    second.diff_focus_priority_sq_sum = 40.0
    second.diff_focus_priority_min = 3.0
    second.diff_focus_priority_max = 6.0

    _buffer_add_completed_game(
        buf=buf,
        game_batch=first,
        now_s=100.0,
        model_sha="abc123",
        model_step=7,
    )
    _buffer_add_completed_game(
        buf=buf,
        game_batch=second,
        now_s=101.0,
        model_sha="abc123",
        model_step=7,
    )
    shard_path, _elapsed_s = _flush_upload_buffer_to_pending(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=102.0,
    )

    assert shard_path is not None
    _arrs, meta = load_shard_arrays(shard_path)
    assert meta["diff_focus_records"] == 5
    assert meta["diff_focus_kept"] == 3
    assert meta["diff_focus_keep_prob_sum"] == 2.5
    assert meta["diff_focus_keep_limited"] == 2
    assert meta["diff_focus_sample_weight_sum"] == 2.5
    assert meta["diff_focus_sample_weight_limited"] == 1
    assert meta["diff_focus_priority_sum"] == 17.0
    assert meta["diff_focus_priority_sq_sum"] == 75.0
    assert meta["diff_focus_priority_min"] == 1.0
    assert meta["diff_focus_priority_max"] == 6.0


def test_worker_buffer_flushes_on_send_age_even_if_small(tmp_path) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(1),
        now_s=200.0,
        model_sha="def456",
        model_step=9,
    )
    assert _buffer_should_flush(
        buf=buf,
        now_s=261.0,
        last_send_s=200.0,
        target_positions=500,
        flush_seconds=60.0,
    )

    shard_path, elapsed_s = _flush_upload_buffer_to_pending(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=261.0,
    )
    assert shard_path is not None
    assert elapsed_s == 61.0


def test_worker_buffer_maybe_flushes_and_resets(tmp_path) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(3),
        now_s=300.0,
        model_sha="abc999",
        model_step=11,
    )

    shard_path, elapsed_s = _maybe_flush_upload_buffer(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=361.0,
        last_send_s=300.0,
        target_positions=500,
        flush_seconds=60.0,
        force=False,
    )

    assert shard_path is not None
    assert elapsed_s == 61.0
    _arrs, meta = load_shard_arrays(shard_path)
    samples = arrays_to_samples(_arrs)
    assert len(samples) == 3
    assert meta["positions"] == 3
    assert buf.positions == 0
    assert not buf.samples


def test_worker_buffer_force_flushes_even_below_threshold(tmp_path) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
        now_s=400.0,
        model_sha="force123",
        model_step=12,
    )

    shard_path, elapsed_s = _maybe_flush_upload_buffer(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=401.0,
        last_send_s=400.0,
        target_positions=500,
        flush_seconds=60.0,
        force=True,
    )

    assert shard_path is not None
    assert elapsed_s == 1.0


def test_worker_buffer_drops_new_games_above_max_positions_cap(caplog) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(3),
        now_s=100.0,
        model_sha="cap123",
        model_step=5,
        max_positions=5,
    )
    assert buf.positions == 3

    with caplog.at_level("WARNING"):
        _buffer_add_completed_game(
            buf=buf,
            game_batch=_game_batch(3),
            now_s=101.0,
            model_sha="cap123",
            model_step=5,
            max_positions=5,
        )

    assert buf.positions == 3
    assert len(buf.samples) == 3
    assert any("dropping" in rec.message for rec in caplog.records)


def test_worker_buffer_no_cap_when_max_positions_is_zero() -> None:
    buf = _BufferedUpload()
    for _ in range(5):
        _buffer_add_completed_game(
            buf=buf,
            game_batch=_game_batch(10),
            now_s=100.0,
            model_sha="nocap",
            model_step=7,
            max_positions=0,
        )
    assert buf.positions == 50


def test_worker_buffer_preserves_original_model_metadata_across_retries(tmp_path) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
        now_s=500.0,
        model_sha="oldmodel",
        model_step=21,
    )

    shard_path, _ = _flush_upload_buffer_to_pending(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=501.0,
    )

    assert shard_path is not None
    _, meta = load_shard_arrays(shard_path)
    assert meta["model_sha256"] == "oldmodel"
    assert meta["model_step"] == 21


def test_worker_pending_shard_names_are_unique_with_same_metadata_and_time(tmp_path) -> None:
    paths = []
    for _ in range(2):
        buf = _BufferedUpload()
        _buffer_add_completed_game(
            buf=buf,
            game_batch=_game_batch(2),
            now_s=500.0,
            model_sha="same-model",
            model_step=21,
        )
        shard_path, _ = _flush_upload_buffer_to_pending(
            pending_dir=tmp_path,
            username="worker",
            buf=buf,
            now_s=501.0,
        )
        assert shard_path is not None
        paths.append(shard_path)

    assert paths[0] != paths[1]
    assert all(path.exists() for path in paths)


def test_worker_pending_sidecar_failure_keeps_buffer_and_removes_shard(
    monkeypatch, tmp_path,
) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
        now_s=500.0,
        model_sha="model",
        model_step=1,
    )
    original_write_text = Path.write_text

    def _fail_elapsed_write(path, *args, **kwargs):
        if path.name.endswith(".elapsed_s"):
            raise OSError("sidecar write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fail_elapsed_write)

    with pytest.raises(OSError, match="sidecar write failed"):
        _flush_upload_buffer_to_pending(
            pending_dir=tmp_path,
            username="worker",
            buf=buf,
            now_s=501.0,
        )

    assert buf.positions == 2
    assert len(buf.samples) == 2
    assert not list(tmp_path.glob("*.zarr"))
    assert not list(tmp_path.glob("*.elapsed_s"))


def test_worker_buffer_cap_counts_detached_positions() -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
        now_s=500.0,
        model_sha="model",
        model_step=1,
        max_positions=5,
        buffered_positions_offset=4,
    )
    assert buf.positions == 0

def test_worker_buffer_tags_pending_shards_with_trial_id(tmp_path) -> None:
    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
        now_s=600.0,
        model_sha="trialmodel",
        model_step=22,
    )

    shard_path, _ = _flush_upload_buffer_to_pending(
        pending_dir=tmp_path,
        username="worker",
        buf=buf,
        now_s=601.0,
        trial_id="trial_a",
    )

    assert shard_path is not None
    _, meta = load_shard_arrays(shard_path)
    assert meta["run_id"] == "trial_a"
