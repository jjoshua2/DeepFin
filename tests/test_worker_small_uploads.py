from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


def _game_batch(positions: int) -> SimpleNamespace:
    return SimpleNamespace(
        samples=[_sample() for _ in range(positions)],
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
    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
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

    _buffer_add_completed_game(
        buf=buf,
        game_batch=_game_batch(2),
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
    assert not buf.samples
    assert buf.positions == 0


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
