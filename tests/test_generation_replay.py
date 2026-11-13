from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay.generation import GenerationChunk, GenerationReplay
from chess_anti_engine.train.trainer import Trainer


def _chunk(number: int, count: int) -> GenerationChunk:
    x = np.zeros((count, 146, 8, 8), dtype=np.float16)
    x[:, 0, 0, 0] = number * 100 + np.arange(count)
    policy = np.zeros((count, POLICY_SIZE), dtype=np.float16)
    policy[:, 0] = 1
    return GenerationChunk(f"{number:064x}", {
        "x": x, "policy_target": policy,
        "wdl_target": np.zeros(count, dtype=np.int8),
        "priority": np.ones(count, dtype=np.float32),
        "has_policy": np.ones(count, dtype=np.uint8),
    })


def test_whole_generation_fifo_and_draw_lineage_preserve_rng() -> None:
    rng = np.random.default_rng(123)
    draws = []
    replay = GenerationReplay(generations=4, rng=rng, on_draw=draws.append)
    before = rng.bit_generator.state
    counts = [3, 2, 7, 1, 9]
    for generation, count in enumerate(counts, 1):
        replay.admit_generation(generation, [_chunk(generation, count)])
        assert rng.bit_generator.state == before
    assert replay.rng is rng
    assert replay.membership == tuple((g, f"{g:064x}", counts[g - 1]) for g in range(2, 6))
    assert len(replay) == sum(counts[1:])
    ids = np.concatenate([g * 100 + np.arange(counts[g - 1]) for g in range(2, 6)])
    expected = np.random.default_rng(123).choice(len(ids), size=128, replace=True)
    batch = replay.sample_batch_arrays(128)
    np.testing.assert_array_equal(batch["x"][:, 0, 0, 0], ids[expected])
    assert [r.generation * 100 + r.row_index for r in draws[0]] == ids[expected].tolist()
    assert all(r.source_sha256 == f"{r.generation:064x}" for r in draws[0])


def test_real_trainer_sampling_ignores_priorities_and_result_classes(tmp_path) -> None:
    trainer = Trainer(
        torch.nn.Linear(1, 1), device="cpu", lr=1e-3, use_amp=False,
        log_dir=tmp_path, prefetch_batches=False,
    )
    sampled = []
    for changed in (False, True):
        chunk = _chunk(1, 11)
        if changed:
            chunk.arrays["priority"][0] = 1e20
            chunk.arrays["wdl_target"][:] = [0] * 9 + [1, 2]
        seen = []
        replay = GenerationReplay(generations=4, rng=np.random.default_rng(42), on_draw=seen.append)
        replay.admit_generation(1, [chunk])
        batch = trainer._sample_batch_host(cast(Any, replay), batch_size=128, mirror_prob=0.0)
        assert isinstance(batch, dict)
        sampled.append(batch["x"][:, 0, 0, 0])
        assert [r.row_index for r in seen[0]] == np.random.default_rng(42).choice(11, size=128).tolist()
    np.testing.assert_array_equal(sampled[0], sampled[1])
    trainer.writer.close()


def test_failed_refill_keeps_window_rng_and_admission_cursor() -> None:
    replay = GenerationReplay(generations=1, rng=np.random.default_rng(7))
    replay.admit_generation(1, [_chunk(1, 3)])
    before = replay.rng.bit_generator.state
    broken = _chunk(2, 2)
    del broken.arrays["policy_target"]
    with pytest.raises(ValueError, match="missing required fields"):
        replay.admit_generation(2, [broken])
    assert replay.membership == ((1, f"{1:064x}", 3),)
    assert replay.rng.bit_generator.state == before
    replay.admit_generation(2, [_chunk(2, 2)])
    assert len(replay) == 2
    # Eviction does not permit ingestion of the same physical source again.
    with pytest.raises(ValueError, match="already admitted"):
        replay.admit_generation(3, [_chunk(1, 3)])


def test_multiple_chunks_map_local_physical_indices() -> None:
    seen = []
    replay = GenerationReplay(generations=1, rng=np.random.default_rng(3), on_draw=seen.append)
    replay.admit_generation(1, [_chunk(10, 2), _chunk(20, 3)])
    batch = replay.sample_batch_arrays(16)
    markers = [int(r.source_sha256, 16) * 100 + r.row_index for r in seen[0]]
    np.testing.assert_array_equal(batch["x"][:, 0, 0, 0], markers)
    assert all(r.generation == 1 for r in seen[0])


@pytest.mark.parametrize("number", [0, -1, True, 1.5])
def test_invalid_window_rejected(number) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GenerationReplay(generations=number, rng=np.random.default_rng(1))


def test_invalid_admission_and_draw_are_refused() -> None:
    replay = GenerationReplay(generations=4, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="empty"):
        replay.sample_batch_arrays(1)
    with pytest.raises(ValueError, match="consecutive"):
        replay.admit_generation(2, [_chunk(1, 1)])
    with pytest.raises(ValueError, match="nonempty"):
        replay.admit_generation(1, [])
    with pytest.raises(ValueError, match="contain rows"):
        replay.admit_generation(1, [_chunk(1, 0)])
    with pytest.raises(ValueError, match="SHA256"):
        replay.admit_generation(1, [GenerationChunk("wrong", _chunk(1, 1).arrays)])
    with pytest.raises(ValueError, match="already admitted"):
        replay.admit_generation(1, [_chunk(1, 1), _chunk(1, 1)])
    with pytest.raises(ValueError, match="positive integer"):
        replay.sample_batch_arrays(0)
