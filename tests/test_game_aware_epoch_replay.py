from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay import game_epoch as game_epoch_module
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
)


def _sample(*, game: int | None, row: int, planes: int = 146) -> ReplaySample:
    policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    policy[row % COMPACT_POLICY_SIZE] = 1.0
    legal = np.zeros_like(policy, dtype=np.uint8)
    legal[row % COMPACT_POLICY_SIZE] = 1
    return ReplaySample(
        x=np.full((planes, 8, 8), row % 17, dtype=np.float32),
        policy_target=policy,
        legal_mask=legal,
        wdl_target=row % 3,
        priority=float(row + 1),
        has_policy=True,
        game_id=game,
        ply_index=row,
    )


def _write(
    shard_dir: Path,
    shards: Sequence[Sequence[tuple[int | None, int]]],
    *,
    planes: int = 146,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(shards):
        samples = [_sample(game=game, row=row, planes=planes) for game, row in rows]
        save_local_shard_arrays(
            shard_dir / f"shard_{index:06d}.zarr",
            arrs=samples_to_arrays(samples),
            meta=ShardMeta(
                positions=len(samples),
                policy_encoding="lc0_1858",
                policy_size=COMPACT_POLICY_SIZE,
            ),
        )
    return shard_dir


def _open(
    shard_dir: Path,
    *,
    seed: int = 7,
    batch_size: int = 4,
    load_workers: int = 2,
    mirror_augmentation: bool = False,
    max_working_set_bytes: int | None = None,
) -> GameAwareEpochBuffer:
    kwargs = {}
    if max_working_set_bytes is not None:
        kwargs["max_working_set_bytes"] = int(max_working_set_bytes)
    return GameAwareEpochBuffer(
        shard_dir=shard_dir,
        batch_size=batch_size,
        seed=seed,
        input_planes=146,
        mirror_augmentation=mirror_augmentation,
        plan_workers=2,
        load_workers=load_workers,
        **kwargs,
    )


def _drain(buf: GameAwareEpochBuffer) -> list[list[tuple[int, int]]]:
    batches: list[list[tuple[int, int]]] = []
    for _ in range(buf.num_batches):
        arrays = buf.sample_batch_arrays(buf.plan.batch_size)
        games = np.asarray(arrays["game_id"], dtype=np.int64)
        rows = np.asarray(arrays["ply_index"], dtype=np.int64)
        batches.append(list(zip(games.tolist(), rows.tolist(), strict=True)))
    return batches


def test_exact_epoch_uses_every_row_once_and_never_repeats_a_game_in_batch(
    tmp_path: Path,
) -> None:
    expected = [(game, row) for game in range(7) for row in range(game * 3, game * 3 + 3)]
    shard_dir = _write(tmp_path / "replay", [expected[:9], expected[9:16], expected[16:]])
    buf = _open(shard_dir)

    batches = _drain(buf)

    for batch in batches:
        games = [game for game, _ in batch]
        assert len(games) == len(set(games))
    realized = [row for batch in batches for row in batch]
    assert sorted(realized) == sorted(expected)
    assert len(realized) == len(set(realized))
    assert sum(buf.plan.batch_rows.tolist()) == len(expected)
    receipt = buf.receipt()
    expected_receipt = {
        **buf.plan.as_dict(),
        "plan_workers": 2,
        "load_workers": 2,
        "rows_realized": len(expected),
        "batches_realized": buf.num_batches,
        "same_game_repeats_max": 0,
        "realized_sha256": buf.plan.plan_sha256,
        "complete": True,
    }
    assert {key: receipt[key] for key in expected_receipt} == expected_receipt
    assert int(receipt["peak_decoded_rows"]) <= len(expected)
    assert int(receipt["peak_decoded_shards"]) <= 3
    assert receipt["decoded_rows_resident"] == 0
    with pytest.raises(StopIteration, match="exhausted"):
        buf.sample_batch_arrays(4)


def test_schedule_is_seed_deterministic_and_augmentation_rng_cannot_move_it(
    tmp_path: Path,
) -> None:
    rows = [(game, game * 10 + ply) for game in range(12) for ply in range(4)]
    shard_dir = _write(tmp_path / "replay", [rows[:16], rows[16:32], rows[32:]])
    first = _open(shard_dir, seed=11)
    second = _open(shard_dir, seed=11)
    changed = _open(shard_dir, seed=12)

    # Trainer consumes this public stream for mirror augmentation. It is not a
    # schedule stream, so arbitrary augmentation draws cannot change the epoch.
    first.rng.random(10_000)
    first_batches = _drain(first)
    second_batches = _drain(second)
    changed_batches = _drain(changed)

    assert first_batches == second_batches
    assert first.plan.plan_sha256 == second.plan.plan_sha256
    assert changed_batches != first_batches
    assert changed.plan.plan_sha256 != first.plan.plan_sha256


def test_positions_are_shuffled_within_each_shard_game_segment(
    tmp_path: Path,
) -> None:
    """Mutation target: deleting `_row_rng.shuffle(indices)` makes every
    game's observed plies monotonically increasing and turns successive
    batches into opening/middlegame/endgame bands."""
    rows = [(game, game * 100 + ply) for game in range(16) for ply in range(12)]
    buf = _open(_write(tmp_path / "replay", [rows]), seed=19, batch_size=8)

    by_game: dict[int, list[int]] = {}
    batches = _drain(buf)
    for batch in batches:
        # Absolute ply is encoded as game*100 + within-game ply so the returned
        # row remains unique while this reads the actual within-game phase.
        phases = [row % 100 for _, row in batch]
        if len(batch) == 8:
            assert len(set(phases)) >= 4
        for game, row in batch:
            by_game.setdefault(game, []).append(row % 100)

    non_monotonic = sum(
        sequence != sorted(sequence) for sequence in by_game.values()
    )
    assert non_monotonic >= 14, by_game


def test_cross_shard_game_order_reports_its_segment_local_shuffle_contract(
    tmp_path: Path,
) -> None:
    first_segment = [(0, ply) for ply in range(8)]
    second_segment = [(0, 100 + ply) for ply in range(8)]
    # Singleton games give every batch enough independent rows while game 0
    # crosses the same fixed row-count boundary used by the converter.
    fillers = [(game, 1_000 + game) for game in range(1, 17)]
    buf = _open(
        _write(tmp_path / "replay", [first_segment, second_segment + fillers]),
        seed=7,
        batch_size=4,
    )

    batches = _drain(buf)
    game_zero_rows = [
        row for batch in batches for game, row in batch if game == 0
    ]
    segment_ids = [int(row >= 100) for row in game_zero_rows]

    # Rows are shuffled inside each loaded segment, but the bounded sequential
    # loader deliberately does not turn a split game into random shard reads.
    assert sum(a != b for a, b in pairwise(segment_ids)) == 1
    assert buf.receipt()["row_order"] == (
        "seeded_shuffle_within_shard_game_segments"
    )


def test_tail_is_ragged_instead_of_reusing_a_game_to_fill_it(tmp_path: Path) -> None:
    rows = [(0, ply) for ply in range(6)] + [(1, 100 + ply) for ply in range(2)]
    buf = _open(_write(tmp_path / "replay", [rows]), batch_size=4)

    batches = _drain(buf)

    assert [len(batch) for batch in batches] == [2, 2, 1, 1, 1, 1]
    assert all(len({game for game, _ in batch}) == len(batch) for batch in batches)
    assert buf.plan.ragged_batches == len(batches)


def test_long_games_are_deadline_scheduled_and_the_remainder_is_spread(
    tmp_path: Path,
) -> None:
    rows = [(0, ply) for ply in range(4)] + [
        (game, 100 + game) for game in range(1, 10)
    ]
    buf = _open(_write(tmp_path / "replay", [rows]), batch_size=4)

    batches = _drain(buf)

    assert [len(batch) for batch in batches] == [4, 3, 3, 3]
    assert all(0 in {game for game, _ in batch} for batch in batches)
    assert buf.plan.min_batch_rows == 3


def test_deadline_forced_game_is_loaded_directly_not_through_unrelated_prefix(
    tmp_path: Path,
) -> None:
    # Seed 3 initially puts shard 0 last in the six-shard permutation. Game 0
    # has one row due in every batch, so a sequential-prefix implementation
    # decodes the entire corpus before it can return batch 0.
    shards = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 10)],
        [(2, 20)],
        [(3, 30)],
        [(4, 40)],
        [(5, 50)],
    ]
    buf = _open(_write(tmp_path / "replay", shards), seed=3, batch_size=3)

    first = buf.sample_batch_arrays(3)

    assert 0 in np.asarray(first["game_id"], dtype=np.int64)
    assert int(buf.plan.load_counts[0]) == 3
    assert int(buf.receipt()["peak_decoded_rows"]) == 5
    assert int(buf.receipt()["peak_decoded_rows"]) < buf.plan.rows


def test_multiple_deadline_games_are_brought_forward_and_plan_still_closes(
    tmp_path: Path,
) -> None:
    shards = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 10), (1, 11), (1, 12)],
        [(2, 20)],
        [(3, 30)],
        [(4, 40)],
        [(5, 50)],
    ]
    buf = _open(_write(tmp_path / "replay", shards), seed=3, batch_size=4)

    first = buf.sample_batch_arrays(4)

    assert {0, 1}.issubset(set(np.asarray(first["game_id"], dtype=np.int64)))
    assert int(buf.plan.load_counts[0]) == 4
    assert int(buf.receipt()["peak_decoded_rows"]) == 8
    remaining = []
    for _ in range(buf.num_batches - 1):
        arrays = buf.sample_batch_arrays(buf.plan.batch_size)
        remaining.append(list(zip(
            np.asarray(arrays["game_id"], dtype=np.int64).tolist(),
            np.asarray(arrays["ply_index"], dtype=np.int64).tolist(),
            strict=True,
        )))
    assert sum(len(batch) for batch in remaining) + len(first["x"]) == buf.plan.rows
    assert buf.receipt()["realized_sha256"] == buf.plan.plan_sha256
    assert buf.receipt()["complete"] is True


def test_duplicate_only_prefix_is_skipped_for_diversity_and_refill_is_bounded(
    tmp_path: Path,
) -> None:
    # Seed 3's first 21 paths all carry the same deadline-forced game. The 63
    # remaining one-row shards each introduce a singleton. A prefix walk loads
    # 24 full shards to build batch 0; progress ordering needs exactly four.
    seed = 3
    shard_count = 84
    path_order = np.random.default_rng(
        np.random.SeedSequence([seed, 0]),
    ).permutation(shard_count)
    duplicate_paths = {int(index) for index in path_order[:21]}
    next_game = 1
    shards: list[list[tuple[int, int]]] = []
    for index in range(shard_count):
        if index in duplicate_paths:
            shards.append([(0, index)])
        else:
            shards.append([(next_game, 1_000 + index)])
            next_game += 1

    shard_dir = _write(tmp_path / "replay", shards)
    probe = _open(shard_dir, seed=seed, batch_size=4, load_workers=4)

    assert int(probe.plan.load_counts[0]) == 4
    assert int(probe.plan.load_counts.max(initial=0)) <= 4
    tight_limit = int(probe.plan.peak_working_set_bytes)
    bounded = _open(
        shard_dir,
        seed=seed,
        batch_size=4,
        load_workers=4,
        max_working_set_bytes=tight_limit,
    )
    batches = _drain(bounded)

    assert sum(len(batch) for batch in batches) == shard_count
    receipt = bounded.receipt()
    assert receipt["complete"] is True
    assert receipt["realized_sha256"] == bounded.plan.plan_sha256
    assert int(receipt["peak_working_set_bytes"]) <= tight_limit


def test_skewed_long_game_corpus_is_refused_by_memory_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each refill is already only one shard, yet seven unserved rows from the
    # cross-shard long game accumulate per early batch. This is the case a
    # per-refill load-count bound alone cannot make safe.
    shards = [
        [(0, index * 100 + ply) for ply in range(8)]
        + [(index + 1, 10_000 + index)]
        for index in range(20)
    ]
    shard_dir = _write(tmp_path / "replay", shards)
    probe = _open(shard_dir, seed=3, batch_size=2)
    largest_shard = max(record.decoded_bytes for record in probe._records)
    low_limit = int(largest_shard * 4)

    assert int(probe.plan.load_counts.max(initial=0)) <= 2
    assert int(probe.plan.peak_working_set_bytes) > low_limit
    full_decodes = 0

    def unexpected_decode(
        _self: GameAwareEpochBuffer, _record: object,
    ) -> dict[str, np.ndarray]:
        nonlocal full_decodes
        full_decodes += 1
        raise AssertionError("memory refusal must precede full shard decode")

    monkeypatch.setattr(GameAwareEpochBuffer, "_load_one", unexpected_decode)
    with pytest.raises(ValueError, match="working-set preflight"):
        _open(
            shard_dir,
            seed=3,
            batch_size=2,
            load_workers=4,
            max_working_set_bytes=low_limit,
        )
    assert full_decodes == 0


def test_single_shard_larger_than_memory_cap_is_refused_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_dir = _write(
        tmp_path / "replay",
        [[(game, game) for game in range(4)]],
    )
    probe = _open(shard_dir, batch_size=4)
    limit = int(probe._records[0].decoded_bytes - 1)

    def unexpected_decode(
        _self: GameAwareEpochBuffer, _record: object,
    ) -> dict[str, np.ndarray]:
        raise AssertionError("oversized shard must fail during metadata planning")

    monkeypatch.setattr(GameAwareEpochBuffer, "_load_one", unexpected_decode)
    with pytest.raises(ValueError, match="working-set preflight"):
        _open(
            shard_dir,
            batch_size=4,
            load_workers=4,
            max_working_set_bytes=limit,
        )


def test_epoch_slice_returns_each_single_take_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = {
        "x": np.arange(24, dtype=np.float32).reshape(6, 4),
        "game_id": np.arange(6, dtype=np.int64),
        "_policy_size": np.asarray(1858, dtype=np.int64),
    }
    allocated: dict[int, np.ndarray] = {}
    original_take = np.take

    def tracked_take(
        value: np.ndarray, indices: np.ndarray, *, axis: int,
    ) -> np.ndarray:
        result = original_take(value, indices, axis=axis)
        allocated[id(value)] = result
        return result

    monkeypatch.setattr(game_epoch_module.np, "take", tracked_take)
    sliced = game_epoch_module._slice_epoch_arrays(
        arrays, np.asarray([4, 1, 3], dtype=np.int64),
    )

    # The object allocated by np.take is the returned field itself. Wrapping
    # advanced indexing in np.array(copy=True) would return a second object and
    # transiently consume twice the batch/compaction bytes priced by the plan.
    assert sliced["x"] is allocated[id(arrays["x"])]
    assert sliced["game_id"] is allocated[id(arrays["game_id"])]
    assert sliced["x"].flags.owndata
    assert not np.shares_memory(sliced["x"], arrays["x"])
    assert not np.shares_memory(sliced["_policy_size"], arrays["_policy_size"])


def test_single_shard_materialization_obeys_exact_modeled_peak(
    tmp_path: Path,
) -> None:
    shard_dir = _write(
        tmp_path / "replay",
        [[(game, game) for game in range(8)]],
    )
    probe = _open(shard_dir, batch_size=4)
    record = probe._records[0]
    batch_bytes = int(4 * record.row_bytes + record.scalar_bytes)
    exact_peak = int(record.decoded_bytes + batch_bytes)

    assert probe.plan.peak_working_set_bytes == exact_peak
    with pytest.raises(ValueError, match="batch 0 materialization"):
        _open(
            shard_dir,
            batch_size=4,
            max_working_set_bytes=exact_peak - 1,
        )
    bounded = _open(
        shard_dir, batch_size=4, max_working_set_bytes=exact_peak,
    )
    batches = _drain(bounded)

    assert [len(batch) for batch in batches] == [4, 4]
    assert bounded.receipt()["peak_working_set_bytes"] == exact_peak
    assert bounded.receipt()["complete"] is True


def test_tight_cap_keeps_multichunk_compaction_in_lockstep_with_plan(
    tmp_path: Path,
) -> None:
    shards = [
        [(0, ply) for ply in range(100)] + [(1, 10_000)],
        [(2, 20_000 + ply) for ply in range(100)] + [(3, 30_000)],
    ]
    shard_dir = _write(tmp_path / "replay", shards)
    probe = _open(shard_dir, seed=3, batch_size=2)
    record = probe._records[0]
    planned_batch_bytes = int(2 * record.row_bytes + 2 * record.scalar_bytes)
    first_compaction_peak = int(
        2 * record.decoded_bytes
        + planned_batch_bytes
        + 25 * np.dtype(np.int64).itemsize
        + record.scalar_bytes
        + 25 * record.row_bytes
    )
    assert probe.plan.peak_working_set_bytes == first_compaction_peak
    exact = _open(
        shard_dir,
        seed=3,
        batch_size=2,
        max_working_set_bytes=first_compaction_peak,
    )
    _drain(exact)
    assert exact.receipt()["peak_working_set_bytes"] == first_compaction_peak
    assert exact.receipt()["complete"] is True

    # One byte below the first optional compaction peak makes the planner skip
    # it at batch 75. The realized multi-chunk batch is slightly smaller than
    # its conservative per-shard scalar budget; using that smaller value at
    # runtime would compact anyway and immediately diverge from preflight.
    limit = first_compaction_peak - 1
    bounded = _open(
        shard_dir,
        seed=3,
        batch_size=2,
        max_working_set_bytes=limit,
    )
    actual_batch_bytes = 0
    for batch_index in range(77):
        batch = bounded.sample_batch_arrays(2)
        if batch_index == 75:
            actual_batch_bytes = bounded._arrays_nbytes(batch)
            assert actual_batch_bytes < planned_batch_bytes
            assert bounded._resident_bytes == 2 * record.decoded_bytes
        assert bounded._resident_bytes == int(
            bounded.plan.resident_bytes_after_batch[batch_index],
        )

    assert actual_batch_bytes > 0
    assert bounded._resident_bytes < 2 * record.decoded_bytes
    for _ in range(bounded.num_batches - 77):
        bounded.sample_batch_arrays(2)
    assert bounded.receipt()["peak_working_set_bytes"] == (
        bounded.plan.peak_working_set_bytes
    )
    assert int(bounded.receipt()["peak_working_set_bytes"]) <= limit
    assert bounded.receipt()["complete"] is True


def test_consumed_rows_are_compacted_out_of_long_lived_shards(tmp_path: Path) -> None:
    rows = [(game, game * 100 + ply) for game in range(10) for ply in range(10)]
    buf = _open(_write(tmp_path / "replay", [rows]), batch_size=5)

    for _ in range(15):
        buf.sample_batch_arrays(5)

    receipt = buf.receipt()
    assert receipt["peak_decoded_rows"] == 100
    assert receipt["decoded_rows_resident"] == 25


def test_game_ids_are_namespaced_across_independent_conversion_outputs(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path / "source_a", [[(0, 0), (0, 1), (1, 10), (1, 11)]])
    second = _write(tmp_path / "source_b", [[(0, 20), (0, 21), (1, 30), (1, 31)]])
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "shard_000000.zarr").symlink_to(
        first / "shard_000000.zarr",
    )
    (staged / "shard_000001.zarr").symlink_to(
        second / "shard_000000.zarr",
    )

    buf = _open(staged, batch_size=4)
    first_batch = buf.sample_batch_arrays(4)

    # The immutable shards retain source-local ids; the sampled batch exposes
    # the corpus-wide identity used by the scheduler. Treating raw ids as
    # global would see only games {0, 1} and return two rows here.
    assert buf.plan.source_count == 2
    assert buf.plan.game_count == 4
    assert first_batch["x"].shape[0] == 4
    assert sorted(np.asarray(first_batch["game_id"]).tolist()) == [0, 1, 2, 3]
    assert buf.receipt()["same_game_repeats_max"] == 0


def test_partially_missing_game_identity_is_refused_during_plan(tmp_path: Path) -> None:
    shard_dir = _write(tmp_path / "replay", [[(0, 0), (None, 1), (1, 2)]])

    with pytest.raises(ValueError, match="without game_id"):
        _open(shard_dir)


def test_zero_row_index_reservation_is_not_scheduled(tmp_path: Path) -> None:
    shard_dir = _write(
        tmp_path / "replay",
        [[(0, 0), (1, 1), (2, 2), (3, 3)]],
    )
    arrs, _ = load_shard_arrays(shard_dir / "shard_000000.zarr")
    rows = int(arrs["x"].shape[0])
    empty = {
        name: (
            np.asarray(value)[:0]
            if np.asarray(value).ndim >= 1
            and np.asarray(value).shape[0] == rows
            else np.asarray(value)
        )
        for name, value in arrs.items()
    }
    save_local_shard_arrays(shard_dir / "shard_000001.zarr", arrs=empty)

    buf = _open(shard_dir, batch_size=4)

    batches = _drain(buf)

    assert len(batches) == 1
    assert sorted(batches[0]) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert buf.plan.shard_count == 1
    assert buf.receipt()["complete"] is True


def test_batch_size_and_input_shape_are_frozen_by_the_plan(tmp_path: Path) -> None:
    rows = [(game, game) for game in range(5)]
    shard_dir = _write(tmp_path / "replay", [rows], planes=175)
    with pytest.raises(ValueError, match="carries 175 input planes"):
        GameAwareEpochBuffer(
            shard_dir=shard_dir,
            batch_size=4,
            seed=0,
            input_planes=146,
            mirror_augmentation=False,
            plan_workers=1,
            load_workers=1,
        )

    correct = GameAwareEpochBuffer(
        shard_dir=shard_dir,
        batch_size=4,
        seed=0,
        input_planes=175,
        mirror_augmentation=False,
        plan_workers=1,
        load_workers=1,
    )
    with pytest.raises(ValueError, match="planned for batch_size=4"):
        correct.sample_batch_arrays(3)


def test_mixed_input_plane_corpus_is_refused_before_any_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_dir = _write(
        tmp_path / "replay",
        [[(0, 0), (1, 1)]],
        planes=175,
    )
    legacy = [_sample(game=2, row=2, planes=146)]
    save_local_shard_arrays(
        shard_dir / "shard_000001.zarr",
        arrs=samples_to_arrays(legacy),
        meta=ShardMeta(
            positions=1,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
        ),
    )

    def unexpected_decode(
        _self: GameAwareEpochBuffer, _record: object,
    ) -> dict[str, np.ndarray]:
        raise AssertionError("plane mismatch must fail before a full decode")

    monkeypatch.setattr(GameAwareEpochBuffer, "_load_one", unexpected_decode)
    with pytest.raises(ValueError, match="carries 146 input planes"):
        GameAwareEpochBuffer(
            shard_dir=shard_dir,
            batch_size=2,
            seed=0,
            input_planes=175,
            mirror_augmentation=False,
            plan_workers=1,
            load_workers=1,
        )


def test_mixed_policy_width_corpus_is_refused_before_any_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_dir = _write(
        tmp_path / "replay",
        [[(0, 0), (1, 1)]],
        planes=175,
    )
    full = _sample(game=2, row=2, planes=175)
    full.policy_target = np.zeros((POLICY_SIZE,), dtype=np.float32)
    full.policy_target[2] = 1.0
    full.legal_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    full.legal_mask[2] = 1
    save_local_shard_arrays(
        shard_dir / "shard_000001.zarr",
        arrs=samples_to_arrays([full]),
        meta=ShardMeta(
            positions=1,
            policy_encoding="az_4672",
            policy_size=POLICY_SIZE,
        ),
    )

    def unexpected_decode(
        _self: GameAwareEpochBuffer, _record: object,
    ) -> dict[str, np.ndarray]:
        raise AssertionError("policy mismatch must fail before a full decode")

    monkeypatch.setattr(GameAwareEpochBuffer, "_load_one", unexpected_decode)
    with pytest.raises(ValueError, match=r"mixes policy widths \[1858, 4672\]"):
        GameAwareEpochBuffer(
            shard_dir=shard_dir,
            batch_size=2,
            seed=0,
            input_planes=175,
            mirror_augmentation=False,
            plan_workers=1,
            load_workers=1,
        )


def test_mirror_augmentation_is_preflighted_in_working_set(tmp_path: Path) -> None:
    shard_dir = _write(
        tmp_path / "replay",
        [[(game, game) for game in range(4)]],
    )
    plain = _open(shard_dir, batch_size=4, mirror_augmentation=False)
    mirrored = _open(shard_dir, batch_size=4, mirror_augmentation=True)

    assert mirrored.plan.peak_working_set_bytes > plain.plan.peak_working_set_bytes
    assert mirrored.plan.plan_sha256 == plain.plan.plan_sha256
    assert mirrored.plan.mirror_augmentation is True
    assert mirrored.plan.mirror_working_set_batch_copies == 7
    too_small = mirrored.plan.peak_working_set_bytes - 1
    with pytest.raises(ValueError, match="materialization"):
        _open(
            shard_dir,
            batch_size=4,
            mirror_augmentation=True,
            max_working_set_bytes=too_small,
        )
