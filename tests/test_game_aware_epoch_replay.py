from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay import game_epoch as epoch_tool
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


def _open(shard_dir: Path, *, seed: int = 7, batch_size: int = 4) -> GameAwareEpochBuffer:
    return GameAwareEpochBuffer(
        shard_dir=shard_dir,
        batch_size=batch_size,
        seed=seed,
        input_planes=146,
        plan_workers=2,
        load_workers=2,
    )


def _drain(buf: GameAwareEpochBuffer) -> list[list[tuple[int, int]]]:
    batches: list[list[tuple[int, int]]] = []
    for _ in range(buf.num_batches):
        arrays = buf.sample_batch_arrays(buf.plan.batch_size)
        games = np.asarray(arrays["game_id"], dtype=np.int64)
        rows = np.asarray(arrays["ply_index"], dtype=np.int64)
        batches.append(list(zip(games.tolist(), rows.tolist(), strict=True)))
    return batches


def test_game_choice_is_independent_of_active_dictionary_insertion_order() -> None:
    """Planner and decoded-loader insertion order cannot move the schedule."""
    counts = {0: 1, 1: 8, 2: 4, 3: 11}
    planner_order = dict(counts)
    loader_order = {key: counts[key] for key in reversed(counts)}

    planned = epoch_tool._choose_games(
        planner_order,
        3,
        epoch_tool._seeded_rng(0, 1),
        remaining=dict(counts),
        forced={0},
    )
    realized = epoch_tool._choose_games(
        loader_order,
        3,
        epoch_tool._seeded_rng(0, 1),
        remaining=dict(counts),
        forced={0},
    )

    np.testing.assert_array_equal(realized, planned)


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
    wrong_planes = GameAwareEpochBuffer(
        shard_dir=shard_dir,
        batch_size=4,
        seed=0,
        input_planes=146,
        plan_workers=1,
        load_workers=1,
    )
    with pytest.raises(ValueError, match="carries 175 input planes"):
        wrong_planes.sample_batch_arrays(4)

    correct = GameAwareEpochBuffer(
        shard_dir=shard_dir,
        batch_size=4,
        seed=0,
        input_planes=175,
        plan_workers=1,
        load_workers=1,
    )
    with pytest.raises(ValueError, match="planned for batch_size=4"):
        correct.sample_batch_arrays(3)
