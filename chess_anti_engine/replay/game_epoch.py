"""Exact, game-aware epochs for finite offline replay corpora.

The rolling :class:`~chess_anti_engine.replay.disk_buffer.DiskReplayBuffer`
intentionally samples a changing window with replacement.  That is the wrong
contract for a frozen supervised corpus which is meant to receive one view:
replacement leaves rows unseen, repeats others, and can put several positions
from one game in the same optimizer batch.

``GameAwareEpochBuffer`` has the finite-corpus contract instead:

* every stored row is returned exactly once;
* a batch contains at most one row for each semantic game (source directory +
  its source-local ``game_id``);
* shard order, game choice, and within-game row order are deterministic from
  the seed; and
* the complete schedule is planned before training, so the launcher can refuse
  a step budget which would truncate the epoch or wrap into a second one.

Only the lightweight ``game_id`` columns are read during planning. Full
training arrays are decoded once, just before their rows become eligible;
consumed portions of long-lived shards are compacted away. This keeps I/O
sequential at shard granularity and memory bounded instead of turning a
globally shuffled epoch into 512 random zarr reads per batch.
"""
from __future__ import annotations

import hashlib
import struct
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .disk_buffer import _concat_sparse_batches, _slice_array_batch
from .shard import densify_chunk, iter_shard_paths, load_shard_arrays


DEFAULT_PLAN_WORKERS = 16
DEFAULT_LOAD_WORKERS = 4


@dataclass(frozen=True)
class _ShardGames:
    path: Path
    rows: int
    # Raw ids are only unique within one conversion output.  The driver can
    # stage several independently converted directories, each of which starts
    # numbering games at zero, so scheduling uses the namespaced keys.
    game_ids: np.ndarray
    game_keys: np.ndarray
    game_counts: np.ndarray


@dataclass(frozen=True)
class GameEpochPlan:
    """The cheap schedule receipt computed before any full shard is decoded."""

    rows: int
    batches: int
    full_batches: int
    ragged_batches: int
    min_batch_rows: int
    shard_count: int
    source_count: int
    game_count: int
    batch_size: int
    seed: int
    plan_sha256: str
    load_counts: np.ndarray = field(repr=False)
    batch_rows: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, int | str]:
        return {
            "mode": "game_epoch",
            "rows_planned": int(self.rows),
            "batches_planned": int(self.batches),
            "full_batches_planned": int(self.full_batches),
            "ragged_batches_planned": int(self.ragged_batches),
            "min_batch_rows_planned": int(self.min_batch_rows),
            "shards": int(self.shard_count),
            "sources": int(self.source_count),
            "games": int(self.game_count),
            "game_identity": "resolved_shard_parent+game_id",
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "plan_sha256": self.plan_sha256,
        }


@dataclass
class _Segment:
    chunk_id: int
    indices: np.ndarray
    cursor: int = 0

    @property
    def remaining(self) -> int:
        return int(self.indices.shape[0]) - int(self.cursor)

    def pop(self) -> int:
        if self.cursor >= self.indices.shape[0]:
            raise RuntimeError("game segment is empty")
        index = int(self.indices[self.cursor])
        self.cursor += 1
        return index


@dataclass
class _GameRows:
    total: int = 0
    segments: deque[_Segment] = field(default_factory=deque)

    def append(self, segment: _Segment) -> None:
        self.segments.append(segment)
        self.total += segment.remaining

    def pop(self) -> tuple[int, int]:
        while self.segments and self.segments[0].remaining == 0:
            self.segments.popleft()
        if not self.segments or self.total <= 0:
            raise RuntimeError("active game has no rows")
        segment = self.segments[0]
        index = segment.pop()
        self.total -= 1
        if segment.remaining == 0:
            self.segments.popleft()
        return segment.chunk_id, index


@dataclass
class _LoadedChunk:
    arrays: dict[str, np.ndarray]
    remaining: int
    capacity: int


def _seeded_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _scan_shard(path: Path) -> _ShardGames:
    arrs, _ = load_shard_arrays(path, lazy=True)
    if "x" not in arrs:
        raise ValueError(f"{path} carries no x array")
    rows = int(arrs["x"].shape[0])
    if "game_id" not in arrs or "has_game_id" not in arrs:
        raise ValueError(
            f"{path} carries {rows} rows but no game_id/has_game_id columns; "
            "game-aware sampling cannot guess the independence unit",
        )
    has_game = np.asarray(arrs["has_game_id"], dtype=bool)
    if has_game.shape != (rows,):
        raise ValueError(
            f"{path} has_game_id shape {has_game.shape}, expected ({rows},)",
        )
    missing = int(rows - np.count_nonzero(has_game))
    if missing:
        raise ValueError(
            f"{path} has {missing}/{rows} rows without game_id; exact game-aware "
            "sampling refuses a partially identified corpus",
        )
    game_id = np.asarray(arrs["game_id"], dtype=np.int64)
    if game_id.shape != (rows,):
        raise ValueError(
            f"{path} game_id shape {game_id.shape}, expected ({rows},)",
        )
    games, counts = np.unique(game_id, return_counts=True)
    return _ShardGames(
        path=path,
        rows=rows,
        game_ids=np.asarray(games, dtype=np.int64),
        game_keys=np.asarray(games, dtype=np.int64),
        game_counts=np.asarray(counts, dtype=np.int64),
    )


def _scan_shards(paths: Sequence[Path], workers: int) -> list[_ShardGames]:
    n_workers = max(1, min(int(workers), len(paths)))
    if n_workers <= 1:
        records = [_scan_shard(path) for path in paths]
    else:
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="game-epoch-plan",
        ) as pool:
            # map preserves submission order; filesystem timing cannot change
            # the namespace assignment, seeded permutation, or plan hash.
            records = list(pool.map(_scan_shard, paths))

    # `lc0_data_to_rows convert` numbers games from zero on every invocation.
    # `stage_shards` can combine several conversion outputs, so the raw int64
    # alone is not a corpus-wide identity.  Namespace it by the resolved source
    # directory without rewriting the immutable shards.
    key_for_game: dict[tuple[str, int], int] = {}
    namespaced: list[_ShardGames] = []
    for record in records:
        source = str(record.path.resolve().parent)
        keys: list[int] = []
        for raw_game in record.game_ids.tolist():
            identity = (source, int(raw_game))
            key = key_for_game.setdefault(identity, len(key_for_game))
            keys.append(key)
        namespaced.append(
            _ShardGames(
                path=record.path,
                rows=record.rows,
                game_ids=record.game_ids,
                game_keys=np.asarray(keys, dtype=np.int64),
                game_counts=record.game_counts,
            ),
        )
    return namespaced


def _update_choice_hash(digest: Any, chosen_games: np.ndarray) -> None:
    digest.update(struct.pack("<I", int(chosen_games.shape[0])))
    digest.update(np.asarray(chosen_games, dtype="<i8").tobytes(order="C"))


def _choose_games(
    active: dict[int, int] | dict[int, _GameRows],
    count: int,
    rng: np.random.Generator,
    *,
    remaining: dict[int, int],
    forced: set[int],
) -> np.ndarray:
    if not forced.issubset(active):
        missing = sorted(forced.difference(active))
        raise RuntimeError(f"forced games are not decoded: {missing[:8]}")
    if len(forced) > int(count):
        raise RuntimeError(
            f"{len(forced)} games are due in a batch with room for {count}",
        )
    candidates = [key for key in active if key not in forced]
    random_count = int(count) - len(forced)
    if random_count > len(candidates):
        raise RuntimeError(
            f"asked for {count} distinct games from {len(active)} active games",
        )
    chosen = np.asarray(sorted(forced), dtype=np.int64)
    if random_count:
        keys = np.asarray(candidates, dtype=np.int64)
        weights = np.fromiter(
            (remaining[key] for key in candidates),
            dtype=np.float64,
            count=len(candidates),
        )
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise RuntimeError("active game weights must be finite and positive")
        indices = rng.choice(
            keys.shape[0],
            size=random_count,
            replace=False,
            p=weights / weights.sum(),
        )
        chosen = np.concatenate((chosen, keys[np.asarray(indices, dtype=np.int64)]))
    rng.shuffle(chosen)
    return chosen


def _game_totals(records: Sequence[_ShardGames]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for record in records:
        for game, count in zip(
            record.game_keys.tolist(), record.game_counts.tolist(), strict=True,
        ):
            key = int(game)
            totals[key] = totals.get(key, 0) + int(count)
    return totals


def _remaining_buckets(remaining: dict[int, int]) -> dict[int, set[int]]:
    buckets: dict[int, set[int]] = {}
    for game, count in remaining.items():
        buckets.setdefault(int(count), set()).add(int(game))
    return buckets


def _consume_games(
    chosen: np.ndarray,
    remaining: dict[int, int],
    buckets: dict[int, set[int]],
) -> None:
    for game in chosen.tolist():
        key = int(game)
        old = remaining[key]
        bucket = buckets[old]
        bucket.remove(key)
        if not bucket:
            del buckets[old]
        new = old - 1
        if new:
            remaining[key] = new
            buckets.setdefault(new, set()).add(key)
        else:
            del remaining[key]


def _balanced_batch_rows(*, rows: int, batch_size: int, max_game_rows: int) -> np.ndarray:
    # At most one position from a game can enter a batch, so a game with M rows
    # requires at least M batches.  Apart from that constraint, ceil(N/B)
    # batches suffice.  Spreading the remainder across every batch avoids a
    # final 1-row optimizer update when N % B == 1.
    batches = max(
        int(max_game_rows),
        (int(rows) + int(batch_size) - 1) // int(batch_size),
    )
    base, larger = divmod(int(rows), batches)
    sizes = np.full((batches,), base, dtype=np.int32)
    sizes[:larger] += 1
    if np.any(sizes <= 0) or np.any(sizes > int(batch_size)):
        raise RuntimeError(
            f"cannot distribute {rows} rows across {batches} batches of at most "
            f"{batch_size}",
        )
    return sizes


def _plan_epoch(
    records: Sequence[_ShardGames], *, batch_size: int, seed: int,
) -> GameEpochPlan:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not records:
        raise ValueError("game-aware epoch corpus holds no shards")

    path_rng = _seeded_rng(seed, 0)
    order = path_rng.permutation(len(records))
    shuffled = [records[int(index)] for index in order]
    choice_rng = _seeded_rng(seed, 1)
    active: dict[int, int] = {}
    next_shard = 0
    rows = sum(record.rows for record in shuffled)
    if rows <= 0:
        raise ValueError("game-aware epoch corpus holds no rows")
    sources = {str(record.path.resolve().parent) for record in shuffled}
    remaining = _game_totals(shuffled)
    buckets = _remaining_buckets(remaining)
    batch_array = _balanced_batch_rows(
        rows=rows,
        batch_size=batch_size,
        max_game_rows=max(remaining.values()),
    )

    load_counts: list[int] = []
    digest = hashlib.sha256()
    digest.update(struct.pack("<qqq", int(seed), int(batch_size), int(rows)))

    consumed = 0
    for batch_index, batch_rows in enumerate(batch_array.tolist()):
        batches_left = int(batch_array.shape[0] - batch_index)
        forced = buckets.get(batches_left, set())
        loaded = 0
        while (
            len(active) < int(batch_rows)
            or not forced.issubset(active)
        ) and next_shard < len(shuffled):
            record = shuffled[next_shard]
            next_shard += 1
            loaded += 1
            resolved = str(record.path.resolve()).encode("utf-8")
            digest.update(struct.pack("<I", len(resolved)))
            digest.update(resolved)
            digest.update(struct.pack("<q", int(record.rows)))
            for game, count in zip(
                record.game_keys.tolist(), record.game_counts.tolist(), strict=True,
            ):
                key = int(game)
                active[key] = active.get(key, 0) + int(count)

        if len(active) < int(batch_rows) or not forced.issubset(active):
            raise RuntimeError(
                f"epoch plan cannot fill batch {batch_index} ({batch_rows} rows) "
                f"after {consumed}/{rows} rows; {len(active)} decoded games, "
                f"{len(forced.difference(active))} due games unavailable",
            )
        chosen = _choose_games(
            active,
            int(batch_rows),
            choice_rng,
            remaining=remaining,
            forced=forced,
        )
        _update_choice_hash(digest, chosen)
        for game in chosen.tolist():
            key = int(game)
            left = active[key] - 1
            if left:
                active[key] = left
            else:
                del active[key]
        _consume_games(chosen, remaining, buckets)
        load_counts.append(loaded)
        consumed += int(batch_rows)

    if active or remaining or buckets or next_shard != len(shuffled):
        raise RuntimeError(
            f"epoch plan consumed {consumed}/{rows} rows but ended with "
            f"{len(active)} active games, {len(remaining)} unfinished games, "
            f"and {len(shuffled) - next_shard} unloaded shards",
        )

    load_array = np.asarray(load_counts, dtype=np.int32)
    full = int(np.count_nonzero(batch_array == int(batch_size)))
    ragged = int(batch_array.shape[0] - full)
    return GameEpochPlan(
        rows=int(rows),
        batches=int(batch_array.shape[0]),
        full_batches=full,
        ragged_batches=ragged,
        min_batch_rows=int(batch_array.min(initial=batch_size)),
        shard_count=len(shuffled),
        source_count=len(sources),
        game_count=len(_game_totals(shuffled)),
        batch_size=int(batch_size),
        seed=int(seed),
        plan_sha256=digest.hexdigest(),
        load_counts=load_array,
        batch_rows=batch_array,
    )


class GameAwareEpochBuffer:
    """A one-shot replay buffer over a frozen, fully game-identified corpus."""

    exact_without_replacement = True

    def __init__(
        self,
        *,
        shard_dir: Path,
        batch_size: int,
        seed: int,
        input_planes: int | None,
        plan_workers: int = DEFAULT_PLAN_WORKERS,
        load_workers: int = DEFAULT_LOAD_WORKERS,
    ) -> None:
        paths = iter_shard_paths(shard_dir)
        records = _scan_shards(paths, int(plan_workers))
        path_rng = _seeded_rng(seed, 0)
        order = path_rng.permutation(len(records))
        self._records = [records[int(index)] for index in order]
        self.plan = _plan_epoch(records, batch_size=int(batch_size), seed=int(seed))
        self._batch_size = int(batch_size)
        self._input_planes = None if input_planes is None else int(input_planes)
        self._plan_workers = max(1, int(plan_workers))
        self._load_workers = max(1, int(load_workers))
        self._choice_rng = _seeded_rng(seed, 1)
        self._row_rng = _seeded_rng(seed, 2)
        # Public rng is consumed by Trainer's mirror augmentation. Keeping it
        # off the schedule streams makes augmentation probability unable to
        # change which rows the epoch contains or how games are batched.
        self.rng = _seeded_rng(seed, 3)

        self._next_shard = 0
        self._next_chunk_id = 0
        self._batch_index = 0
        self._rows_yielded = 0
        self._chunks: dict[int, _LoadedChunk] = {}
        self._active: dict[int, _GameRows] = {}
        self._remaining = _game_totals(self._records)
        self._remaining_buckets = _remaining_buckets(self._remaining)
        self._resident_rows = 0
        self._peak_resident_rows = 0
        self._peak_resident_chunks = 0
        self._realized_digest = hashlib.sha256()
        self._realized_digest.update(
            struct.pack("<qqq", int(seed), int(batch_size), int(self.plan.rows)),
        )
        self._max_same_game_repeats = 0
        self._closed = False

    def __len__(self) -> int:
        return int(self.plan.rows)

    @property
    def num_batches(self) -> int:
        return int(self.plan.batches)

    def _load_one(self, record: _ShardGames) -> dict[str, np.ndarray]:
        arrs, _ = load_shard_arrays(record.path, lazy=False, validate=True)
        rows = int(arrs["x"].shape[0])
        if rows != record.rows:
            raise RuntimeError(
                f"{record.path} planned {record.rows} rows and decoded {rows}; "
                "the corpus changed after the epoch was planned",
            )
        if self._input_planes is not None and int(arrs["x"].shape[1]) != self._input_planes:
            raise ValueError(
                f"{record.path} carries {int(arrs['x'].shape[1])} input planes, "
                f"the exact-epoch trainer requires {self._input_planes}; this "
                "mode does not silently pad or reinterpret a static corpus",
            )
        return {name: np.asarray(value) for name, value in arrs.items()}

    def _add_loaded_chunk(
        self, record: _ShardGames, arrs: dict[str, np.ndarray],
    ) -> None:
        rows = int(arrs["x"].shape[0])
        raw_game_ids = np.asarray(arrs["game_id"], dtype=np.int64)
        has_game = np.asarray(arrs["has_game_id"], dtype=bool)
        if (
            raw_game_ids.shape != (rows,)
            or has_game.shape != (rows,)
            or not np.all(has_game)
        ):
            raise RuntimeError(
                "a shard's game identity changed between planning and full decode",
            )
        positions = np.searchsorted(record.game_ids, raw_game_ids)
        if (
            np.any(positions >= record.game_ids.shape[0])
            or not np.array_equal(record.game_ids[positions], raw_game_ids)
        ):
            raise RuntimeError(
                "a shard's game ids changed between planning and full decode",
            )
        game_keys = record.game_keys[positions]
        # Expose the same corpus-wide identity the scheduler enforces. Keeping
        # source-local ids here would make a valid multi-source batch appear to
        # repeat games to every downstream audit that inspects the batch.
        arrs["game_id"] = np.asarray(game_keys, dtype=np.int64)
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._chunks[chunk_id] = _LoadedChunk(
            arrays=arrs, remaining=rows, capacity=rows,
        )
        self._resident_rows += rows
        self._peak_resident_rows = max(self._peak_resident_rows, self._resident_rows)
        self._peak_resident_chunks = max(self._peak_resident_chunks, len(self._chunks))

        order = np.argsort(game_keys, kind="stable")
        sorted_games = game_keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_games[1:] != sorted_games[:-1]])
        stops = np.r_[starts[1:], rows]
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            indices = np.asarray(order[int(start):int(stop)], dtype=np.int64).copy()
            self._row_rng.shuffle(indices)
            game = int(sorted_games[int(start)])
            state = self._active.setdefault(game, _GameRows())
            state.append(_Segment(chunk_id=chunk_id, indices=indices))

    def _compact_chunk(self, chunk_id: int) -> None:
        """Release consumed rows from a long-lived shard allocation.

        A single long game can keep an otherwise exhausted 8k-row shard alive
        for hundreds of batches.  Without compaction, those stragglers make a
        100M-row epoch converge on holding the whole decoded corpus in RAM.
        Compact only after at least 75% has been consumed; repeated copies then
        form a geometric series below one third of the original shard volume.
        """
        chunk = self._chunks[chunk_id]
        if chunk.remaining <= 0 or chunk.remaining * 4 > chunk.capacity:
            return
        segments = [
            segment
            for state in self._active.values()
            for segment in state.segments
            if segment.chunk_id == chunk_id and segment.remaining > 0
        ]
        if sum(segment.remaining for segment in segments) != chunk.remaining:
            raise RuntimeError(
                f"chunk {chunk_id} tracks {chunk.remaining} rows but its game "
                "segments disagree",
            )
        keep = np.concatenate([
            segment.indices[segment.cursor:] for segment in segments
        ])
        old_capacity = chunk.capacity
        chunk.arrays = _slice_array_batch(chunk.arrays, keep)
        cursor = 0
        for segment in segments:
            stop = cursor + segment.remaining
            segment.indices = np.arange(cursor, stop, dtype=np.int64)
            segment.cursor = 0
            cursor = stop
        chunk.capacity = chunk.remaining
        self._resident_rows -= old_capacity - chunk.capacity

    def _load_next(self, count: int) -> None:
        n = int(count)
        if n <= 0:
            return
        stop = self._next_shard + n
        if stop > len(self._records):
            raise RuntimeError(
                f"epoch plan requests {n} shards with only "
                f"{len(self._records) - self._next_shard} left",
            )
        records = self._records[self._next_shard:stop]
        self._next_shard = stop
        workers = min(self._load_workers, len(records))
        if workers <= 1:
            loaded = [self._load_one(record) for record in records]
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="game-epoch-load",
            ) as pool:
                loaded = list(pool.map(self._load_one, records))
        for record, arrs in zip(records, loaded, strict=True):
            resolved = str(record.path.resolve()).encode("utf-8")
            self._realized_digest.update(struct.pack("<I", len(resolved)))
            self._realized_digest.update(resolved)
            self._realized_digest.update(struct.pack("<q", int(record.rows)))
            self._add_loaded_chunk(record, arrs)

    def _gather(self, selected: list[tuple[int, int]]) -> dict[str, np.ndarray]:
        by_chunk: dict[int, list[int]] = {}
        for chunk_id, row_index in selected:
            by_chunk.setdefault(int(chunk_id), []).append(int(row_index))
        dense_parts: list[dict[str, np.ndarray]] = []
        for chunk_id, indices in by_chunk.items():
            chunk = self._chunks[chunk_id]
            sparse = _slice_array_batch(
                chunk.arrays, np.asarray(indices, dtype=np.int64),
            )
            policy_size = int(np.asarray(
                sparse.get("_policy_size", sparse["policy_target"].shape[1]),
            ).item())
            dense_parts.append(densify_chunk(sparse, policy_size=policy_size))
        return _concat_sparse_batches(dense_parts)

    def sample_batch_arrays(
        self, batch_size: int, *, wdl_balance: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return the next planned batch; this object cannot wrap epochs.

        ``wdl_balance`` is accepted for the common replay-buffer protocol.  An
        exact epoch necessarily preserves the corpus's row distribution and
        therefore performs no over/under-sampling for either spelling.
        """
        _ = wdl_balance
        if self._closed:
            raise RuntimeError("GameAwareEpochBuffer is closed")
        if int(batch_size) != self._batch_size:
            raise ValueError(
                f"epoch was planned for batch_size={self._batch_size}, got {batch_size}",
            )
        if self._batch_index >= self.plan.batches:
            raise StopIteration("the exact game-aware epoch is exhausted")

        self._load_next(int(self.plan.load_counts[self._batch_index]))
        take = int(self.plan.batch_rows[self._batch_index])
        batches_left = int(self.plan.batches - self._batch_index)
        forced = self._remaining_buckets.get(batches_left, set())
        if take > len(self._active):
            raise RuntimeError(
                f"planned batch {self._batch_index} needs {take} distinct games "
                f"but the realized pool has {len(self._active)}",
            )
        chosen = _choose_games(
            self._active,
            take,
            self._choice_rng,
            remaining=self._remaining,
            forced=forced,
        )
        _update_choice_hash(self._realized_digest, chosen)
        same_game_repeats = int(chosen.shape[0] - np.unique(chosen).shape[0])
        self._max_same_game_repeats = max(
            self._max_same_game_repeats, same_game_repeats,
        )
        if same_game_repeats:
            raise RuntimeError(
                f"planned batch {self._batch_index} repeated {same_game_repeats} game(s)",
            )

        selected: list[tuple[int, int]] = []
        touched_chunks: set[int] = set()
        for game in chosen.tolist():
            key = int(game)
            state = self._active[key]
            chunk_id, row_index = state.pop()
            selected.append((chunk_id, row_index))
            touched_chunks.add(chunk_id)
            self._chunks[chunk_id].remaining -= 1
            if state.total == 0:
                del self._active[key]
        _consume_games(chosen, self._remaining, self._remaining_buckets)

        batch = self._gather(selected)
        realized_rows = int(np.asarray(batch["x"]).shape[0])
        if realized_rows != take:
            raise RuntimeError(
                f"planned batch {self._batch_index} selected {take} rows but "
                f"the materialized batch carries {realized_rows}",
            )
        for chunk_id in touched_chunks:
            if self._chunks[chunk_id].remaining == 0:
                self._resident_rows -= self._chunks[chunk_id].capacity
                del self._chunks[chunk_id]
            else:
                self._compact_chunk(chunk_id)

        self._batch_index += 1
        self._rows_yielded += take
        if self._batch_index == self.plan.batches:
            problems: list[str] = []
            if self._rows_yielded != self.plan.rows:
                problems.append(
                    f"yielded {self._rows_yielded}/{self.plan.rows} rows",
                )
            if self._active:
                problems.append(f"{len(self._active)} games still active")
            if self._remaining or self._remaining_buckets:
                problems.append(f"{len(self._remaining)} games still unfinished")
            if self._chunks:
                problems.append(f"{len(self._chunks)} decoded shards still hold rows")
            if self._next_shard != len(self._records):
                problems.append(
                    f"loaded {self._next_shard}/{len(self._records)} shards",
                )
            if self._realized_digest.hexdigest() != self.plan.plan_sha256:
                problems.append("realized game/shard schedule hash differs from the plan")
            if problems:
                raise RuntimeError(
                    "exact game-aware epoch did not close cleanly: " + "; ".join(problems),
                )
        return batch

    def receipt(self) -> dict[str, int | str | bool]:
        complete = (
            self._batch_index == self.plan.batches
            and self._rows_yielded == self.plan.rows
            and not self._active
            and not self._remaining
            and not self._remaining_buckets
            and not self._chunks
            and self._next_shard == len(self._records)
            and self._realized_digest.hexdigest() == self.plan.plan_sha256
        )
        return {
            **self.plan.as_dict(),
            "plan_workers": int(self._plan_workers),
            "load_workers": int(self._load_workers),
            "rows_realized": int(self._rows_yielded),
            "batches_realized": int(self._batch_index),
            "same_game_repeats_max": int(self._max_same_game_repeats),
            "peak_decoded_rows": int(self._peak_resident_rows),
            "peak_decoded_shards": int(self._peak_resident_chunks),
            "decoded_rows_resident": int(self._resident_rows),
            "realized_sha256": self._realized_digest.hexdigest(),
            "complete": bool(complete),
        }

    def close(self) -> None:
        self._closed = True
        self._chunks.clear()
        self._active.clear()
        self._remaining.clear()
        self._remaining_buckets.clear()
        self._resident_rows = 0
