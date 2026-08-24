"""Stratified FEN sampling for the NNUE parity gate and throughput benchmark.

The sample is stratified on the two axes the big net's cost and code paths
actually vary along:

* **piece count**, because it selects the layer stack outright
  (``bucket = (piece_count - 1) / 4``) — an unstratified sample from random play
  would put almost everything in bucket 7 and leave six layer stacks untested;
* **threat density**, i.e. how many ``FullThreats`` features are active, because
  that is what the accumulator gather loops over. Scoping measured 24–40 active
  threats typical with a hard cap of 128; a sample that never reaches the tail
  never exercises the wide case.

⚑ IN-CHECK POSITIONS ARE EXCLUDED BY CONSTRUCTION, and the excluded fraction is
reported rather than silently dropped. The NNUE evaluation is undefined in
check: Stockfish asserts ``!pos.checkers()`` and its ``eval`` command prints
``Final evaluation: none (in check)`` with no network lines at all, so an
in-check FEN in the sample would not be a hard case — it would be an unanswerable
one. Our evaluator refuses them at the seam for the same reason.

Positions come from seeded random playouts with a capture bias, so the sampler
is self-contained (no data files, no network) and reproducible from the seed
alone. The capture bias is what reaches the low-piece-count buckets; pure
uniform-random play almost never does.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import chess

from scripts.nnue_reference import position_view, threat_active_indices

#: Threat-density bin edges (number of active FullThreats features, perspective
#: WHITE). Chosen around the 24–40 typical band measured during scoping so the
#: sparse and the wide tail both get their own cells.
THREAT_BIN_EDGES: tuple[int, ...] = (16, 32, 48)

N_BUCKETS = 8
N_THREAT_BINS = len(THREAT_BIN_EDGES) + 1


def threat_bin(active: int) -> int:
    """Which density bin a position with ``active`` threat features lands in."""
    for i, edge in enumerate(THREAT_BIN_EDGES):
        if active < edge:
            return i
    return len(THREAT_BIN_EDGES)


@dataclass
class SampleStats:
    """What the sampler saw, so a caller can report it instead of assuming it."""

    considered: int = 0
    in_check_excluded: int = 0
    accepted: int = 0
    cell_counts: Counter[tuple[int, int]] = field(default_factory=Counter)

    @property
    def in_check_fraction(self) -> float:
        return self.in_check_excluded / self.considered if self.considered else 0.0

    def coverage_report(self) -> str:
        lines = [
            f"considered        : {self.considered:,}",
            f"in-check excluded : {self.in_check_excluded:,} "
            f"({100.0 * self.in_check_fraction:.2f}% of considered)",
            f"accepted          : {self.accepted:,}",
            "cells (rows = piece-count bucket, cols = threat-density bin):",
        ]
        header = "        " + "".join(f"{b:>10}" for b in range(N_THREAT_BINS))
        lines.append(header)
        for bucket in range(N_BUCKETS):
            row = "".join(f"{self.cell_counts[(bucket, tb)]:>10}" for tb in range(N_THREAT_BINS))
            lines.append(f"  b{bucket}   {row}")
        empty = [
            (b, t)
            for b in range(N_BUCKETS)
            for t in range(N_THREAT_BINS)
            if self.cell_counts[(b, t)] == 0
        ]
        if empty:
            lines.append(f"  EMPTY CELLS: {empty}")
        return "\n".join(lines)


def _playout(rng: random.Random, capture_bias: float) -> list[chess.Board]:
    """One seeded random game; returns a snapshot of every position visited."""
    board = chess.Board()
    seen: list[chess.Board] = [board.copy(stack=False)]
    for _ in range(300):
        if board.is_game_over(claim_draw=False):
            break
        moves = list(board.legal_moves)
        captures = [m for m in moves if board.is_capture(m)]
        pool = captures if (captures and rng.random() < capture_bias) else moves
        board.push(rng.choice(pool))
        seen.append(board.copy(stack=False))
    return seen


def sample_fens(
    count: int,
    seed: int = 20260824,
    capture_bias: float = 0.55,
    max_playouts: int = 200_000,
    stall_limit: int = 150,
) -> tuple[list[str], SampleStats]:
    """Return up to ``count`` distinct legal, not-in-check FENs plus sampling stats.

    Candidates are collected into (bucket, threat-bin) cells and then drawn
    ROUND-ROBIN across cells, so a cell random play floods (deep endgame shuffles
    land in bucket 0 by the thousand) cannot crowd out a thin one — a flat
    arrival-order sample would be ~all bucket 0/1 and would leave six layer
    stacks untested.

    Some cells are structurally thin or unreachable — a 3-piece board cannot
    carry 48 active threats — so the stall counter is PER CELL. ⚑ A single
    global counter reset on any cell's growth, which meant one thin cell
    trickling in a position every few playouts kept the whole loop alive until
    every OTHER cell hit its cap: at count=50000 that turned a two-minute sample
    into an hour of work nobody asked for. A cell is finished when it is full OR
    when it has gone ``stall_limit`` playouts without growing, and collection
    ends once every cell is finished.

    The realised cell counts are reported, never assumed.
    """
    rng = random.Random(seed)
    stats = SampleStats()
    per_cell_cap = max(4, 4 * count // (N_BUCKETS * N_THREAT_BINS))
    by_cell: dict[tuple[int, int], list[str]] = {}
    cell_stall: dict[tuple[int, int], int] = {}
    seen_fens: set[str] = set()

    playouts = 0
    while playouts < max_playouts:
        playouts += 1
        added = 0
        grew: set[tuple[int, int]] = set()
        for board in _playout(rng, capture_bias):
            stats.considered += 1
            if board.is_check():
                stats.in_check_excluded += 1
                continue
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                continue
            # Cheap pre-filter: the bucket is a popcount, the threat bin is not.
            # Random play floods the low buckets with endgame shuffles, so
            # skipping a saturated bucket before classifying its threats is the
            # difference between minutes and an hour at 50k.
            bucket = (bin(board.occupied).count("1") - 1) // 4
            if all(
                len(by_cell.get((bucket, tb), ())) >= per_cell_cap for tb in range(N_THREAT_BINS)
            ):
                continue
            pos = position_view(board)
            cell = (bucket, threat_bin(len(threat_active_indices(0, pos))))
            cell_list = by_cell.setdefault(cell, [])
            if len(cell_list) >= per_cell_cap:
                continue
            fen = board.fen()
            if fen in seen_fens:
                continue
            cell_list.append(fen)
            seen_fens.add(fen)
            grew.add(cell)
            added += 1

        for cell in by_cell:
            cell_stall[cell] = 0 if cell in grew else cell_stall.get(cell, 0) + 1
        finished = all(
            len(values) >= per_cell_cap or cell_stall.get(cell, 0) >= stall_limit
            for cell, values in by_cell.items()
        )
        drawable = sum(len(v) for v in by_cell.values())
        if by_cell and (finished or (drawable >= count and _cells_balanced(by_cell, count))):
            break

    ordered: list[str] = []
    index = 0
    while len(ordered) < count:
        drew = False
        for cell in sorted(by_cell):
            values = by_cell[cell]
            if index < len(values):
                drew = True
                ordered.append(values[index])
                stats.cell_counts[cell] += 1
                if len(ordered) == count:
                    break
        if not drew:
            break
        index += 1

    stats.accepted = len(ordered)
    return ordered, stats


def _cells_balanced(by_cell: dict[tuple[int, int], list[str]], count: int) -> bool:
    """True once every reachable cell holds its round-robin share of ``count``."""
    if not by_cell:
        return False
    share = max(1, count // len(by_cell))
    return all(len(v) >= share for v in by_cell.values())


def sample_fens_pooled(
    count: int,
    seeds: int,
    base_seed: int = 20260824,
    **kwargs: object,
) -> tuple[list[str], SampleStats]:
    """Pool ``seeds`` independent stratified samples into one of ``count`` FENs.

    ⚑ THIS EXISTS BECAUSE A SINGLE SEED DOES NOT SCALE, and the reason is worth
    stating rather than hiding behind a knob. Some (bucket, threat-density) cells
    are structurally thin — random play produces a 20-piece board with 48 active
    threats rarely — so the per-cell target that a 5k sample fills in four
    minutes would take hours at 50k, and all of the extra time goes into chasing
    cells that will never fill. Ten independent 5k draws reach 50k in the time
    ten 5k draws take, cover the same cells, and carry MORE seed diversity than
    one long run would.

    Each sub-sample is reproducible from ``base_seed + i`` alone, so the pooled
    sample is reproducible from (count, seeds, base_seed).
    """
    per_seed = -(-count // max(1, seeds))
    pooled: list[str] = []
    seen: set[str] = set()
    total = SampleStats()
    for i in range(seeds):
        chunk, stats = sample_fens(per_seed, seed=base_seed + i, **kwargs)  # pyright: ignore[reportArgumentType]
        total.considered += stats.considered
        total.in_check_excluded += stats.in_check_excluded
        for fen in chunk:
            if fen in seen:
                continue
            seen.add(fen)
            pooled.append(fen)
        total.cell_counts.update(stats.cell_counts)
        if len(pooled) >= count:
            break
    pooled = pooled[:count]
    total.accepted = len(pooled)
    return pooled, total
