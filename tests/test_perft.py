"""Perft: exact published node counts through CBoard's own move generator.

⚑ WHY THIS FILE EXISTS AT ALL. The table-backed sliders
(`encoding/_slider_attacks_impl.h`) replace the generator's ray walkers, and the
whole change rests on an oracle. Until this file landed the repo had no perft
test of any kind, so the strongest available oracle for a move generator — the
externally published, independently reproduced node counts of the standard
positions — was going unused while the acceptance criterion in
`docs/nnue_speed_plan.md` §3 named it explicitly.

⚑ WHY PERFT AND NOT MORE PARITY TESTS. `tests/test_cboard_move_parity.py`
already diffs CBoard against python-chess, and that is a good test, but it is an
INTERNAL comparison against another implementation in this same process. Perft
counts are EXTERNAL: they were computed by other engines, by other people, and
they are the same numbers regardless of what either library here believes. A
rule that is wrong in both CBoard and python-chess — or a corpus that never
reaches the position that would expose it — cannot be caught by parity, and can
be caught here. Depth 3+ is where that bites: a castling, en-passant or
promotion rule that is subtly wrong shows up as a node-count difference several
plies below where the defect lives.

The counts below are the canonical Chess Programming Wiki perft suite
(startpos plus "Kiwipete" and positions 3-6). They are NOT derived from this
repo, and must never be regenerated from it — a self-generated expectation
would turn this file into a change detector for whatever the generator happens
to do today. If a count here disagrees with the code, the code is wrong.

Depths are chosen to keep the whole file well inside a CI budget (~520k leaf
nodes, a few seconds) while still reaching the depth at which each position's
characteristic rule interactions compose.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard

#: (label, fen, [(depth, nodes), ...]) — Chess Programming Wiki perft results.
PERFT_SUITE: list[tuple[str, str, list[tuple[int, int]]]] = [
    (
        "startpos",
        chess.STARTING_FEN,
        [(1, 20), (2, 400), (3, 8902), (4, 197281)],
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [(1, 48), (2, 2039), (3, 97862)],
    ),
    (
        "position3_rook_endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        [(1, 14), (2, 191), (3, 2812), (4, 43238)],
    ),
    (
        "position4_promotions",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [(1, 6), (2, 264), (3, 9467)],
    ),
    (
        "position5_promotion_race",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [(1, 44), (2, 1486), (3, 62379)],
    ),
    (
        "position6_symmetric_middlegame",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [(1, 46), (2, 2079), (3, 89890)],
    ),
]


def _perft(board: CBoard, depth: int) -> int:
    """Count leaf nodes at `depth` using CBoard's generator alone.

    ⚑ `push_index` mutates IN PLACE and returns None, so the recursion has to
    descend into `copy()`. Pushing on the shared board and expecting an unmake
    would silently count a different tree.
    """
    indices = board.legal_move_indices()
    if depth == 1:
        return len(indices)
    total = 0
    for index in indices:
        child = board.copy()
        child.push_index(int(index))
        total += _perft(child, depth - 1)
    return total


@pytest.mark.parametrize(
    ("label", "fen", "expectations"),
    PERFT_SUITE,
    ids=[row[0] for row in PERFT_SUITE],
)
def test_perft_matches_published_node_counts(
    label: str, fen: str, expectations: list[tuple[int, int]]
) -> None:
    board = chess.Board(fen)
    for depth, expected in expectations:
        actual = _perft(CBoard.from_board(board), depth)
        assert actual == expected, (
            f"{label} perft({depth}) = {actual}, published count is {expected}"
        )


@pytest.mark.parametrize(
    ("label", "fen"),
    [(row[0], row[1]) for row in PERFT_SUITE],
    ids=[row[0] for row in PERFT_SUITE],
)
def test_policy_indices_are_one_per_legal_move(label: str, fen: str) -> None:
    """Perft through a policy-index space is only valid if the map is injective.

    ⚑ This is the assumption that would make every count above wrong in the same
    direction and still look self-consistent. `legal_move_indices()` returns
    sorted indices in the 4672 action space; if two distinct legal moves ever
    collapsed onto one index the array would be short, perft would undercount at
    every depth, and the failure would read as a movegen bug rather than as an
    encoding bug. Depth-1 agreement with python-chess pins it directly.
    """
    board = chess.Board(fen)
    indices = CBoard.from_board(board).legal_move_indices()

    assert len(set(map(int, indices))) == len(indices), (
        f"{label}: duplicate policy indices — the action encoding collapsed two "
        "distinct legal moves onto one index"
    )
    assert len(indices) == board.legal_moves.count(), (
        f"{label}: CBoard generated {len(indices)} moves, python-chess "
        f"{board.legal_moves.count()}"
    )
    assert np.array_equal(indices, np.sort(indices))
