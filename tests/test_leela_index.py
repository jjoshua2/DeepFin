"""Our compact-1858 order is not Leela's, and the remap must survive the two
places the conventions genuinely disagree: which promotion shares the plain
back-rank slot, and how castling is spelled."""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import (
    castling_from_lc0_planes,
    lc0_gather_context_from_planes,
    pawn_mask_from_lc0_planes,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_MOVE_STRS
from chess_anti_engine.moves.leela_index import (
    AMBIGUOUS_COMPACT_SLOTS,
    COMPACT_TO_LEELA_PAWN,
    COMPACT_TO_LEELA_SLIDE,
    compact_index_for_move,
    compact_move_str,
    leela_gather_indices,
    leela_index_for_move,
    leela_policy_to_compact,
)
from chess_anti_engine.utils.bitboards import orient_square

CASTLING_FENS = (
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
)
PROMOTION_FENS = (
    "8/PPPPPPPP/8/8/8/7k/8/K7 w - - 0 1",
    "k7/8/7K/8/8/8/pppppppp/8 b - - 0 1",
    # non-pawn slides onto the back rank: the case a static remap gets wrong
    "7k/R7/8/8/8/8/8/7K w - - 0 1",
    "6k1/3Q4/8/8/8/8/8/6K1 w - - 0 1",
    "7K/r7/8/8/8/8/8/7k b - - 0 1",
)


def _random_boards(n_games: int = 25, seed: int = 11) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    for _ in range(n_games):
        board = chess.Board()
        for _ply in range(rng.randint(0, 110)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            boards.append(board.copy(stack=8))
    return boards


ALL_BOARDS = [chess.Board(f) for f in CASTLING_FENS + PROMOTION_FENS] + _random_boards()


def test_our_compact_order_is_not_leelas() -> None:
    """The premise: reading a Leela policy through our table is a permutation
    error, so the remap is load-bearing rather than decorative."""
    ours = [compact_move_str(c) for c in range(COMPACT_POLICY_SIZE)]
    same = sum(1 for a, b in zip(ours, LC0_1858_MOVE_STRS, strict=True) if a == b)
    assert same < 100, "if the orderings agreed, leela_index would be pointless"
    # Same move SET, different spelling of the two shared back-rank slots.
    assert set(ours) - set(LC0_1858_MOVE_STRS) == {
        s for s in ours if s.endswith("n")
    }
    assert set(LC0_1858_MOVE_STRS) - set(ours) == {
        s for s in LC0_1858_MOVE_STRS if s.endswith("q")
    }


def test_every_legal_move_maps_to_the_same_uci_in_both_conventions() -> None:
    checked = 0
    kinds: dict[str, int] = {}
    for board in ALL_BOARDS:
        if not any(board.legal_moves):
            continue
        planes = encode_position(board, add_features=False, input_history_encoding="lc0_root")
        gather = leela_gather_indices(*lc0_gather_context_from_planes(planes))[0]
        for move in board.legal_moves:
            reference = leela_index_for_move(board, move)
            assert reference >= 0, f"{board.fen()} {move.uci()} has no Leela slot"
            got = int(gather[compact_index_for_move(board, move)])
            assert got == reference, (
                f"{board.fen()} {move.uci()}: gather says "
                f"{LC0_1858_MOVE_STRS[got]}, reference says {LC0_1858_MOVE_STRS[reference]}"
            )
            kind = ("castle" if board.is_castling(move)
                    else f"promo_{move.promotion}" if move.promotion else "normal")
            kinds[kind] = kinds.get(kind, 0) + 1
            checked += 1
    assert checked > 10_000
    # A pass that never exercised the disagreeing cases proves nothing.
    assert kinds.get("castle", 0) > 0
    assert {f"promo_{p}" for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)} <= set(kinds)


def test_back_rank_slot_follows_the_piece_not_a_fixed_guess() -> None:
    """The 22 shared slots resolve to the queen-promotion slot for a pawn and
    to the plain slide for anything else — a static table cannot do both."""
    pawn_board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    rook_board = chess.Board("7k/R7/8/8/8/8/8/7K w - - 0 1")
    slot = compact_index_for_move(pawn_board, chess.Move.from_uci("a7a8q"))
    assert slot in set(AMBIGUOUS_COMPACT_SLOTS.tolist())
    assert slot == compact_index_for_move(rook_board, chess.Move.from_uci("a7a8"))

    def gathered(board: chess.Board) -> str:
        planes = encode_position(board, add_features=False, input_history_encoding="lc0_root")
        gather = leela_gather_indices(*lc0_gather_context_from_planes(planes))[0]
        return LC0_1858_MOVE_STRS[int(gather[slot])]

    assert gathered(pawn_board) == "a7a8q"
    assert gathered(rook_board) == "a7a8"
    assert COMPACT_TO_LEELA_PAWN[slot] != COMPACT_TO_LEELA_SLIDE[slot]


def test_castling_uses_leelas_king_takes_rook_slot() -> None:
    board = chess.Board(CASTLING_FENS[0])
    planes = encode_position(board, add_features=False, input_history_encoding="lc0_root")
    gather = leela_gather_indices(*lc0_gather_context_from_planes(planes))[0]
    for uci, leela in (("e1g1", "e1h1"), ("e1c1", "e1a1")):
        slot = compact_index_for_move(board, chess.Move.from_uci(uci))
        assert LC0_1858_MOVE_STRS[int(gather[slot])] == leela
    # Without the castling context the same slot falls back to the slide entry,
    # which is what makes passing it load-bearing rather than cosmetic.
    plain = leela_gather_indices(pawn_mask_from_lc0_planes(planes))[0]
    slot = compact_index_for_move(board, chess.Move.from_uci("e1g1"))
    assert LC0_1858_MOVE_STRS[int(plain[slot])] == "e1g1"


def test_plane_context_matches_the_board() -> None:
    for board in ALL_BOARDS[:60]:
        planes = encode_position(board, add_features=False, input_history_encoding="lc0_root")
        mask = pawn_mask_from_lc0_planes(planes)[0]
        want = {int(orient_square(sq, board.turn)) for sq in board.pieces(chess.PAWN, board.turn)}
        assert {int(i) for i in np.flatnonzero(mask)} == want
        kingside, queenside = castling_from_lc0_planes(planes)[0]
        assert bool(kingside) == board.has_kingside_castling_rights(board.turn)
        assert bool(queenside) == board.has_queenside_castling_rights(board.turn)


def test_leela_policy_to_compact_moves_each_logit_to_its_move() -> None:
    board = chess.Board(CASTLING_FENS[2])
    planes = encode_position(board, add_features=False, input_history_encoding="lc0_root")
    leela_policy = np.arange(COMPACT_POLICY_SIZE, dtype=np.float32)[None, :]
    ours = leela_policy_to_compact(leela_policy, *lc0_gather_context_from_planes(planes))
    for move in board.legal_moves:
        slot = compact_index_for_move(board, move)
        assert float(ours[0, slot]) == float(leela_index_for_move(board, move))


def test_gather_rejects_malformed_context() -> None:
    with pytest.raises(ValueError, match="pawn_mask must be"):
        leela_gather_indices(np.zeros((2, 63), dtype=bool))
    with pytest.raises(ValueError, match="castling must be"):
        leela_gather_indices(np.zeros((2, 64), dtype=bool), np.zeros((3, 2), dtype=bool))
