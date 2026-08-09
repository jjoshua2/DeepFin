"""Board-aware permutation between our compact-1858 policy order and Leela's.

Our ``lc0_1858`` encoding is NOT Leela's 1858 ordering. Both enumerate 1858
slots, but ours is "the geometrically valid subset of AZ-4672, in AZ-4672
order" (``COMPACT_TO_FULL_POLICY``) while Leela's is the verbatim
``kMoveStrs[]`` table (:mod:`chess_anti_engine.moves.lc0_1858_movestrs`).
They agree on only 46 of 1858 slots — reading a real Leela net's policy
through our table is a silent 1812-slot permutation error.

The two conventions also differ in WHICH promotion shares the plain
from/to slot:

===================  ================================  ==========================
back-rank slot       ours (AZ-4672)                    Leela (kMoveStrs)
===================  ================================  ==========================
``a7a8``             non-pawn slide **or** promo-to-Q  non-pawn slide **or** =N
``a7a8q``            (no such slot)                    promotion to queen
``a7a8n``            promotion to knight               (no such slot)
``a7a8b/r``          promotion to bishop/rook          promotion to bishop/rook
===================  ================================  ==========================

So the 22 shared back-rank slots (8th-rank targets from the 7th rank, in
side-to-move-oriented coordinates) cannot be mapped without knowing whether
the piece on the from-square is a pawn. ``onnx/load.py``'s static
``build_lc0_policy_remap`` guesses "always the queen-promotion slot", which
misreads a rook/queen/king slide onto the back rank.

The second disagreement is castling: Leela spells it king-takes-rook
(``e1h1`` / ``e1a1``) even in standard chess, while our AZ-4672 slot is the
king's two-square move (``e1g1`` / ``e1c1``). Leela's table also *has* an
``e1g1`` entry — an ordinary slide — so the static remap silently reads the
wrong logit rather than failing.

Measured over 8,972 random self-play positions (275,943 legal moves), the
static remap mis-maps at least one legal move in 9.3% of positions: 1,048
back-rank slides and 319 castles. The maps here take the two bits of board
context those cases need and are exact everywhere.

All UCI strings are in side-to-move-oriented coordinates (rank-flipped for
black), which is what both ``move_to_index`` and Leela's policy head use.
"""
from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    PLANE_COUNT,
    UNDERPROMO_DFS,
    UNDERPROMO_PIECES,
    _PLANE_TO_DELTA,
    compact_policy_index,
    move_to_index,
)
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX

_PROMO_CHARS = {chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r"}


def compact_move_str(compact_index: int) -> str:
    """Oriented UCI for one of OUR compact-1858 slots, in OUR promo semantics.

    A bare 4-char string means "queen-like move" — which on the 7th→8th rank
    covers both a non-pawn slide and a promotion to queen.
    """
    full = int(COMPACT_TO_FULL_POLICY[int(compact_index)])
    from_sq = full // PLANE_COUNT
    plane = full % PLANE_COUNT
    ff = chess.square_file(from_sq)
    fr = chess.square_rank(from_sq)
    if plane >= 64:
        rel = plane - 64
        to_sq = chess.square(ff + UNDERPROMO_DFS[rel % 3], 7)
        promo = _PROMO_CHARS[UNDERPROMO_PIECES[rel // 3]]
        return chess.square_name(from_sq) + chess.square_name(to_sq) + promo
    df, dr = _PLANE_TO_DELTA[plane]
    to_sq = chess.square(ff + df, fr + dr)
    return chess.square_name(from_sq) + chess.square_name(to_sq)


def _build_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(slide-map, pawn-map, ambiguous compact slots, their from-squares)."""
    slide = np.full((COMPACT_POLICY_SIZE,), -1, dtype=np.int64)
    pawn = np.full((COMPACT_POLICY_SIZE,), -1, dtype=np.int64)
    amb_slots: list[int] = []
    amb_from: list[int] = []
    for c in range(COMPACT_POLICY_SIZE):
        uci = compact_move_str(c)
        bare = uci[:4]
        if len(uci) == 5:
            # Bishop/rook underpromotions have their own slot in both
            # conventions; knight promotion resolves through the "...n" alias
            # onto Leela's bare entry.
            slide[c] = pawn[c] = LC0_1858_UCI_TO_IDX[uci]
            continue
        leela_bare = LC0_1858_UCI_TO_IDX[bare]
        slide[c] = leela_bare
        promo_q = bare + "q"
        if promo_q in LC0_1858_UCI_TO_IDX:
            # Shared back-rank slot: a pawn here is promoting to a queen.
            pawn[c] = LC0_1858_UCI_TO_IDX[promo_q]
            amb_slots.append(c)
            amb_from.append(int(COMPACT_TO_FULL_POLICY[c]) // PLANE_COUNT)
        else:
            pawn[c] = leela_bare
    if int((slide < 0).sum()) or int((pawn < 0).sum()):
        raise RuntimeError("compact→Leela map has unmapped slots")
    return slide, pawn, np.array(amb_slots, dtype=np.int64), np.array(amb_from, dtype=np.int64)


(
    COMPACT_TO_LEELA_SLIDE,
    COMPACT_TO_LEELA_PAWN,
    AMBIGUOUS_COMPACT_SLOTS,
    AMBIGUOUS_FROM_SQUARES,
) = _build_maps()
"""``COMPACT_TO_LEELA_SLIDE[c]`` / ``[PAWN][c]``: Leela slot for our compact
slot ``c`` when the from-square holds a non-pawn / a pawn. They differ only on
``AMBIGUOUS_COMPACT_SLOTS`` (the 22 shared back-rank slots), whose oriented
from-squares are ``AMBIGUOUS_FROM_SQUARES``."""


# Leela spells castling king-takes-rook (chess960 style) even in standard chess:
# the policy slot for O-O is "e1h1", NOT the "e1g1" our AZ-4672 king move uses.
# The classic 112-plane input carries castling as four boolean planes with no
# rook file, so chess960 rook placement is not recoverable from planes at all;
# standard placement (h-file / a-file rook, king on e1 in oriented coordinates)
# is the only thing that format can mean.
CASTLE_KINGSIDE_COMPACT: int = compact_policy_index(
    move_to_index(chess.Move.from_uci("e1g1"), chess.Board()),
)
CASTLE_QUEENSIDE_COMPACT: int = compact_policy_index(
    move_to_index(chess.Move.from_uci("e1c1"), chess.Board()),
)
CASTLE_KINGSIDE_LEELA: int = LC0_1858_UCI_TO_IDX["e1h1"]
CASTLE_QUEENSIDE_LEELA: int = LC0_1858_UCI_TO_IDX["e1a1"]


def leela_gather_indices(
    pawn_mask: np.ndarray, castling: np.ndarray | None = None,
) -> np.ndarray:
    """(B, 1858) int64 Leela slots for our compact slots, per position.

    ``pawn_mask`` is (B, 64) truthy: side-to-move's pawns indexed by ORIENTED
    square (``rank * 8 + file`` after the black rank-flip), i.e. exactly plane
    0 of an ``lc0_root`` encoding flattened. ``castling`` is (B, 2) truthy —
    the side to move's (kingside, queenside) rights. Both come straight off the
    112-plane input; see
    :func:`chess_anti_engine.encoding.lc0.lc0_gather_context_from_planes`.

    Without ``castling`` the O-O / O-O-O slots resolve to Leela's ordinary
    ``e1g1`` / ``e1c1`` *slide* slots, which are illegal whenever castling is
    legal — so the net's castling logit is missed and the move gets junk prior.
    """
    mask = np.asarray(pawn_mask)
    if mask.ndim != 2 or mask.shape[1] != 64:
        raise ValueError(f"pawn_mask must be (B, 64); got {mask.shape}")
    out = np.repeat(COMPACT_TO_LEELA_SLIDE[None, :], mask.shape[0], axis=0)
    has_pawn = mask[:, AMBIGUOUS_FROM_SQUARES] > 0  # (B, 22)
    out[:, AMBIGUOUS_COMPACT_SLOTS] = np.where(
        has_pawn,
        COMPACT_TO_LEELA_PAWN[AMBIGUOUS_COMPACT_SLOTS][None, :],
        COMPACT_TO_LEELA_SLIDE[AMBIGUOUS_COMPACT_SLOTS][None, :],
    )
    if castling is not None:
        cst = np.asarray(castling)
        if cst.shape != (mask.shape[0], 2):
            raise ValueError(f"castling must be ({mask.shape[0]}, 2); got {cst.shape}")
        rows = np.flatnonzero(cst[:, 0] > 0)
        out[rows, CASTLE_KINGSIDE_COMPACT] = CASTLE_KINGSIDE_LEELA
        rows = np.flatnonzero(cst[:, 1] > 0)
        out[rows, CASTLE_QUEENSIDE_COMPACT] = CASTLE_QUEENSIDE_LEELA
    return out


def leela_policy_to_compact(
    policy_leela: np.ndarray, pawn_mask: np.ndarray, castling: np.ndarray | None = None,
) -> np.ndarray:
    """Reorder a real Leela (B, 1858) policy into OUR compact-1858 order."""
    pol = np.asarray(policy_leela, dtype=np.float32)
    if pol.ndim != 2 or pol.shape[1] != COMPACT_POLICY_SIZE:
        raise ValueError(f"policy must be (B, {COMPACT_POLICY_SIZE}); got {pol.shape}")
    return np.take_along_axis(pol, leela_gather_indices(pawn_mask, castling), axis=1)


def leela_index_for_move(board: chess.Board, move: chess.Move) -> int:
    """Leela 1858 slot for ``move`` on ``board`` — the independent reference.

    Derived straight from the move (orient, then apply Leela's castling and
    promotion spelling), NOT from our compact tables, so a test comparing this
    against :func:`leela_gather_indices` genuinely cross-checks the two
    conventions rather than restating one of them.
    """
    to_square = move.to_square
    if board.is_castling(move):
        rank_mask = chess.BB_RANK_1 if chess.square_rank(move.from_square) == 0 else chess.BB_RANK_8
        rooks = board.castling_rights & rank_mask
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        to_square = chess.msb(rooks) if kingside else chess.lsb(rooks)
    from_o = chess.square_name(_orient(move.from_square, board.turn))
    to_o = chess.square_name(_orient(to_square, board.turn))
    # Leela spells queen/rook/bishop promotions with a suffix and knight
    # promotion as the bare move.
    suffix = "" if move.promotion is None else _LEELA_PROMO_SUFFIX[int(move.promotion)]
    return LC0_1858_UCI_TO_IDX.get(from_o + to_o + suffix, -1)


_LEELA_PROMO_SUFFIX: dict[int, str] = {
    chess.QUEEN: "q",
    chess.ROOK: "r",
    chess.BISHOP: "b",
    chess.KNIGHT: "",
}


def _orient(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def compact_index_for_move(board: chess.Board, move: chess.Move) -> int:
    """Our compact-1858 slot for ``move`` (via the AZ-4672 index)."""
    return compact_policy_index(move_to_index(move, board))
