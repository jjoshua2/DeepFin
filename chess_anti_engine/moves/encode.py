from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.utils.bitboards import orient_square

try:
    from chess_anti_engine.encoding._lc0_ext import (
        legal_move_policy_indices as _c_legal_move_policy_indices,
    )
    _HAS_LC0_C_EXT = True
except ImportError:
    _HAS_LC0_C_EXT = False

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import (  # noqa: F401,F811
        legal_move_policy_indices as _c_legal_move_policy_indices,
    )

# LC0 / AlphaZero-style policy encoding: 8x8x73 = 4672.
#
# Indexing:
# - First orient the move into side-to-move perspective (white-to-move).
# - from_sq is in oriented coordinates (0..63, a1=0 .. h8=63).
# - plane is 0..72:
#   - 0..55 : queen-like moves = 8 directions x 7 distances
#   - 56..63: knight moves = 8
#   - 64..72: underpromotions = 3 piece types (N,B,R) x 3 directions (left,forward,right)
# - flat index = from_sq * 73 + plane
#
# Data augmentation note:
# We also support mirroring moves across the *vertical axis* (left-right file flip)
# in oriented (side-to-move) coordinates.

PLANE_COUNT = 73
POLICY_SIZE = 64 * PLANE_COUNT  # 4672

QUEEN_DIRS: list[tuple[int, int]] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]

KNIGHT_DELTAS: list[tuple[int, int]] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

UNDERPROMO_TO_IDX = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
UNDERPROMO_DFS: list[int] = [-1, 0, 1]  # left, forward, right
DF_TO_UNDERPROMO_DIR = {-1: 0, 0: 1, 1: 2}

_DELTA_TO_PLANE: dict[tuple[int, int], int] = {}
_PLANE_TO_DELTA: dict[int, tuple[int, int]] = {}

_plane = 0
for dfi, dri in QUEEN_DIRS:
    for dist in range(1, 8):
        df, dr = dfi * dist, dri * dist
        _DELTA_TO_PLANE[(df, dr)] = _plane
        _PLANE_TO_DELTA[_plane] = (df, dr)
        _plane += 1

for i, (df, dr) in enumerate(KNIGHT_DELTAS):
    p = 56 + i
    _DELTA_TO_PLANE[(df, dr)] = p
    _PLANE_TO_DELTA[p] = (df, dr)


def mirror_oriented_square(sq: chess.Square) -> chess.Square:
    """Mirror an *oriented* square left-right.

    Oriented squares follow the policy encoding convention (side-to-move as white):
    a1<->h1, a8<->h8.

    This is distinct from python-chess's `square_mirror`, which mirrors ranks.
    """
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return chess.square(7 - f, r)


def mirror_policy_index(index: int) -> int:
    """Map a policy index to the corresponding index under a left-right mirror."""
    idx = int(index)
    from_o = idx // PLANE_COUNT
    plane = idx % PLANE_COUNT

  # Mirror the origin square.
    f_o = chess.Square(from_o)
    f_m = mirror_oriented_square(f_o)

    if plane >= 64:
  # Underpromotions: mirror df direction (left<->right), keep piece type.
        rel = plane - 64
        piece_idx = rel // 3
        dir_idx = rel % 3
        dir_m = 2 - int(dir_idx)  # 0<->2, 1 stays
        plane_m = 64 + int(piece_idx) * 3 + int(dir_m)
        return int(f_m) * PLANE_COUNT + int(plane_m)

    delta = _PLANE_TO_DELTA.get(int(plane))
    if delta is None:
  # Should be unreachable for valid indices.
        return idx
    df, dr = delta
    plane_m = _DELTA_TO_PLANE.get((-int(df), int(dr)))
    if plane_m is None:
        return idx

    return int(f_m) * PLANE_COUNT + int(plane_m)


# Flat index permutations for mirroring in oriented coordinates.
#
# Convention:
# - MIRROR_POLICY_MAP[old] = new
# - MIRROR_POLICY_INV[new] = old (inverse permutation)
MIRROR_POLICY_MAP = np.array([mirror_policy_index(i) for i in range(POLICY_SIZE)], dtype=np.int32)
MIRROR_POLICY_INV = np.empty((POLICY_SIZE,), dtype=np.int32)
MIRROR_POLICY_INV[MIRROR_POLICY_MAP] = np.arange(POLICY_SIZE, dtype=np.int32)


# Precomputed reverse LUT: policy index → (from_sq, to_sq, promotion) in real coordinates.
# Shape: (2, POLICY_SIZE, 3) — axis 0: int(turn) (0=BLACK, 1=WHITE).
# [:,idx,0] = from_sq, [:,idx,1] = to_sq, [:,idx,2] = promotion piece type (0=none, 2=knight, 3=bishop, 4=rook, 5=queen)
_INDEX_TO_MOVE_LUT = np.full((2, POLICY_SIZE, 3), -1, dtype=np.int16)

for _turn in (chess.WHITE, chess.BLACK):
    _ti = int(_turn)
    for _idx in range(POLICY_SIZE):
        _from_o = _idx // PLANE_COUNT
        _plane = _idx % PLANE_COUNT
        if _plane >= 64:
            _rel = _plane - 64
            _piece_idx = _rel // 3
            _dir_idx = _rel % 3
            _df = UNDERPROMO_DFS[_dir_idx]
            _dr = 1
            _ff, _fr = chess.square_file(_from_o), chess.square_rank(_from_o)
            _tf, _tr = _ff + _df, _fr + _dr
            if 0 <= _tf <= 7 and 0 <= _tr <= 7:
                _to_o = chess.square(_tf, _tr)
                _f_real = orient_square(_from_o, _turn)
                _t_real = orient_square(_to_o, _turn)
                _promo = [chess.KNIGHT, chess.BISHOP, chess.ROOK][_piece_idx]
                _INDEX_TO_MOVE_LUT[_ti, _idx] = [_f_real, _t_real, _promo]
        else:
            _delta = _PLANE_TO_DELTA.get(_plane)
            if _delta is not None:
                _df, _dr = _delta
                _ff, _fr = chess.square_file(_from_o), chess.square_rank(_from_o)
                _tf, _tr = _ff + _df, _fr + _dr
                if 0 <= _tf <= 7 and 0 <= _tr <= 7:
                    _to_o = chess.square(_tf, _tr)
                    _f_real = orient_square(_from_o, _turn)
                    _t_real = orient_square(_to_o, _turn)
  # Check for queen promotion (pawn reaching last rank)
                    _promo_val = 0
                    _real_rank = chess.square_rank(_t_real)
                    if _real_rank == 7 or _real_rank == 0:
  # Could be pawn promotion — mark as queen
                        _promo_val = chess.QUEEN
                    _INDEX_TO_MOVE_LUT[_ti, _idx] = [_f_real, _t_real, _promo_val]


def index_to_move_fast(index: int, board: chess.Board) -> chess.Move:
    """Fast index → move using precomputed LUT."""
    entry = _INDEX_TO_MOVE_LUT[int(board.turn), int(index)]
    f, t, promo = int(entry[0]), int(entry[1]), int(entry[2])
    if f < 0:
        return next(iter(board.legal_moves))

  # Only apply promotion if a pawn is actually on the from square
    promotion = None
    if promo > 0:
        piece = board.piece_at(f)
        if piece is not None and piece.piece_type == chess.PAWN:
            promotion = promo

    m = chess.Move(f, t, promotion=promotion)
    if m in board.legal_moves:
        return m
  # Fallback (rare edge cases)
    for lm in board.legal_moves:
        if move_to_index(lm, board) == index:
            return lm
    return next(iter(board.legal_moves))


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a (POLICY_SIZE,) policy vector left-right.

    Accepts any float dtype; returns float32.
    """
    p = np.asarray(policy)
    if p.shape != (POLICY_SIZE,):
        raise ValueError(f"policy must be ({POLICY_SIZE},), got {p.shape}")
  # new[j] = old[inv[j]]
    return p[MIRROR_POLICY_INV].astype(np.float32, copy=False)


def mirror_policy_batch(policies: np.ndarray) -> np.ndarray:
    p = np.asarray(policies)
    if p.ndim != 2 or int(p.shape[1]) != int(POLICY_SIZE):
        raise ValueError(f"policies must be (N,{POLICY_SIZE}), got {p.shape}")
    return p[:, MIRROR_POLICY_INV].astype(np.float32, copy=False)

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    turn = board.turn
    f = orient_square(move.from_square, turn)
    t = orient_square(move.to_square, turn)

    ff, fr = chess.square_file(f), chess.square_rank(f)
    tf, tr = chess.square_file(t), chess.square_rank(t)
    df = tf - ff
    dr = tr - fr

    if move.promotion is not None and move.promotion in UNDERPROMO_TO_IDX:
        dir_idx = DF_TO_UNDERPROMO_DIR.get(df, 1)
        piece_idx = UNDERPROMO_TO_IDX[move.promotion]
        plane = 64 + piece_idx * 3 + dir_idx
        return int(f) * PLANE_COUNT + int(plane)

    plane = _DELTA_TO_PLANE.get((df, dr))
    if plane is None:
        raise ValueError(f"Unencodable move delta df={df}, dr={dr} for move={move}")

    return int(f) * PLANE_COUNT + int(plane)


_UNDERPROMO_CHAR_TO_IDX = {"n": 0, "b": 1, "r": 2}


def uci_to_policy_index(uci: str, turn: bool) -> int:
    """Convert a UCI move string to a policy index without python-chess.

    ``turn`` is True for White, False for Black (same as chess.WHITE/BLACK).
    Returns -1 if the move is unencodable.
    """
    if len(uci) < 4 or not uci[0].isalpha():
        return -1
    from_sq = (ord(uci[0]) - ord("a")) + (ord(uci[1]) - ord("1")) * 8
    to_sq = (ord(uci[2]) - ord("a")) + (ord(uci[3]) - ord("1")) * 8

  # Underpromotion: 5-char UCI like "a7a8n"
    promo_char = uci[4].lower() if len(uci) == 5 else ""
    if promo_char in _UNDERPROMO_CHAR_TO_IDX:
  # Orient squares
        f = from_sq if turn else (from_sq ^ 56)
        t = to_sq if turn else (to_sq ^ 56)
        ff, tf = f % 8, t % 8
        df = tf - ff
        dir_idx = DF_TO_UNDERPROMO_DIR.get(df, 1)
        piece_idx = _UNDERPROMO_CHAR_TO_IDX[promo_char]
        plane = 64 + piece_idx * 3 + dir_idx
        return int(f) * PLANE_COUNT + int(plane)

  # Queen promotion or normal move — use precomputed LUT
    idx = int(_MOVE_INDEX_LUT[int(turn)][from_sq][to_sq])
    return idx


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a policy index back to a chess.Move. Uses precomputed LUT."""
    return index_to_move_fast(index, board)


# Precomputed (from_sq, to_sq) → policy index for non-underpromotion moves.
# Shape: (2, 64, 64) — axis 0: int(turn) (0=BLACK, 1=WHITE).
_MOVE_INDEX_LUT = np.full((2, 64, 64), -1, dtype=np.int32)

for _turn in (chess.WHITE, chess.BLACK):
    _ti = int(_turn)
    for _from_sq in range(64):
        _f = orient_square(_from_sq, _turn)
        _ff = chess.square_file(_f)
        _fr = chess.square_rank(_f)
        for _to_sq in range(64):
            if _from_sq == _to_sq:
                continue
            _t = orient_square(_to_sq, _turn)
            _tf = chess.square_file(_t)
            _tr = chess.square_rank(_t)
            _df = _tf - _ff
            _dr = _tr - _fr
            _p = _DELTA_TO_PLANE.get((_df, _dr))
            if _p is not None:
                _MOVE_INDEX_LUT[_ti][_from_sq][_to_sq] = int(_f) * PLANE_COUNT + int(_p)


def _extract_c_legal_args(board: chess.Board) -> tuple:
    """Extract bitboard args for C legal move generation (shared by mask + indices)."""
    turn = board.turn
    us_occ = int(board.occupied_co[turn])
    them_occ = int(board.occupied_co[not turn])
    pawns, knights, bishops = board.pawns, board.knights, board.bishops
    rooks, queens, kings = board.rooks, board.queens, board.kings
    return (
        int(pawns & us_occ), int(knights & us_occ),
        int(bishops & us_occ), int(rooks & us_occ),
        int(queens & us_occ), int(kings & us_occ),
        int(pawns & them_occ), int(knights & them_occ),
        int(bishops & them_occ), int(rooks & them_occ),
        int(queens & them_occ), int(kings & them_occ),
        1 if turn else 0,
        int(board.has_kingside_castling_rights(turn)),
        int(board.has_queenside_castling_rights(turn)),
        int(board.has_kingside_castling_rights(not turn)),
        int(board.has_queenside_castling_rights(not turn)),
        -1 if board.ep_square is None else board.ep_square,
    )


def legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
    if _HAS_LC0_C_EXT:
        idx = _c_legal_move_policy_indices(*_extract_c_legal_args(board))
        if idx.size > 0:
            mask[idx] = True
        return mask
    lut = _MOVE_INDEX_LUT[int(board.turn)]
    for m in board.legal_moves:
        if m.promotion is not None and m.promotion in UNDERPROMO_TO_IDX:
            mask[move_to_index(m, board)] = True
        else:
            idx = int(lut[m.from_square, m.to_square])
            if idx >= 0:
                mask[idx] = True
    return mask


def _legal_move_indices_c(board: chess.Board) -> np.ndarray:
    """C-accelerated legal move index generation (bypasses python-chess move gen)."""
    return _c_legal_move_policy_indices(*_extract_c_legal_args(board))


def _legal_move_indices_py(board: chess.Board) -> np.ndarray:
    """Python fallback for legal move index generation."""
    lut = _MOVE_INDEX_LUT[int(board.turn)]
    indices: list[int] = []
    for m in board.legal_moves:
        if m.promotion is not None and m.promotion in UNDERPROMO_TO_IDX:
            indices.append(move_to_index(m, board))
        else:
            idx = int(lut[m.from_square, m.to_square])
            if idx >= 0:
                indices.append(idx)
    return np.array(indices, dtype=np.int32)


def legal_move_indices(board: chess.Board) -> np.ndarray:
    """Return sorted int32 array of legal policy indices."""
    if _HAS_LC0_C_EXT:
        return _legal_move_indices_c(board)
    return _legal_move_indices_py(board)


@dataclass(frozen=True)
class PolicyGatherTables:
    to_sq: np.ndarray  # int64 (64,64)
    valid: np.ndarray  # bool (64,64)


def build_policy_gather_tables() -> PolicyGatherTables:
    to_sq = np.zeros((64, 64), dtype=np.int64)
    valid = np.zeros((64, 64), dtype=np.bool_)

    for from_sq in range(64):
        ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)

        p = 0
        for dfi, dri in QUEEN_DIRS:
            for dist in range(1, 8):
                tf, tr = ff + dfi * dist, fr + dri * dist
                if 0 <= tf <= 7 and 0 <= tr <= 7:
                    to_sq[from_sq, p] = chess.square(tf, tr)
                    valid[from_sq, p] = True
                p += 1

        for i, (df, dr) in enumerate(KNIGHT_DELTAS):
            p = 56 + i
            tf, tr = ff + df, fr + dr
            if 0 <= tf <= 7 and 0 <= tr <= 7:
                to_sq[from_sq, p] = chess.square(tf, tr)
                valid[from_sq, p] = True

    return PolicyGatherTables(to_sq=to_sq, valid=valid)


_ARANGE_POLICY = np.arange(POLICY_SIZE, dtype=np.intp)


def sample_move_from_logits(
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    temperature: float = 1.0,
    rng: np.random.Generator,
) -> int:
    assert logits.shape == (POLICY_SIZE,)
    legal_logits = logits.copy()
    legal_logits[~mask] = -1e9

    if temperature <= 0:
        return int(np.argmax(legal_logits))

    z = legal_logits / float(temperature)
    z = z - np.max(z)
    p = np.exp(z)
    p[~mask] = 0.0
    s = float(p.sum())
    if s <= 0:
        return int(np.argmax(legal_logits))
    p /= s
    return int(rng.choice(_ARANGE_POLICY, p=p))
