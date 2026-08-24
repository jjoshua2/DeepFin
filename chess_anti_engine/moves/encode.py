from __future__ import annotations

from dataclasses import dataclass
import sys
import threading
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
    from chess_anti_engine.encoding._lc0_ext import (
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
POLICY_SIZE = 64 * PLANE_COUNT  # 4672 — MCTS/CBoard action-id space (not model output)
COMPACT_POLICY_SIZE = 1858

POLICY_ENCODING_AZ_4672 = "az_4672"
POLICY_ENCODING_LC0_1858 = "lc0_1858"
# Trained nets and production shards always use compact 1858. Full az_4672 is
# the fixed search/action-id space (CBoard, MCTS tree edges); convert only at
# the model↔search boundary. Legacy 4672-wide shards / ONNX LC0 nets may still
# declare az_4672 as a *storage/output width*, not a ChessNet config option.
POLICY_ENCODINGS = (POLICY_ENCODING_AZ_4672, POLICY_ENCODING_LC0_1858)
MODEL_POLICY_ENCODING = POLICY_ENCODING_LC0_1858
MODEL_POLICY_SIZE = COMPACT_POLICY_SIZE

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
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
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


def _policy_index(from_oriented: int, plane: int) -> int:
    return int(from_oriented) * PLANE_COUNT + int(plane)


def _underpromo_plane(piece_idx: int, dir_idx: int) -> int:
    return 64 + int(piece_idx) * 3 + int(dir_idx)


def _underpromo_dir_idx(df: int) -> int:
    return DF_TO_UNDERPROMO_DIR.get(int(df), 1)


def _oriented_move_delta(move: chess.Move, turn: chess.Color) -> tuple[int, int, int]:
    from_oriented = orient_square(move.from_square, turn)
    to_oriented = orient_square(move.to_square, turn)
    df = chess.square_file(to_oriented) - chess.square_file(from_oriented)
    dr = chess.square_rank(to_oriented) - chess.square_rank(from_oriented)
    return int(from_oriented), int(df), int(dr)


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

    f_o = chess.Square(from_o)
    f_m = mirror_oriented_square(f_o)

    if plane >= 64:
        # Underpromotions: mirror df direction (left<->right), keep piece type.
        rel = plane - 64
        piece_idx = rel // 3
        dir_idx = rel % 3
        dir_m = 2 - int(dir_idx)  # 0<->2, 1 stays
        plane_m = _underpromo_plane(piece_idx, dir_m)
        return _policy_index(f_m, plane_m)

    delta = _PLANE_TO_DELTA.get(int(plane))
    if delta is None:
        # Should be unreachable for valid indices.
        return idx
    df, dr = delta
    plane_m = _DELTA_TO_PLANE.get((-int(df), int(dr)))
    if plane_m is None:
        return idx

    return _policy_index(f_m, plane_m)


# Flat index permutations for mirroring in oriented coordinates.
#
# Convention:
# - MIRROR_POLICY_MAP[old] = new
# - MIRROR_POLICY_INV[new] = old (inverse permutation)
MIRROR_POLICY_MAP = np.array([mirror_policy_index(i) for i in range(POLICY_SIZE)], dtype=np.int32)
MIRROR_POLICY_INV = np.empty((POLICY_SIZE,), dtype=np.int32)
MIRROR_POLICY_INV[MIRROR_POLICY_MAP] = np.arange(POLICY_SIZE, dtype=np.int32)


def _is_geometrically_valid_policy_index(index: int) -> bool:
    idx = int(index)
    if idx < 0 or idx >= POLICY_SIZE:
        return False
    from_o = idx // PLANE_COUNT
    plane = idx % PLANE_COUNT
    ff, fr = chess.square_file(from_o), chess.square_rank(from_o)
    if plane >= 64:
        rel = plane - 64
        if rel < 0 or rel >= 9:
            return False
        df = UNDERPROMO_DFS[rel % 3]
        tf, tr = ff + df, fr + 1
        return 0 <= tf <= 7 and tr == 7
    delta = _PLANE_TO_DELTA.get(int(plane))
    if delta is None:
        return False
    df, dr = delta
    tf, tr = ff + df, fr + dr
    return 0 <= tf <= 7 and 0 <= tr <= 7


COMPACT_TO_FULL_POLICY = np.array(
    [idx for idx in range(POLICY_SIZE) if _is_geometrically_valid_policy_index(idx)],
    dtype=np.int32,
)
if int(COMPACT_TO_FULL_POLICY.shape[0]) != COMPACT_POLICY_SIZE:
    raise RuntimeError(
        f"compact policy map has {COMPACT_TO_FULL_POLICY.shape[0]} entries, "
        f"expected {COMPACT_POLICY_SIZE}",
    )

FULL_TO_COMPACT_POLICY = np.full((POLICY_SIZE,), -1, dtype=np.int32)
FULL_TO_COMPACT_POLICY[COMPACT_TO_FULL_POLICY] = np.arange(COMPACT_POLICY_SIZE, dtype=np.int32)

COMPACT_MIRROR_POLICY_MAP = FULL_TO_COMPACT_POLICY[MIRROR_POLICY_MAP[COMPACT_TO_FULL_POLICY]]
COMPACT_MIRROR_POLICY_INV = np.empty((COMPACT_POLICY_SIZE,), dtype=np.int32)
COMPACT_MIRROR_POLICY_INV[COMPACT_MIRROR_POLICY_MAP] = np.arange(COMPACT_POLICY_SIZE, dtype=np.int32)


def normalize_policy_encoding(policy_encoding: str | None) -> str:
    """Canonicalize a policy *width* name (shard/manifest metadata only).

    Model output and train targets are always compact 1858. ``az_4672`` remains
    only as the name for full-width search vectors / legacy shards / ONNX nets.
    """
    enc = str(policy_encoding or MODEL_POLICY_ENCODING).lower()
    aliases = {
        "az": POLICY_ENCODING_AZ_4672,
        "az_4672": POLICY_ENCODING_AZ_4672,
        "lc0_4672": POLICY_ENCODING_AZ_4672,
        "4672": POLICY_ENCODING_AZ_4672,
        "8x8x73": POLICY_ENCODING_AZ_4672,
        "lc0": POLICY_ENCODING_LC0_1858,
        "lc0_1858": POLICY_ENCODING_LC0_1858,
        "compact": POLICY_ENCODING_LC0_1858,
        "compact_1858": POLICY_ENCODING_LC0_1858,
        "1858": POLICY_ENCODING_LC0_1858,
    }
    try:
        return aliases[enc]
    except KeyError as exc:
        raise ValueError(f"Unsupported policy_encoding {policy_encoding!r}") from exc


def policy_size_for_encoding(policy_encoding: str | None) -> int:
    enc = normalize_policy_encoding(policy_encoding)
    return COMPACT_POLICY_SIZE if enc == POLICY_ENCODING_LC0_1858 else POLICY_SIZE


def compact_policy_index(full_index: int) -> int:
    """Map a full 4672 action id → compact 1858 (-1 if not in the vocabulary)."""
    idx = int(full_index)
    if idx < 0 or idx >= POLICY_SIZE:
        return -1
    return int(FULL_TO_COMPACT_POLICY[idx])


def full_policy_index(compact_index: int) -> int:
    """Map a compact 1858 index → full 4672 action id (-1 if out of range)."""
    idx = int(compact_index)
    if idx < 0 or idx >= COMPACT_POLICY_SIZE:
        return -1
    return int(COMPACT_TO_FULL_POLICY[idx])


# Aliases kept for call-site stability: train space is always compact.
def policy_index_for_encoding(full_index: int, *, policy_encoding: str | None = None) -> int:
    del policy_encoding
    return compact_policy_index(full_index)


def policy_indices_for_encoding(
    full_indices: np.ndarray,
    *,
    policy_encoding: str | None = None,
) -> np.ndarray:
    del policy_encoding
    idx = np.asarray(full_indices, dtype=np.int32)
    mapped = FULL_TO_COMPACT_POLICY[idx]
    return mapped[mapped >= 0].astype(np.int32, copy=False)


def move_to_index_for_encoding(
    move: chess.Move,
    board: chess.Board,
    *,
    policy_encoding: str | None = None,
) -> int:
    del policy_encoding
    return compact_policy_index(move_to_index(move, board))


def index_to_move_for_encoding(
    index: int,
    board: chess.Board,
    *,
    policy_encoding: str | None = None,
) -> chess.Move:
    del policy_encoding
    full_index = full_policy_index(index)
    if full_index < 0:
        raise ValueError(f"invalid compact policy index {index}")
    return index_to_move(full_index, board)


def legal_move_indices_for_encoding(
    board: chess.Board,
    *,
    policy_encoding: str | None = None,
) -> np.ndarray:
    del policy_encoding
    return policy_indices_for_encoding(legal_move_indices(board))


def legal_move_mask_for_encoding(
    board: chess.Board,
    *,
    policy_encoding: str | None = None,
) -> np.ndarray:
    del policy_encoding
    mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.bool_)
    idx = legal_move_indices_for_encoding(board)
    if idx.size > 0:
        mask[idx] = True
    return mask


def _renormalize_policy_vector(policy: np.ndarray) -> np.ndarray:
    out = np.asarray(policy, dtype=np.float32)
    total = float(out.sum(dtype=np.float32))
    if total > 0.0:
        out = out / total
    return out.astype(np.asarray(policy).dtype, copy=False)


def _renormalize_policy_batch(policies: np.ndarray) -> np.ndarray:
    out = np.asarray(policies, dtype=np.float32)
    totals = out.sum(axis=1, keepdims=True, dtype=np.float32)
    np.divide(out, totals, out=out, where=totals > 0.0)
    return out.astype(np.asarray(policies).dtype, copy=False)


def policy_vector_to_encoding(
    policy: np.ndarray,
    *,
    policy_encoding: str | None = None,
) -> np.ndarray:
    """Project a policy vector into compact 1858 train/model space.

    Already-compact vectors pass through. Full 4672 vectors are gathered onto
    the geometrically valid subset and renormalized. ``policy_encoding`` is
    ignored (kept for call-site compatibility).
    """
    del policy_encoding
    p = np.asarray(policy)
    if p.shape == (COMPACT_POLICY_SIZE,):
        return p
    if p.shape != (POLICY_SIZE,):
        raise ValueError(
            f"policy must be ({POLICY_SIZE},) or ({COMPACT_POLICY_SIZE},), got {p.shape}",
        )
    return _renormalize_policy_vector(p[COMPACT_TO_FULL_POLICY])


def policy_mask_to_encoding(
    mask: np.ndarray,
    *,
    policy_encoding: str | None = None,
) -> np.ndarray:
    """Project a legal/value mask into compact 1858 space (no renormalize)."""
    del policy_encoding
    m = np.asarray(mask)
    if m.shape == (COMPACT_POLICY_SIZE,):
        return m
    if m.shape != (POLICY_SIZE,):
        raise ValueError(
            f"policy mask must be ({POLICY_SIZE},) or ({COMPACT_POLICY_SIZE},), got {m.shape}",
        )
    return m[COMPACT_TO_FULL_POLICY]


def policy_batch_to_encoding(
    policies: np.ndarray,
    *,
    policy_encoding: str | None = None,
) -> np.ndarray:
    """Batch form of :func:`policy_vector_to_encoding`."""
    del policy_encoding
    p = np.asarray(policies)
    if p.ndim != 2:
        raise ValueError(f"policies must be 2D, got {p.shape}")
    if int(p.shape[1]) == int(COMPACT_POLICY_SIZE):
        return p
    if int(p.shape[1]) != int(POLICY_SIZE):
        raise ValueError(
            f"policies must be (N,{POLICY_SIZE}) or (N,{COMPACT_POLICY_SIZE}), got {p.shape}",
        )
    return _renormalize_policy_batch(p[:, COMPACT_TO_FULL_POLICY])


def policy_vector_to_full(
    policy: np.ndarray,
    *,
    policy_encoding: str | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Expand compact 1858 → full 4672 (or pass through full)."""
    del policy_encoding
    p = np.asarray(policy)
    if p.shape == (POLICY_SIZE,):
        return p
    if p.shape != (COMPACT_POLICY_SIZE,):
        raise ValueError(
            f"policy must be ({POLICY_SIZE},) or ({COMPACT_POLICY_SIZE},), got {p.shape}",
        )
    out = np.full((POLICY_SIZE,), fill_value, dtype=p.dtype)
    out[COMPACT_TO_FULL_POLICY] = p
    return out


def policy_batch_to_full(
    policies: np.ndarray,
    *,
    policy_encoding: str | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Batch form of :func:`policy_vector_to_full`."""
    del policy_encoding
    p = np.asarray(policies)
    if p.ndim != 2:
        raise ValueError(f"policies must be 2D, got {p.shape}")
    if int(p.shape[1]) == int(POLICY_SIZE):
        return p
    if int(p.shape[1]) != int(COMPACT_POLICY_SIZE):
        raise ValueError(
            f"policies must be (N,{POLICY_SIZE}) or (N,{COMPACT_POLICY_SIZE}), got {p.shape}",
        )
    out = np.full((int(p.shape[0]), POLICY_SIZE), fill_value, dtype=p.dtype)
    out[:, COMPACT_TO_FULL_POLICY] = p
    return out


def policy_batch_to_full_if_needed(
    policies: np.ndarray,
    *,
    policy_encoding: str | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Shape-based expand for search: full stays full, compact scatters to 4672."""
    del policy_encoding
    return policy_batch_to_full(policies, fill_value=fill_value)


def _build_index_to_move_lut() -> np.ndarray:
    """Precomputed reverse LUT: policy index → (from_sq, to_sq, promotion).

    Shape ``(2, POLICY_SIZE, 3)``; axis 0 is ``int(turn)`` (0=BLACK, 1=WHITE).
    ``lut[:, idx, 0]`` = from_sq, ``lut[:, idx, 1]`` = to_sq,
    ``lut[:, idx, 2]`` = promotion piece type (0=none, 2=knight, 3=bishop,
    4=rook, 5=queen).
    """
    lut = np.full((2, POLICY_SIZE, 3), -1, dtype=np.int16)
    for turn in (chess.WHITE, chess.BLACK):
        ti = int(turn)
        for idx in range(POLICY_SIZE):
            from_o = idx // PLANE_COUNT
            plane = idx % PLANE_COUNT
            ff, fr = chess.square_file(from_o), chess.square_rank(from_o)
            if plane >= 64:
                rel = plane - 64
                df = UNDERPROMO_DFS[rel % 3]
                tf, tr = ff + df, fr + 1
                if not (0 <= tf <= 7 and 0 <= tr <= 7):
                    continue
                to_o = chess.square(tf, tr)
                promo = UNDERPROMO_PIECES[rel // 3]
                lut[ti, idx] = [orient_square(from_o, turn), orient_square(to_o, turn), promo]
                continue
            delta = _PLANE_TO_DELTA.get(plane)
            if delta is None:
                continue
            df, dr = delta
            tf, tr = ff + df, fr + dr
            if not (0 <= tf <= 7 and 0 <= tr <= 7):
                continue
            to_o = chess.square(tf, tr)
            f_real = orient_square(from_o, turn)
            t_real = orient_square(to_o, turn)
            # Pawn reaching last rank: encode queen-promotion candidate; the
            # real-board piece-type check happens at decode time in
            # ``index_to_move_fast``.
            promo_val = chess.QUEEN if chess.square_rank(t_real) in (0, 7) else 0
            lut[ti, idx] = [f_real, t_real, promo_val]
    return lut


_INDEX_TO_MOVE_LUT = _build_index_to_move_lut()


class ActionDecodeError(ValueError):
    """An action id did not name a legal move on this board.

    Raised only by :func:`index_to_move_strict`. The resilient decoders
    substitute the first legal move instead and count the event — see
    :func:`decode_fallback_count`.
    """

    # ⚑ ValueError base, so a play loop wrapped in `except ValueError` swallows
    # this and re-launders the failure it exists to surface. No path does that
    # today (checked 2026-08-23); a new one must catch ActionDecodeError FIRST
    # and re-raise, or void the measurement.

    def __init__(self, index: int, board: chess.Board, detail: str) -> None:
        self.index = int(index)
        self.fen = board.fen()
        self.turn = bool(board.turn)
        self.detail = str(detail)
        super().__init__(
            f"action id {self.index} does not decode to a legal move: {self.detail} "
            f"(side to move {'white' if self.turn else 'black'}, fen {self.fen!r})",
        )


# Every first-legal substitution made by `index_to_move_fast` (hence by
# `index_to_move`) since process start. A substitution silently changes the move
# that gets played, so an action-space regression surfaces here before it
# surfaces anywhere else — and until this counter existed nothing recorded that
# it had happened at all.
_DECODE_FALLBACKS = 0
_DECODE_FALLBACK_WARNED = False
# High-water mark already published into the outcome_stats metric stream by
# `drain_decode_fallback_count`; the lock covers drain-vs-drain only.
_DECODE_FALLBACKS_PUBLISHED = 0
_DECODE_FALLBACK_DRAIN_LOCK = threading.Lock()


def _decode_action(index: int, board: chess.Board) -> chess.Move | None:
    """LUT decode shared by the resilient and strict decoders.

    Returns ``None`` for the two cases that have no legal decode: the id has no
    LUT entry for this side to move, and the constructed move is illegal with no
    legal move encoding back to this id. Handing those back as ``None`` rather
    than substituting here is what lets the caller choose between resilience and
    a raise without either one re-deriving the decode rule.
    """
    entry = _INDEX_TO_MOVE_LUT[int(board.turn), int(index)]
    f, t, promo = int(entry[0]), int(entry[1]), int(entry[2])
    if f < 0:
        return None

    # Only apply promotion if a pawn is actually on the from square.
    promotion = None
    if promo > 0:
        piece = board.piece_at(f)
        if piece is not None and piece.piece_type == chess.PAWN:
            promotion = promo

    m = chess.Move(f, t, promotion=promotion)
    if m in board.legal_moves:
        return m
    # Fallback for rare edge cases like ambiguous queen-promotion candidates.
    for lm in board.legal_moves:
        if move_to_index(lm, board) == index:
            return lm
    return None


def _decode_failure_detail(index: int, board: chess.Board) -> str:
    """Why :func:`_decode_action` returned ``None``. Error/warning paths only."""
    entry = _INDEX_TO_MOVE_LUT[int(board.turn), int(index)]
    f, t, promo = int(entry[0]), int(entry[1]), int(entry[2])
    if f < 0:
        return "no LUT entry for this side to move"
    return (
        f"LUT produced {chess.SQUARE_NAMES[f]}{chess.SQUARE_NAMES[t]} "
        f"(promotion piece type {promo}), which is illegal here, and no legal "
        f"move encodes back to this id"
    )


def _note_decode_fallback(index: int, board: chess.Board) -> None:
    """Count a first-legal substitution; describe the first one on stderr.

    O(1) on the fallback path: the LUT re-read and the FEN are built only inside
    the one-shot branch. stderr and not stdout because this module is also on
    the UCI engine's move path, whose stdout is the protocol channel.
    """
    global _DECODE_FALLBACKS, _DECODE_FALLBACK_WARNED
    _DECODE_FALLBACKS += 1
    if not _DECODE_FALLBACK_WARNED:
        _DECODE_FALLBACK_WARNED = True
        print(
            f"[moves] action id {int(index)} did not decode to a legal move; "
            f"SUBSTITUTED the first legal move and kept playing. "
            f"{_decode_failure_detail(index, board)}; side to move "
            f"{'white' if board.turn else 'black'}, fen {board.fen()!r}. This "
            "changes the move that is actually played, so an action-space "
            "regression shows up here first. Further occurrences are counted, "
            "not logged; read the total with "
            "chess_anti_engine.moves.decode_fallback_count().",
            file=sys.stderr,
            flush=True,
        )


def decode_fallback_count() -> int:
    """First-legal substitutions made by the resilient decoders, since process start."""
    return _DECODE_FALLBACKS


def drain_decode_fallback_count() -> int:
    """Substitutions since the last drain, for publishing into a metric stream.

    :func:`decode_fallback_count` stays cumulative for anyone who asks; this is
    the delta form, so a publisher that runs once per finalized game reports
    each substitution exactly once instead of re-reporting the running total.
    Selfplay finalize (`selfplay/finalize.py`) is that publisher — a stderr line
    in a worker log is not part of the experiment metric stream, and a counter
    no production path reads is the same defect the count exists to expose.

    Process-wide, not per-game: concurrent games in one worker share the
    counter, and the shard sums them anyway. The lock makes two concurrent
    drains exact; the increment itself is an unlocked ``+=`` like the guard
    counters in ``mcts/gumbel_c.py``.
    """
    global _DECODE_FALLBACKS_PUBLISHED
    with _DECODE_FALLBACK_DRAIN_LOCK:
        total = _DECODE_FALLBACKS
        delta = total - _DECODE_FALLBACKS_PUBLISHED
        _DECODE_FALLBACKS_PUBLISHED = total
    return delta


def index_to_move_fast(index: int, board: chess.Board) -> chess.Move:
    """Fast index → move using precomputed LUT.

    Resilient by design: an id that names no legal move is replaced with the
    first legal one so a search never dies mid-playout. That substitution is
    counted, because it is otherwise invisible — nothing downstream can tell the
    substituted move from a chosen one. Measurement paths want
    :func:`index_to_move_strict` instead.
    """
    m = _decode_action(index, board)
    if m is not None:
        return m
    # Count only once a substitute EXISTS. On a board with no legal moves
    # `next(iter(...))` raises, and recording "SUBSTITUTED ... kept playing"
    # before that would bank a move that never happened.
    substitute = next(iter(board.legal_moves))
    _note_decode_fallback(index, board)
    return substitute


def index_to_move_strict(index: int, board: chess.Board) -> chess.Move:
    """Index → move that RAISES :class:`ActionDecodeError` instead of substituting.

    Identical decode to :func:`index_to_move_fast` — they share
    :func:`_decode_action` and differ only in what they do when it fails.
    """
    m = _decode_action(index, board)
    if m is None:
        raise ActionDecodeError(int(index), board, _decode_failure_detail(index, board))
    return m


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a policy vector left-right.

    Accepts either the legacy 4672 or compact 1858 move encoding and any float
    dtype; returns float32.
    """
    p = np.asarray(policy)
    if p.shape == (COMPACT_POLICY_SIZE,):
        return p[COMPACT_MIRROR_POLICY_INV].astype(np.float32, copy=False)
    if p.shape != (POLICY_SIZE,):
        raise ValueError(
            f"policy must be ({POLICY_SIZE},) or ({COMPACT_POLICY_SIZE},), got {p.shape}",
        )
    # new[j] = old[inv[j]]
    return p[MIRROR_POLICY_INV].astype(np.float32, copy=False)


def mirror_policy_batch(policies: np.ndarray) -> np.ndarray:
    p = np.asarray(policies)
    if p.ndim != 2:
        raise ValueError(f"policies must be 2D, got {p.shape}")
    if int(p.shape[1]) == int(COMPACT_POLICY_SIZE):
        return p[:, COMPACT_MIRROR_POLICY_INV].astype(np.float32, copy=False)
    if int(p.shape[1]) != int(POLICY_SIZE):
        raise ValueError(
            f"policies must be (N,{POLICY_SIZE}) or (N,{COMPACT_POLICY_SIZE}), got {p.shape}",
        )
    return p[:, MIRROR_POLICY_INV].astype(np.float32, copy=False)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    from_oriented, df, dr = _oriented_move_delta(move, board.turn)

    if move.promotion is not None and move.promotion in UNDERPROMO_TO_IDX:
        dir_idx = _underpromo_dir_idx(df)
        piece_idx = UNDERPROMO_TO_IDX[move.promotion]
        return _policy_index(from_oriented, _underpromo_plane(piece_idx, dir_idx))

    plane = _DELTA_TO_PLANE.get((df, dr))
    if plane is None:
        raise ValueError(f"Unencodable move delta df={df}, dr={dr} for move={move}")

    return _policy_index(from_oriented, plane)


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

    promo_char = uci[4].lower() if len(uci) == 5 else ""
    if promo_char in _UNDERPROMO_CHAR_TO_IDX:
        f = from_sq if turn else (from_sq ^ 56)
        t = to_sq if turn else (to_sq ^ 56)
        ff, tf = f % 8, t % 8
        df = tf - ff
        dir_idx = _underpromo_dir_idx(df)
        piece_idx = _UNDERPROMO_CHAR_TO_IDX[promo_char]
        return _policy_index(f, _underpromo_plane(piece_idx, dir_idx))

    # Queen promotion or normal move: use the precomputed non-underpromotion LUT.
    idx = int(_MOVE_INDEX_LUT[int(turn)][from_sq][to_sq])
    return idx


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert a policy index back to a chess.Move. Uses precomputed LUT.

    Alias of :func:`index_to_move_fast`, so it substitutes and counts rather
    than raising; a caller that must not silently play a different move wants
    :func:`index_to_move_strict`.
    """
    return index_to_move_fast(index, board)


def _build_move_index_lut() -> np.ndarray:
    """Precomputed (from_sq, to_sq) → policy index for non-underpromotion moves.

    Shape ``(2, 64, 64)``; axis 0 is ``int(turn)`` (0=BLACK, 1=WHITE).
    Returns -1 for unreachable (df, dr) deltas.
    """
    lut = np.full((2, 64, 64), -1, dtype=np.int32)
    for turn in (chess.WHITE, chess.BLACK):
        ti = int(turn)
        for from_sq in range(64):
            f = orient_square(from_sq, turn)
            ff, fr = chess.square_file(f), chess.square_rank(f)
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                t = orient_square(to_sq, turn)
                df = chess.square_file(t) - ff
                dr = chess.square_rank(t) - fr
                p = _DELTA_TO_PLANE.get((df, dr))
                if p is not None:
                    lut[ti][from_sq][to_sq] = int(f) * PLANE_COUNT + int(p)
    return lut


_MOVE_INDEX_LUT = _build_move_index_lut()


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
