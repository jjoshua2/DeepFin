"""Extra (non-LC0) input feature planes.

Two versions exist, selected by ``model.input_extra_features``:

- ``v1``: the original 34 planes (positional half of a Stockfish-style eval)
- ``v2_threats``: v1 plus 29 threat-family planes appended AFTER index 34.
  Existing planes are never reordered or renumbered.
- ``v3_checks``: v2_threats plus 4 opponent-safe-check planes appended AFTER
  index 63 (the mirror of the side-to-move safe-checks family [54:58]).
  Existing planes are never reordered or renumbered.
- ``v3_xray``: v2_threats plus 6 per-slider-type x-ray attack planes appended
  AFTER index 63. SEPARATE v3 family from ``v3_checks`` (it does NOT include
  the opponent-safe-check planes). Existing planes are never reordered.
- ``v3_see``: v2_threats plus 2 graded Static-Exchange-Evaluation planes
  appended AFTER index 63 (the continuous replacement for v2's binary
  cheaper-attacked planes [52:53]). SEPARATE v3 family. Existing planes are
  never reordered.
- ``v3_passers``: v2_threats plus 8 passed-pawn-quality planes appended AFTER
  index 63 (4 per side). SEPARATE v3 family. Existing planes are never
  reordered.

Plane layout (indices within the extra block; absolute index = 112 + idx):

| idx     | family                | contents                                          |
|---------|-----------------------|---------------------------------------------------|
| [0:10]  | king-zone safety (v1) | per side: king zone + N/B/R/Q attacker overlaps   |
| [10:16] | pins (v1)             | per side: pinned, pin rays, discovered attackers  |
| [16:24] | pawn structure (v1)   | per side: passed, isolated, backward, connected   |
| [24:30] | mobility (v1)         | per piece type: normalized move counts at source  |
| [30:34] | outposts (v1)         | per side: outpost squares, space control          |
| [34]    | attacks (v2)          | all squares attacked by us (union over types)     |
| [35]    | attacks (v2)          | all squares attacked by them                      |
| [36:42] | attacks (v2)          | squares attacked by our P, N, B, R, Q, K          |
| [42:48] | attacks (v2)          | squares attacked by their P, N, B, R, Q, K        |
| [48]    | attackers (v2)        | our attacker count per square (/4, clamped to 1)  |
| [49]    | attackers (v2)        | their attacker count per square (/4, clamped)     |
| [50]    | hanging (v2)          | our pieces attacked by them and undefended by us  |
| [51]    | hanging (v2)          | their pieces attacked by us and undefended        |
| [52]    | threats (v2)          | our N/B/R/Q attacked by a strictly cheaper piece  |
| [53]    | threats (v2)          | their N/B/R/Q attacked by a strictly cheaper piece|
| [54:58] | safe checks (v2)      | side to move: safe N, B, R, Q check-origin squares|
| [58]    | control (v2)          | (our − their attacker count)/4, clamped to [-1,1] |
| [59]    | pawn tension (v2)     | our pawns that can capture an enemy pawn          |
| [60]    | pawn tension (v2)     | their pawns that can capture one of our pawns     |
| [61]    | pawn storm (v2)       | enemy pawns on files around OUR king, scaled by   |
|         |                       | rank closeness to the king (1 − rank_dist/7)      |
| [62]    | pawn storm (v2)       | our pawns storming THEIR king, same scaling       |
| [63:67] | opp safe checks (v3_checks) | opponent's safe N, B, R, Q check-origin       |
|         |                       | squares against OUR king (mirror of [54:58])      |
| [63:69] | x-ray attacks (v3_xray) | per-slider-type x-ray attacks (squares seen     |
|         |                       | THROUGH the first blocker on each ray): our B, R, |
|         |                       | Q then their B, R, Q. SEPARATE v3 family from     |
|         |                       | v3_checks — same start index, different version.  |
| [63:65] | SEE (v3_see)          | graded Static Exchange Evaluation, normalized     |
|         |                       | clip(see_cp/8, -1, 1): [63] our pieces' SEE when  |
|         |                       | the OPPONENT initiates a capture (<=0 = we lose   |
|         |                       | material); [64] their pieces' SEE when WE initiate|
|         |                       | (>=0 = we win material). SEPARATE v3 family.      |
| [63:71] | passers (v3_passers)  | passed-pawn quality, 4 planes per side (us 63-66, |
|         |                       | them 67-70): rank-advancement (rel-rank/7), safe  |
|         |                       | passer (stop square unattacked & empty), blocked  |
|         |                       | passer (any piece on stop square), promotion path |
|         |                       | controlled by enemy (any file-ahead square the    |
|         |                       | enemy attacks). SEPARATE v3 family.               |

Semantics notes:
- All planes are side-to-move oriented (``bitboards_to_planes(..., turn)``).
- Attack maps / attacker counts are direct attacks at current occupancy
  (no x-rays); pawn attacks are capture squares only (no en passant).
- X-ray (v3_xray) of a slider at square ``s`` with occupancy ``occ`` is the
  set of squares revealed beyond the FIRST blocker on each ray:
  ``direct ^ att(s, occ ^ (direct & occ))`` where ``direct = att(s, occ)``.
  These squares are disjoint from the direct-attack squares by construction.
  Unioned per side over each slider type (bishop, rook, queen). Queen x-ray
  is bishop-xray(from queen sq) | rook-xray(from queen sq).
- "Defended" in the hanging planes means attacked by any friendly piece.
- Cheaper-attacker classes: pawn → N/B/R/Q, minor → R/Q, rook → Q.
- Safe checks follow Stockfish semantics: squares a piece of ours could
  check the enemy king from (slider attacks from the king square at current
  occupancy) that are neither attacked by the enemy nor occupied by us.
  Computed for the side to move only.

The C twin of this layout lives in ``_features_impl.h`` — keep both tables
in sync (tests/test_threat_planes.py enforces value parity).
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.utils.bitboards import bitboards_to_planes

EXTRA_FEATURES_V1 = "v1"
EXTRA_FEATURES_V2_THREATS = "v2_threats"
EXTRA_FEATURES_V3_CHECKS = "v3_checks"
EXTRA_FEATURES_V3_XRAY = "v3_xray"
EXTRA_FEATURES_V3_SEE = "v3_see"
EXTRA_FEATURES_V3_PASSERS = "v3_passers"
EXTRA_FEATURE_VERSIONS = (
    EXTRA_FEATURES_V1,
    EXTRA_FEATURES_V2_THREATS,
    EXTRA_FEATURES_V3_CHECKS,
    EXTRA_FEATURES_V3_XRAY,
    EXTRA_FEATURES_V3_SEE,
    EXTRA_FEATURES_V3_PASSERS,
)

_EXTRA_FEATURE_PLANES = {
    EXTRA_FEATURES_V1: 34,
    EXTRA_FEATURES_V2_THREATS: 63,
    EXTRA_FEATURES_V3_CHECKS: 67,
    EXTRA_FEATURES_V3_XRAY: 69,
    EXTRA_FEATURES_V3_SEE: 65,
    EXTRA_FEATURES_V3_PASSERS: 71,
}

# Distinct plane counts are load-bearing: version_for_input_planes maps a total
# plane count back to a version by first match, so two versions sharing a count
# would silently collide. Fail loudly here if a future addition reuses a count.
assert len(set(_EXTRA_FEATURE_PLANES.values())) == len(_EXTRA_FEATURE_PLANES), (
    "EXTRA_FEATURE_PLANES values must be unique (version_for_input_planes "
    f"collides otherwise): {_EXTRA_FEATURE_PLANES}"
)


@functools.lru_cache(maxsize=32)
def normalize_extra_features_encoding(value: str | None) -> str:
    """Map a config value to a canonical extra-features version (default v1)."""
    if value is None:
        return EXTRA_FEATURES_V1
    v = str(value).strip().lower()
    if v in ("", "v1", "legacy"):
        return EXTRA_FEATURES_V1
    if v in ("v2", "v2_threats"):
        return EXTRA_FEATURES_V2_THREATS
    if v in ("v3", "v3_checks"):
        return EXTRA_FEATURES_V3_CHECKS
    if v in ("v3_xray", "xray"):
        return EXTRA_FEATURES_V3_XRAY
    if v in ("v3_see", "see"):
        return EXTRA_FEATURES_V3_SEE
    if v in ("v3_passers", "passers", "v3_passed"):
        return EXTRA_FEATURES_V3_PASSERS
    raise ValueError(
        f"unknown input_extra_features {value!r}; expected one of {EXTRA_FEATURE_VERSIONS}"
    )


@functools.lru_cache(maxsize=32)
def extra_feature_plane_count(version: str | None = None) -> int:
    """Number of extra feature planes for a (possibly unnormalized) version."""
    return _EXTRA_FEATURE_PLANES[normalize_extra_features_encoding(version)]


RELATION_COUNT = 5


def relation_matrices(board: chess.Board) -> np.ndarray:
    """Dynamic board-relation matrices: (RELATION_COUNT, 64, 64) uint8.

    Side-to-move oriented exactly like the feature planes (rank flip when
    black to move), row = from-square, col = to-square:

      R0 attacks(from, to)          piece on ``from`` attacks square ``to``
                                    (pawns: capture squares only, no EP)
      R1 defends(from, to)          R0 restricted to ``to`` occupied by a
                                    same-color piece
      R2 pinned_by(from, to)        piece on ``from`` is absolutely pinned to
                                    its own king by the enemy slider on ``to``
      R3 shares_open_line(from, to) both squares occupied, aligned on a file
                                    or diagonal, all squares strictly between
                                    empty (symmetric; ranks excluded)
      R4 pawn_tension(from, to)     pawn on ``from`` can capture the enemy
                                    pawn on ``to``

    python-chess reference implementation; the C twin is
    ``compute_relations`` in ``_features_impl.h`` (parity-tested).
    """
    out = np.zeros((RELATION_COUNT, 64, 64), dtype=np.uint8)
    flip = 0 if board.turn == chess.WHITE else 56

    occ = int(board.occupied)
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            att = int(chess.BB_PAWN_ATTACKS[piece.color][sq])
        else:
            att = int(board.attacks_mask(sq))
        own_occ = int(board.occupied_co[piece.color])
        f = sq ^ flip
        for to in chess.scan_forward(att):
            out[0, f, to ^ flip] = 1
        for to in chess.scan_forward(att & own_occ):
            out[1, f, to ^ flip] = 1

  # R2: walk the 8 rays from each king (king .. own blocker .. enemy slider).
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        own_occ = int(board.occupied_co[color])
        opp = not color
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        for d, (df, dr) in enumerate(
            ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        ):
            sliders = int(
                board.pieces_mask(chess.QUEEN, opp)
                | board.pieces_mask(chess.ROOK if d < 4 else chess.BISHOP, opp)
            )
            blocker = -1
            for dist in range(1, 8):
                ff, rr = kf + df * dist, kr + dr * dist
                if not (0 <= ff <= 7 and 0 <= rr <= 7):
                    break
                cur = chess.square(ff, rr)
                bit = int(chess.BB_SQUARES[cur])
                if not occ & bit:
                    continue
                if own_occ & bit:
                    if blocker >= 0:
                        break
                    blocker = cur
                else:
                    if blocker >= 0 and sliders & bit:
                        out[2, blocker ^ flip, cur ^ flip] = 1
                    break

  # R3: first occupied square along files and diagonals.
    for sq in chess.scan_forward(occ):
        f0, r0 = chess.square_file(sq), chess.square_rank(sq)
        for df, dr in ((0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            for dist in range(1, 8):
                ff, rr = f0 + df * dist, r0 + dr * dist
                if not (0 <= ff <= 7 and 0 <= rr <= 7):
                    break
                to = chess.square(ff, rr)
                if occ & int(chess.BB_SQUARES[to]):
                    out[3, sq ^ flip, to ^ flip] = 1
                    break

  # R4: mutual pawn-capture pairs (each direction marked from its mover).
    for color in (chess.WHITE, chess.BLACK):
        enemy_pawns = int(board.pieces_mask(chess.PAWN, not color))
        for sq in chess.scan_forward(int(board.pieces_mask(chess.PAWN, color))):
            for to in chess.scan_forward(int(chess.BB_PAWN_ATTACKS[color][sq]) & enemy_pawns):
                out[4, sq ^ flip, to ^ flip] = 1

    return out


def relation_matrices_c(board: chess.Board) -> np.ndarray:
    """C-accelerated twin of :func:`relation_matrices`."""
    turn = board.turn
    us, them = turn, not turn
    pieces_us = np.array(
        [int(board.pieces_mask(pt, us)) for pt in chess.PIECE_TYPES], dtype=np.uint64
    )
    pieces_them = np.array(
        [int(board.pieces_mask(pt, them)) for pt in chess.PIECE_TYPES], dtype=np.uint64
    )
    _ks_us = board.king(us)
    _ks_them = board.king(them)
    return _c_compute_relations(
        pieces_us, pieces_them, int(board.occupied),
        _ks_us if _ks_us is not None else -1,
        _ks_them if _ks_them is not None else -1,
        turn == chess.WHITE,
    )

try:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,
        compute_relation_matrices as _c_compute_relations,
    )
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False

if TYPE_CHECKING:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,
        compute_relation_matrices as _c_compute_relations,
    )



def _ray_step(src: int, dst: int) -> int | None:
    sf = chess.square_file(src)
    sr = chess.square_rank(src)
    df = chess.square_file(dst)
    dr = chess.square_rank(dst)
    dx = df - sf
    dy = dr - sr
    if dx == 0 and dy != 0:
        return 8 if dy > 0 else -8
    if dy == 0 and dx != 0:
        return 1 if dx > 0 else -1
    if abs(dx) == abs(dy) and dx != 0:
        if dx > 0 and dy > 0:
            return 9
        if dx > 0 and dy < 0:
            return -7
        if dx < 0 and dy > 0:
            return 7
        return -9
    return None


def _is_slider_aligned(src: int, dst: int, piece_type: chess.PieceType) -> bool:
    sf = chess.square_file(src)
    sr = chess.square_rank(src)
    df = chess.square_file(dst)
    dr = chess.square_rank(dst)
    dx = df - sf
    dy = dr - sr
    if piece_type == chess.BISHOP:
        return abs(dx) == abs(dy) and dx != 0
    if piece_type == chess.ROOK:
        return (dx == 0) ^ (dy == 0)
    if piece_type == chess.QUEEN:
        return (dx == 0) ^ (dy == 0) or (abs(dx) == abs(dy) and dx != 0)
    return False


def _discovered_attack_mask(board: chess.Board, color: chess.Color) -> int:
    """Squares of own pieces whose removal leaves the enemy king attacked.

    This preserves the existing training signal semantics:
    - if the enemy king is already attacked, mark all own pieces whose
      removal still leaves the king attacked
    - otherwise, mark the unique own blocker on any slider ray to the king
    """
    opp_king = board.king(not color)
    if opp_king is None:
        return 0

    if board.is_attacked_by(color, opp_king):
  # Optimized in-check path: avoid board copies per piece.
  # Any non-attacker piece's removal doesn't change the check.
  # Multiple attackers → removing any single one leaves the others.
        attackers = board.attackers_mask(color, opp_king)
        n_attackers = chess.popcount(attackers)
        all_own = int(board.occupied_co[color])

        if n_attackers >= 2:
  # Double+ check: removing any piece leaves ≥1 attacker.
            return all_own

  # Single attacker: all non-attackers stay in mask. For the single
  # attacker itself, check if removing it reveals a hidden slider.
        attacker_sq = chess.lsb(attackers)
        b2 = board.copy(stack=False)
        b2.remove_piece_at(attacker_sq)
        if b2.is_attacked_by(color, opp_king):
            return all_own  # hidden slider revealed
        return all_own & ~int(attackers)

  # No sliders → no discovered attacks possible.
    has_sliders = (
        board.pieces_mask(chess.BISHOP, color)
        | board.pieces_mask(chess.ROOK, color)
        | board.pieces_mask(chess.QUEEN, color)
    )
    if not has_sliders:
        return 0

    occ = int(board.occupied)
    discovered_mask = 0
    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if not _is_slider_aligned(sq, opp_king, pt):
                continue
            step = _ray_step(sq, opp_king)
            if step is None:
                continue
            cur = sq + step
            blocker_sq = -1
            blocker_count = 0
            while cur != opp_king:
                if occ & chess.BB_SQUARES[cur]:
                    blocker_sq = cur
                    blocker_count += 1
                    if blocker_count > 1:
                        break
                cur += step
            if blocker_count == 1:
                piece = board.piece_at(blocker_sq)
                if piece is not None and piece.color == color:
                    discovered_mask |= chess.BB_SQUARES[blocker_sq]
    return discovered_mask


def _build_adjacent_file_masks() -> list[int]:
    out: list[int] = []
    for file_idx in range(8):
        mask = 0
        if file_idx > 0:
            mask |= int(chess.BB_FILES[file_idx - 1])
        if file_idx < 7:
            mask |= int(chess.BB_FILES[file_idx + 1])
        out.append(mask)
    return out


_ADJACENT_FILE_MASKS = _build_adjacent_file_masks()

_PASSED_PAWN_MASKS = {
    chess.WHITE: [0] * 64,
    chess.BLACK: [0] * 64,
}
_CONNECTED_NEIGHBOR_MASKS = [0] * 64
_BACKWARD_SUPPORT_MASKS = {
    chess.WHITE: [0] * 64,
    chess.BLACK: [0] * 64,
}
_PAWN_SINGLE_PUSH_MASK = {
    chess.WHITE: [0] * 64,
    chess.BLACK: [0] * 64,
}
_PAWN_DOUBLE_PUSH_MASK = {
    chess.WHITE: [0] * 64,
    chess.BLACK: [0] * 64,
}
_ORIENT_COORDS: dict[bool, list[tuple[int, int]]] = {
  # Out-of-range sentinel: uninitialized accesses surface-fault rather than
  # silently returning a plausible square. _build_square_tables() fills every slot.
    chess.WHITE: [(-1, -1)] * 64,
    chess.BLACK: [(-1, -1)] * 64,
}

def _build_square_tables() -> None:
    for sq in chess.SQUARES:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        _ORIENT_COORDS[chess.WHITE][sq] = (r, f)
        _ORIENT_COORDS[chess.BLACK][sq] = (7 - r, f)

        conn_mask = 0
        for df in (-1, 1):
            f2 = f + df
            if not (0 <= f2 <= 7):
                continue
            for dr in (-1, 0, 1):
                r2 = r + dr
                if 0 <= r2 <= 7:
                    conn_mask |= chess.BB_SQUARES[chess.square(f2, r2)]
        _CONNECTED_NEIGHBOR_MASKS[sq] = int(conn_mask)

        for color in (chess.WHITE, chess.BLACK):
            passed = 0
            support = 0
            direction = 1 if color == chess.WHITE else -1

            for ff in range(max(0, f - 1), min(7, f + 1) + 1):
                rr = r + direction
                while 0 <= rr <= 7:
                    passed |= chess.BB_SQUARES[chess.square(ff, rr)]
                    rr += direction

            for af in (f - 1, f + 1):
                if not (0 <= af <= 7):
                    continue
                if color == chess.WHITE:
                    for rr in range(r, 8):
                        support |= chess.BB_SQUARES[chess.square(af, rr)]
                else:
                    for rr in range(r + 1):
                        support |= chess.BB_SQUARES[chess.square(af, rr)]

            _PASSED_PAWN_MASKS[color][sq] = int(passed)
            _BACKWARD_SUPPORT_MASKS[color][sq] = int(support)

            single = 0
            double = 0
            r1 = r + direction
            if 0 <= r1 <= 7:
                single = int(chess.BB_SQUARES[chess.square(f, r1)])
                start_rank = 1 if color == chess.WHITE else 6
                r2 = r + 2 * direction
                if r == start_rank and 0 <= r2 <= 7:
                    double = int(chess.BB_SQUARES[chess.square(f, r2)])
            _PAWN_SINGLE_PUSH_MASK[color][sq] = single
            _PAWN_DOUBLE_PUSH_MASK[color][sq] = double


_build_square_tables()

_CENTER_FILES = frozenset({2, 3, 4, 5})


def _king_zone(board: chess.Board, color: chess.Color) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]

  # Add squares 1-2 ranks in front of king (toward opponent)
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    drs = (1, 2) if color == chess.WHITE else (-1, -2)

    for df in (-1, 0, 1):
        f = kf + df
        if not (0 <= f <= 7):
            continue
        for dr in drs:
            r = kr + dr
            if 0 <= r <= 7:
                zone |= chess.BB_SQUARES[chess.square(f, r)]

    return zone



def _passed_pawns(board: chess.Board, color: chess.Color) -> int:
    passed = 0
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)

    for sq in chess.scan_forward(int(board.pieces_mask(chess.PAWN, color))):
        if not (_PASSED_PAWN_MASKS[color][sq] & int(enemy_pawns)):
            passed |= chess.BB_SQUARES[sq]

    return passed


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    isolated = 0
    own_pawns = board.pieces_mask(chess.PAWN, color)

    for sq in chess.scan_forward(int(own_pawns)):
        f = chess.square_file(sq)
        if not (_ADJACENT_FILE_MASKS[f] & int(own_pawns)):
            isolated |= chess.BB_SQUARES[sq]

    return isolated


def _connected_pawns(board: chess.Board, color: chess.Color) -> int:
    """Heuristic: pawn with a friendly pawn on an adjacent file within ±1 rank."""
    connected = 0
    own_pawns = int(board.pieces_mask(chess.PAWN, color))

    for sq in chess.scan_forward(own_pawns):
        if _CONNECTED_NEIGHBOR_MASKS[sq] & own_pawns:
            connected |= chess.BB_SQUARES[sq]

    return connected


def _backward_pawns(board: chess.Board, color: chess.Color) -> int:
    """Heuristic backward pawn detector.

    Marks a pawn as backward if:
    - it's not isolated
    - the square directly in front is controlled by an enemy pawn
    - and there is no friendly pawn on adjacent files that is at least as advanced
    """
    backward = 0
    direction = 1 if color == chess.WHITE else -1

    own_pawns = int(board.pieces_mask(chess.PAWN, color))
    enemy_pawns = int(board.pieces_mask(chess.PAWN, not color))

    for sq in chess.scan_forward(own_pawns):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

  # not isolated
        if not (_ADJACENT_FILE_MASKS[f] & own_pawns):
            continue

  # in-front square
        r1 = r + direction
        if not (0 <= r1 <= 7):
            continue
        front_sq = chess.square(f, r1)

  # attacked by enemy pawn?
        if not (chess.BB_PAWN_ATTACKS[color][front_sq] & enemy_pawns):
            continue

  # no adjacent pawn at least as advanced
        if _BACKWARD_SUPPORT_MASKS[color][sq] & own_pawns:
            continue

        backward |= chess.BB_SQUARES[sq]

    return backward



_MOBILITY_MAX = {
    chess.PAWN: 4.0,
    chess.KNIGHT: 8.0,
    chess.BISHOP: 13.0,
    chess.ROOK: 14.0,
    chess.QUEEN: 27.0,
    chess.KING: 8.0,
}



def extra_feature_planes_c(
    board: chess.Board, *, version: str | None = None,
) -> np.ndarray:
    """C-accelerated version: returns (n_extra, 8, 8) float32 directly.

    Extracts bitboards from python-chess Board and delegates all planes
    to the native C extension.
    """
    turn = board.turn
    us, them = turn, not turn
    n_extra = extra_feature_plane_count(version)

  # Build uint64[6] piece-bitboard arrays for each side.
    pieces_us = np.array(
        [int(board.pieces_mask(pt, us)) for pt in chess.PIECE_TYPES], dtype=np.uint64
    )
    pieces_them = np.array(
        [int(board.pieces_mask(pt, them)) for pt in chess.PIECE_TYPES], dtype=np.uint64
    )

    occupied = int(board.occupied)
    _ks_us = board.king(us)
    king_sq_us = _ks_us if _ks_us is not None else -1
    _ks_them = board.king(them)
    king_sq_them = _ks_them if _ks_them is not None else -1
    turn_white = turn == chess.WHITE
    ep_square = board.ep_square if board.ep_square is not None else -1

    return _c_compute(
        pieces_us, pieces_them, occupied, king_sq_us, king_sq_them,
        turn_white, ep_square, n_extra,
    )


def _collect_king_safety_bitboards(board: chess.Board, us: bool, them: bool) -> list[int]:
    """10 bitboards: per-color (king zone + 4 attacker overlaps for KNIGHT/BISHOP/ROOK/QUEEN)."""
    bbs: list[int] = []
    for color in (us, them):
        kz = _king_zone(board, color)
        bbs.append(kz)
        opp = not color
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            overlap = 0
            for sq in board.pieces(pt, opp):
                overlap |= board.attacks_mask(sq) & kz
            bbs.append(overlap)
    return bbs


def _collect_pin_bitboards(board: chess.Board, us: bool, them: bool) -> list[int]:
    """6 bitboards: per-color (pinned squares, pin-ray, discovered-attack mask)."""
    bbs: list[int] = []
    for color in (us, them):
        pinned_mask = 0
        pin_ray_mask = 0
        for sq in chess.scan_forward(int(board.occupied_co[color])):
            pin = board.pin_mask(color, sq)
            if pin != chess.BB_ALL:
                pinned_mask |= chess.BB_SQUARES[sq]
                pin_ray_mask |= pin
        bbs.append(pinned_mask)
        bbs.append(pin_ray_mask)
        bbs.append(_discovered_attack_mask(board, color))
    return bbs


def _pawn_mobility_count(
    sq: int, *, color: chess.Color, occ: int, opp_occ: int, ep_mask: int,
) -> int:
    """Pawn-specific mobility: forward push (single + maybe double) + diagonal captures + en passant."""
    mobility = 0
    single_mask = _PAWN_SINGLE_PUSH_MASK[color][sq]
    if single_mask and not (occ & single_mask):
        mobility += 1
        double_mask = _PAWN_DOUBLE_PUSH_MASK[color][sq]
        if double_mask and not (occ & double_mask):
            mobility += 1
    capture_mask = chess.BB_PAWN_ATTACKS[color][sq]
    mobility += chess.popcount(capture_mask & opp_occ)
    if ep_mask and (capture_mask & ep_mask):
        mobility += 1
    return mobility


class _AttackAccum:
    """Per-color attack maps + per-square attacker counts.

    Harvested from the mobility pass so the v2 threat planes reuse the
    attack computations instead of recomputing them.
    """

    __slots__ = ("by_type", "counts")

    def __init__(self) -> None:
        self.by_type: dict[chess.Color, dict[chess.PieceType, int]] = {
            chess.WHITE: dict.fromkeys(chess.PIECE_TYPES, 0),
            chess.BLACK: dict.fromkeys(chess.PIECE_TYPES, 0),
        }
        self.counts: dict[chess.Color, np.ndarray] = {
            chess.WHITE: np.zeros(64, dtype=np.int32),
            chess.BLACK: np.zeros(64, dtype=np.int32),
        }

    def add(self, color: chess.Color, pt: chess.PieceType, attacks: int) -> None:
        self.by_type[color][pt] |= attacks
        counts = self.counts[color]
        for sq in chess.scan_forward(attacks):
            counts[sq] += 1

    def union(self, color: chess.Color) -> int:
        bb = 0
        for mask in self.by_type[color].values():
            bb |= mask
        return bb


def _fill_mobility_planes(
    board: chess.Board, *, turn: chess.Color, out: np.ndarray,
    attack_accum: _AttackAccum | None = None,
) -> None:
    """Fill out[24:30] in place. Each piece type's plane carries that piece's
    move-count (normalized by _MOBILITY_MAX[pt]) at its source square.

    When ``attack_accum`` is given (v2_threats), the per-piece attack masks
    this pass already computes are also ORed into the accumulator.
    """
    orient_coords = _ORIENT_COORDS[turn]
    occ = int(board.occupied)
    ep_mask = int(chess.BB_SQUARES[board.ep_square]) if board.ep_square is not None else 0

    for pi, pt in enumerate((chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING, chess.PAWN)):
        plane = out[24 + pi]
        max_m = _MOBILITY_MAX[pt]
        for color in (chess.WHITE, chess.BLACK):
            own_occ = int(board.occupied_co[color])
            opp_occ = int(board.occupied_co[not color])
            for sq in chess.scan_forward(int(board.pieces_mask(pt, color))):
                if pt == chess.PAWN:
                    attacks = int(chess.BB_PAWN_ATTACKS[color][sq])
                    mobility = _pawn_mobility_count(
                        sq, color=color, occ=occ, opp_occ=opp_occ, ep_mask=ep_mask,
                    )
                else:
                    attacks = int(board.attacks_mask(sq))
                    mobility = chess.popcount(attacks & ~own_occ)
                if attack_accum is not None:
                    attack_accum.add(color, pt, attacks)
                row, col = orient_coords[sq]
                plane[row, col] = np.float32(float(mobility) / max_m)


def _collect_outpost_bitboards(board: chess.Board, us: bool, them: bool) -> list[int]:
    """4 bitboards: per-color (outpost = pawn-attack covered & not enemy-attacked, space-control)."""
    bbs: list[int] = []
    for color in (us, them):
        own_att = 0
        for sq in board.pieces(chess.PAWN, color):
            own_att |= chess.BB_PAWN_ATTACKS[color][sq]
        enemy_att = 0
        for sq in board.pieces(chess.PAWN, not color):
            enemy_att |= chess.BB_PAWN_ATTACKS[not color][sq]
        bbs.append(own_att & ~enemy_att)

        space = 0
        direction = -1 if color == chess.WHITE else 1
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            if f not in _CENTER_FILES:
                continue
            r = chess.square_rank(sq)
            for dr in (direction, 2 * direction):
                r2 = r + dr
                if 0 <= r2 <= 7:
                    space |= chess.BB_SQUARES[chess.square(f, r2)]
        bbs.append(space)
    return bbs


def _slider_attacks(sq: int, occ: int, *, diagonal: bool) -> int:
    if diagonal:
        return int(chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ])
    return int(
        chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
        | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    )


def _oriented_value_plane(values: np.ndarray, *, turn: chess.Color) -> np.ndarray:
    """(64,) per-square values → side-to-move oriented (8, 8) float32."""
    plane = values.reshape(8, 8).astype(np.float32)
    if turn == chess.WHITE:
        return plane
    return plane[::-1, :].copy()


def _pawn_tension(board: chess.Board, color: chess.Color) -> int:
    """Pawns of ``color`` that can capture an enemy pawn (no en passant)."""
    enemy_pawns = int(board.pieces_mask(chess.PAWN, not color))
    tension = 0
    for sq in chess.scan_forward(int(board.pieces_mask(chess.PAWN, color))):
        if chess.BB_PAWN_ATTACKS[color][sq] & enemy_pawns:
            tension |= chess.BB_SQUARES[sq]
    return tension


def _pawn_storm_values(board: chess.Board, defender: chess.Color) -> np.ndarray:
    """Per-square value of enemy pawns storming ``defender``'s king.

    Enemy pawns on the king file ±1 score ``1 − rank_dist(pawn, king)/7``;
    everything else is 0.
    """
    values = np.zeros(64, dtype=np.float32)
    king_sq = board.king(defender)
    if king_sq is None:
        return values
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    for sq in chess.scan_forward(int(board.pieces_mask(chess.PAWN, not defender))):
        if abs(chess.square_file(sq) - kf) <= 1:
            values[sq] = 1.0 - abs(chess.square_rank(sq) - kr) / 7.0
    return values


def _fill_threat_planes(
    board: chess.Board, *, turn: chess.Color, out: np.ndarray, accum: _AttackAccum,
) -> None:
    """Fill the v2 threat planes out[34:63] from the mobility-pass attacks."""
    us, them = turn, not turn
    by_us = accum.by_type[us]
    by_them = accum.by_type[them]
    all_us = accum.union(us)
    all_them = accum.union(them)
    us_occ = int(board.occupied_co[us])
    them_occ = int(board.occupied_co[them])
    occ = int(board.occupied)

    out[34:48] = bitboards_to_planes(
        [all_us, all_them]
        + [by_us[pt] for pt in chess.PIECE_TYPES]
        + [by_them[pt] for pt in chess.PIECE_TYPES],
        turn=turn,
    )

  # min(count, 7) mirrors the C path's 3-bit saturating counters; the count
  # planes clamp at 4 and the margin at ±4, so the cap is lossless there.
    counts_us = np.minimum(accum.counts[us], 7).astype(np.float32)
    counts_them = np.minimum(accum.counts[them], 7).astype(np.float32)
    out[48] = _oriented_value_plane(np.clip(counts_us / 4.0, 0.0, 1.0), turn=turn)
    out[49] = _oriented_value_plane(np.clip(counts_them / 4.0, 0.0, 1.0), turn=turn)

    def pieces(pt: chess.PieceType, color: chess.Color) -> int:
        return int(board.pieces_mask(pt, color))

    def cheaper_attacked(victim_color: chess.Color, attacker_maps: dict) -> int:
        n = pieces(chess.KNIGHT, victim_color)
        b = pieces(chess.BISHOP, victim_color)
        r = pieces(chess.ROOK, victim_color)
        q = pieces(chess.QUEEN, victim_color)
        return (
            ((n | b | r | q) & attacker_maps[chess.PAWN])
            | ((r | q) & (attacker_maps[chess.KNIGHT] | attacker_maps[chess.BISHOP]))
            | (q & attacker_maps[chess.ROOK])
        )

    hanging_us = us_occ & all_them & ~all_us
    hanging_them = them_occ & all_us & ~all_them
    out[50:54] = bitboards_to_planes(
        [
            hanging_us,
            hanging_them,
            cheaper_attacked(us, by_them),
            cheaper_attacked(them, by_us),
        ],
        turn=turn,
    )

  # Safe checks for the side to move: squares an N/B/R/Q of ours could check
  # the enemy king from (attacks projected FROM the king square at current
  # occupancy), excluding squares the enemy attacks or that we occupy.
    king_sq_them = board.king(them)
    if king_sq_them is None:
        check_n = check_b = check_r = check_q = 0
    else:
        check_n = int(chess.BB_KNIGHT_ATTACKS[king_sq_them])
        check_b = _slider_attacks(king_sq_them, occ, diagonal=True)
        check_r = _slider_attacks(king_sq_them, occ, diagonal=False)
        check_q = check_b | check_r
    safe = ~all_them & ~us_occ & chess.BB_ALL
    out[54:58] = bitboards_to_planes(
        [check_n & safe, check_b & safe, check_r & safe, check_q & safe],
        turn=turn,
    )

    out[58] = _oriented_value_plane(
        np.clip((counts_us - counts_them) / 4.0, -1.0, 1.0), turn=turn,
    )

    out[59:61] = bitboards_to_planes(
        [_pawn_tension(board, us), _pawn_tension(board, them)], turn=turn,
    )

    out[61] = _oriented_value_plane(_pawn_storm_values(board, us), turn=turn)
    out[62] = _oriented_value_plane(_pawn_storm_values(board, them), turn=turn)


def _fill_opponent_safe_check_planes(
    board: chess.Board, *, turn: chess.Color, out: np.ndarray, accum: _AttackAccum,
) -> None:
    """Fill the v3_checks planes out[63:67] — the OPPONENT's safe checks.

    The mirror of the side-to-move safe-checks block in :func:`_fill_threat_planes`:
    squares an N/B/R/Q of THEIRS could check OUR king from (attacks projected
    FROM our king square at current occupancy), excluding squares we attack or
    that they occupy. Side-to-move oriented like everything else.
    """
    us, them = turn, not turn
    all_us = accum.union(us)
    them_occ = int(board.occupied_co[them])
    occ = int(board.occupied)

    king_sq_us = board.king(us)
    if king_sq_us is None:
        check_n = check_b = check_r = check_q = 0
    else:
        check_n = int(chess.BB_KNIGHT_ATTACKS[king_sq_us])
        check_b = _slider_attacks(king_sq_us, occ, diagonal=True)
        check_r = _slider_attacks(king_sq_us, occ, diagonal=False)
        check_q = check_b | check_r
    safe_opp = ~all_us & ~them_occ & chess.BB_ALL
    out[63:67] = bitboards_to_planes(
        [check_n & safe_opp, check_b & safe_opp, check_r & safe_opp, check_q & safe_opp],
        turn=turn,
    )


def _slider_xray(sq: int, occ: int, *, diagonal: bool) -> int:
    """X-ray of one slider: squares seen THROUGH the first blocker on each ray.

    ``direct ^ att(s, occ ^ (direct & occ))`` — remove the squares the slider
    currently attacks-and-are-occupied (i.e. each ray's first blocker), recompute
    the attack, and XOR with the direct attack. The result is the squares beyond
    the first blocker on each ray, disjoint from the direct-attack squares.
    """
    direct = _slider_attacks(sq, occ, diagonal=diagonal)
    occ_xray = occ ^ (direct & occ)
    return direct ^ _slider_attacks(sq, occ_xray, diagonal=diagonal)


def _xray_union(board: chess.Board, color: chess.Color, occ: int) -> tuple[int, int, int]:
    """Per-type x-ray unions for ``color``: (bishop, rook, queen).

    Queen x-ray = bishop-xray(from queen sq) | rook-xray(from queen sq).
    """
    bishop = 0
    for sq in board.pieces(chess.BISHOP, color):
        bishop |= _slider_xray(sq, occ, diagonal=True)
    rook = 0
    for sq in board.pieces(chess.ROOK, color):
        rook |= _slider_xray(sq, occ, diagonal=False)
    queen = 0
    for sq in board.pieces(chess.QUEEN, color):
        queen |= _slider_xray(sq, occ, diagonal=True) | _slider_xray(sq, occ, diagonal=False)
    return bishop, rook, queen


def _fill_xray_planes(
    board: chess.Board, *, turn: chess.Color, out: np.ndarray,
) -> None:
    """Fill the v3_xray planes out[63:69] — per-slider-type x-ray attacks.

    Order: our bishop/rook/queen x-ray, then their bishop/rook/queen x-ray.
    Side-to-move oriented like everything else.
    """
    us, them = turn, not turn
    occ = int(board.occupied)
    us_b, us_r, us_q = _xray_union(board, us, occ)
    them_b, them_r, them_q = _xray_union(board, them, occ)
    out[63:69] = bitboards_to_planes(
        [us_b, us_r, us_q, them_b, them_r, them_q],
        turn=turn,
    )


# Static-exchange-evaluation piece values (pawns). King is "large" so it is
# never voluntarily traded into a losing recapture but can still deliver the
# final capture in the swap-off. Indexed by python-chess PieceType (1..6).
_SEE_PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
}


def _least_valuable_attacker(
    board: chess.Board, square: int, side: chess.Color,
) -> int | None:
    """Square of ``side``'s least-valued piece attacking ``square`` (or None).

    Direct attackers at current occupancy only — the simplified swap-off does
    not discover x-ray re-attackers revealed once a front attacker is removed.
    """
    best_sq = None
    best_val = None
    for sq in chess.scan_forward(int(board.attackers_mask(side, square))):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = _SEE_PIECE_VALUE[piece.piece_type]
        if best_val is None or val < best_val:
            best_val = val
            best_sq = sq
    return best_sq


def _see_capture(board: chess.Board, square: int, initiator: chess.Color) -> int:
    """Static exchange evaluation (pawns) for ``initiator`` capturing on ``square``.

    Standard iterative swap-off: the initiator captures the piece on ``square``
    with its least-valuable attacker, then sides alternate recapturing with
    their least-valuable attacker; each side stops recapturing once continuing
    would worsen its own balance (the usual ``max(0, ...)`` negamax pruning).
    Returns the value the initiator nets, in pawns (>= 0).

    Simplification: attackers are enumerated at current occupancy via
    ``board.attackers_mask`` after removing the pieces already captured. This
    captures sliders revealed once a closer same-line attacker is consumed
    (board state is mutated), but not x-ray batteries that require recomputing
    through a removed enemy attacker on the SAME square — i.e. the common
    "rook behind rook on the file" battery IS handled (the removed piece sat on
    a different square), while a fully general SEE is not strictly guaranteed.
    Good enough for a graded threat signal; documented for parity.
    """
    target = board.piece_at(square)
    if target is None:
        return 0

    work = board.copy(stack=False)
    # Classic swap-off list (Chess Programming Wiki SEE). gain[0] is the value
    # of the piece captured by the initiator. gain[d] = value[attacker_d] −
    # gain[d-1], where attacker_d is the least-valuable piece of the side to
    # move that captures on ``square`` at step d. Removing each attacker from
    # the working occupancy reveals rear sliders along its ray.
    gain: list[int] = [_SEE_PIECE_VALUE[target.piece_type]]
    side = initiator

    while True:
        att_sq = _least_valuable_attacker(work, square, side)
        if att_sq is None:
            break
        attacker = work.piece_at(att_sq)
        assert attacker is not None
        gain.append(_SEE_PIECE_VALUE[attacker.piece_type] - gain[-1])
        work.remove_piece_at(att_sq)
        side = not side

    # Negamax fold-back (CPW: ``while (--d) gain[d-1] = -max(-gain[d-1],
    # gain[d])``): each side declines a recapture that would worsen its own
    # outcome. With d_max = len(gain)-1 captures, fold i from d_max-2 down to 0
    # — the deepest capture (gain[-1]) stands; a lone capture (d_max==1) is
    # never folded, so an undefended piece nets its full value.
    for i in range(len(gain) - 3, -1, -1):
        gain[i] = -max(-gain[i], gain[i + 1])
    return gain[0]


def _fill_see_planes(
    board: chess.Board, *, turn: chess.Color, out: np.ndarray,
) -> None:
    """Fill the v3_see planes out[63:65] — graded SEE, side-to-move oriented.

    [63] OUR pieces' SEE-when-captured: for each of our occupied squares with an
         enemy attacker, the value the OPPONENT nets if THEY initiate, expressed
         from OUR perspective (negated; <= 0 when we lose material).
    [64] THEIR pieces' SEE-when-captured: same for their pieces if WE initiate
         (>= 0 when we win material).
    Both normalized clip(see/8, -1, 1).
    """
    us, them = turn, not turn
    vals_us = np.zeros(64, dtype=np.float32)
    vals_them = np.zeros(64, dtype=np.float32)

    for sq in chess.scan_forward(int(board.occupied_co[us])):
        if not board.attackers_mask(them, sq):
            continue
        # Opponent initiates a capture on our piece; we net the negation.
        see = _see_capture(board, sq, them)
        vals_us[sq] = np.clip(-see / 8.0, -1.0, 1.0)

    for sq in chess.scan_forward(int(board.occupied_co[them])):
        if not board.attackers_mask(us, sq):
            continue
        see = _see_capture(board, sq, us)
        vals_them[sq] = np.clip(see / 8.0, -1.0, 1.0)

    out[63] = _oriented_value_plane(vals_us, turn=turn)
    out[64] = _oriented_value_plane(vals_them, turn=turn)


def _fill_passer_planes(
    board: chess.Board, *, turn: chess.Color, out: np.ndarray,
) -> None:
    """Fill the v3_passers planes out[63:71] — passed-pawn quality.

    4 planes per side (us 63:67, them 67:71), each side-to-move oriented:
      +0 rank-advancement      = (relative rank advanced)/7 in [0, 1]
      +1 safe passer           = stop square not enemy-attacked and not occupied
      +2 blocked passer        = any piece on the stop square
      +3 promo-path enemy ctrl = any file-ahead square is enemy-attacked

    KNOWN LIMITATION (documented, intentionally not fixed): the safe/promo-path
    attack tests use the CURRENT occupancy, with the passer still on its origin.
    Self-blocking can therefore mark a push "safe" that becomes attacked once the
    origin is vacated (e.g. a white passer on e4 screens a black rook on e1 from
    e5; after e4-e5 the rook attacks e5). Recomputing enemy attacks with the
    passer removed from its origin is the correct semantics, but v3_passers is a
    confirmed-negative experiment (+0.036 eval_loss vs v2) we will not promote,
    and the C twin in ``_features_impl.h`` shares this exact behavior — fixing
    only Python would break the parity test for a feature we are dropping. Left
    as-is to keep C/Python parity; revisit only if v3_passers is ever revived.
    """
    occ = int(board.occupied)
    for side_idx, color in enumerate((turn, not turn)):
        base = 63 + 4 * side_idx
        enemy = not color
        direction = 1 if color == chess.WHITE else -1
        passers = _passed_pawns(board, color)
        rank_vals = np.zeros(64, dtype=np.float32)
        safe = 0
        blocked = 0
        promo_ctrl = 0
        for sq in chess.scan_forward(int(passers)):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            rel_rank = r if color == chess.WHITE else 7 - r
            rank_vals[sq] = rel_rank / 7.0

            stop_r = r + direction
            if 0 <= stop_r <= 7:
                stop_sq = chess.square(f, stop_r)
                stop_bit = int(chess.BB_SQUARES[stop_sq])
                if occ & stop_bit:
                    blocked |= int(chess.BB_SQUARES[sq])
                elif not board.is_attacked_by(enemy, stop_sq):
                    safe |= int(chess.BB_SQUARES[sq])

            rr = r + direction
            while 0 <= rr <= 7:
                ahead_sq = chess.square(f, rr)
                if board.is_attacked_by(enemy, ahead_sq):
                    promo_ctrl |= int(chess.BB_SQUARES[sq])
                    break
                rr += direction

        out[base] = _oriented_value_plane(rank_vals, turn=turn)
        out[base + 1:base + 4] = bitboards_to_planes(
            [safe, blocked, promo_ctrl], turn=turn,
        )


def extra_feature_planes_fast(
    board: chess.Board, *, version: str | None = None,
) -> np.ndarray:
    """Optimized version: returns (n_extra, 8, 8) float32 directly.

    Collects all bitboard masks first, converts in a single batch operation,
    then fills in the mobility planes (float values) separately. See the
    module docstring for the plane layout.
    """
    turn = board.turn
    us, them = turn, not turn
    n_extra = extra_feature_plane_count(version)
    out = np.zeros((n_extra, 8, 8), dtype=np.float32)

    bbs: list[int] = []
    bbs.extend(_collect_king_safety_bitboards(board, us, them))
    bbs.extend(_collect_pin_bitboards(board, us, them))
    for color in (us, them):
        bbs.append(_passed_pawns(board, color))
        bbs.append(_isolated_pawns(board, color))
        bbs.append(_backward_pawns(board, color))
        bbs.append(_connected_pawns(board, color))
    assert len(bbs) == 24
    out[:24] = bitboards_to_planes(bbs, turn=turn)

    accum = _AttackAccum() if n_extra > 34 else None
    _fill_mobility_planes(board, turn=turn, out=out, attack_accum=accum)
    out[30:34] = bitboards_to_planes(_collect_outpost_bitboards(board, us, them), turn=turn)

    if accum is not None:
        _fill_threat_planes(board, turn=turn, out=out, accum=accum)
        if n_extra == 67:  # v3_checks: opponent safe checks at [63:67]
            _fill_opponent_safe_check_planes(board, turn=turn, out=out, accum=accum)
        elif n_extra == 69:  # v3_xray: per-slider x-ray attacks at [63:69]
            _fill_xray_planes(board, turn=turn, out=out)
        elif n_extra == 65:  # v3_see: graded SEE at [63:65]
            _fill_see_planes(board, turn=turn, out=out)
        elif n_extra == 71:  # v3_passers: passed-pawn quality at [63:71]
            _fill_passer_planes(board, turn=turn, out=out)
    return out
