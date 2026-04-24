from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.utils.bitboards import bitboards_to_planes

try:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,
    )
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False

if TYPE_CHECKING:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,  # noqa: F401,F811
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



def extra_feature_planes_c(board: chess.Board) -> np.ndarray:
    """C-accelerated version: returns (34, 8, 8) float32 directly.

    Extracts bitboards from python-chess Board and delegates all 34 planes
    to the native C extension.
    """
    turn = board.turn
    us, them = turn, not turn

  # Build uint64[6] arrays for each side: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    piece_types = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
    pieces_us = np.array(
        [int(board.pieces_mask(pt, us)) for pt in piece_types], dtype=np.uint64
    )
    pieces_them = np.array(
        [int(board.pieces_mask(pt, them)) for pt in piece_types], dtype=np.uint64
    )

    occupied = int(board.occupied)
    _ks_us = board.king(us)
    king_sq_us = _ks_us if _ks_us is not None else -1
    _ks_them = board.king(them)
    king_sq_them = _ks_them if _ks_them is not None else -1
    turn_white = turn == chess.WHITE
    ep_square = board.ep_square if board.ep_square is not None else -1

    return _c_compute(pieces_us, pieces_them, occupied, king_sq_us, king_sq_them, turn_white, ep_square)


def extra_feature_planes_fast(board: chess.Board) -> np.ndarray:
    """Optimized version: returns (34, 8, 8) float32 directly.

    Collects all bitboard masks first, converts in a single batch operation,
    then fills in the mobility planes (float values) separately.

    Layout: [0:10] king safety, [10:16] pins, [16:24] pawns, [24:30] mobility, [30:34] outpost
    """
    turn = board.turn
    us, them = turn, not turn
    out = np.zeros((34, 8, 8), dtype=np.float32)

  # --- Collect bitboard-based planes ---
  # Phase 1: king safety (10) + pins (6) + pawn structure (8) = 24 bitboards
    bbs: list[int] = []

  # King safety: 10 planes
    for color in (us, them):
        kz = _king_zone(board, color)
        bbs.append(kz)
        opp = not color
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            overlap = 0
            for sq in board.pieces(pt, opp):
                overlap |= board.attacks_mask(sq) & kz
            bbs.append(overlap)

  # Pins/x-rays/discovered: 6 planes
    for color in (us, them):
        pinned_mask = 0
        pin_ray_mask = 0
        for sq in chess.scan_forward(int(board.occupied_co[color])):
            pin = board.pin_mask(color, sq)
            if pin != chess.BB_ALL:
                pinned_mask |= chess.BB_SQUARES[sq]
                pin_ray_mask |= pin
        discovered_mask = _discovered_attack_mask(board, color)
        bbs.append(pinned_mask)
        bbs.append(pin_ray_mask)
        bbs.append(discovered_mask)

  # Pawn structure: 8 planes
    for color in (us, them):
        bbs.append(_passed_pawns(board, color))
        bbs.append(_isolated_pawns(board, color))
        bbs.append(_backward_pawns(board, color))
        bbs.append(_connected_pawns(board, color))

  # Convert first 24 bitboard planes in one batch → out[0:24]
    assert len(bbs) == 24
    out[:24] = bitboards_to_planes(bbs, turn=turn)

  # --- Mobility: 6 planes [24:30] (float values, not bitboard) ---
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
                else:
                    attacks = board.attacks_mask(sq)
                    mobility = chess.popcount(attacks & ~own_occ)
                row, col = orient_coords[sq]
                plane[row, col] = np.float32(float(mobility) / max_m)

  # --- Outpost/space: 4 planes [30:34] ---
    outpost_bbs: list[int] = []
    for color in (us, them):
        own_att = 0
        for sq in board.pieces(chess.PAWN, color):
            own_att |= chess.BB_PAWN_ATTACKS[color][sq]
        enemy_att = 0
        for sq in board.pieces(chess.PAWN, not color):
            enemy_att |= chess.BB_PAWN_ATTACKS[not color][sq]
        outpost_bbs.append(own_att & ~enemy_att)

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
        outpost_bbs.append(space)

    out[30:34] = bitboards_to_planes(outpost_bbs, turn=turn)

    return out
