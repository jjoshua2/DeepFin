from __future__ import annotations

import numpy as np
import chess

from chess_anti_engine.utils.bitboards import bitboard_to_plane


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


def king_safety_planes(board: chess.Board) -> list[np.ndarray]:
    """10 planes: for (us, them): king zone (1) + attacks on zone by N,B,R,Q (4)."""
    turn = board.turn
    us, them = turn, not turn
    planes: list[np.ndarray] = []

    for color in (us, them):
        kz = _king_zone(board, color)
        planes.append(bitboard_to_plane(kz, turn=turn))

        # Attacks on that zone by opponent pieces
        opp = not color
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            overlap = 0
            for sq in board.pieces(pt, opp):
                overlap |= board.attacks_mask(sq) & kz
            planes.append(bitboard_to_plane(overlap, turn=turn))

    return planes


def pin_and_xray_planes(board: chess.Board) -> list[np.ndarray]:
    """6 planes: for (us, them): pinned pieces mask, pin ray mask, discovered-check potential."""
    turn = board.turn
    us, them = turn, not turn

    planes: list[np.ndarray] = []

    for color in (us, them):
        pinned_mask = 0
        pin_ray_mask = 0

        # pinned pieces & rays
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != color:
                continue
            pin = board.pin_mask(color, sq)
            if pin != chess.BB_ALL:
                pinned_mask |= chess.BB_SQUARES[sq]
                pin_ray_mask |= pin

        # discovered attack/check potential (expensive but ok for phase 1)
        discovered_mask = 0
        opp_king = board.king(not color)
        if opp_king is not None:
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None or piece.color != color:
                    continue
                b2 = board.copy(stack=False)
                b2.remove_piece_at(sq)
                if b2.is_attacked_by(color, opp_king):
                    discovered_mask |= chess.BB_SQUARES[sq]

        planes.append(bitboard_to_plane(pinned_mask, turn=turn))
        planes.append(bitboard_to_plane(pin_ray_mask, turn=turn))
        planes.append(bitboard_to_plane(discovered_mask, turn=turn))

    return planes


def _passed_pawns(board: chess.Board, color: chess.Color) -> int:
    passed = 0
    direction = 1 if color == chess.WHITE else -1
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)

    for sq in board.pieces(chess.PAWN, color):
        f0 = chess.square_file(sq)
        r0 = chess.square_rank(sq)
        blocking = 0
        for f in range(max(0, f0 - 1), min(7, f0 + 1) + 1):
            r = r0 + direction
            while 0 <= r <= 7:
                blocking |= chess.BB_SQUARES[chess.square(f, r)]
                r += direction
        if not (blocking & enemy_pawns):
            passed |= chess.BB_SQUARES[sq]

    return passed


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    isolated = 0
    own_pawns = board.pieces_mask(chess.PAWN, color)

    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        adjacent_files = 0
        if f > 0:
            adjacent_files |= chess.BB_FILES[f - 1]
        if f < 7:
            adjacent_files |= chess.BB_FILES[f + 1]
        if not (adjacent_files & own_pawns):
            isolated |= chess.BB_SQUARES[sq]

    return isolated


def _connected_pawns(board: chess.Board, color: chess.Color) -> int:
    """Heuristic: pawn with a friendly pawn on an adjacent file within ±1 rank."""
    connected = 0
    own_pawns = list(board.pieces(chess.PAWN, color))
    own_set = set(own_pawns)

    for sq in own_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        ok = False
        for df in (-1, 1):
            f2 = f + df
            if not (0 <= f2 <= 7):
                continue
            for dr in (-1, 0, 1):
                r2 = r + dr
                if not (0 <= r2 <= 7):
                    continue
                if chess.square(f2, r2) in own_set:
                    ok = True
                    break
            if ok:
                break
        if ok:
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

    own_pawns = list(board.pieces(chess.PAWN, color))
    own_pawns_by_file: dict[int, list[int]] = {f: [] for f in range(8)}
    for sq in own_pawns:
        own_pawns_by_file[chess.square_file(sq)].append(chess.square_rank(sq))

    enemy_pawns = board.pieces_mask(chess.PAWN, not color)

    for sq in own_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        # not isolated
        adjacent_files = []
        if f > 0:
            adjacent_files.append(f - 1)
        if f < 7:
            adjacent_files.append(f + 1)
        if all(len(own_pawns_by_file[af]) == 0 for af in adjacent_files):
            continue

        # in-front square
        r1 = r + direction
        if not (0 <= r1 <= 7):
            continue
        front_sq = chess.square(f, r1)

        # attacked by enemy pawn?
        # enemy pawn attacks depend on enemy color
        enemy = not color
        attacked_by_enemy_pawn = False
        if enemy == chess.WHITE:
            # enemy pawn attacks upwards (to higher ranks)
            for df in (-1, 1):
                f2 = f + df
                r2 = r1 - 1
                if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                    if enemy_pawns & chess.BB_SQUARES[chess.square(f2, r2)]:
                        attacked_by_enemy_pawn = True
                        break
        else:
            for df in (-1, 1):
                f2 = f + df
                r2 = r1 + 1
                if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                    if enemy_pawns & chess.BB_SQUARES[chess.square(f2, r2)]:
                        attacked_by_enemy_pawn = True
                        break
        if not attacked_by_enemy_pawn:
            continue

        # no adjacent pawn at least as advanced
        def is_at_least_as_advanced(rank_other: int) -> bool:
            return rank_other >= r if color == chess.WHITE else rank_other <= r

        if any(any(is_at_least_as_advanced(rr) for rr in own_pawns_by_file[af]) for af in adjacent_files):
            continue

        backward |= chess.BB_SQUARES[sq]

    return backward


def pawn_structure_planes(board: chess.Board) -> list[np.ndarray]:
    """8 planes: passed/isolated/backward/connected × (us, them)."""
    turn = board.turn
    us, them = turn, not turn

    planes: list[np.ndarray] = []
    for color in (us, them):
        planes.append(bitboard_to_plane(_passed_pawns(board, color), turn=turn))
        planes.append(bitboard_to_plane(_isolated_pawns(board, color), turn=turn))
        planes.append(bitboard_to_plane(_backward_pawns(board, color), turn=turn))
        planes.append(bitboard_to_plane(_connected_pawns(board, color), turn=turn))

    return planes


_MOBILITY_MAX = {
    chess.PAWN: 4.0,
    chess.KNIGHT: 8.0,
    chess.BISHOP: 13.0,
    chess.ROOK: 14.0,
    chess.QUEEN: 27.0,
    chess.KING: 8.0,
}


def mobility_planes(board: chess.Board) -> list[np.ndarray]:
    """6 planes: per piece type mobility.

    For each piece of a given type (both colors), put a normalized mobility value
    on the piece's square.

    Note: mobility is computed against the current occupancy, and excludes
    destinations occupied by same-color pieces.
    """
    turn = board.turn
    planes: list[np.ndarray] = []

    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING, chess.PAWN):
        plane = np.zeros((8, 8), dtype=np.float32)
        max_m = _MOBILITY_MAX[pt]

        for color in (chess.WHITE, chess.BLACK):
            for sq in board.pieces(pt, color):
                if pt == chess.PAWN:
                    # pawn: forward (1/2) if empty + captures (+ EP)
                    mobility = 0
                    direction = 1 if color == chess.WHITE else -1
                    r = chess.square_rank(sq)
                    f = chess.square_file(sq)
                    r1 = r + direction
                    if 0 <= r1 <= 7:
                        sq1 = chess.square(f, r1)
                        if board.piece_at(sq1) is None:
                            mobility += 1
                            start_rank = 1 if color == chess.WHITE else 6
                            r2 = r + 2 * direction
                            if r == start_rank and 0 <= r2 <= 7:
                                sq2 = chess.square(f, r2)
                                if board.piece_at(sq2) is None:
                                    mobility += 1
                        for df in (-1, 1):
                            f2 = f + df
                            if 0 <= f2 <= 7:
                                cap_sq = chess.square(f2, r1)
                                p = board.piece_at(cap_sq)
                                if p is not None and p.color != color:
                                    mobility += 1

                    if board.ep_square is not None:
                        if chess.square_rank(board.ep_square) == (r + direction):
                            if abs(chess.square_file(board.ep_square) - f) == 1:
                                mobility += 1
                else:
                    attacks = board.attacks_mask(sq)
                    own_occ = board.occupied_co[color]
                    mobility = chess.popcount(attacks & ~own_occ)

                val = float(mobility) / max_m
                osq = sq if turn == chess.WHITE else chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))
                plane[chess.square_rank(osq), chess.square_file(osq)] = np.float32(val)

        planes.append(plane)

    return planes


def outpost_and_space_planes(board: chess.Board) -> list[np.ndarray]:
    """4 planes: outpost and space for (us, them).

    Outpost heuristic:
    - square supported by own pawn attack
    - square not attacked by enemy pawn attack

    Space heuristic:
    - squares behind own pawns on center files (c-f), 1-2 ranks behind.
    """
    turn = board.turn
    us, them = turn, not turn
    planes: list[np.ndarray] = []

    for color in (us, them):
        # python-chess represents pawn attacks as lookup tables per square, so we build via iteration
        own_att = 0
        for sq in board.pieces(chess.PAWN, color):
            own_att |= chess.BB_PAWN_ATTACKS[color][sq]
        enemy_att = 0
        for sq in board.pieces(chess.PAWN, not color):
            enemy_att |= chess.BB_PAWN_ATTACKS[not color][sq]

        outpost = own_att & ~enemy_att
        planes.append(bitboard_to_plane(outpost, turn=turn))

        # space (very rough): behind pawns on files c-f
        space = 0
        center_files = {2, 3, 4, 5}
        direction = -1 if color == chess.WHITE else 1  # behind toward own side
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            if f not in center_files:
                continue
            r = chess.square_rank(sq)
            for dr in (direction, 2 * direction):
                r2 = r + dr
                if 0 <= r2 <= 7:
                    space |= chess.BB_SQUARES[chess.square(f, r2)]
        planes.append(bitboard_to_plane(space, turn=turn))

    return planes


def extra_feature_planes(board: chess.Board) -> list[np.ndarray]:
    """Return the additional classical feature planes.

    Target count: 34 planes.
    - King safety: 10
    - Pins/x-rays/discovered: 6
    - Pawn structure: 8
    - Mobility: 6
    - Outpost/space: 4
    """
    planes: list[np.ndarray] = []
    planes.extend(king_safety_planes(board))          # 10
    planes.extend(pin_and_xray_planes(board))         # 6
    planes.extend(pawn_structure_planes(board))       # 8
    planes.extend(mobility_planes(board))             # 6
    planes.extend(outpost_and_space_planes(board))    # 4
    return planes
