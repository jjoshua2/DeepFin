"""Strict standard-chess position and stored-history overlap, not full provenance."""

import chess
import numpy as np


def decode(row):
    if (
        row.shape != (175, 8, 8)
        or row.dtype != np.float32
        or not np.isfinite(row).all()
    ):
        raise ValueError("finite stored175 float32 required")
    if not np.array_equal(row, row.astype(np.float16).astype(np.float32)):
        raise ValueError("exact raw-float16 representability required")
    h = row[:104].reshape(8, 13, 8, 8)
    if not np.isin(h, (0, 1)).all():
        raise ValueError("binary history required")
    for slot in h:
        if not (slot[12] == slot[12, 0, 0]).all():
            raise ValueError("constant repetition flags required")
    for i in (*range(104, 109), 111):
        if not np.isin(row[i], (0, 1)).all() or not (row[i] == row[i, 0, 0]).all():
            raise ValueError("constant binary metadata required")
    if row[111, 0, 0] != 1:
        raise ValueError("bias required")
    turn = chess.BLACK if row[108, 0, 0] else chess.WHITE
    b = chess.Board(None)
    b.turn = turn
    for i in range(12):
        color = turn if i < 6 else not turn
        for rank, file in np.argwhere(row[i] == 1):
            square = chess.square(int(file), int(rank) if turn else 7 - int(rank))
            if b.piece_at(square):
                raise ValueError("overlapping pieces")
            b.set_piece_at(square, chess.Piece(i % 6 + 1, color))
    for i, (color, file) in enumerate(
        ((turn, 0), (turn, 7), (not turn, 0), (not turn, 7))
    ):
        if row[104 + i, 0, 0]:
            b.castling_rights |= chess.BB_SQUARES[chess.square(file, 0 if color else 7)]
    ep = row[110]
    if (
        not np.isin(ep, (0, 1)).all()
        or not (ep == ep[0]).all()
        or int(ep[0].sum()) not in (0, 1)
    ):
        raise ValueError("exact EP column required")
    if ep.any():
        b.ep_square = chess.square(int(np.flatnonzero(ep[0])[0]), 5 if turn else 2)
    clock = row[109]
    if not (clock == clock[0, 0]).all():
        raise ValueError("constant clock required")
    if clock[0, 0] == 1:
        raise ValueError("saturated clock unresolved")
    candidates = [
        i for i in range(100) if np.float32(np.float16(i / 100)) == clock[0, 0]
    ]
    if len(candidates) != 1:
        raise ValueError("quantized clock invalid")
    b.halfmove_clock = candidates[0]
    if not b.is_valid():
        raise ValueError("invalid position metadata")
    return b


def identity(b):
    return (b.board_fen(), b.turn, b.castling_rights, b.ep_square, b.halfmove_clock)


def validate_pair(parent, child):
    try:
        p, c = decode(parent), decode(child)
        if p.turn == c.turn:
            raise ValueError("turn did not alternate")
        ph = parent[:104].reshape(8, 13, 8, 8)
        ch = child[:104].reshape(8, 13, 8, 8)
        flipped = np.concatenate(
            (ph[:7, 6:12, ::-1], ph[:7, :6, ::-1], ph[:7, 12:13, ::-1]), axis=1
        )
        if not np.array_equal(flipped, ch[1:]):
            raise ValueError("stored history overlap mismatch")
        matches = []
        for move in p.legal_moves:
            q = p.copy(stack=False)
            q.push(move)
            if identity(q) == identity(c):
                matches.append(move.uci())
        if len(matches) != 1:
            raise ValueError("unique legal transition absent")
        return {"status": "qualified_narrow", "move_uci": matches[0]}
    except ValueError as exc:
        return {"status": "unresolved", "reason": str(exc)}


def validate_panel(x, pairs):
    if x.shape != (254, 175, 8, 8) or x.dtype != np.float32 or len(pairs) != 227:
        raise ValueError("fixed panel and pair count required")
    frozen = tuple(tuple(p) for p in pairs)
    if (
        len(set(frozen)) != 227
        or len({p[0] for p in frozen}) != 227
        or any(
            len(p) != 2 or any(type(i) is not int or not 0 <= i < 254 for i in p)
            for p in frozen
        )
    ):
        raise ValueError("fixed unique pair indices required")
    return {
        "prior_missing_successors": 27,
        "classifications": [
            {"pair": p, **validate_pair(x[p[0]], x[p[1]])} for p in frozen
        ],
        "scope": "Legal position plus stored seven-slot overlap only; older history, repetition truth, fullmove and draw adjudication unverified; no scores",
    }
