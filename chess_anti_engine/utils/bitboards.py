from __future__ import annotations

import chess
import numpy as np


def orient_square(sq: chess.Square, turn: chess.Color) -> chess.Square:
    """Map a square into side-to-move perspective.

    Convention:
    - If it's White to move: identity.
    - If it's Black to move: flip ranks (mirror vertically).

    This matches the common LC0 convention that the board is viewed from the
    side-to-move's perspective.
    """
    if turn == chess.WHITE:
        return sq
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return chess.square(f, 7 - r)


def bitboards_to_planes(bbs: list[int], *, turn: chess.Color) -> np.ndarray:
    """Convert multiple bitboards to oriented (N, 8, 8) float32 planes in one batch."""
    n = len(bbs)
    if n == 0:
        return np.zeros((0, 8, 8), dtype=np.float32)
    raw_bytes = b''.join(int(bb).to_bytes(8, 'big') for bb in bbs)
    raw = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).reshape(n, 8, 8)
    if turn == chess.WHITE:
        return raw[:, ::-1, ::-1].astype(np.float32)
    return raw[:, :, ::-1].astype(np.float32)


def file_to_plane(file_idx: int) -> np.ndarray:
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[:, file_idx] = 1.0
    return plane


def unsearchable_king_reason(board: chess.Board) -> str | None:
    """Why ``board``'s kings make it unsearchable, or None if they are fine.

    ``chess.Board(fen)`` is a STRUCTURAL parse: it raises only on a malformed
    FEN string, never on an impossible position. ``4k3/8/8/8/8/8/8/8 w - - 0 1``
    parses happily. ``Board.legal_moves`` will also happily CAPTURE A KING when
    the side not to move is in check, so a legal starting position plus legal
    pushes can still arrive here kingless.

    The condition checked is deliberately narrow: **python-chess's
    ``Board.king(color)`` and the C ``lsb64(bb[KING] & occ[color])`` must
    provably designate the same square.** Every king-safety answer on either
    side of the Python/C boundary is computed relative to that square, so when
    the two disagree -- or when one of them does not exist -- the two
    implementations are answering about different positions. Measured, all three
    ways of breaking it diverge, and all three in the key-MERGING direction (we
    drop an en-passant term python-chess keeps, which invents a repetition):

      * no king             ``4k3/8/8/3pP3/8/8/8/8 w - d6``   -- ``king()`` is
        None, so python-chess's ``is_into_check`` short-circuits to False and
        calls every ep capture legal, while ``bitboards_have_legal_ep`` returns
        0 on ``!us_kings``;
      * king marked promoted ``4k3/8/8/K~2pP2r/8/8/8/8 w - d6`` -- ``Board.king``
        masks with ``~promoted`` and returns None, while the C reads the raw
        king bitboard;
      * two kings of a colour ``4k3/8/7K/r2pP3/8/8/8/K7 w - d6`` -- ``Board.king``
        takes ``msb``, the C takes ``lsb``, and they pick different squares.

    Both colours are checked although the C precondition is about the MOVER:
    search pushes moves, so a position whose OPPONENT lacks a king produces
    children whose MOVER lacks one. Root-level agreement does not survive a ply.

    ⚑ This is NOT ``board.status() == Status.VALID`` and must not become it.
    Its callers exist to serve EPD, puzzle and blind-spot drivers, which feed
    routinely weird-but-legal positions: pawns in odd files, lopsided material,
    the side not to move already in check, an ep square that no double push
    could have produced. ``VALID`` rejects all of those and would gut those
    callers. This predicate says nothing about any of them -- it fires on 0 of
    5.9M real FENs across the blindspot, audit, wac.epd and lichess-puzzle
    corpora. ``selfplay/opening.py::_fen_reject_reason`` DOES use the full
    ``is_valid()``, correctly, since a training seed is meant to be a real
    position -- which is why the training path never had this hole.

    Lives here rather than beside either caller because two copies of a rule is
    the defect class this whole PR exists to remove.
    """
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        king_bb = board.kings & board.occupied_co[color]
        n_kings = chess.popcount(king_bb)
        if n_kings != 1:
            return f"{name} has {n_kings} kings, need exactly 1"
        if board.king(color) is None:
  # popcount is 1, so the square exists but carries a '~' promoted marker and
  # Board.king()'s `& ~promoted` drops it. The C reads the raw king bitboard.
            return f"{name}'s only king is marked promoted, so python-chess sees none"
    return None
