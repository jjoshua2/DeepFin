"""Status-invalid but answerable positions, through the real engine process.

`position` normalises what has a legal answer (stale castling rights, an
impossible en-passant square) instead of refusing it, so a GUI or EPD driver
that sends such a FEN still gets a legal move -- never ``bestmove 0000``.
One engine process (tiny net, CPU) answers every case.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import chess
import pytest

from chess_anti_engine.uci.subprocess_client import LineReader, send_line
from tests.test_uci_smoke import _make_tiny_checkpoint, _spawn_engine

# The reviewer's four: main played the move shown; a blanket status rejection
# forfeited each with 0000.
_REGRESSIONS = (
    "fen 4k3/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "fen r3k2r/8/8/8/8/8/8/4K3 w KQkq - 0 1",
    "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1",
    "fen 4k3/8/8/8/8/8/8/R3K2R b KQkq - 0 1 moves e8d8",
)

# Follow-up review: an "impossible" ep square on which a capture is legal is
# kept, so the capture in the move list is accepted (pre-#1294 main: e1d1).
_KEPT_EP = (
    "fen 4k3/8/8/8/3pP3/8/4P3/4K3 b - e3 0 1 moves d4e3",
    "fen 4k3/8/8/8/3p4/8/8/4K3 b - e3 0 1 moves d4e3",
    "fen 4k3/8/8/8/3pP3/8/4P3/4K3 b - e3 0 1",
    "fen 4k3/8/8/8/3p4/8/8/4K3 b - e3 0 1",
)

_HARD = (
    chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING | chess.STATUS_TOO_MANY_KINGS
    | chess.STATUS_PAWNS_ON_BACKRANK | chess.STATUS_OPPOSITE_CHECK | chess.STATUS_EMPTY
)


class _Engine:
    def __init__(self, checkpoint: Path) -> None:
        self.proc = _spawn_engine(checkpoint)
        self.reader = LineReader(self.proc)
        send_line(self.proc, "uci")
        self.reader.read_until("uciok")
        send_line(self.proc, "isready")
        self.reader.read_until("readyok", timeout_s=120.0)

    def answer(self, position: str) -> tuple[str, list[str]]:
        send_line(self.proc, f"position {position}")
        send_line(self.proc, "go nodes 16")
        lines = self.reader.read_until("bestmove", timeout_s=60.0)
        return lines[-1].split()[1], lines

    def close(self) -> None:
        try:
            send_line(self.proc, "quit")
            self.proc.wait(timeout=10)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> Iterator[_Engine]:
    eng = _Engine(_make_tiny_checkpoint(tmp_path_factory.mktemp("ckpt")))
    try:
        yield eng
    finally:
        eng.close()


def _gui_board(position: str) -> chess.Board:
    """The position the GUI means, as python-chess plays it (its move
    generator already ignores stale rights and impossible ep squares)."""
    tokens = position.split()
    assert tokens[0] == "fen"
    end = tokens.index("moves") if "moves" in tokens else len(tokens)
    board = chess.Board(" ".join(tokens[1:end]))
    for uci in tokens[end + 1:]:
        board.push_uci(uci)
    return board


@pytest.mark.parametrize("position", _REGRESSIONS)
def test_the_reviewer_regressions_get_a_legal_move(engine: _Engine, position: str) -> None:
    move, lines = engine.answer(position)
    assert move != "0000", lines
    assert chess.Move.from_uci(move) in _gui_board(position).legal_moves, lines
    assert any("position: normalised castling rights" in ln for ln in lines), lines
    assert not any("rejected" in ln for ln in lines), lines


def _corrupted_fens(seed: int, count: int) -> list[str]:
    """Positions from random play, then made status-invalid in a way that
    still has an answer: phantom castling rights, an impossible ep square,
    an extra pawn (9 pawns) -- any combination. Kept only if no hard flag is
    left after normalisation and the side to move has a legal move."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randrange(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        fields = board.fen().split()
        kind = rng.randrange(1, 8)
        if kind & 1:
            fields[2] = "KQkq"
        if kind & 2:
            rank = "3" if board.turn == chess.WHITE else "6"  # wrong side's rank
            fields[3] = rng.choice("abcdefgh") + rank
        if kind & 4:
            empty = [sq for sq in chess.SQUARES
                     if board.piece_at(sq) is None and 1 <= chess.square_rank(sq) <= 6]
            placed = chess.Board(" ".join(fields))
            pawns = chess.popcount(placed.pawns & placed.occupied_co[board.turn])
            for sq in rng.sample(empty, k=min(len(empty), max(0, 9 - pawns))):
                placed.set_piece_at(sq, chess.Piece(chess.PAWN, board.turn))
            fields[0] = placed.board_fen()
        fen = " ".join(fields)
        try:
            corrupted = chess.Board(fen)
        except ValueError:
            continue
        if corrupted.status() == chess.STATUS_VALID or corrupted.status() & _HARD:
            continue
        clean = corrupted.copy()
        clean.castling_rights = clean.clean_castling_rights()
        if clean.status() & chess.STATUS_INVALID_EP_SQUARE:
            clean.ep_square = None
        if not any(clean.legal_moves):
            continue
        out.append(fen)
    return out


@pytest.mark.parametrize("fen", _corrupted_fens(seed=20261114, count=24))
def test_status_invalid_but_playable_fens_always_get_a_legal_move(engine: _Engine, fen: str) -> None:
    move, lines = engine.answer(f"fen {fen}")
    assert move != "0000", lines
    clean = chess.Board(fen)
    clean.castling_rights = clean.clean_castling_rights()
    if clean.status() & chess.STATUS_INVALID_EP_SQUARE:
        clean.ep_square = None
    assert chess.Move.from_uci(move) in clean.legal_moves, (fen, lines)


@pytest.mark.parametrize("position", _KEPT_EP)
def test_a_capturable_unreachable_ep_square_is_kept(engine: _Engine, position: str) -> None:
    move, lines = engine.answer(position)
    assert move != "0000", lines
    assert chess.Move.from_uci(move) in _gui_board(position).legal_moves, lines
    assert any("kept en-passant square e3" in ln for ln in lines), lines
    assert not any("rejected" in ln for ln in lines), lines


def test_a_move_list_reaching_a_board_the_c_generator_truncates_is_rejected(engine: _Engine) -> None:
    """The root agrees; after e4h7 python-chess has 267 legal moves and the C
    generator caps at 256 (CBOARD_MAX_LEGAL_MOVES): rejected, `bestmove 0000`."""
    move, lines = engine.answer(
        "fen QQQQQQnk/Q5RP/Q6Q/Q6Q/Q3b2Q/Q6Q/Q6Q/KQQQQQQQ b - - 0 1 moves e4h7")
    assert move == "0000", lines
    assert any("after move 1 'e4h7'" in ln and "256 vs 267 legal moves" in ln for ln in lines), lines
