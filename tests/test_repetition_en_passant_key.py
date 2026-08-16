"""The repetition key must treat en passant exactly as python-chess does.

python-chess 1.11.2, ``Board._transposition_key`` — the key ``is_repetition()``
compares:

    return (self.pawns, self.knights, self.bishops, self.rooks,
            self.queens, self.kings,
            self.occupied_co[WHITE], self.occupied_co[BLACK],
            self.turn, self.clean_castling_rights(),
            self.ep_square if self.has_legal_en_passant() else None)

``ep_square`` is dropped when ``has_legal_en_passant()`` is false and KEPT when
it is true. Our C hash used to drop it unconditionally, and three comments
justified that by claiming it matched python-chess. It does not: dropping it
unconditionally makes a legal-ep position compare EQUAL to the same
pieces/turn/castling without the ep right, which is a false repetition.

Why this is not cosmetic: ``cboard_search_terminal`` (mcts/_mcts_tree.c) answers
a repetition with SOLVED_DRAW, and ``tree_resolve_from_children`` lets a single
drawn child turn a proven-LOST node into a proven-DRAWN one. SOLVED is terminal,
so extra visits never correct it and the error propagates upward.

The fixtures below are chosen to separate THREE implementations, not two:

  (a) ep set, no pawn can capture at all   -> key must NOT split
  (b) ep set, capture pseudo-legal but ILLEGAL (pinned) -> key must NOT split
  (c) ep set, capture genuinely legal      -> key MUST split

(c) fails against the old ep-blind key. (b) fails against the tempting shortcut
of reusing ``cboard_ep_capture_available``, which is pseudo-legal by design — it
would turn today's false POSITIVES into false NEGATIVES (a real draw missed).
(a) is the case the old comment was actually right about, pinned so a future
"just always include ep" cannot pass.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.lc0 import LC0_FULL, encode_lc0_full
from chess_anti_engine.moves import move_to_index

# encode_full plane 103 is the current-position repetition plane on the
# production C path — the plane the training input actually carries.
REP_PLANE = 103


def _play(fen: str, ucis: Sequence[str]) -> tuple[chess.Board, CBoard]:
    """Play ``ucis`` from ``fen`` on a python-chess board and a CBoard in step."""
    board = chess.Board(fen)
    cb = CBoard.from_board(board)
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"{uci} illegal in {board.fen()}"
        cb.push_index(move_to_index(move, board))
        board.push(move)
    return board, cb


def _plane103(cb: CBoard) -> bool:
    planes = np.asarray(cb.encode_full()).reshape(-1, 8, 8)
    return bool(planes[REP_PLANE].any())


# ---------------------------------------------------------------------------
# The reported reproduction, with both controls.
#
# A test that only asserted the ep case would be satisfied by an always-clear
# plane, so the two controls are part of the regression test, not decoration.
# ---------------------------------------------------------------------------

_REPRO_FEN = "rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.mark.parametrize(
    ("label", "ucis", "expected"),
    [
        # e2e4 gives black a LEGAL ep (d4xe3); the knight shuffle restores the
        # pieces, turn and castling. python-chess: NOT a repetition.
        ("legal ep must not repeat", ["e2e4", "g8f6", "g1f3", "f6g8", "f3g1"], False),
        # Control: same shape, no repetition at all -> plane must be CLEAR.
        ("control non-repetition", ["b1c3", "g8f6", "c3b1", "f6g8", "g1f3"], False),
        # Control: a genuine repetition -> plane must be SET. Without this the
        # test above passes against an encoder that never sets the plane.
        ("control genuine repetition", ["b1c3", "g8f6", "c3b1", "f6g8"], True),
    ],
)
def test_repro_double_push_then_knight_shuffle(
    label: str, ucis: Sequence[str], expected: bool,
) -> None:
    board, cb = _play(_REPRO_FEN, ucis)
    assert board.is_repetition(2) is expected, f"oracle disagrees on {label!r}"
    assert cb.is_repetition() is expected, label
    assert _plane103(cb) is expected, label


# ---------------------------------------------------------------------------
# The three discriminating ep fixtures.
#
# (b) and (c) are the SAME position but for a black rook on e8 pinning the
# capturing pawn: arm against arm, so the only thing that can explain a
# different verdict is ep LEGALITY.
# ---------------------------------------------------------------------------

# White pushes d2d4; black has no pawn that attacks d3. Shuffle is black-first
# because the double push leaves black to move.
_NO_CAPTURER = ("7k/8/8/8/8/8/3P4/4K3 w - - 0 1",
                ["d2d4", "h8g8", "e1f1", "g8h8", "f1e1"])
# Black pushes d7d5; white's e5 pawn attacks d6 but is pinned to Ke1 by Re8,
# so exd6 is pseudo-legal and ILLEGAL.
_PINNED_CAPTURER = ("4r2k/3p4/8/4P3/8/8/8/4K3 b - - 0 1",
                    ["d7d5", "e1f1", "h8g8", "f1e1", "g8h8"])
# Identical, minus the pinning rook: exd6 is genuinely legal.
_LEGAL_CAPTURER = ("7k/3p4/8/4P3/8/8/8/4K3 b - - 0 1",
                   ["d7d5", "e1f1", "h8g8", "f1e1", "g8h8"])


@pytest.mark.parametrize(
    ("label", "case", "ep_legal", "ep_pseudo", "repeats"),
    [
        ("(a) ep set, no capturer", _NO_CAPTURER, False, False, True),
        ("(b) ep pseudo-legal but pinned", _PINNED_CAPTURER, False, True, True),
        ("(c) ep genuinely legal", _LEGAL_CAPTURER, True, True, False),
    ],
)
def test_ep_legality_decides_the_repetition_key(
    label: str,
    case: tuple[str, Sequence[str]],
    ep_legal: bool,
    ep_pseudo: bool,
    repeats: bool,
) -> None:
    fen, ucis = case

    # 1. The fixture is what it claims to be. Without this the three cases could
    #    silently collapse onto the same ep status and stop discriminating.
    board, cb = _play(fen, ucis[:1])
    assert board.ep_square is not None, f"{label}: fixture sets no ep square"
    assert board.has_legal_en_passant() is ep_legal, label
    assert any(True for _ in board.generate_pseudo_legal_ep()) is ep_pseudo, label
    # Our C predicate must agree with python-chess's, case by case.
    assert cb.has_legal_en_passant() is ep_legal, label

    # 2. The repetition verdict that ep status decides.
    board, cb = _play(fen, ucis)
    assert board.is_repetition(2) is repeats, f"oracle disagrees on {label!r}"
    assert cb.is_repetition() is repeats, label
    assert _plane103(cb) is repeats, label


def test_occupied_ep_target_is_not_a_legal_en_passant() -> None:
    """An ep square that is OCCUPIED cannot be captured onto.

    python-chess guards this in ``generate_pseudo_legal_ep``::

        if BB_SQUARES[self.ep_square] & self.occupied:
            return

    Unreachable from a real double push, but reachable through ``from_raw`` and
    ``from_board``: python-chess keeps ``Board.ep_square`` set on a hand-written
    FEN whose target is occupied — ``fen()`` prints ``-`` while the attribute
    stays 43 — so ``from_board`` reads it straight through. Found by review.
    """
    fen = "7k/8/3n4/3pP3/8/8/8/4K3 w - d6 0 1"
    board = chess.Board(fen)
    # The fixture only bites while python-chess still exposes the ep square.
    assert board.ep_square == chess.D6
    assert board.piece_at(chess.D6) is not None, "ep target must be occupied"
    assert board.has_legal_en_passant() is False
    assert list(board.generate_pseudo_legal_ep()) == []

    assert CBoard.from_board(board).has_legal_en_passant() is False

    # ...and through from_raw, where the ep square is supplied directly.
    raw = CBoard.from_raw(
        board.pawns, board.knights, board.bishops, board.rooks,
        board.queens, board.kings,
        int(board.occupied_co[chess.WHITE]), int(board.occupied_co[chess.BLACK]),
        int(board.turn), 0, chess.D6, 0,
    )
    assert raw.has_legal_en_passant() is False


def test_pinned_and_legal_fixtures_differ_only_by_the_rook() -> None:
    """Guard the (b)/(c) pair: one added rook is the whole difference.

    If the two FENs ever drift apart in another way, the pair stops isolating ep
    legality and the (b) case silently stops proving anything about the
    pseudo-legal shortcut.
    """
    pinned = chess.Board(_PINNED_CAPTURER[0])
    legal = chess.Board(_LEGAL_CAPTURER[0])
    assert _PINNED_CAPTURER[1] == _LEGAL_CAPTURER[1]
    diff = {
        sq for sq in chess.SQUARES
        if pinned.piece_at(sq) != legal.piece_at(sq)
    }
    assert diff == {chess.E8}
    assert pinned.piece_at(chess.E8) == chess.Piece(chess.ROOK, chess.BLACK)
    assert legal.piece_at(chess.E8) is None


# ---------------------------------------------------------------------------
# Resume-record compatibility: the persisted identity hash must stay EP-blind.
#
# selfplay/resume.py stores zobrist_hash as pos_hash / final_pos_hash and
# re-derives it on replay; a mismatch raises ResumeStateError("position_mismatch"),
# which is NOT in _PRESERVE_FILE_REASONS, so the game is DISCARDED rather than
# deferred. The fix therefore layers the ep term on top of the hash instead of
# folding it in, and this test is what stops a later change from folding it in.
# ---------------------------------------------------------------------------


def test_zobrist_identity_hash_is_unchanged_by_ep() -> None:
    with_ep, cb_with = _play(_LEGAL_CAPTURER[0], ["d7d5"])
    assert with_ep.has_legal_en_passant()

    # The same position reached without a double push: no ep right.
    without = chess.Board("7k/8/8/3pP3/8/8/8/4K3 w - - 0 1")
    cb_without = CBoard.from_board(without)
    assert not without.has_legal_en_passant()

    assert cb_with.zobrist_hash == cb_without.zobrist_hash, (
        "zobrist_hash is the PERSISTED position-identity hash; folding ep into "
        "it would invalidate every in-flight selfplay resume record"
    )
    # ...while the repetition question about them still differs, which is the
    # whole point of keeping the two keys separate.
    assert cb_with.has_legal_en_passant() is True
    assert cb_without.has_legal_en_passant() is False


# ---------------------------------------------------------------------------
# Python-path parity: encoding/lc0.py::_check_repetitions carried the same wrong
# justification, so before the fix the two paths agreed on the WRONG rule.
# ---------------------------------------------------------------------------


def _py_rep_plane(board: chess.Board) -> bool:
    planes = encode_lc0_full(board)
    return bool(planes[LC0_FULL.legacy_repetition_base].any())


@pytest.mark.parametrize(
    ("label", "case", "repeats"),
    [
        ("(a) ep set, no capturer", _NO_CAPTURER, True),
        ("(b) ep pseudo-legal but pinned", _PINNED_CAPTURER, True),
        ("(c) ep genuinely legal", _LEGAL_CAPTURER, False),
        ("repro", (_REPRO_FEN, ["e2e4", "g8f6", "g1f3", "f6g8", "f3g1"]), False),
        ("repro control", (_REPRO_FEN, ["b1c3", "g8f6", "c3b1", "f6g8"]), True),
    ],
)
def test_python_encoder_agrees_with_python_chess(
    label: str, case: tuple[str, Sequence[str]], repeats: bool,
) -> None:
    fen, ucis = case
    board, _ = _play(fen, ucis)
    assert board.is_repetition(2) is repeats, f"oracle disagrees on {label!r}"
    assert _py_rep_plane(board) is repeats, label


# ---------------------------------------------------------------------------
# The instrument that found the bug, kept as a test: python-chess as an external
# oracle over a corpus built to contain the event.
#
# A uniform-random corpus is NOT usable here — from the start position a double
# push almost never creates a LEGAL ep (it needs an enemy pawn already adjacent
# on the fifth rank), so ~200k random plies yield well under 100 legal-ep
# positions and roughly zero ep-then-repeat events. The generator below builds
# the pawn geometry directly and seeds sliders so the pinned case occurs too.
# ---------------------------------------------------------------------------


def _synthetic_ep_position(rng: random.Random) -> tuple[chess.Board, chess.Move] | None:
    for _ in range(200):
        board = chess.Board.empty()
        file_ = rng.randrange(8)
        board.set_piece_at(chess.square(file_, 1), chess.Piece(chess.PAWN, chess.WHITE))
        adj = file_ + rng.choice([-1, 1])
        if not 0 <= adj <= 7:
            continue
        board.set_piece_at(chess.square(adj, 3), chess.Piece(chess.PAWN, chess.BLACK))
        free = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
        rng.shuffle(free)
        wk, bk = free.pop(), free.pop()
        if chess.square_distance(wk, bk) <= 1:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        for _ in range(rng.randrange(2, 6)):
            if not free:
                break
            board.set_piece_at(free.pop(), chess.Piece(
                rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]),
                rng.choice([chess.WHITE, chess.BLACK]),
            ))
        board.turn = chess.WHITE
        board.castling_rights = 0
        if not board.is_valid():
            continue
        push = chess.Move(chess.square(file_, 1), chess.square(file_, 3))
        if push in board.legal_moves:
            return board, push
    return None


def test_external_oracle_sweep_over_a_corpus_that_contains_the_event() -> None:
    rng = random.Random(4242)
    mismatches: list[str] = []
    legal_ep = pseudo_only_ep = true_reps = plies = 0

    built = 0
    while built < 400:
        made = _synthetic_ep_position(rng)
        if made is None:
            continue
        board, push = made
        built += 1
        cb = CBoard.from_board(board)
        cb.push_index(move_to_index(push, board))
        board.push(push)

        for step in range(11):
            if step:
                if board.is_game_over(claim_draw=False):
                    break
                moves = list(board.legal_moves)
                quiet = [
                    m for m in moves
                    if not board.is_capture(m)
                    and board.piece_type_at(m.from_square) != chess.PAWN
                ]
                move = rng.choice(quiet or moves)
                cb.push_index(move_to_index(move, board))
                board.push(move)

            plies += 1
            if board.ep_square is not None:
                if board.has_legal_en_passant():
                    legal_ep += 1
                elif any(True for _ in board.generate_pseudo_legal_ep()):
                    pseudo_only_ep += 1
            truth2 = board.is_repetition(2)
            true_reps += truth2
            if cb.has_legal_en_passant() is not board.has_legal_en_passant():
                mismatches.append(f"ep predicate: {board.fen()}")
            if cb.is_repetition() is not truth2:
                mismatches.append(f"2-fold: {board.fen()}")
            if cb.is_threefold_repetition() is not board.is_repetition(3):
                mismatches.append(f"3-fold: {board.fen()}")
            if _plane103(cb) is not truth2:
                mismatches.append(f"plane {REP_PLANE}: {board.fen()}")

    # The corpus must actually contain the discriminating populations, or the
    # zero-mismatch assertion below is vacuous.
    assert plies > 3000, plies
    assert legal_ep > 300, f"corpus has too few legal-ep positions: {legal_ep}"
    assert pseudo_only_ep > 0, "corpus contains no pinned/in-check ep positions"
    assert true_reps > 0, "corpus contains no genuine repetitions"
    assert not mismatches, mismatches[:10]
