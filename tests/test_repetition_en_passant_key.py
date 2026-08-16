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


# ---------------------------------------------------------------------------
# The hist_ep plumbing.
#
# Review finding B2: the whole hist_ep path -- the CBoard field, the reset
# helper, py_read_ep_square, cboard_hist_hash's ep parameter and its call sites
# -- could be deleted with every existing test still green, while 21 measured
# false positives came back. Everything above drives the C board with
# push_index() from an EMPTY history, and cboard_push records hist_hash from
# cboard_repetition_key directly; it never reaches cboard_hist_hash, which is
# the only consumer of hist_ep.
#
# cboard_hist_hash is reached by:
#   - CBoard.from_board(), which rebuilds hash_stack, hist_hash and (with the
#     rep fix on) the per-slot hist_was_rep flags from python-chess _BoardState
#     snapshots -- and production builds boards this way;
#   - cboard_fill_lc0_112_root's default branch, which reconstructs each history
#     slot's key from b->hist_ep[idx].
#
# Both are covered below. A CBoard's repetition planes are compared against
# python-chess's own key over the same game, slot by slot, so what is asserted
# is WHICH slot changed -- not merely that a number moved.
# ---------------------------------------------------------------------------

# 12 = the current-position repetition plane of the lc0-root layout;
# (hi+1) * 13 + 12 = history frame hi's, newest first (cboard_set_hist_rep_plane).
_LC0_ROOT_HIST_FRAMES = 7


@pytest.fixture
def restore_rep_fix():
    """history_rep_fix is a process-global in the C extension."""
    from chess_anti_engine.encoding import _lc0_ext

    yield _lc0_ext.set_history_rep_fix
    _lc0_ext.set_history_rep_fix(False)


def _lc0_root_rep_planes(cb: CBoard) -> list[bool]:
    """[current, frame 0, ... frame 6] repetition flags, production encoding."""
    planes = np.asarray(cb.encode_full(2, 63)).reshape(-1, 8, 8)
    return [bool(planes[12].any())] + [
        bool(planes[(hi + 1) * 13 + 12].any()) for hi in range(_LC0_ROOT_HIST_FRAMES)
    ]


def _pychess_rep_flags(fen: str, ucis: Sequence[str]) -> list[bool]:
    """The same vector, from python-chess's OWN Board._transposition_key().

    Newest first: element i is "the position i plies back had already occurred
    earlier in this game".
    """
    board = chess.Board(fen)
    keys = [board._transposition_key()]  # the oracle's OWN key
    for uci in ucis:
        board.push(chess.Move.from_uci(uci))
        keys.append(board._transposition_key())
    flags = [key in keys[:i] for i, key in enumerate(keys)]
    return list(reversed(flags))[: _LC0_ROOT_HIST_FRAMES + 1]


# e2e4 gives black a LEGAL ep (d4xe3), so the position 4 plies later -- same
# pieces, turn and castling, no ep -- is NOT a repetition of it. Each further
# knight cycle repeats what came before, which is what puts a True on either
# side of the discriminating False and stops an always-clear (or always-set)
# plane from passing.
_EP_HISTORY_UCIS = ["e2e4", "g8f6", "g1f3", "f6g8", "f3g1", "g8f6", "g1f3", "f6g8"]


@pytest.mark.parametrize("n_plies", [7, 8])
@pytest.mark.parametrize("rep_fix", [False, True])
@pytest.mark.parametrize("construction", ["push", "from_board"])
def test_history_slot_repetition_planes_match_python_chess(
    restore_rep_fix, construction: str, rep_fix: bool, n_plies: int,
) -> None:
    """Every history slot's repetition flag, against python-chess, slot by slot.

    ``from_board`` is the case that pins hist_ep: it rebuilds each slot's key
    from a _BoardState, and a _BoardState's bitboards do not say whether an ep
    capture was legal -- only its ep_square does. ``push`` + rep-fix-off pins the
    other consumer, cboard_fill_lc0_112_root's reconstruction from
    ``b->hist_ep[idx]``.
    """
    restore_rep_fix(rep_fix)
    ucis = _EP_HISTORY_UCIS[:n_plies]
    expected = _pychess_rep_flags(_REPRO_FEN, ucis)

    # The fixture must actually discriminate: a True adjacent to the False, and
    # the False must be the ep case. Without this the assertion below could be
    # satisfied by a plane that is never set.
    assert expected[0] is True, expected
    assert True in expected, expected
    assert False in expected, expected

    ep_ply = chess.Board(_REPRO_FEN)
    ep_ply.push(chess.Move.from_uci(ucis[0]))
    assert ep_ply.has_legal_en_passant(), "fixture no longer creates a legal ep"

    board = chess.Board(_REPRO_FEN)
    if construction == "push":
        cb = CBoard.from_board(board)
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            cb.push_index(move_to_index(move, board))
            board.push(move)
    else:
        for uci in ucis:
            board.push(chess.Move.from_uci(uci))
        cb = CBoard.from_board(board)

    got = _lc0_root_rep_planes(cb)
    assert got == expected, (
        f"{construction} rep_fix={rep_fix} n_plies={n_plies}: "
        f"got {got}, python-chess says {expected}"
    )


@pytest.mark.parametrize(
    ("label", "ucis", "expected"),
    [
        ("legal ep must not repeat", ["e2e4", "g8f6", "g1f3", "f6g8", "f3g1"], False),
        ("control non-repetition", ["b1c3", "g8f6", "c3b1", "f6g8", "g1f3"], False),
        ("control genuine repetition", ["b1c3", "g8f6", "c3b1", "f6g8"], True),
    ],
)
def test_from_board_rebuilds_the_hash_stack_on_the_ep_aware_key(
    label: str, ucis: Sequence[str], expected: bool,
) -> None:
    """The same reproduction, but constructing the CBoard AFTER the moves.

    ``_play`` builds the CBoard first and pushes into it, so its hash_stack comes
    from cboard_push. Constructing from a board that already has a move stack
    routes every entry through cboard_hist_hash + py_read_ep_square instead --
    the path selfplay and MCTS take whenever a CBoard is made from a live
    python-chess board.
    """
    board = chess.Board(_REPRO_FEN)
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"{uci} illegal in {board.fen()}"
        board.push(move)

    cb = CBoard.from_board(board)
    assert board.is_repetition(2) is expected, f"oracle disagrees on {label!r}"
    assert cb.is_repetition() is expected, label
    assert _plane103(cb) is expected, label


# ---------------------------------------------------------------------------
# Review finding B1: the pseudo-legal half of cboard_has_legal_ep must be a
# LITERAL transcription of python-chess's generate_pseudo_legal_ep, because a
# caller-supplied ep square (from_raw / from_board on a hand-written FEN) need
# not be consistent with the pieces.
#
# Two divergences were found and fixed. Both are named below, because a comment
# claiming "matches python-chess" is worth nothing without a test that fails
# when it stops being true.
# ---------------------------------------------------------------------------

_INCONSISTENT_EP_FENS = [
    # Missing captured pawn. python-chess does NOT require it -- and this one
    # survives a fen() round-trip (fen() prints "d6", because python-chess reads
    # the ep as legal), so a FEN file can carry it. We required it and answered
    # False where the oracle answers True.
    ("no pawn to capture", "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1", chess.D6, True),
    # No capturer-rank mask. python-chess restricts capturers to
    # BB_RANKS[4 if turn else 3]; without it a b2 pawn "captured" on c3.
    ("capturer off the ep rank", "Nr1n4/3N4/6P1/k3P3/8/8/1PpnR1RK/8 w - c3 0 1",
     chess.C3, False),
    # Codex's P2, kept here so all three inconsistent-ep cases read together.
    ("occupied ep target", "7k/8/3n4/3pP3/8/8/8/4K3 w - d6 0 1", chess.D6, False),
]


@pytest.mark.parametrize(
    ("label", "fen", "ep_square", "oracle"),
    [(lab, fen, ep, ora) for lab, fen, ep, ora in _INCONSISTENT_EP_FENS],
)
def test_c_matches_python_chess_on_inconsistent_ep_fens(
    label: str, fen: str, ep_square: int, oracle: bool,
) -> None:
    board = chess.Board(fen)
    # The fixture only bites while python-chess still exposes the ep square.
    assert board.ep_square == ep_square, f"{label}: python-chess dropped the ep"
    assert board.has_legal_en_passant() is oracle, (
        f"{label}: the ORACLE moved, not us -- re-derive the C rule before "
        f"changing this expectation"
    )

    # from_board reads Board.ep_square straight through...
    assert CBoard.from_board(board).has_legal_en_passant() is oracle, label
    # ...and from_raw takes it as an argument.
    raw = CBoard.from_raw(
        board.pawns, board.knights, board.bishops, board.rooks,
        board.queens, board.kings,
        int(board.occupied_co[chess.WHITE]), int(board.occupied_co[chess.BLACK]),
        int(board.turn), 0, ep_square, 0,
    )
    assert raw.has_legal_en_passant() is oracle, f"{label} (from_raw)"


def _random_sparse_board(rng: random.Random) -> chess.Board:
    """A random sparse position, which SOMETIMES OMITS A KING.

    ⚑ The omission is the point. An earlier revision always placed both kings, so
    720,000 samples reported zero key-MERGING divergences while one sat just
    outside the generator's reach — the sweep was a gate that structurally could
    not fail, which is the defect class this whole file exists to close. Review
    finding B5. The kingless arm is counted and asserted non-empty below, so it
    cannot quietly stop being generated either.
    """
    board = chess.Board.empty()
    squares = list(chess.SQUARES)
    rng.shuffle(squares)
    roll = rng.random()
    if roll >= 0.10:   # 10% have no white king
        board.set_piece_at(squares.pop(), chess.Piece(chess.KING, chess.WHITE))
    if roll < 0.85 or roll >= 0.95:   # 10% have no black king
        board.set_piece_at(squares.pop(), chess.Piece(chess.KING, chess.BLACK))
    for _ in range(rng.randint(2, 12)):
        square = squares.pop()
        piece_type = rng.choice([chess.PAWN, chess.PAWN, chess.PAWN, chess.KNIGHT,
                                 chess.BISHOP, chess.ROOK, chess.QUEEN])
        if piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
            piece_type = chess.KNIGHT
        board.set_piece_at(square, chess.Piece(
            piece_type, rng.choice([chess.WHITE, chess.BLACK])))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = 0
    return board


def test_ep_predicate_oracle_sweep_over_inconsistent_ep_fields() -> None:
    """Sweep the class B1 lives in: ep squares that contradict the pieces.

    The claim being pinned is exact agreement with python-chess wherever the ep
    field is PAWN-CONSISTENT -- the captured square empty, or holding an enemy
    pawn. That domain covers every position reachable from a legal double push
    plus every hand-written FEN whose ep square merely lacks its pawn (finding
    B1's own case).

    Outside it -- a non-pawn standing on the captured square, which asserts a
    pawn double-pushed onto a square another piece occupies -- we answer exact
    chess legality of the ep capture while python-chess answers its own
    approximation (pin_mask + _ep_skewered, both valid only when a pawn is
    there). The residual is asserted to stay in that class AND to stay in the
    key-SPLITTING direction: we can miss a repetition there, never invent one,
    which is the direction this whole fix exists to remove.

    ⚑ Second precondition, added after review B5 found the first one false: the
    SIDE TO MOVE must have a king. Boards where it does not are generated on
    purpose (``_random_sparse_board``), counted, and asserted to diverge in
    exactly the documented shape rather than swept into the "consistent" bucket
    -- an earlier revision of this test could not produce one at all, so its zero
    was a gate that could not fire.
    """
    rng = random.Random(20260816)
    n = consistent = pawn_consistent_mismatch = 0
    residual_split = residual_merge = 0
    pop_legal_ep = pop_cap_empty = pop_cap_pawn = pop_cap_nonpawn = 0
    pop_no_mover_king = kingless_merge = kingless_agree = 0
    kingless_unexpected: list[str] = []
    examples: list[str] = []

    while n < 30_000:
        board = _random_sparse_board(rng)
        roll = rng.random()
        if roll < 0.6:
            ep = chess.square(rng.randrange(8), 5 if board.turn == chess.WHITE else 2)
        elif roll < 0.85:
            ep = chess.square(rng.randrange(8), 2 if board.turn == chess.WHITE else 5)
        else:
            ep = rng.randrange(64)
        n += 1

        captured = ep - 8 if board.turn == chess.WHITE else ep + 8
        in_board = 0 <= captured < 64
        cap_pawn = in_board and bool(
            chess.BB_SQUARES[captured] & board.pawns
            & board.occupied_co[not board.turn])
        cap_empty = in_board and not (chess.BB_SQUARES[captured] & board.occupied)
        pop_cap_pawn += cap_pawn
        pop_cap_empty += cap_empty
        pop_cap_nonpawn += in_board and not cap_pawn and not cap_empty

        probe = board.copy(stack=False)
        probe.ep_square = ep
        oracle = probe.has_legal_en_passant()
        pop_legal_ep += oracle

        ours = CBoard.from_raw(
            board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings,
            int(board.occupied_co[chess.WHITE]), int(board.occupied_co[chess.BLACK]),
            int(board.turn), 0, ep, 0,
        ).has_legal_en_passant()

        if board.king(board.turn) is None:
            # Documented, accepted divergence: python-chess's is_into_check
            # answers False with no king, so it calls the ep capture legal; we
            # have no king to test for exposure and answer 0. Assert the SHAPE,
            # not just "something differs" -- the only tolerated disagreement
            # here is ours=False / python-chess=True.
            pop_no_mover_king += 1
            if ours is oracle:
                kingless_agree += 1
            elif ours is False and oracle is True:
                kingless_merge += 1
            else:
                kingless_unexpected.append(
                    f"{probe.board_fen()} {'w' if board.turn else 'b'} "
                    f"ep={chess.SQUARE_NAMES[ep]} ours={ours} py={oracle}")
            continue

        if (not in_board) or cap_pawn or cap_empty:
            consistent += 1
            if ours is not oracle:
                pawn_consistent_mismatch += 1
                if len(examples) < 8:
                    examples.append(
                        f"{probe.board_fen()} "
                        f"{'w' if board.turn else 'b'} ep={chess.SQUARE_NAMES[ep]} "
                        f"ours={ours} python-chess={oracle}")
        elif ours is not oracle:
            if ours:
                residual_split += 1
            else:
                residual_merge += 1
                if len(examples) < 8:
                    examples.append(
                        f"MERGE-DIRECTION {probe.board_fen()} "
                        f"{'w' if board.turn else 'b'} ep={chess.SQUARE_NAMES[ep]}")

    # The corpus must contain each discriminating population, or the zeros below
    # are vacuous.
    assert consistent > 15_000, consistent
    assert pop_legal_ep > 100, f"too few legal-ep positions: {pop_legal_ep}"
    assert pop_cap_empty > 1_000, f"too few missing-pawn ep fields: {pop_cap_empty}"
    assert pop_cap_pawn > 20, f"too few consistent ep fields: {pop_cap_pawn}"
    assert pop_cap_nonpawn > 1_000, f"too few non-pawn captured squares: {pop_cap_nonpawn}"
    # ...including the one the previous revision structurally could not produce.
    assert pop_no_mover_king > 1_000, (
        "the generator stopped emitting boards whose MOVER has no king, so the "
        f"kingless divergence is untested again: {pop_no_mover_king}")
    assert kingless_merge > 0, (
        "no kingless MERGE observed -- either the ep geometry stopped occurring "
        "in that arm, or the C predicate changed and "
        "test_kingless_board_is_a_known_accepted_divergence should have failed too")
    assert not kingless_unexpected, (
        "a kingless divergence in an UNDOCUMENTED shape (expected ours=False, "
        "python-chess=True)", kingless_unexpected[:5])

    assert pawn_consistent_mismatch == 0, examples
    assert residual_merge == 0, (
        "a divergence in the key-MERGING direction -- we would invent a "
        "repetition python-chess does not see", examples)
    # residual_split is allowed and documented in bitboards_have_legal_ep; it is
    # deliberately NOT asserted to be zero, and deliberately NOT left unbounded.
    assert residual_split < 0.01 * pop_cap_nonpawn, (
        f"residual grew: {residual_split} of {pop_cap_nonpawn} non-pawn cases")


# ---------------------------------------------------------------------------
# The kingless boundary (review B5).
#
# python-chess's is_into_check opens
#
#     king = self.king(self.turn)
#     if king is None:
#         return False          # -> the ep capture is LEGAL
#
# so has_legal_en_passant() is True on a board whose mover has no king.
# bitboards_have_legal_ep answers 0: there is no king whose exposure the capture
# could create. That is the key-MERGING direction -- we drop an ep term the oracle
# keeps -- and it is accepted ONLY because a kingless board cannot arise from
# play_batch, selfplay, MCTS or UCI parsing, never because it is rare.
#
# ⚑ This test asserts the divergence EXISTS. That is deliberate: it is the only
# thing that will speak up if someone later "fixes" the predicate to match the
# oracle here, or moves the king check and changes the answer by accident. A
# boundary nobody asserts is a boundary that moves silently -- which is what let
# the original en-passant defect live for months, and what let the 720,000-sample
# sweep report zero merges while this case sat outside its generator.
# ---------------------------------------------------------------------------

# Mover has no king, and a black pawn stands on d5 -- so the ep field is
# PAWN-CONSISTENT, i.e. this sits inside the domain the predicate's comment
# claims exactness over on every axis except the king.
_KINGLESS_FEN = "8/8/8/3pP3/8/8/8/8 w - d6 0 1"
_MOVER_KINGLESS_FEN = "8/8/8/3pP3/8/8/8/4k3 w - d6 0 1"
_OPPONENT_KINGLESS_FEN = "4K3/8/8/3pP3/8/8/8/8 w - d6 0 1"
_BOTH_KINGS_FEN = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"


def _c_has_legal_ep_both_ways(board: chess.Board) -> tuple[bool, bool]:
    from_board = CBoard.from_board(board).has_legal_en_passant()
    from_raw = CBoard.from_raw(
        board.pawns, board.knights, board.bishops, board.rooks,
        board.queens, board.kings,
        int(board.occupied_co[chess.WHITE]), int(board.occupied_co[chess.BLACK]),
        int(board.turn), 0,
        board.ep_square if board.ep_square is not None else -1, 0,
    ).has_legal_en_passant()
    return from_board, from_raw


@pytest.mark.parametrize(
    ("label", "fen", "oracle", "ours"),
    [
        ("no kings at all", _KINGLESS_FEN, True, False),
        ("mover has no king", _MOVER_KINGLESS_FEN, True, False),
        # The precondition is about the MOVER. An absent OPPONENT king is not a
        # divergence, and pinning that is what stops the carve-out being widened
        # to "any missing king" by someone reading the two rows above.
        ("opponent has no king", _OPPONENT_KINGLESS_FEN, True, True),
        ("control: both kings", _BOTH_KINGS_FEN, True, True),
    ],
)
def test_kingless_board_is_a_known_accepted_divergence(
    label: str, fen: str, oracle: bool, ours: bool,
) -> None:
    board = chess.Board(fen)

    # The fixture must still be the thing it claims to be, on every axis but the
    # king -- otherwise it could pass for the wrong reason.
    assert board.ep_square == chess.D6, f"{label}: python-chess dropped the ep"
    assert board.piece_at(chess.D5) == chess.Piece(chess.PAWN, chess.BLACK), (
        f"{label}: the ep field must stay PAWN-CONSISTENT, or this stops being "
        f"a case inside the claimed domain")
    assert board.has_legal_en_passant() is oracle, (
        f"{label}: the ORACLE moved. Re-derive the C rule before editing this.")

    from_board, from_raw = _c_has_legal_ep_both_ways(board)
    assert from_board is ours, (
        f"{label}: from_board answered {from_board}, the recorded boundary is "
        f"{ours}. If this change was deliberate, update bitboards_have_legal_ep's "
        f"header comment -- the claim and the code must move together.")
    assert from_raw is ours, f"{label}: from_raw disagrees with from_board"


def test_the_kingless_divergence_is_in_the_key_merging_direction() -> None:
    """Name the direction, because it is the bad one and must not be softened.

    The two boards below differ ONLY in the ep right. python-chess separates
    them; we do not. Asserting the keys directly rather than the predicate is
    what ties the carve-out to its actual consequence -- a repetition we would
    report that python-chess would not.
    """
    with_ep = chess.Board(_KINGLESS_FEN)
    without_ep = chess.Board(_KINGLESS_FEN.replace(" d6 ", " - "))

    assert with_ep.ep_square == chess.D6
    assert without_ep.ep_square is None
    assert with_ep._transposition_key() != without_ep._transposition_key(), (
        "python-chess separates these two positions")

    cb_with = CBoard.from_board(with_ep)
    cb_without = CBoard.from_board(without_ep)
    # Neither is credited with a legal ep, so neither gets the ep term and the
    # repetition keys coincide. zobrist_hash is the ep-blind base of both.
    assert cb_with.has_legal_en_passant() is False
    assert cb_without.has_legal_en_passant() is False
    assert cb_with.zobrist_hash == cb_without.zobrist_hash, (
        "the two boards must differ only in the ep right")
