from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.moves.encode import mirror_policy_index


PLANE_COUNT = 73
POLICY_SIZE = 64 * PLANE_COUNT

QUEEN_DIRS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
KNIGHT_DELTAS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
UNDERPROMO_TO_PIECE = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
}


def _orient_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    if turn == chess.WHITE:
        return square
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def _spec_plane_for_delta(df: int, dr: int) -> int:
    plane = 0
    for dir_file, dir_rank in QUEEN_DIRS:
        for dist in range(1, 8):
            if df == dir_file * dist and dr == dir_rank * dist:
                return plane
            plane += 1
    for offset, (dir_file, dir_rank) in enumerate(KNIGHT_DELTAS):
        if df == dir_file and dr == dir_rank:
            return 56 + offset
    raise AssertionError(f"unencodable delta df={df} dr={dr}")


def _spec_policy_index(move: chess.Move, turn: chess.Color) -> int:
    from_oriented = _orient_square(move.from_square, turn)
    to_oriented = _orient_square(move.to_square, turn)
    from_file = chess.square_file(from_oriented)
    from_rank = chess.square_rank(from_oriented)
    to_file = chess.square_file(to_oriented)
    to_rank = chess.square_rank(to_oriented)
    df = to_file - from_file
    dr = to_rank - from_rank

    if move.promotion in UNDERPROMO_TO_PIECE:
        direction = {-1: 0, 0: 1, 1: 2}[df]
        plane = 64 + UNDERPROMO_TO_PIECE[move.promotion] * 3 + direction
    else:
        plane = _spec_plane_for_delta(df, dr)

    index = int(from_oriented) * PLANE_COUNT + plane
    assert 0 <= index < POLICY_SIZE
    return index


def _mirror_square_files(square: chess.Square) -> chess.Square:
    return chess.square(7 - chess.square_file(square), chess.square_rank(square))


def _mirror_move_files(move: chess.Move) -> chess.Move:
    return chess.Move(
        _mirror_square_files(move.from_square),
        _mirror_square_files(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def _expected_legal_indices(board: chess.Board) -> set[int]:
    return {_spec_policy_index(move, board.turn) for move in board.legal_moves}


def _cboard_legal_indices(board: chess.Board) -> set[int]:
    cb = CBoard.from_board(board)
    indices = cb.legal_move_indices()
    assert indices.dtype == np.int32
    assert np.array_equal(indices, np.sort(indices))
    return {int(index) for index in indices}


def _assert_cboard_matches_board(cb: CBoard, board: chess.Board) -> None:
    assert cb.fen() == board.fen()
    assert bool(cb.turn) == bool(board.turn)
    assert cb.ep_square == board.ep_square
    assert int(cb.halfmove_clock) == int(board.halfmove_clock)
    assert int(cb.ply) == int(board.ply())


def _assert_cboard_current_fields_match_board(cb: CBoard, board: chess.Board) -> None:
    cboard_fields = cb.fen().split()
    board_fields = board.fen().split()
    assert cboard_fields[:5] == board_fields[:5]
    assert bool(cb.turn) == bool(board.turn)
    assert cb.ep_square == board.ep_square
    assert int(cb.halfmove_clock) == int(board.halfmove_clock)


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
        "4k3/8/8/r1pPK3/8/8/8/8 w - c6 0 1",
        "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
        "4k3/P6p/8/8/8/8/p6P/4K3 b - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
        "k7/8/2K5/8/8/8/8/1Q6 b - - 0 1",
    ],
)
def test_cboard_legal_indices_match_independent_policy_spec(fen: str) -> None:
    board = chess.Board(fen)

    assert _cboard_legal_indices(board) == _expected_legal_indices(board)


@pytest.mark.parametrize(
    ("fen", "uci"),
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1", "e5d6"),
        ("4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1", "a7a8q"),
        ("4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1", "a7a8n"),
        ("4k3/P6p/8/8/8/8/p6P/4K3 b - - 0 1", "a2a1r"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "h1h8"),
    ],
)
def test_cboard_push_index_matches_board_for_special_moves(fen: str, uci: str) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves

    cb = CBoard.from_board(board)
    cb.push_index(_spec_policy_index(move, board.turn))
    board.push(move)

    _assert_cboard_matches_board(cb, board)
    assert set(map(int, cb.legal_move_indices())) == _expected_legal_indices(board)


def test_cboard_push_index_matches_board_through_deterministic_walk() -> None:
    rng = np.random.default_rng(20260506)
    board = chess.Board()
    cb = CBoard.from_board(board)

    for _ply in range(96):
        assert set(map(int, cb.legal_move_indices())) == _expected_legal_indices(board)
        if board.is_game_over(claim_draw=True):
            break

        legal_moves = list(board.legal_moves)
        move = legal_moves[int(rng.integers(0, len(legal_moves)))]
        cb.push_index(_spec_policy_index(move, board.turn))
        board.push(move)
        _assert_cboard_matches_board(cb, board)


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
        "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
        "4k3/P6p/8/8/8/8/p6P/4K3 b - - 0 1",
    ],
)
def test_mirror_policy_index_matches_independent_file_mirror(fen: str) -> None:
    board = chess.Board(fen)
    mirrored_board = board.transform(chess.flip_horizontal)

    for move in board.legal_moves:
        mirrored_move = _mirror_move_files(move)
        if not board.is_castling(move):
            assert mirrored_move in mirrored_board.legal_moves

        index = _spec_policy_index(move, board.turn)
        mirrored_index = _spec_policy_index(mirrored_move, board.turn)
        assert mirror_policy_index(index) == mirrored_index


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
        "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
        "4k3/P6p/8/8/8/8/p6P/4K3 b - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
        "k7/8/2K5/8/8/8/8/1Q6 b - - 0 1",
    ],
)
def test_fast_cboard_constructor_matches_current_position_contract(fen: str) -> None:
    board = chess.Board(fen)
    cb = cboard_from_board_fast(board)

    _assert_cboard_current_fields_match_board(cb, board)
    assert int(cb.hist_len) == 0
    assert int(cb.hash_stack_len) == 0
    assert cb.is_game_over() == board.is_game_over(claim_draw=True)
    assert set(map(int, cb.legal_move_indices())) == _expected_legal_indices(board)


@pytest.mark.parametrize(
    ("fen", "uci"),
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
        ("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1", "e5d6"),
        ("4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1", "a7a8b"),
        ("4k3/P6p/8/8/8/8/p6P/4K3 b - - 0 1", "a2a1q"),
    ],
)
def test_fast_cboard_constructor_can_seed_push_index_contract(fen: str, uci: str) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves

    cb = cboard_from_board_fast(board)
    cb.push_index(_spec_policy_index(move, board.turn))
    board.push(move)

    _assert_cboard_current_fields_match_board(cb, board)
    assert set(map(int, cb.legal_move_indices())) == _expected_legal_indices(board)


def test_cboard_rejects_direct_construction():
    """CBoard(...) must raise, not silently return a zeroed (empty) board.

    ``tp_new`` is ``PyType_GenericNew`` with no ``tp_init``, so before the
    guard ``CBoard("anything")`` accepted and ignored the argument and handed
    back an all-empty board that looked parsed but wasn't (UBSAN-found via the
    differential fuzzer). Only ``from_board`` / ``from_raw`` may construct.
    """
    with pytest.raises(TypeError):
        # Deliberate misuse: the stub (correctly) exposes no constructor args.
        CBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        CBoard()
    # The supported path still works and is populated.
    cb = CBoard.from_board(chess.Board())
    assert bin(int(cb.occ_white) | int(cb.occ_black)).count("1") == 32


def test_push_index_rejects_move_from_empty_square():
    """push_index of an index whose source square is empty must raise.

    Previously this indexed the Zobrist table with the ``piece_type_at == -1``
    sentinel (UBSAN: negative index into ``uint64_t[12][64]``). Now it raises
    ValueError at the Python boundary instead of corrupting the board.
    """
    cb = CBoard.from_board(chess.Board())
    legal = {int(i) for i in cb.legal_move_indices()}
    # Find an index that decodes to an empty source square for the start pos.
    # Indices over the back two ranks for white's side-to-move that aren't legal
    # and whose from-square is empty (e.g. a move originating from rank 3-6).
    raised = False
    for idx in range(4672):
        if idx in legal:
            continue
        probe = CBoard.from_board(chess.Board())
        try:
            probe.push_index(idx)
        except ValueError:
            raised = True
            break
    assert raised, "expected at least one index to decode to an empty source square"


def test_from_raw_sanitizes_out_of_range_ep_square():
    """from_raw must normalize ep_square to the documented -1/0..63 invariant.

    Several C paths shift by ``ep_square`` after only an ``>= 0`` check (e.g.
    ``1ULL << ep`` in FEN emission), so storing an out-of-range value like 100
    verbatim — the old behavior — was undefined behaviour downstream. Such
    values now collapse to "no ep square" at the boundary.
    """
    b = chess.Board()
    for bad_ep in (64, 100, 127, -5):
        cb = CBoard.from_raw(
            int(b.pawns), int(b.knights), int(b.bishops),
            int(b.rooks), int(b.queens), int(b.kings),
            int(b.occupied_co[chess.WHITE]), int(b.occupied_co[chess.BLACK]),
            1, 0xF, bad_ep, 0,
        )
        assert cb.ep_square is None
        assert cb.fen().split()[3] == "-"


def test_push_index_rejects_unused_lut_slot():
    """push_index of an unused POLICY_LUT slot (off-board move) must raise.

    Unused slots keep the ``memset(-1)`` sentinel, so ``from_sq == -1``.
    Previously the empty-square guard probed occupancy with that square first
    (UBSAN: shift by -1 in ``sq_bit``). Now the decoded squares are
    bounds-checked before any occupancy probe.
    """
    # Underpromotion planes (64-72) always step toward rank +1 in oriented
    # coordinates, so from the oriented 8th rank (square 56 = a8) the target
    # rank is off-board and the LUT slot is never filled, for either side.
    idx = 56 * 73 + 65  # straight-push knight underpromotion from a8
    for board in (chess.Board(), chess.Board("k7/8/8/8/8/8/8/K7 b - - 0 1")):
        cb = CBoard.from_board(board)
        with pytest.raises(ValueError, match="on-board"):
            cb.push_index(idx)
