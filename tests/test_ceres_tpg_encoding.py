"""The Ceres ``squares`` [64, 137] encoding, field by field.

A wrong TPG encoder does not crash — it produces a plausible-looking policy and
a net that merely reads weak. Every assertion here therefore pins a value that
was transcribed from a named Ceres source file, not a shape or a "looks sane".
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.ceres_tpg import (
    CAN_OO,
    CAN_OOO,
    CERES_TPG_HISTORY_POS,
    CERES_TPG_NUM_FEATURES,
    CERES_TPG_NUM_SQUARES,
    CERES_TPG_PIECES_PER_SLOT,
    FILE_ONE_HOT_BASE,
    HISTORY_REPETITION_BASE,
    IS_EN_PASSANT,
    MOVE50_COUNT,
    OPPONENT_CAN_OO,
    OPPONENT_CAN_OOO,
    PIECE_HISTORY_BASE,
    PLY_SINCE_LAST_MOVE,
    Q_NEGATIVE_BLUNDERS,
    Q_POSITIVE_BLUNDERS,
    RANK_ONE_HOT_BASE,
    ceres_tpg_gather_context,
    encode_ceres_tpg,
    encode_ceres_tpg_batch,
    encode_ceres_tpg_bytes,
)

ONE = 100  # ByteScaled.SetOne()


def _slot(record: np.ndarray, square: int, slot: int) -> int:
    """The piece one-hot code written for ``square`` in history ``slot``."""
    base = PIECE_HISTORY_BASE + slot * CERES_TPG_PIECES_PER_SLOT
    row = record[square, base:base + CERES_TPG_PIECES_PER_SLOT]
    assert int(row.sum()) == ONE, f"slot {slot} of square {square} is not one-hot: {row}"
    return int(np.argmax(row))


def test_field_offsets_tile_the_record_exactly() -> None:
    """The 137 bytes are a packed C struct: a gap or an overlap is a silent
    off-by-one that shifts every field after it."""
    spans = [
        (PIECE_HISTORY_BASE, CERES_TPG_HISTORY_POS * CERES_TPG_PIECES_PER_SLOT),
        (HISTORY_REPETITION_BASE, CERES_TPG_HISTORY_POS),
        (CAN_OO, 1), (CAN_OOO, 1), (OPPONENT_CAN_OO, 1), (OPPONENT_CAN_OOO, 1),
        (MOVE50_COUNT, 1), (PLY_SINCE_LAST_MOVE, 1), (IS_EN_PASSANT, 1),
        (Q_POSITIVE_BLUNDERS, 1), (Q_NEGATIVE_BLUNDERS, 1),
        (RANK_ONE_HOT_BASE, 8), (FILE_ONE_HOT_BASE, 8),
    ]
    cursor = 0
    for offset, width in spans:
        assert offset == cursor, f"field at {offset} does not follow the previous one ({cursor})"
        cursor += width
    assert cursor == CERES_TPG_NUM_FEATURES


def test_shape_and_one_hot_structure() -> None:
    record = encode_ceres_tpg_bytes(chess.Board())
    assert record.shape == (CERES_TPG_NUM_SQUARES, CERES_TPG_NUM_FEATURES)
    assert record.dtype == np.uint8
    for slot in range(CERES_TPG_HISTORY_POS):
        base = PIECE_HISTORY_BASE + slot * CERES_TPG_PIECES_PER_SLOT
        block = record[:, base:base + CERES_TPG_PIECES_PER_SLOT]
        np.testing.assert_array_equal(block.sum(axis=1), np.full(64, ONE))
    np.testing.assert_array_equal(record[:, RANK_ONE_HOT_BASE:RANK_ONE_HOT_BASE + 8].sum(1),
                                  np.full(64, ONE))
    np.testing.assert_array_equal(record[:, FILE_ONE_HOT_BASE:FILE_ONE_HOT_BASE + 8].sum(1),
                                  np.full(64, ONE))


def test_rank_and_file_one_hots_name_the_record_index() -> None:
    """``WriteSquareEncoding`` writes the square the record describes, and the
    normalised converter never reverses the array order — so record i is a1+i.
    The policy remap decodes the square from these bytes, so if they drifted
    from the array index the remap would read another square's pawn."""
    record = encode_ceres_tpg_bytes(chess.Board())
    rank = np.argmax(record[:, RANK_ONE_HOT_BASE:RANK_ONE_HOT_BASE + 8], axis=1)
    file = np.argmax(record[:, FILE_ONE_HOT_BASE:FILE_ONE_HOT_BASE + 8], axis=1)
    np.testing.assert_array_equal(rank * 8 + file, np.arange(64))


def test_startpos_piece_codes() -> None:
    """0 empty, 1..6 our pawn..king, 7..12 theirs (``WritePieceEncoding`` with
    ``Ceres.PieceType`` Pawn=1..King=6)."""
    record = encode_ceres_tpg_bytes(chess.Board())
    assert _slot(record, chess.E1, 0) == 6      # our king
    assert _slot(record, chess.E8, 0) == 12     # their king
    assert _slot(record, chess.A2, 0) == 1      # our pawn
    assert _slot(record, chess.B1, 0) == 2      # our knight
    assert _slot(record, chess.C1, 0) == 3      # our bishop
    assert _slot(record, chess.A1, 0) == 4      # our rook
    assert _slot(record, chess.D1, 0) == 5      # our queen
    assert _slot(record, chess.A8, 0) == 10     # their rook
    assert _slot(record, chess.E4, 0) == 0      # empty


def test_black_to_move_is_the_mirrored_white_position() -> None:
    """``ConvertToTPGRecordSquaresFromCompactHistory`` reverses every slot when
    the root side to move is black, and ``Position.Reversed`` is exactly
    ``Board.mirror()``. So a black-to-move board and its colour-swapped,
    rank-flipped twin must encode to the SAME bytes — the canonicalisation is
    not a claim about the code, it is a checkable identity."""
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 6 5"
    black = chess.Board(fen)
    white_twin = black.mirror()
    assert white_twin.turn is chess.WHITE
    np.testing.assert_array_equal(
        encode_ceres_tpg_bytes(black), encode_ceres_tpg_bytes(white_twin),
    )


def test_black_to_move_castling_is_side_relative() -> None:
    """Only black has rights here; from black's POV they must land in CanOO /
    CanOOO, not in the opponent fields."""
    record = encode_ceres_tpg_bytes(chess.Board("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"))
    assert int(record[0, CAN_OO]) == ONE
    assert int(record[0, CAN_OOO]) == ONE
    assert int(record[0, OPPONENT_CAN_OO]) == 0
    assert int(record[0, OPPONENT_CAN_OOO]) == 0


def test_white_to_move_castling_is_side_relative() -> None:
    record = encode_ceres_tpg_bytes(chess.Board("r3k2r/8/8/8/8/8/8/R3K3 w Qkq - 0 1"))
    assert int(record[0, CAN_OO]) == 0
    assert int(record[0, CAN_OOO]) == ONE
    assert int(record[0, OPPONENT_CAN_OO]) == ONE
    assert int(record[0, OPPONENT_CAN_OOO]) == ONE


@pytest.mark.parametrize(
    ("halfmove_clock", "expected_byte"),
    # Move50CountEncoded = min(count, 100) / 50, then ByteScaled x100.
    [(0, 0), (1, 2), (37, 74), (50, 100), (100, 200), (150, 200)],
)
def test_move50_encoding(halfmove_clock: int, expected_byte: int) -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    board.halfmove_clock = halfmove_clock
    assert int(encode_ceres_tpg_bytes(board)[0, MOVE50_COUNT]) == expected_byte


def test_ply_since_last_move_is_zero() -> None:
    """``NNEvaluatorOptionsCeres.PlySinceLastMoveMode`` defaults to ``Zero`` and
    its handler returns without touching the field."""
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert not encode_ceres_tpg_bytes(board)[:, PLY_SINCE_LAST_MOVE].any()


def test_q_blunder_fields_default_and_scale() -> None:
    record = encode_ceres_tpg_bytes(chess.Board())
    assert int(record[0, Q_POSITIVE_BLUNDERS]) == 3   # DEFAULT_Q_BLUNDER 0.03 x 100
    assert int(record[0, Q_NEGATIVE_BLUNDERS]) == 3
    custom = encode_ceres_tpg_bytes(
        chess.Board(), q_positive_blunders=0.5, q_negative_blunders=0.17,
    )
    assert int(custom[0, Q_POSITIVE_BLUNDERS]) == 50
    assert int(custom[0, Q_NEGATIVE_BLUNDERS]) == 17
    # ByteScaled.MAX_VALUE = 2.54 clamps rather than wrapping the byte.
    clamped = encode_ceres_tpg_bytes(chess.Board(), q_positive_blunders=9.0)
    assert int(clamped[0, Q_POSITIVE_BLUNDERS]) == 255


def test_en_passant_marks_the_capturable_pawn_only() -> None:
    """``WritePosPieces`` flags the OPPONENT pawn standing on canonical rank 5
    (index 4), not the ep target square and not the capturing pawn."""
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push_uci(uci)
    assert board.ep_square == chess.D6
    record = encode_ceres_tpg_bytes(board)
    flagged = np.flatnonzero(record[:, IS_EN_PASSANT] > 0)
    # White to move, so the canonical frame is the real board: the black pawn
    # that just double-pushed sits on d5.
    np.testing.assert_array_equal(flagged, np.array([chess.D5]))
    assert _slot(record, chess.D5, 0) == 7  # their pawn


def test_en_passant_is_dropped_without_history() -> None:
    """A single real position has no prior board to diff against, so the Ceres
    converter clears the right; keeping it would feed a feature the net never
    sees for a FEN-only input."""
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 4")
    assert board.ep_square == chess.F6
    assert not encode_ceres_tpg_bytes(board)[:, IS_EN_PASSANT].any()


def test_history_slots_hold_the_previous_boards() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    record = encode_ceres_tpg_bytes(board)
    # White to move again, so no mirroring: slot 0 is the live board.
    assert _slot(record, chess.E4, 0) == 1
    assert _slot(record, chess.E5, 0) == 7
    # Slot 1 is the position after 1.e4 (black to move) reversed back into the
    # root frame, so our e4 pawn is there but their e5 pawn is not yet.
    assert _slot(record, chess.E4, 1) == 1
    assert _slot(record, chess.E5, 1) == 0
    # Slot 2 is the start position.
    assert _slot(record, chess.E2, 2) == 1
    assert _slot(record, chess.E4, 2) == 0


def test_short_history_fill_repeats_the_oldest_board() -> None:
    """``FillInHistory`` copies ``historyPositions[numPositions - 1]`` into every
    unused slot; that is what a FEN-only position gets from Ceres's own
    ``EncodedPositionWithHistory.FromPosition`` default."""
    board = chess.Board()
    record = encode_ceres_tpg_bytes(board)
    first = record[:, PIECE_HISTORY_BASE:PIECE_HISTORY_BASE + CERES_TPG_PIECES_PER_SLOT]
    for slot in range(1, CERES_TPG_HISTORY_POS):
        base = PIECE_HISTORY_BASE + slot * CERES_TPG_PIECES_PER_SLOT
        np.testing.assert_array_equal(
            record[:, base:base + CERES_TPG_PIECES_PER_SLOT], first,
        )


def test_short_history_fill_uses_the_oldest_real_board_not_the_newest() -> None:
    """``historyPositions[numPositions - 1]`` is the OLDEST real slot. With a
    single real position the two readings coincide, which is why this needs a
    partial history to be a test at all."""
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    record = encode_ceres_tpg_bytes(board)  # 3 real positions, 5 filled
    oldest = 2
    assert _slot(record, chess.E4, 0) == 1, "live board has our pawn on e4"
    assert _slot(record, chess.E4, oldest) == 0, "oldest board is the start position"
    for slot in range(3, CERES_TPG_HISTORY_POS):
        base = PIECE_HISTORY_BASE + slot * CERES_TPG_PIECES_PER_SLOT
        old_base = PIECE_HISTORY_BASE + oldest * CERES_TPG_PIECES_PER_SLOT
        np.testing.assert_array_equal(
            record[:, base:base + CERES_TPG_PIECES_PER_SLOT],
            record[:, old_base:old_base + CERES_TPG_PIECES_PER_SLOT],
            err_msg=f"filled slot {slot} is not a copy of the oldest real slot",
        )


def test_no_fill_history_alternates_mirrored_copies() -> None:
    """The ``FillInHistory == false`` branch is ``historyPositions[k - 1].Reversed``,
    which is a genuinely different input, not a cosmetic variant."""
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    filled = encode_ceres_tpg_bytes(board, fill_in_history=True)
    alternating = encode_ceres_tpg_bytes(board, fill_in_history=False)
    assert not np.array_equal(filled, alternating)
    # Slot 1 is the mirror: our pawn on e2 becomes their pawn on e7.
    assert _slot(alternating, chess.E2, 1) == 0
    assert _slot(alternating, chess.E7, 1) == 7
    # Slot 2 mirrors back onto slot 0.
    assert _slot(alternating, chess.E2, 2) == 1


def test_repetition_flag_fires_only_on_a_repeated_position() -> None:
    board = chess.Board()
    fresh = encode_ceres_tpg_bytes(board)
    assert not fresh[:, HISTORY_REPETITION_BASE:HISTORY_REPETITION_BASE
                     + CERES_TPG_HISTORY_POS].any()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    record = encode_ceres_tpg_bytes(board)
    reps = record[0, HISTORY_REPETITION_BASE:HISTORY_REPETITION_BASE + CERES_TPG_HISTORY_POS]
    # Slot 0 is the start position seen a second time; slots 1..3 are the
    # single-occurrence knight positions; slot 4 is the original start position.
    assert int(reps[0]) == ONE
    assert [int(v) for v in reps[1:4]] == [0, 0, 0]
    assert int(reps[4]) == 0


def test_float_input_is_the_bytes_over_one_hundred() -> None:
    """``ConvertToFlatTPG`` divides the byte buffer by
    ``TPGSquareRecord.SQUARE_BYTES_DIVISOR``; the ``-I8`` nets take the same
    bytes undivided, so the two paths must differ by exactly this factor."""
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
    raw = encode_ceres_tpg_bytes(board)
    floats = encode_ceres_tpg(board)
    assert floats.dtype == np.float32
    np.testing.assert_allclose(floats, raw.astype(np.float32) / 100.0, rtol=0, atol=0)
    assert float(floats.max()) <= 2.55


def test_batch_encoder_matches_single() -> None:
    boards = [chess.Board(), chess.Board("8/8/8/8/8/5k2/6q1/7K w - - 0 1")]
    batch = encode_ceres_tpg_batch(boards)
    assert batch.shape == (2, CERES_TPG_NUM_SQUARES, CERES_TPG_NUM_FEATURES)
    for i, board in enumerate(boards):
        np.testing.assert_array_equal(batch[i], encode_ceres_tpg(board))


def test_history_len_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="history_len"):
        encode_ceres_tpg_bytes(chess.Board(), history_len=0)
    with pytest.raises(ValueError, match="history_len"):
        encode_ceres_tpg_bytes(chess.Board(), history_len=9)


@pytest.mark.parametrize(
    "fen",
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
        "8/PPPPPPPP/8/8/8/7k/8/K7 w - - 0 1",
        "k7/8/7K/8/8/8/pppppppp/8 b - - 0 1",
    ],
)
def test_gather_context_matches_the_board(fen: str) -> None:
    """The remap's two bits of board context, decoded back out of the very
    tensor the net is fed. ``pawn_mask`` is by ORIENTED square, so for a
    black-to-move board it must be the MIRRORED pawn squares."""
    board = chess.Board(fen)
    pawn_mask, castling = ceres_tpg_gather_context(encode_ceres_tpg(board))
    want = np.zeros(64, dtype=bool)
    for square in board.pieces(chess.PAWN, board.turn):
        want[square if board.turn else chess.square_mirror(square)] = True
    np.testing.assert_array_equal(pawn_mask[0], want)
    np.testing.assert_array_equal(
        castling[0],
        np.array([board.has_kingside_castling_rights(board.turn),
                  board.has_queenside_castling_rights(board.turn)]),
    )


def test_gather_context_rejects_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="square records"):
        ceres_tpg_gather_context(np.zeros((1, 112, 8, 8), dtype=np.float32))
