"""Ceres TPG square-record encoder — the ``squares`` [B, 64, 137] input.

Ceres C1/C2/C3 nets do NOT take LC0's 112 planes. They take one 137-byte
record per square (``TPGSquareRecord``), 64 of them per position. This is a
different FEATURE SET, not a reshape of the LC0 planes: 64x137 = 8768 values
against LC0's 112*8*8 = 7168, and the extra capacity is spent on per-square
rank/file one-hots, a per-square en-passant flag and two scalar
"assumed future blunder" knobs that have no LC0 counterpart.

Everything below is transcribed from the Ceres sources (github.com/dje-dev/
Ceres @ main, read 2026-08-13), not inferred from the number 137:

``src/Ceres.Chess/NNEvaluators/Ceres/TPG/TPGSquareRecord.cs``
    field order and widths (the ``#region Raw fields`` block), the piece
    one-hot convention (``TPGRecordUtils.WritePieceEncoding``), the en-passant
    rule and the rank/file one-hot write.
``src/Ceres.Chess/NNEvaluators/Ceres/TPG/TPGRecordConverter.cs``
    ``ConvertToTPGRecordSquaresFromCompactHistory`` — history ordering, the
    side-to-move canonicalisation, the short-history fill and the
    "drop en passant when there is only one real position" rule.
``src/Ceres.Chess/NNEvaluators/Ceres/TPG/TPGRecordEncoding.cs``
    ``Move50CountEncoded``.
``src/Ceres.Base/DataTypes/ByteScaled.cs``
    ``SCALING_FACTOR = 100``; every field is stored as a byte and the value
    the network sees is ``byte / 100``.
``src/Ceres.Chess/NNEvaluators/Ceres/TPG/TPGConvertersToFlat.cs``
    ``ConvertToFlatTPG`` divides the byte buffer by
    ``TPGSquareRecord.SQUARE_BYTES_DIVISOR`` (= 100) to build the float
    ``squares`` input; the ``-I8`` nets take the SAME bytes undivided through
    a ``squares_byte`` input instead.

Because the byte buffer is the ground truth and the float input is derived
from it, :func:`encode_ceres_tpg_bytes` is the primitive here and
:func:`encode_ceres_tpg` is ``bytes / 100``. Quantisation therefore happens
in exactly the place Ceres puts it, and the ``-I8`` variant is the byte array
fed straight through (that path is NOT wired up here — see the PR).

Field layout of one 137-byte record (offset: width — field):

===========  =====  ==========================================================
offset       width  field
===========  =====  ==========================================================
0            104    ``pieceTypeHistoryAllHistoryPositions`` — 8 history slots
                    x 13 one-hot: 0 empty, 1..6 OUR pawn/knight/bishop/rook/
                    queen/king, 7..12 THEIR same order
104          8      ``historyRepetitionCounts`` — per slot, ``min(1, reps)``
112          1      ``CanOO``           (side to move, kingside)
113          1      ``CanOOO``          (side to move, queenside)
114          1      ``OpponentCanOO``
115          1      ``OpponentCanOOO``
116          1      ``Move50Count``     = ``min(halfmove_clock, 100) / 50``
117          1      ``PlySinceLastMove`` (0 under Ceres's default inference
                    mode ``PlySinceLastMoveModeEnum.Zero``)
118          1      ``IsEnPassant``     — on the capturable enemy PAWN's square
119          1      ``QPositiveBlunders``
120          1      ``QNegativeBlunders``
121          8      ``rankEncoding`` one-hot (back rank = 0)
129          8      ``fileEncoding`` one-hot (A = 0)
===========  =====  ==========================================================

Side-to-move canonicalisation: ``ConvertToTPGRecordSquaresFromCompactHistory``
reverses each history position whose side to move disagrees with its slot
parity, which for a real alternating game history means "mirror every slot
iff the ROOT side to move is black". ``Position.Reversed`` flips ranks, swaps
colours and swaps castling rights — i.e. ``chess.Board.mirror()``. After that
normalisation the current position is always white to move, so
``TPGSquareRecord.WritePosPieces``'s black branch (which would write records
in reverse array order) is dead on the inference path and record ``i`` always
describes canonical square ``i`` (a1 = 0 ... h8 = 63).
"""
from __future__ import annotations

import chess
import numpy as np

CERES_TPG_NUM_SQUARES = 64
"""Records per position (``TPGRecord.NUM_SQUARES``)."""

CERES_TPG_NUM_FEATURES = 137
"""Bytes per record (``TPGRecord.BYTES_PER_SQUARE_RECORD``)."""

CERES_TPG_HISTORY_POS = 8
"""History slots per record (``TPGSquareRecord.NUM_HISTORY_POS``)."""

CERES_TPG_PIECES_PER_SLOT = 13
"""One-hot width per history slot (``NUM_BYTES_PER_HISTORY_PLANE``)."""

CERES_TPG_BYTE_SCALE = 100.0
"""``ByteScaled.SCALING_FACTOR`` — the divisor between bytes and net input."""

CERES_TPG_MAX_BYTE = 255
"""Storage ceiling of a ``ByteScaled``."""

CERES_TPG_DEFAULT_Q_BLUNDER = 0.03
"""``NNEvaluatorOptionsCeres.DEFAULT_Q_BLUNDER`` — the inference-time value for
both blunder fields. Zero is out of distribution ("the training rarely saw
values of exactly zero")."""

# Byte offsets within one record. Names mirror the C# fields.
PIECE_HISTORY_BASE = 0
HISTORY_REPETITION_BASE = 104
CAN_OO = 112
CAN_OOO = 113
OPPONENT_CAN_OO = 114
OPPONENT_CAN_OOO = 115
MOVE50_COUNT = 116
PLY_SINCE_LAST_MOVE = 117
IS_EN_PASSANT = 118
Q_POSITIVE_BLUNDERS = 119
Q_NEGATIVE_BLUNDERS = 120
RANK_ONE_HOT_BASE = 121
FILE_ONE_HOT_BASE = 129

_ONE_BYTE = int(CERES_TPG_BYTE_SCALE)  # ByteScaled.SetOne()
_PIECE_BITBOARD_ATTRS = ("pawns", "knights", "bishops", "rooks", "queens", "kings")
_EN_PASSANT_PAWN_RANK = 4  # canonical (white-to-move) rank index of the capturable pawn


def _scaled_byte(value: float) -> int:
    """``ByteScaled.Value`` setter: round(value * 100), clamped to the byte range."""
    return int(min(float(CERES_TPG_MAX_BYTE), max(0.0, round(value * CERES_TPG_BYTE_SCALE))))


def _piece_codes(board: chess.Board) -> np.ndarray:
    """(64,) uint8 of ``WritePieceEncoding`` codes for a CANONICAL board.

    0 empty, 1..6 white (= side to move at the root) by
    ``chess``/``Ceres.PieceType`` order pawn..king, 7..12 black.
    """
    codes = np.zeros(CERES_TPG_NUM_SQUARES, dtype=np.uint8)
    white = int(board.occupied_co[chess.WHITE])
    black = int(board.occupied_co[chess.BLACK])
    for piece_type_minus_one, attr in enumerate(_PIECE_BITBOARD_ATTRS):
        piece_bb = int(getattr(board, attr))
        for offset, occupancy in ((1, white), (7, black)):
            bb = piece_bb & occupancy
            while bb:
                square = (bb & -bb).bit_length() - 1
                codes[square] = piece_type_minus_one + offset
                bb &= bb - 1
    return codes


def _history_boards_and_repetitions(
    board: chess.Board, history_len: int,
) -> tuple[list[chess.Board], list[bool]]:
    """Real history (newest first) plus each slot's ``RepetitionCount > 0`` flag.

    Repetition is "this exact position already occurred earlier in the game",
    which is what LC0 training data records and what ``WritePieceHistory``
    clamps to 1. Determining it needs the WHOLE game, not just the window, so
    the move stack is walked to its root even though only ``history_len``
    boards are kept.
    """
    work = board.copy()
    boards: list[chess.Board] = []
    keys: list[object] = []
    while True:
        if len(boards) < history_len:
            boards.append(work.copy(stack=False))
        # No public accessor; `lc0.py` reads `board._stack` the same way.
        keys.append(work._transposition_key())
        if not work.move_stack:
            break
        work.pop()
    seen: set[object] = set()
    repeated: list[bool] = []
    # keys[0] is the current position, keys[-1] the game root; a slot repeats
    # when its key also appears strictly EARLIER in the game (later in `keys`).
    for index in range(len(keys) - 1, -1, -1):
        repeated.append(keys[index] in seen)
        seen.add(keys[index])
    repeated.reverse()
    return boards, repeated[: len(boards)]


def _canonical_slots(
    boards: list[chess.Board], *, history_len: int, fill_in_history: bool,
) -> list[chess.Board]:
    """Ceres's normalisation + short-history fill, applied to newest-first boards."""
    slots: list[chess.Board] = []
    for slot_index, real in enumerate(boards):
        # `if (pos.IsWhite != (k % 2 == 0)) pos = pos.Reversed;`
        slots.append(real if real.turn == (slot_index % 2 == 0) else real.mirror())
    oldest = slots[-1]
    for slot_index in range(len(boards), history_len):
        slots.append(oldest if fill_in_history else slots[slot_index - 1].mirror())
    return slots


def encode_ceres_tpg_bytes(
    board: chess.Board,
    *,
    history_len: int = CERES_TPG_HISTORY_POS,
    fill_in_history: bool = True,
    q_negative_blunders: float = CERES_TPG_DEFAULT_Q_BLUNDER,
    q_positive_blunders: float = CERES_TPG_DEFAULT_Q_BLUNDER,
) -> np.ndarray:
    """(64, 137) uint8 — the raw ``TPGSquareRecord`` bytes for ``board``.

    ``fill_in_history=True`` is Ceres's own behaviour for a position given
    without a move history: ``EncodedPositionWithHistory.FromPosition``
    defaults to filling the empty history planes with copies of the current
    board, which ``CompactHistoryDerivation`` then reports as
    ``FillInHistory = true``, and the converter fills the unused slots with
    the OLDEST real position. ``False`` reproduces the no-fill branch
    (alternating mirrored copies), which is what a producer that leaves the
    history planes empty gets.
    """
    if history_len < 1 or history_len > CERES_TPG_HISTORY_POS:
        raise ValueError(f"history_len must be in [1, {CERES_TPG_HISTORY_POS}]; got {history_len}")
    boards, repetitions = _history_boards_and_repetitions(board, history_len)
    slots = _canonical_slots(
        boards, history_len=CERES_TPG_HISTORY_POS, fill_in_history=fill_in_history,
    )
    # Every filled slot copies a real board (directly under fill-in, through the
    # mirrored chain otherwise) and `Position.Reversed` carries MiscInfo across,
    # so a filled slot inherits the oldest real slot's repetition flag either way.
    repetition_flags = list(repetitions) + [repetitions[-1]] * (
        CERES_TPG_HISTORY_POS - len(repetitions)
    )

    out = np.zeros((CERES_TPG_NUM_SQUARES, CERES_TPG_NUM_FEATURES), dtype=np.uint8)
    all_squares = np.arange(CERES_TPG_NUM_SQUARES)
    for slot_index, slot in enumerate(slots):
        base = PIECE_HISTORY_BASE + slot_index * CERES_TPG_PIECES_PER_SLOT
        codes = _piece_codes(slot).astype(np.int64)
        out[all_squares, base + codes] = _ONE_BYTE
        if repetition_flags[slot_index]:
            out[:, HISTORY_REPETITION_BASE + slot_index] = _ONE_BYTE

    current = slots[0]
    for offset, has_right in (
        (CAN_OO, current.has_kingside_castling_rights(chess.WHITE)),
        (CAN_OOO, current.has_queenside_castling_rights(chess.WHITE)),
        (OPPONENT_CAN_OO, current.has_kingside_castling_rights(chess.BLACK)),
        (OPPONENT_CAN_OOO, current.has_queenside_castling_rights(chess.BLACK)),
    ):
        if has_right:
            out[:, offset] = _ONE_BYTE

    out[:, MOVE50_COUNT] = _scaled_byte(min(int(current.halfmove_clock), 100) / 50.0)
    # PLY_SINCE_LAST_MOVE stays 0: NNEvaluatorOptionsCeres.PlySinceLastMoveMode
    # defaults to `Zero`, whose handler explicitly leaves the field at zero.
    out[:, Q_POSITIVE_BLUNDERS] = _scaled_byte(q_positive_blunders)
    out[:, Q_NEGATIVE_BLUNDERS] = _scaled_byte(q_negative_blunders)

    # A single real position has no prior board to diff against, so Ceres
    # deliberately drops its en-passant right; with real history it is kept.
    if current.ep_square is not None and len(boards) > 1:
        pawn_square = chess.square(chess.square_file(current.ep_square), _EN_PASSANT_PAWN_RANK)
        if current.piece_at(pawn_square) != chess.Piece(chess.PAWN, chess.BLACK):
            raise ValueError(
                f"en passant right on {chess.square_name(current.ep_square)} but no "
                f"capturable pawn on {chess.square_name(pawn_square)}: {current.fen()}",
            )
        out[pawn_square, IS_EN_PASSANT] = _ONE_BYTE

    out[all_squares, RANK_ONE_HOT_BASE + all_squares // 8] = _ONE_BYTE
    out[all_squares, FILE_ONE_HOT_BASE + all_squares % 8] = _ONE_BYTE
    return out


def encode_ceres_tpg(
    board: chess.Board,
    *,
    history_len: int = CERES_TPG_HISTORY_POS,
    fill_in_history: bool = True,
    q_negative_blunders: float = CERES_TPG_DEFAULT_Q_BLUNDER,
    q_positive_blunders: float = CERES_TPG_DEFAULT_Q_BLUNDER,
) -> np.ndarray:
    """(64, 137) float32 ``squares`` input — the byte record divided by 100."""
    raw = encode_ceres_tpg_bytes(
        board,
        history_len=history_len,
        fill_in_history=fill_in_history,
        q_negative_blunders=q_negative_blunders,
        q_positive_blunders=q_positive_blunders,
    )
    return raw.astype(np.float32) / np.float32(CERES_TPG_BYTE_SCALE)


def encode_ceres_tpg_batch(
    boards: list[chess.Board],
    *,
    history_len: int = CERES_TPG_HISTORY_POS,
    fill_in_history: bool = True,
    q_negative_blunders: float = CERES_TPG_DEFAULT_Q_BLUNDER,
    q_positive_blunders: float = CERES_TPG_DEFAULT_Q_BLUNDER,
) -> np.ndarray:
    """(B, 64, 137) float32 ``squares`` input for a list of boards."""
    if not boards:
        return np.zeros((0, CERES_TPG_NUM_SQUARES, CERES_TPG_NUM_FEATURES), dtype=np.float32)
    return np.stack([
        encode_ceres_tpg(
            b,
            history_len=history_len,
            fill_in_history=fill_in_history,
            q_negative_blunders=q_negative_blunders,
            q_positive_blunders=q_positive_blunders,
        )
        for b in boards
    ])


def _as_square_batch(squares: np.ndarray) -> np.ndarray:
    x = np.asarray(squares)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[1:] != (CERES_TPG_NUM_SQUARES, CERES_TPG_NUM_FEATURES):
        raise ValueError(
            f"expected (B, {CERES_TPG_NUM_SQUARES}, {CERES_TPG_NUM_FEATURES}) square "
            f"records; got {np.shape(squares)}",
        )
    return x


def ceres_tpg_gather_context(squares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pawn_mask, castling)`` for the Leela policy remap, read off ``squares``.

    Same contract as
    :func:`chess_anti_engine.encoding.lc0.lc0_gather_context_from_planes`:
    ``pawn_mask`` is (B, 64) by ORIENTED square (``rank * 8 + file`` in the
    side-to-move canonical frame, which is exactly the TPG canonical square)
    and ``castling`` is (B, 2) side-to-move (kingside, queenside).

    The square index is decoded from each record's own rank/file one-hots
    rather than assumed to equal the record's position in the array, so this
    reads the board the NET saw and cannot drift from it.
    """
    x = _as_square_batch(squares)
    rank = np.argmax(x[:, :, RANK_ONE_HOT_BASE:RANK_ONE_HOT_BASE + 8], axis=-1)
    file = np.argmax(x[:, :, FILE_ONE_HOT_BASE:FILE_ONE_HOT_BASE + 8], axis=-1)
    oriented = rank * 8 + file  # (B, 64)
    our_pawn = x[:, :, PIECE_HISTORY_BASE + 1] > 0.0  # slot 0, code 1 = our pawn
    pawn_mask = np.zeros_like(our_pawn)
    np.put_along_axis(pawn_mask, oriented, our_pawn, axis=1)
    castling = np.stack([x[:, 0, CAN_OO] > 0.0, x[:, 0, CAN_OOO] > 0.0], axis=1)
    return pawn_mask, castling
