"""A pure-numpy reference implementation of the Stockfish big-net NNUE forward pass.

⚑ THIS IS A DEBUGGING ORACLE, NOT THE GATE. Internal Python-vs-C parity cannot
find a rule that is wrong in BOTH implementations; the only gate that can is
Stockfish itself (``scripts/nnue_parity.py``). What this module buys is
localisation: when the C evaluator disagrees with Stockfish, this lets a test
compare the two implementations feature-by-feature and layer-by-layer in a
readable language, and it lets unit tests assert hand-checkable feature indices
on fixed positions without booting an engine.

Everything here mirrors the SCALAR reference paths in the Stockfish sources
(``nnue_feature_transformer.h``'s non-VECTOR branch, ``layers/clipped_relu.h``
and ``layers/sqr_clipped_relu.h``'s tail loops, ``layers/affine_transform.h``).
Stockfish's SIMD kernels are bit-equivalent to those tails, which is why the
scalar form is a legitimate specification of the integer semantics.

Output units: ``psqt // 16 + positional // 16``, side-to-move POV — exactly the
``(Big net) NNUE evaluation <n> (side to move, internal units)`` line printed by
Stockfish's ``eval`` command. Post-processing (small-net selection, optimism
blending, the material and rule50 scaling in ``Eval::evaluate``) is deliberately
NOT reproduced.

⚑ NNUE is UNDEFINED in check: Stockfish asserts ``!pos.checkers()`` before
evaluating and its ``eval`` command refuses outright. ``evaluate()`` raises
``InCheckError`` rather than returning a number a caller could mistake for one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import chess
import numpy as np
import numpy.typing as npt

from scripts.nnue_parse import HALFKA_DIMS, PSQT_BUCKETS, THREAT_DIMS, NnueNet

# --- Stockfish's own colour/piece/square encodings -------------------------
# Colour: WHITE = 0, BLACK = 1  (python-chess uses True/False, so never mix).
# Piece:  W_PAWN = 1 .. W_KING = 6, B_PAWN = 9 .. B_KING = 14.
# Square: A1 = 0 .. H8 = 63, i.e. rank * 8 + file — same as python-chess.
WHITE: Final = 0
BLACK: Final = 1

PAWN: Final = 1
KNIGHT: Final = 2
BISHOP: Final = 3
ROOK: Final = 4
QUEEN: Final = 5
KING: Final = 6

OUTPUT_SCALE: Final = 16
WEIGHT_SCALE_BITS: Final = 6

#: ``FullThreats::numValidTargets``, indexed by Piece.
NUM_VALID_TARGETS: Final = (0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0)

#: ``FullThreats::map``, indexed ``[attacker_type - 1][attacked_type - 1]``.
THREAT_MAP: Final = (
    (0, 1, -1, 2, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (-1, -1, -1, -1, -1, -1),
)

#: ``HalfKAv2_hm::PieceSquareIndex``, indexed ``[perspective][piece]``.
PS_NONE, PS_W_PAWN, PS_B_PAWN = 0, 0, 64
PS_W_KNIGHT, PS_B_KNIGHT = 2 * 64, 3 * 64
PS_W_BISHOP, PS_B_BISHOP = 4 * 64, 5 * 64
PS_W_ROOK, PS_B_ROOK = 6 * 64, 7 * 64
PS_W_QUEEN, PS_B_QUEEN = 8 * 64, 9 * 64
PS_KING: Final = 10 * 64
PS_NB: Final = 11 * 64

PIECE_SQUARE_INDEX: Final = (
    (PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE),
    (PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE),
)

#: ``HalfKAv2_hm::KingBuckets`` (already multiplied by PS_NB), a1..h8.
_KING_BUCKET_IDS: Final = (
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
    8, 9, 10, 11, 11, 10, 9, 8,
    4, 5, 6, 7, 7, 6, 5, 4,
    0, 1, 2, 3, 3, 2, 1, 0,
)
KING_BUCKETS: Final = tuple(v * PS_NB for v in _KING_BUCKET_IDS)

#: ``HalfKAv2_hm::OrientTBL`` — mirror so the king sits on files e..h.
HALFKA_ORIENT: Final = tuple(7 if (sq & 7) < 4 else 0 for sq in range(64))
#: ``FullThreats::OrientTBL`` — the OPPOSITE mirror, king on files a..d.
THREATS_ORIENT: Final = tuple(0 if (sq & 7) < 4 else 7 for sq in range(64))

FILE_A: Final = 0x0101010101010101
FILE_H: Final = 0x8080808080808080


class InCheckError(ValueError):
    """Raised when an evaluation is requested for a position that is in check.

    The NNUE evaluation is undefined there: Stockfish never evaluates a position
    with checkers and its ``eval`` command refuses. Callers must resolve check
    nodes RECURSIVELY before calling eval — an evasion can itself give check, so
    the resolution is a minimax backup over evasions (with repetition and
    50-move terminals handled inside the resolver, and mate when there are no
    evasions) continuing until a non-check position or a terminal is reached.
    This refusal is the enforcement backstop for that invariant, not a
    substitute for it.
    """


# --- bitboard helpers -------------------------------------------------------


def _popcount(bb: int) -> int:
    return bin(bb).count("1")


def _bits(bb: int) -> list[int]:
    out: list[int] = []
    while bb:
        lsb = bb & -bb
        out.append(lsb.bit_length() - 1)
        bb ^= lsb
    return out


def _knight_attacks(sq: int) -> int:
    f, r = sq & 7, sq >> 3
    bb = 0
    for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            bb |= 1 << (nr * 8 + nf)
    return bb


def _king_attacks(sq: int) -> int:
    f, r = sq & 7, sq >> 3
    bb = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= 1 << (nr * 8 + nf)
    return bb


def _ray_attacks(sq: int, occ: int, deltas: tuple[tuple[int, int], ...]) -> int:
    f, r = sq & 7, sq >> 3
    bb = 0
    for df, dr in deltas:
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            target = nr * 8 + nf
            bb |= 1 << target
            if occ >> target & 1:
                break
            nf += df
            nr += dr
    return bb


_BISHOP_DELTAS: Final = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ROOK_DELTAS: Final = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _pawn_attacks(colour: int, sq: int) -> int:
    f, r = sq & 7, sq >> 3
    dr = 1 if colour == WHITE else -1
    bb = 0
    for df in (-1, 1):
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            bb |= 1 << (nr * 8 + nf)
    return bb


def _pawn_push(colour: int, sq: int) -> int:
    r = sq >> 3
    nr = r + (1 if colour == WHITE else -1)
    return 1 << (nr * 8 + (sq & 7)) if 0 <= nr < 8 else 0


def attacks_from(piece_type: int, sq: int, occ: int) -> int:
    """``attacks_bb(pt, s, occupied)`` for the non-pawn piece types."""
    if piece_type == KNIGHT:
        return PSEUDO_ATTACKS[KNIGHT][sq]
    if piece_type == KING:
        return PSEUDO_ATTACKS[KING][sq]
    if piece_type == BISHOP:
        return _ray_attacks(sq, occ, _BISHOP_DELTAS)
    if piece_type == ROOK:
        return _ray_attacks(sq, occ, _ROOK_DELTAS)
    if piece_type == QUEEN:
        return _ray_attacks(sq, occ, _BISHOP_DELTAS) | _ray_attacks(sq, occ, _ROOK_DELTAS)
    raise ValueError(f"attacks_from does not handle piece type {piece_type}")


def _build_pseudo_attacks() -> tuple[tuple[int, ...], ...]:
    """``PseudoAttacks`` — note index 0/1 hold WHITE/BLACK pawn attacks, as in Stockfish."""
    table: list[list[int]] = [[0] * 64 for _ in range(7)]
    for sq in range(64):
        table[WHITE][sq] = _pawn_attacks(WHITE, sq)
        table[BLACK][sq] = _pawn_attacks(BLACK, sq)
        table[KNIGHT][sq] = _knight_attacks(sq)
        table[KING][sq] = _king_attacks(sq)
        table[BISHOP][sq] = _ray_attacks(sq, 0, _BISHOP_DELTAS)
        table[ROOK][sq] = _ray_attacks(sq, 0, _ROOK_DELTAS)
        table[QUEEN][sq] = table[BISHOP][sq] | table[ROOK][sq]
    return tuple(tuple(row) for row in table)


PSEUDO_ATTACKS: Final = _build_pseudo_attacks()

#: ``PawnPushOrAttacks`` — single push OR diagonal attacks, indexed [colour][sq].
PAWN_PUSH_OR_ATTACKS: Final = tuple(
    tuple(_pawn_push(c, sq) | _pawn_attacks(c, sq) for sq in range(64)) for c in (WHITE, BLACK)
)


def _threat_attack_set(piece: int, sq: int) -> int:
    """The empty-board attack set the FullThreats offsets are built from."""
    piece_type = piece & 7
    if piece_type == PAWN:
        return PAWN_PUSH_OR_ATTACKS[piece >> 3][sq]
    return PSEUDO_ATTACKS[piece_type][sq]


# --- FullThreats index lookup tables ---------------------------------------

_ALL_PIECES: Final = (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14)


@dataclass(frozen=True)
class ThreatTables:
    """The three lookup tables ``FullThreats::make_index`` sums."""

    #: ``offsets[piece][from]``
    offsets: tuple[tuple[int, ...], ...]
    #: ``index_lut1[attacker][attacked][from_oriented < to_oriented]``
    lut1: tuple[tuple[tuple[int, int], ...], ...]
    #: ``index_lut2[piece][from][to]``
    lut2: tuple[tuple[tuple[int, ...], ...], ...]
    #: ``helper_offsets[piece].cumulativePieceOffset``
    piece_span: tuple[int, ...]
    #: ``helper_offsets[piece].cumulativeOffset``
    piece_base: tuple[int, ...]
    #: Total feature count; must equal ``THREAT_DIMS``.
    dimensions: int


def build_threat_tables() -> ThreatTables:
    """Rebuild ``FullThreats``' constexpr tables from the attack geometry."""
    offsets = [[0] * 64 for _ in range(16)]
    piece_span = [0] * 16
    piece_base = [0] * 16

    cumulative_offset = 0
    for piece in _ALL_PIECES:
        cumulative_piece_offset = 0
        piece_type = piece & 7
        for from_sq in range(64):
            offsets[piece][from_sq] = cumulative_piece_offset
            if piece_type != PAWN:
                cumulative_piece_offset += _popcount(PSEUDO_ATTACKS[piece_type][from_sq])
            elif 8 <= from_sq <= 55:
                cumulative_piece_offset += _popcount(PAWN_PUSH_OR_ATTACKS[piece >> 3][from_sq])
        piece_span[piece] = cumulative_piece_offset
        piece_base[piece] = cumulative_offset
        cumulative_offset += NUM_VALID_TARGETS[piece] * cumulative_piece_offset

    dimensions = cumulative_offset

    lut1 = [[(dimensions, dimensions)] * 16 for _ in range(16)]
    for attacker in _ALL_PIECES:
        row = list(lut1[attacker])
        for attacked in _ALL_PIECES:
            enemy = (attacker ^ attacked) == 8
            attacker_type = attacker & 7
            attacked_type = attacked & 7
            mapped = THREAT_MAP[attacker_type - 1][attacked_type - 1]
            semi_excluded = attacker_type == attacked_type and (enemy or attacker_type != PAWN)
            feature = piece_base[attacker] + (
                (attacked >> 3) * (NUM_VALID_TARGETS[attacker] // 2) + mapped
            ) * piece_span[attacker]
            lo = dimensions if mapped < 0 else feature
            hi = dimensions if (mapped < 0 or semi_excluded) else feature
            row[attacked] = (lo, hi)
        lut1[attacker] = row

    lut2: list[tuple[tuple[int, ...], ...]] = []
    for piece in range(16):
        if piece not in _ALL_PIECES:
            lut2.append(tuple(tuple([0] * 64) for _ in range(64)))
            continue
        rows: list[tuple[int, ...]] = []
        for from_sq in range(64):
            attacks = _threat_attack_set(piece, from_sq)
            rows.append(tuple(_popcount(((1 << to) - 1) & attacks) for to in range(64)))
        lut2.append(tuple(rows))

    return ThreatTables(
        offsets=tuple(tuple(row) for row in offsets),
        lut1=tuple(tuple(row) for row in lut1),
        lut2=tuple(lut2),
        piece_span=tuple(piece_span),
        piece_base=tuple(piece_base),
        dimensions=dimensions,
    )


THREAT_TABLES: Final = build_threat_tables()


def threat_make_index(
    perspective: int, attacker: int, from_sq: int, to_sq: int, attacked: int, ksq: int
) -> int:
    """``FullThreats::make_index``. Returns ``THREAT_DIMS`` for excluded relations."""
    orientation = THREATS_ORIENT[ksq] ^ (56 * perspective)
    from_o = from_sq ^ orientation
    to_o = to_sq ^ orientation
    swap = 8 * perspective
    attacker_o = attacker ^ swap
    attacked_o = attacked ^ swap
    return (
        THREAT_TABLES.lut1[attacker_o][attacked_o][1 if from_o < to_o else 0]
        + THREAT_TABLES.offsets[attacker_o][from_o]
        + THREAT_TABLES.lut2[attacker_o][from_o][to_o]
    )


def halfka_make_index(perspective: int, sq: int, piece: int, ksq: int) -> int:
    """``HalfKAv2_hm::make_index``."""
    flip = 56 * perspective
    return (
        (sq ^ HALFKA_ORIENT[ksq] ^ flip)
        + PIECE_SQUARE_INDEX[perspective][piece]
        + KING_BUCKETS[ksq ^ flip]
    )


# --- position adapter -------------------------------------------------------


@dataclass(frozen=True)
class PositionView:
    """The Stockfish-flavoured view of a position the feature sets need."""

    #: ``pieces[colour][piece_type]`` bitboards, piece_type 1..6.
    pieces: tuple[tuple[int, ...], tuple[int, ...]]
    occupied: int
    piece_on: tuple[int, ...]
    king_sq: tuple[int, int]
    side_to_move: int
    piece_count: int
    in_check: bool

    @property
    def bucket(self) -> int:
        return (self.piece_count - 1) // 4


def position_view(board: chess.Board) -> PositionView:
    """Convert a ``python-chess`` board into the Stockfish-flavoured view."""
    per_colour: list[tuple[int, ...]] = []
    for colour in (WHITE, BLACK):
        cc = chess.WHITE if colour == WHITE else chess.BLACK
        per_colour.append(
            (
                0,
                board.pawns & board.occupied_co[cc],
                board.knights & board.occupied_co[cc],
                board.bishops & board.occupied_co[cc],
                board.rooks & board.occupied_co[cc],
                board.queens & board.occupied_co[cc],
                board.kings & board.occupied_co[cc],
            )
        )
    piece_on = [0] * 64
    for sq in _bits(board.occupied):
        piece = board.piece_at(sq)
        assert piece is not None
        piece_on[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 8)

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("NNUE evaluation requires both kings on the board")

    return PositionView(
        pieces=(per_colour[0], per_colour[1]),
        occupied=board.occupied,
        piece_on=tuple(piece_on),
        king_sq=(white_king, black_king),
        side_to_move=WHITE if board.turn == chess.WHITE else BLACK,
        piece_count=_popcount(board.occupied),
        in_check=board.is_check(),
    )


def halfka_active_indices(perspective: int, pos: PositionView) -> list[int]:
    """``HalfKAv2_hm::append_active_indices``."""
    ksq = pos.king_sq[perspective]
    return [
        halfka_make_index(perspective, sq, pos.piece_on[sq], ksq) for sq in _bits(pos.occupied)
    ]


def _shift(bb: int, delta: int) -> int:
    """``shift<D>`` — file-masked so wraps are impossible."""
    if delta == 9:
        return (bb & ~FILE_H) << 9 & 0xFFFFFFFFFFFFFFFF
    if delta == 7:
        return (bb & ~FILE_A) << 7 & 0xFFFFFFFFFFFFFFFF
    if delta == -7:
        return (bb & ~FILE_H) >> 7
    if delta == -9:
        return (bb & ~FILE_A) >> 9
    if delta == 8:
        return bb << 8 & 0xFFFFFFFFFFFFFFFF
    if delta == -8:
        return bb >> 8
    raise ValueError(f"unsupported shift {delta}")


def threat_active_indices(perspective: int, pos: PositionView) -> list[int]:
    """``FullThreats::append_active_indices``, excluded relations already dropped."""
    ksq = pos.king_sq[perspective]
    occupied = pos.occupied
    pawns = pos.pieces[WHITE][PAWN] | pos.pieces[BLACK][PAWN]
    active: list[int] = []

    def push(attacker: int, from_sq: int, to_sq: int) -> None:
        index = threat_make_index(
            perspective, attacker, from_sq, to_sq, pos.piece_on[to_sq], ksq
        )
        if index < THREAT_DIMS:
            active.append(index)

    for colour in (WHITE, BLACK):
        for piece_type in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN):
            c = perspective ^ colour
            attacker = piece_type + 8 * c
            bb = pos.pieces[c][piece_type]

            if piece_type == PAWN:
                right = 9 if c == WHITE else -9
                left = 7 if c == WHITE else -7
                attacks_left = _shift(bb, right) & occupied
                attacks_right = _shift(bb, left) & occupied
                for to_sq in _bits(attacks_left):
                    push(attacker, to_sq - right, to_sq)
                for to_sq in _bits(attacks_right):
                    push(attacker, to_sq - left, to_sq)
                # Pawns blocked from moving by a pawn directly in front of them.
                pushers = _shift(pawns, -8 if c == WHITE else 8) & pos.pieces[c][PAWN]
                for from_sq in _bits(pushers):
                    push(attacker, from_sq, from_sq + (8 if c == WHITE else -8))
            else:
                for from_sq in _bits(bb):
                    for to_sq in _bits(attacks_from(piece_type, from_sq, occupied) & occupied):
                        push(attacker, from_sq, to_sq)

    return active


# --- forward pass -----------------------------------------------------------


def _wrap16(values: npt.NDArray[np.int64]) -> npt.NDArray[np.int16]:
    """Wrap to int16 the way Stockfish's int16 accumulators do."""
    return (values & 0xFFFF).astype(np.uint16).view(np.int16)


@dataclass(frozen=True)
class EvalTrace:
    """Per-bucket psqt/positional split, in the same units as ``eval``'s table."""

    bucket: int
    psqt: tuple[int, ...]
    positional: tuple[int, ...]

    @property
    def total(self) -> int:
        return self.psqt[self.bucket] + self.positional[self.bucket]


class ReferenceEvaluator:
    """The big-net forward pass, in numpy."""

    def __init__(self, net: NnueNet) -> None:
        if not net.arch.use_threats:
            raise ValueError(
                f"only the big (threat) architecture is supported; got {net.arch.name}"
            )
        if net.threat_weight is None or net.threat_psqt is None:
            raise ValueError("threat tensors missing from a threat architecture")
        self.net = net
        self.l1 = net.arch.l1
        self.half = self.l1 // 2
        self.fc0_outputs = net.arch.l2 + 1

    # -- accumulators --------------------------------------------------------

    def accumulators(
        self, pos: PositionView
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int32]]:
        """Return ``(accumulation[2][L1], psqt_accumulation[2][8])``."""
        net = self.net
        assert net.threat_weight is not None
        assert net.threat_psqt is not None

        acc = np.zeros((2, self.l1), dtype=np.int16)
        psqt = np.zeros((2, PSQT_BUCKETS), dtype=np.int32)
        for perspective in (WHITE, BLACK):
            halfka = np.asarray(halfka_active_indices(perspective, pos), dtype=np.int64)
            threats = np.asarray(threat_active_indices(perspective, pos), dtype=np.int64)
            if halfka.max(initial=-1) >= HALFKA_DIMS or halfka.min(initial=0) < 0:
                raise AssertionError("HalfKAv2_hm index out of range")

            psq_sum = net.ft_bias.astype(np.int64) + net.ft_weight[halfka].sum(
                axis=0, dtype=np.int64
            )
            threat_sum = (
                net.threat_weight[threats].sum(axis=0, dtype=np.int64)
                if threats.size
                else np.zeros(self.l1, dtype=np.int64)
            )
            # Two independent int16 accumulators, then an int16 add of the pair.
            acc[perspective] = _wrap16(
                _wrap16(psq_sum).astype(np.int64) + _wrap16(threat_sum).astype(np.int64)
            )
            psqt[perspective] = net.ft_psqt[halfka].sum(axis=0, dtype=np.int64).astype(np.int32)
            if threats.size:
                psqt[perspective] += net.threat_psqt[threats].sum(axis=0, dtype=np.int64).astype(
                    np.int32
                )
        return acc, psqt

    # -- feature transformer -------------------------------------------------

    def transform(
        self,
        acc: npt.NDArray[np.int16],
        psqt_acc: npt.NDArray[np.int32],
        stm: int,
        bucket: int,
    ) -> tuple[npt.NDArray[np.uint8], int]:
        perspectives = (stm, stm ^ 1)
        # ⚑ Both the HalfKA and the threat PSQT contributions are already summed
        # into psqt_acc, so this is the single ``(a - b) / 2`` of transform().
        psqt = int(psqt_acc[perspectives[0]][bucket]) - int(psqt_acc[perspectives[1]][bucket])
        psqt = _trunc_div(psqt, 2)  # C integer division truncates toward zero

        out = np.empty(self.l1, dtype=np.uint8)
        for p in (0, 1):
            side = acc[perspectives[p]].astype(np.int64)
            sum0 = np.clip(side[: self.half], 0, 255)
            sum1 = np.clip(side[self.half :], 0, 255)
            out[self.half * p : self.half * (p + 1)] = ((sum0 * sum1) // 512).astype(np.uint8)
        return out, psqt

    # -- layer stack ---------------------------------------------------------

    def propagate(self, features: npt.NDArray[np.uint8], bucket: int) -> int:
        stack = self.net.stacks[bucket]
        x = features.astype(np.int64)

        fc0 = stack.fc0_bias.astype(np.int64) + stack.fc0_weight[:, : self.l1].astype(
            np.int64
        ) @ x

        n = self.net.arch.l2  # FC_0_OUTPUTS
        fc1_in = np.zeros(stack.fc1_weight.shape[1], dtype=np.int64)
        # SqrClippedReLU over the first n outputs, then ClippedReLU appended.
        fc1_in[:n] = np.minimum(127, (fc0[:n] * fc0[:n]) >> (2 * WEIGHT_SCALE_BITS + 7))
        fc1_in[n : 2 * n] = np.clip(fc0[:n] >> WEIGHT_SCALE_BITS, 0, 127)

        fc1 = stack.fc1_bias.astype(np.int64) + stack.fc1_weight.astype(np.int64) @ fc1_in
        ac1 = np.clip(fc1 >> WEIGHT_SCALE_BITS, 0, 127)

        fc2 = int(stack.fc2_bias[0]) + int(stack.fc2_weight[0, : ac1.size].astype(np.int64) @ ac1)

        # fc0[n] carries a forward-skip term on a different scale: 1.0 there is
        # 127 << WeightScaleBits, but the output wants 1.0 == 600 * OutputScale.
        fwd = _trunc_div(int(fc0[n]) * (600 * OUTPUT_SCALE), 127 * (1 << WEIGHT_SCALE_BITS))
        return fc2 + fwd

    # -- public API ----------------------------------------------------------

    def evaluate_view(self, pos: PositionView) -> int:
        """Internal units, side-to-move POV. Raises on an in-check position."""
        if pos.in_check:
            raise InCheckError("NNUE evaluation is undefined for a position in check")
        acc, psqt_acc = self.accumulators(pos)
        bucket = pos.bucket
        features, psqt = self.transform(acc, psqt_acc, pos.side_to_move, bucket)
        positional = self.propagate(features, bucket)
        return _trunc_div(psqt, OUTPUT_SCALE) + _trunc_div(positional, OUTPUT_SCALE)

    def evaluate(self, board: chess.Board) -> int:
        return self.evaluate_view(position_view(board))

    def trace_view(self, pos: PositionView) -> EvalTrace:
        """Every bucket's psqt/positional split — the localisation instrument."""
        if pos.in_check:
            raise InCheckError("NNUE evaluation is undefined for a position in check")
        acc, psqt_acc = self.accumulators(pos)
        psqts: list[int] = []
        positionals: list[int] = []
        for bucket in range(PSQT_BUCKETS):
            features, psqt = self.transform(acc, psqt_acc, pos.side_to_move, bucket)
            psqts.append(_trunc_div(psqt, OUTPUT_SCALE))
            positionals.append(_trunc_div(self.propagate(features, bucket), OUTPUT_SCALE))
        return EvalTrace(bucket=pos.bucket, psqt=tuple(psqts), positional=tuple(positionals))

    def trace(self, board: chess.Board) -> EvalTrace:
        return self.trace_view(position_view(board))


def _trunc_div(value: int, divisor: int) -> int:
    """C integer division: truncate toward zero, not floor."""
    q = abs(value) // divisor
    return q if value >= 0 else -q
