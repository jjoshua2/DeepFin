#!/usr/bin/env python3
"""Convert lc0 v6 training records into OUR production training rows.

Offline tooling only — nothing here runs on the training path. It exists to
build a **positive control for the whole training stack**: feed our exact net
and trainer known-good supervised data and see whether it learns. That control
is worthless if the converter silently produces plausible-but-wrong planes, so
this module is written verification-first — every conversion runs the gates in
:class:`VerifyStats` and refuses to write rows when one fails.

What is produced per lc0 record:

* ``x``            (175, 8, 8) ``v2_threats`` planes in the production
                   ``lc0_root_legacy_meta`` history layout,
* ``policy_target``(1858,) in OUR compact ``lc0_1858`` order,
* ``legal_mask``   (1858,) uint8,
* ``wdl_target``   0/1/2 from lc0's ``result_q``/``result_d`` (side-to-move POV),
* ``search_wdl``   (3,) from lc0's ``best_q``/``best_d`` — the search-improved
                   value estimate, our ``search_wdl`` channel's exact analogue,
* ``moves_left``   ``plies_left`` / ``--moves-left-max-plies``.

``sf_wdl`` is deliberately NOT populated; see "The value target" below.

Why a game chain rather than per-record decoding
------------------------------------------------
A v6 record carries 8 history frames, and lc0's own repetition planes are
computed over the WHOLE game. Decoding one record in isolation can only ever
rebuild an 8-ply synthetic history, so its repetition planes are unverifiable.
Instead each ``.gz`` (one lc0 game) is replayed move by move: the first
position is rebuilt from frame 0, and every later position is reached by
pushing the move named by the previous record's ``played_idx``. That gives a
``chess.Board`` with the real move stack, and it makes the round trip a
CHAINED check — a wrong move decode does not cancel out, it desynchronises the
rest of the game and every subsequent plane comparison fails.

The gates
---------
1. ``planes_exact`` — re-encode the rebuilt board with OUR encoder in lc0's own
   ``lc0_root`` layout and compare all 112 planes BIT-EXACT against the planes
   the record itself carries. This is an external referee: the bytes were
   written by lc0, not by us.
2. ``support_exact`` — the set of Leela-1858 slots our
   ``leela_index_for_move`` assigns to the rebuilt board's legal moves must
   equal, exactly, the set of slots lc0 marked legal (``probability >= 0``).
   python-chess computes legality independently of anything in this repo, so a
   mirrored board, a dropped castling right or a permuted index space all show
   up here.
3. ``gather_agrees`` — the vectorised production-shaped path
   (``leela_gather_indices`` off the encoded planes) must agree move-for-move
   with the scalar reference ``leela_index_for_move``. Catches a permuted map.
4. ``argmax_legal`` — the record policy's argmax, mapped back through the
   index space, must be a legal move.
5. ``best_idx_argmax`` — lc0 stores ``best_idx`` redundantly alongside the
   visit distribution; the agreement rate is reported as a parse sanity read
   (it is a search property, not an invariant, so it is reported, not gated).
6. ``played_promotions`` — the count of PLAYED promotions the chain carried,
   split by promoted piece. See "Promotion spelling" below.

The one known divergence — OURS is the wrong side
-------------------------------------------------
Measured over 345,231 positions: 15 rows (1 in 23,000) fail the bit-exact plane
gate, always on a repetition plane. It is not a converter defect, and it is not
an oversight either — it is DELIBERATE and documented as such. Our production
encoder keys a position WITHOUT ``ep_square``: ``encoding/lc0.py``'s
``_check_repetitions`` omits it in a comment calling the false positives
"extremely rare and harmless", and ``_lc0_ext.c:960`` labels the C key
*"repetition key; EP-blind by design"* — with an EP-AWARE ``transposition_key``
defined three lines below it. So the correct key exists and the repetition path
deliberately does not use it. The consequence is that two positions four plies
apart differing only by a legal en-passant right read as a repetition to us and
as distinct positions to lc0 AND to python-chess. ⚑ A comment asserting "by
design" is not evidence the design is right: this simplification was never
reconciled against either external implementation. Both our paths agreeing is
exactly why an internal equivalence check could never have found it.
Such rows are classified by :func:`known_repetition_ep_alias`, COUNTED as
``rep_ep_alias_rows``, and DROPPED — never emitted — while the rest of the game
continues, because the board itself is correct.

⚑ Both index paths used here go through the board-aware
``chess_anti_engine.moves.leela_index``, which is what ``c49b89937`` (PR #376)
put in place. The static ``build_lc0_policy_remap`` this replaced no longer
exists anywhere in the tree; nothing here touches ``onnx/load.py``.

Promotion spelling — proved, not asserted
-----------------------------------------
Leela's 1858 table spells promotion-to-QUEEN with a ``q`` suffix and
promotion-to-KNIGHT as the BARE from/to slot; ours is the other way round.
⚑ A legal-mask or support check CANNOT tell the two conventions apart: either
way a promoting position lists four slots and the sets match. The
discriminating observation is the move that was actually PLAYED. The game
chain supplies it for free and makes it a hard test: ``played_idx`` is decoded
through ``leela_index_for_move``, the decoded move is pushed, and the NEXT
record's piece planes are then compared bit-exact. Under the inverted
convention a played queen promotion would put a knight on the board (and vice
versa) and the very next comparison would fail. ``played_promotions`` reports
how many promotions of each piece were carried this way, so "the convention is
right" is backed by a count of exercised cases rather than by reading the
table.

⚑ No torch. The policy mapping needed here is move -> compact-1858, which
``moves.leela_index`` provides directly off module-level tables; the
device-cached tensors in ``moves/torch_maps.py`` map compact <-> AZ-4672, a
conversion this path does not perform. No lookup table is redefined here.

The value target
----------------
lc0 and we do not train value on the same label, and the choice is a real one:

* **lc0 gives** a hard game outcome (``result_q``/``result_d``) and the search's
  own value estimate (``best_q``/``best_d``). lc0's own recipe trains value on a
  mix of the two.
* **We train** ``wdl`` on ``game_frac``·outcome + ``sf_wdl_frac``·SF cp-logistic
  + ``search_wdl_frac``·own search WDL (``docs/model_heads.md``). There is no
  Stockfish label anywhere in lc0 data, so the SF component cannot be filled.

Three options, and the one taken:

(a) Emit outcome only. Discards lc0's best label and puts the run in exactly the
    deep-outcome regime production avoids.
(b) Write ``best_q``/``best_d`` into ``sf_wdl``. REJECTED — it would launder a
    Leela search estimate into a field every reader treats as a Stockfish eval,
    which is this codebase's signature defect verbatim.
(c) **TAKEN:** emit the outcome as ``wdl_target`` and lc0's ``best_q``/``best_d``
    as ``search_wdl`` (its true analogue: a search-improved root value), and
    leave ``sf_wdl`` absent so ``has_sf_wdl = 0``.

⚑ (c) has a trap, and it is REAL: ``losses.py`` falls the SF component back to
the raw one-hot outcome when ``has_sf_wdl`` is 0 — so running these rows at the
production ``sf_wdl_frac: 0.50`` would train value on ~50% deep game outcome,
silently, with no error. The control run MUST set ``sf_wdl_frac: 0.0`` (and
``sf_wdl_frac_floor: 0.0``) and put the whole non-outcome share on
``search_wdl_frac``.

⚑ That requirement used to live only as prose in a manifest nothing reads,
which is not a guard. ``check-run-config`` now makes it EXECUTABLE: it loads a
training yaml the way production does, asks the shards whether any Stockfish
label exists, and exits 1 when the two disagree. Pointing the production config
at these shards fails it today. It is still operator-invoked — a refusal inside
the trainer would be the real end state, but that is a training-affecting change
and needs its own ledger entry, so it is named as the follow-up rather than
smuggled into an offline-tooling PR.

``categorical_target`` is not emitted (``has_categorical = 0`` masks that head):
it is an outcome-derived auxiliary and the outcome already reaches the loss via
``wdl_target``.

Usage
-----
  PYTHONPATH=. python3 scripts/lc0_data_to_rows.py verify \\
      --data data/lc0_training --limit-games 200
  PYTHONPATH=. python3 scripts/lc0_data_to_rows.py convert \\
      --data data/lc0_training --out data/lc0_rows --limit-games 2000
  PYTHONPATH=. python3 scripts/lc0_data_to_rows.py corrupt-check \\
      --data data/lc0_training
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
import sys
import tarfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import chess
import numpy as np

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    LC0_HISTORY_ROOT,
    LC0_PLANES_PER_FRAME,
    lc0_gather_context_from_planes,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.lc0_1858_movestrs import (
    LC0_1858_MOVE_STRS,
    LC0_1858_UCI_TO_IDX,
)
from chess_anti_engine.moves.leela_index import (
    compact_index_for_move,
    leela_gather_indices,
    leela_index_for_move,
)
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    local_shard_path,
    samples_to_arrays,
    save_local_shard_arrays,
)

# ── the v6 record layout ───────────────────────────────────────────────────────
# lc0's ``V6TrainingData`` (``src/trainingdata/trainingdata.h``), little-endian,
# packed, ``static_assert(sizeof(V6TrainingData) == 8356)``. Offsets are spelled
# out rather than derived so a struct change upstream shows up as a size
# mismatch on the very first record instead of as shifted floats.
V6_RECORD_BYTES = 8356
V6_VERSION = 6
V6_INPUT_FORMAT_CLASSICAL = 1
LEELA_POLICY_SIZE = 1858
V6_HISTORY_PLANES = 104

_OFF_VERSION = 0
_OFF_INPUT_FORMAT = 4
_OFF_PROBABILITIES = 8
_OFF_PLANES = _OFF_PROBABILITIES + LEELA_POLICY_SIZE * 4  # 7440
_OFF_CASTLING = _OFF_PLANES + V6_HISTORY_PLANES * 8       # 8272
_OFF_SIDE_TO_MOVE = _OFF_CASTLING + 4                     # 8276
_OFF_RULE50 = _OFF_SIDE_TO_MOVE + 1                       # 8277
_OFF_INVARIANCE = _OFF_RULE50 + 1                         # 8278
_OFF_FLOATS = _OFF_INVARIANCE + 2                         # 8280 (skip dep_result)
# root_q best_q root_d best_d root_m best_m plies_left result_q result_d
# played_q played_d played_m orig_q orig_d orig_m
_OFF_BEST_Q = _OFF_FLOATS + 4
_OFF_BEST_D = _OFF_FLOATS + 12
_OFF_PLIES_LEFT = _OFF_FLOATS + 24
_OFF_RESULT_Q = _OFF_FLOATS + 28
_OFF_RESULT_D = _OFF_FLOATS + 32
_OFF_VISITS = _OFF_FLOATS + 60                            # 8340
_OFF_PLAYED_IDX = _OFF_VISITS + 4                         # 8344
_OFF_BEST_IDX = _OFF_PLAYED_IDX + 2                       # 8346
_OFF_TAIL_END = _OFF_BEST_IDX + 2 + 4 + 4                 # policy_kld + reserved

if _OFF_TAIL_END != V6_RECORD_BYTES:  # pragma: no cover - structural constant
    raise AssertionError(
        f"v6 offset table sums to {_OFF_TAIL_END}, not {V6_RECORD_BYTES}",
    )

_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# lc0 serialises a plane as a uint64 through ``ReverseBitsInBytes``, so bit j is
# the square (rank j // 8, file 7 - j % 8): the file axis runs h..a within each
# rank. Verified twice on real data — only this reading puts the queen on d1 and
# the king on e1 in a startpos record, and only this reading makes every
# record's legal-move set agree with lc0's own policy support.
_ORIENTED_SQUARE = np.array(
    [chess.square(7 - (j % 8), j // 8) for j in range(64)], dtype=np.int64,
)


@dataclass(frozen=True)
class V6Record:
    """One parsed lc0 v6 training record."""

    version: int
    input_format: int
    probabilities: np.ndarray  # (1858,) float32; < 0 marks an illegal move
    planes: np.ndarray         # (104,) uint64, 8 frames x 13 planes
    castling_us_ooo: int
    castling_us_oo: int
    castling_them_ooo: int
    castling_them_oo: int
    side_to_move: int          # input_format 1: 1 when black is to move
    rule50: int
    invariance_info: int
    best_q: float
    best_d: float
    plies_left: float
    result_q: float
    result_d: float
    visits: int
    played_idx: int
    best_idx: int


def parse_v6_record(buf: bytes) -> V6Record:
    """Parse one 8356-byte v6 record. Raises on any other length."""
    if len(buf) != V6_RECORD_BYTES:
        raise ValueError(f"v6 record must be {V6_RECORD_BYTES} bytes; got {len(buf)}")
    castling = struct.unpack_from("<4B", buf, _OFF_CASTLING)
    played_idx, best_idx = struct.unpack_from("<HH", buf, _OFF_PLAYED_IDX)

    def u32(offset: int) -> int:
        return int(struct.unpack_from("<I", buf, offset)[0])

    def f32(offset: int) -> float:
        return float(struct.unpack_from("<f", buf, offset)[0])

    return V6Record(
        version=u32(_OFF_VERSION),
        input_format=u32(_OFF_INPUT_FORMAT),
        probabilities=np.frombuffer(
            buf, dtype="<f4", count=LEELA_POLICY_SIZE, offset=_OFF_PROBABILITIES,
        ),
        planes=np.frombuffer(buf, dtype="<u8", count=V6_HISTORY_PLANES, offset=_OFF_PLANES),
        castling_us_ooo=int(castling[0]),
        castling_us_oo=int(castling[1]),
        castling_them_ooo=int(castling[2]),
        castling_them_oo=int(castling[3]),
        side_to_move=int(buf[_OFF_SIDE_TO_MOVE]),
        rule50=int(buf[_OFF_RULE50]),
        invariance_info=int(buf[_OFF_INVARIANCE]),
        best_q=f32(_OFF_BEST_Q),
        best_d=f32(_OFF_BEST_D),
        plies_left=f32(_OFF_PLIES_LEFT),
        result_q=f32(_OFF_RESULT_Q),
        result_d=f32(_OFF_RESULT_D),
        visits=u32(_OFF_VISITS),
        played_idx=int(played_idx),
        best_idx=int(best_idx),
    )


def parse_v6_stream(blob: bytes) -> list[V6Record]:
    """Parse a decompressed lc0 training file (one game) into its records."""
    if not blob or len(blob) % V6_RECORD_BYTES:
        raise ValueError(
            f"lc0 training blob of {len(blob)} bytes is not a whole number of "
            f"{V6_RECORD_BYTES}-byte v6 records",
        )
    return [
        parse_v6_record(blob[i:i + V6_RECORD_BYTES])
        for i in range(0, len(blob), V6_RECORD_BYTES)
    ]


def iter_lc0_games(paths: Sequence[Path]) -> Iterator[tuple[str, list[V6Record]]]:
    """Yield ``(name, records)`` per lc0 game from tars, ``.gz`` files or dirs.

    lc0 writes one ``training.NNNNNN.gz`` per game, so a file is a game. Tars
    are streamed member by member; a truncated tar (a download still in flight)
    stops cleanly at the last whole member rather than raising.
    """
    for path in paths:
        if path.is_dir():
            children = sorted(
                p for p in path.iterdir()
                if p.suffix in {".tar", ".gz"} and not p.name.endswith(".part")
            )
            yield from iter_lc0_games(children)
        elif path.suffix == ".tar":
            yield from _iter_tar_games(path)
        elif path.suffix == ".gz":
            yield str(path.name), parse_v6_stream(gzip.decompress(path.read_bytes()))
        else:
            raise ValueError(f"unrecognised lc0 training path {path}")


def _iter_tar_games(path: Path) -> Iterator[tuple[str, list[V6Record]]]:
    with tarfile.open(path) as tar:
        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                yield member.name, parse_v6_stream(gzip.decompress(handle.read()))
        except (tarfile.ReadError, EOFError, gzip.BadGzipFile):
            return  # truncated archive: stop at the last whole member


# ── record -> board ────────────────────────────────────────────────────────────

def mask_to_plane(mask: int) -> np.ndarray:
    """One lc0 plane mask as an (8, 8) float32 plane in OUR (rank, file) frame."""
    bits = np.unpackbits(
        np.array([mask], dtype="<u8").view(np.uint8), bitorder="little",
    )
    return bits.reshape(8, 8)[:, ::-1].astype(np.float32)


def record_reference_planes(rec: V6Record) -> np.ndarray:
    """The (112, 8, 8) input lc0 itself fed the net, rebuilt from the record.

    Planes 0..103 come straight off the stored masks; 104..111 are the aux
    block lc0 derives from the scalar fields (castling us-Q, us-K, them-Q,
    them-K; side to move; raw rule50; a zero legacy movecount plane; all-ones).
    That is exactly ``encode_lc0_full_root``'s layout, which is why the round
    trip can be bit-exact over all 112 rather than only the piece planes.
    """
    ref = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)
    for plane in range(V6_HISTORY_PLANES):
        ref[plane] = mask_to_plane(int(rec.planes[plane]))
    base = LC0_FULL.root_metadata_base
    ref[base + 0] = 1.0 if rec.castling_us_ooo else 0.0
    ref[base + 1] = 1.0 if rec.castling_us_oo else 0.0
    ref[base + 2] = 1.0 if rec.castling_them_ooo else 0.0
    ref[base + 3] = 1.0 if rec.castling_them_oo else 0.0
    ref[base + 4] = 1.0 if rec.side_to_move else 0.0
    ref[base + 5] = float(rec.rule50)
    ref[LC0_FULL.ones_plane] = 1.0
    return ref


def _set_bits(mask: int) -> Iterator[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def board_from_record(rec: V6Record) -> chess.Board:
    """Rebuild the TRUE position of a record's live frame.

    v6 planes are side-to-move oriented (rank-flipped when black moves), so a
    black-to-move record has to be un-mirrored and recoloured, not merely read
    as a white-to-move board: our plane 108 encodes the real side to move and
    would otherwise disagree with the record.

    En passant is NOT representable in lc0's classical 112-plane format, so it
    is left unset here and repaired, when the legal-move set says one is
    needed, by :func:`repair_en_passant`.

    ⚑ Castling rights are four booleans with no rook file, so the rook has to
    be inferred: lc0 means "the outermost own rook on that side of the king",
    which is a1/h1 in standard chess and an arbitrary file under chess960.
    Inferring it that way rather than hard-coding a/h keeps the reconstruction
    faithful; :func:`castling_reconstruction_problem` is what then REFUSES the
    non-standard cases, so a chess960 game is dropped by name instead of
    quietly losing a castling right.
    """
    us = chess.BLACK if rec.side_to_move else chess.WHITE
    them = not us
    board = chess.Board(None)
    for offset, color in ((0, us), (6, them)):
        for i, piece_type in enumerate(_PIECE_ORDER):
            for bit in _set_bits(int(rec.planes[offset + i])):
                oriented = int(_ORIENTED_SQUARE[bit])
                square = oriented if us == chess.WHITE else chess.square_mirror(oriented)
                board.set_piece_at(square, chess.Piece(piece_type, color))
    board.turn = us
    rights = chess.BB_EMPTY
    for has, kingside, color in (
        (rec.castling_us_ooo, False, us),
        (rec.castling_us_oo, True, us),
        (rec.castling_them_ooo, False, them),
        (rec.castling_them_oo, True, them),
    ):
        if has:
            rook = _castling_rook(board, color, kingside=kingside)
            if rook is not None:
                rights |= chess.BB_SQUARES[rook]
    board.castling_rights = rights
    board.halfmove_clock = int(rec.rule50)
    board.fullmove_number = 1
    return board


def _castling_rook(board: chess.Board, color: chess.Color, *, kingside: bool) -> int | None:
    """The outermost own back-rank rook on one side of the king, or None."""
    king = board.king(color)
    if king is None:
        return None
    back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    rooks = board.rooks & board.occupied_co[color] & back_rank
    if kingside:
        candidates = rooks & ~((1 << (king + 1)) - 1)
        return chess.msb(candidates) if candidates else None
    candidates = rooks & ((1 << king) - 1)
    return chess.lsb(candidates) if candidates else None


def castling_reconstruction_problem(rec: V6Record, board: chess.Board) -> str | None:
    """Why ``board``'s castling rights cannot be trusted, or None.

    Two separate failures, kept separate because they mean different things:
    a right the record asserts but no rook backs (the position is unreadable),
    and a right backed by a rook that is not on the a/h file with the king on
    e — chess960, which our AZ-4672 policy encoding has no slot for, so those
    games are dropped rather than half-converted.
    """
    asserted = (
        rec.castling_us_ooo + rec.castling_us_oo
        + rec.castling_them_ooo + rec.castling_them_oo
    )
    if bin(int(board.castling_rights)).count("1") != asserted:
        return "castling right with no backing rook"
    for color in (chess.WHITE, chess.BLACK):
        back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        rights = int(board.castling_rights) & int(back_rank)
        if not rights:
            continue
        home = chess.square(4, 0 if color == chess.WHITE else 7)
        if board.king(color) != home:
            return "chess960"
        if any(chess.square_file(sq) not in (0, 7) for sq in chess.scan_forward(rights)):
            return "chess960"
    return None


def leela_support(rec: V6Record) -> set[int]:
    """The Leela-1858 slots lc0 marked legal in this record."""
    return set(np.flatnonzero(np.asarray(rec.probabilities) >= 0.0).tolist())


def board_leela_slots(board: chess.Board) -> dict[int, chess.Move]:
    """``{leela slot: move}`` over the board's legal moves."""
    return {leela_index_for_move(board, move): move for move in board.legal_moves}


def repair_en_passant(board: chess.Board, rec: V6Record) -> chess.Board | None:
    """Return a board whose legal-move set matches the record, or None.

    Only the first position of a game needs this: every later position is
    reached by pushing a real move, which sets ``ep_square`` itself. The
    candidate set is the squares an en-passant capture could be made on, and
    the repair is accepted only when exactly one candidate reproduces lc0's
    own legal-move support — a unique-witness test, not a search for agreement.
    """
    want = leela_support(rec)
    if set(board_leela_slots(board)) == want:
        return board
    rank = 5 if board.turn == chess.WHITE else 2
    matches: list[chess.Board] = []
    for file_index in range(8):
        candidate = board.copy(stack=False)
        candidate.ep_square = chess.square(file_index, rank)
        if set(board_leela_slots(candidate)) == want:
            matches.append(candidate)
    if len(matches) > 1:
        # ⚑ PROVABLY UNREACHABLE, and it raises rather than picking one anyway.
        # Reaching this line means the early return above did NOT fire, i.e. the
        # target legal set contains an en-passant capture — and distinct ep
        # squares add distinct capture moves, so at most one candidate can
        # reproduce it. `test_at_most_one_en_passant_witness_is_REACHABLE`
        # brute-forces that claim over every ep file rather than asserting it,
        # so if python-chess's semantics ever change the premise, this stops
        # being an unreachable branch and the test says so first.
        raise AssertionError(
            f"ambiguous en-passant reconstruction ({len(matches)} witnesses) — the "
            "uniqueness premise in repair_en_passant no longer holds",
        )
    return matches[0] if matches else None


def first_record_en_passant_risk(board: chess.Board) -> str | None:
    """Why this game's FIRST position may carry an e.p. square we cannot see.

    ⚑ THE GAP THIS CLOSES, AND WHY NO EXISTING GATE COULD SEE IT.
    lc0's classical 112-plane format cannot express en passant at all — MEASURED
    on this corpus, not assumed: T91 is ``version=6, input_format=1``, and over
    4,000 positions the 206 that carry an ``ep_square`` show NO marker anywhere
    in their own record (no pawn on rank 1/8 in frame 0, plane 110 zero in all
    206). So the first position of a game is reconstructed without it and
    :func:`repair_en_passant` recovers the marker only when an e.p. CAPTURE is
    legal — the support set is the only witness. Measured over 60,000 random
    positions by an independent reviewer: **94.5% of positions carrying an
    ``ep_square`` have no legal e.p. capture**, so the early return fires and
    the marker is silently dropped. Production planes 110 and 141 are then both
    wrong for that row.

    ⚑⚑ AND `_production_aux_consistent` CANNOT CATCH IT, BY CONSTRUCTION: it
    compares plane 110 against *the reconstructed board's own* ``ep_square``, so
    it is self-consistent with the very reconstruction that lost the marker —
    and plane 141 sits in the extra-features block, outside the 112 planes it
    walks. This is the SAME failure class as the ``_check_repetitions`` finding
    this whole PR started from: **a check that compares our output against our
    own reconstruction can only ever find transcription errors, never a wrong
    rule.** Only an external referee, or a guard on the premise, can.

    ⚑ And it cannot be recovered from lc0's stored history either — the obvious
    alternative fix. A game's FIRST record has no history to read: measured over
    300 T91 first records, frames 1-7 (planes 13-103) are ALL ZERO in 300 of
    300. There is nothing to diff a double-push against.

    So this guards the PREMISE instead. An e.p. square can exist only if the
    side that just moved double-pushed a pawn, which puts that pawn on one
    specific rank; with no such pawn present, no marker can have been lost and
    the reconstruction is provably exact. Every T91 game starts from the initial
    position, so the check is free there — but that is a property of THIS corpus
    that nothing asserted, and a resumed/adjudicated/book-started corpus is one
    step away from getting a wrong e.p. plane on row 0 of every game with no
    gate able to object. Now it fails loudly instead.
    """
    if board.ep_square is not None:
        return None  # recovered by the unique-witness repair; nothing was lost
    mover = not board.turn
    pawn_rank = 3 if mover == chess.WHITE else 4
    at_risk = board.pawns & board.occupied_co[mover] & chess.BB_RANKS[pawn_rank]
    if not at_risk:
        return None
    return (
        "first record may carry an unexpressible en-passant square "
        "(see first_record_en_passant_risk)"
    )


def value_field_problem(rec: V6Record) -> str | None:
    """Why this record's value scalars cannot be trusted, or None.

    ⚑ ``_wdl_from_q_d`` used to CLIP whatever it was handed, so a corrupt or
    out-of-range ``q``/``d`` became a plausible distribution instead of an
    error — the "accepted and silently ignored" shape. lc0's own invariants are
    ``q in [-1, 1]``, ``d in [0, 1]`` and ``|q| + d <= 1``; a record violating
    them is unreadable, not roundable.
    """
    for label, q, d in (("result", rec.result_q, rec.result_d), ("best", rec.best_q, rec.best_d)):
        if not (np.isfinite(q) and np.isfinite(d)):
            return f"{label}_q/{label}_d not finite ({q}, {d})"
        if not -1.0001 <= q <= 1.0001 or not -0.0001 <= d <= 1.0001:
            return f"{label}_q/{label}_d out of range ({q}, {d})"
        if abs(q) + d > 1.0001:
            return f"|{label}_q| + {label}_d = {abs(q) + d:.4f} exceeds 1"
    if not np.isfinite(rec.plies_left) or rec.plies_left < 0.0:
        return f"plies_left is not a non-negative finite number ({rec.plies_left})"
    return None


# ── record -> targets ──────────────────────────────────────────────────────────

_LEELA_SUFFIX_TO_PIECE: dict[str, int] = {
    "q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "": chess.KNIGHT,
}


def move_from_leela_slot(board: chess.Board, slot: int) -> chess.Move | None:
    """Decode a Leela 1858 slot into a move on ``board`` WITHOUT enumerating legality.

    ⚑ This is the REVERSE of :func:`leela_index_for_move`, and the direction
    matters. A decoder that picks its answer out of ``board.legal_moves``
    cannot produce an illegal move, so asking "is the result legal" would be a
    tautology. This one reads the slot's own UCI string out of Leela's table,
    un-orients it (rank-flip when Black is to move) and applies Leela's two
    spelling conventions from the board alone:

    * the BARE back-rank slot is promotion-to-KNIGHT for Leela (ours is queen);
    * castling is spelled king-takes-rook (``e1h1``), while our ``chess.Move``
      is the king's two-square hop.

    So a mirrored board, a permuted index space or an inverted promotion
    convention all yield a move that is genuinely not legal, and the check that
    consumes this has content. Returns None only for a slot outside the table.
    """
    if not 0 <= slot < LEELA_POLICY_SIZE:
        return None
    uci = LC0_1858_MOVE_STRS[slot]
    unorient = (lambda sq: sq) if board.turn == chess.WHITE else chess.square_mirror
    from_square = unorient(chess.parse_square(uci[0:2]))
    to_square = unorient(chess.parse_square(uci[2:4]))
    mover = board.piece_at(from_square)
    suffix = uci[4:]
    promotion: int | None = None
    if mover is not None and mover.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
        promotion = _LEELA_SUFFIX_TO_PIECE.get(suffix, chess.KNIGHT)
    elif suffix:
        # ⚑ ALIAS BAND, family (ii): a promotion-suffixed slot whose from-square
        # does not hold a pawn is a slot Leela can never populate. Decoding it as
        # the plain move (``c7b8q`` with a QUEEN on c7 -> ``c7b8``) hands back a
        # legal move for an illegal slot and blunts the gate. Measured at 222 of
        # 234 alias cases over 456,833 illegal slots; refusing them is exact.
        return None
    elif mover is not None and mover.piece_type == chess.KING:
        target = board.piece_at(to_square)
        if target is not None and target.color == mover.color and target.piece_type == chess.ROOK:
            kingside = chess.square_file(to_square) > chess.square_file(from_square)
            to_square = chess.square(6 if kingside else 2, chess.square_rank(from_square))
        elif abs(chess.square_file(to_square) - chess.square_file(from_square)) == 2:
            # ⚑ ALIAS BAND, family (i): Leela spells castling king-takes-rook, so
            # the plain two-square king slot (``e1g1``) is an ordinary SLIDE in
            # its space and is never the castling move. Letting python-chess
            # reinterpret it as castling makes an illegal slot decode legal.
            # Unconditional: a king cannot slide two files, so with a king on the
            # from-square this slot is one Leela can never populate — and
            # ``board.is_castling`` would NOT narrow it, since it answers True for
            # any king move of more than one file regardless of rights.
            return None
    return chess.Move(from_square, to_square, promotion=promotion)


def _wdl_from_q_d(q: float, d: float) -> np.ndarray:
    """lc0 (q, d) as our (W, D, L) triple, side-to-move POV."""
    draw = min(max(float(d), 0.0), 1.0)
    win = (1.0 - draw + float(q)) / 2.0
    loss = (1.0 - draw - float(q)) / 2.0
    triple = np.array([win, draw, loss], dtype=np.float32).clip(0.0, 1.0)
    total = float(triple.sum())
    if total <= 0.0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (triple / total).astype(np.float32)


@dataclass(frozen=True)
class PolicyTargets:
    """The compact-1858 policy target and its legal mask."""

    policy: np.ndarray   # (1858,) float32, sums to 1
    legal_mask: np.ndarray  # (1858,) uint8
    gather_agrees: bool


def build_policy_targets(
    board: chess.Board, rec: V6Record, planes_lc0_root: np.ndarray,
) -> PolicyTargets:
    """Reorder lc0's policy into OUR compact-1858 order, both ways, and compare.

    The scalar path walks the legal moves through ``leela_index_for_move`` /
    ``compact_index_for_move``. The vectorised path is the production shape:
    ``leela_gather_indices`` built from the encoded planes alone. They are
    independent readings of the same correspondence, so their agreement is a
    real cross-check on the index space rather than a restatement of it.
    """
    probabilities = np.asarray(rec.probabilities, dtype=np.float64)
    policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float64)
    legal_mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.uint8)
    gather = leela_gather_indices(
        *lc0_gather_context_from_planes(
            planes_lc0_root, input_history_encoding=LC0_HISTORY_ROOT,
        ),
    )[0]
    gather_agrees = True
    for move in board.legal_moves:
        leela = leela_index_for_move(board, move)
        compact = compact_index_for_move(board, move)
        if int(gather[compact]) != leela:
            gather_agrees = False
        policy[compact] = max(float(probabilities[leela]), 0.0)
        legal_mask[compact] = 1
    total = float(policy.sum())
    if total > 0.0:
        policy /= total
    return PolicyTargets(
        policy=policy.astype(np.float32),
        legal_mask=legal_mask,
        gather_agrees=gather_agrees,
    )


# ── verification ───────────────────────────────────────────────────────────────

EP_ALIAS_DROP_REASON = "repetition ep-alias row (dropped, see known_repetition_ep_alias)"
# Sample size below which the chess960 rate check declines to answer.
CHESS960_MIN_GAMES = 400

@dataclass
class VerifyStats:
    """Counters for every gate. ``ok`` is the conversion's go/no-go."""

    games: int = 0
    games_dropped: int = 0
    games_ep_repaired: int = 0
    # Positions that entered the gates. ⚑ The denominator for every rate below:
    # ``rows`` counts only positions that PASSED, so ``planes_exact / rows`` is
    # 100% by construction and could never have shown a failure.
    attempts: int = 0
    rows: int = 0
    rep_ep_alias_rows: int = 0
    planes_exact: int = 0
    planes_mismatch: int = 0
    rows_with_repetition: int = 0
    support_exact: int = 0
    support_mismatch: int = 0
    gather_agrees: int = 0
    gather_disagrees: int = 0
    argmax_legal: int = 0
    argmax_illegal: int = 0
    best_idx_is_argmax: int = 0
    production_aux_matches: int = 0
    production_aux_mismatch: int = 0
    value_range_mismatch: int = 0
    first_divergence: str | None = None
    drop_reasons: dict[str, int] = field(default_factory=dict)
    # Played promotions whose decode was CONFIRMED by the next record's planes,
    # keyed by python-chess piece symbol ("q", "n", "r", "b").
    played_promotions: dict[str, int] = field(default_factory=dict)

    # ⚑ The ONLY drop classes a PASS may contain. Everything else is an
    # integrity failure — see `integrity_drops`. Membership is by explicit
    # enumeration, so a NEW drop reason is failing-by-default rather than
    # silently benign, which is the direction that cannot rot.
    BENIGN_DROP_REASONS: ClassVar[frozenset[str]] = frozenset({"chess960"})

    def integrity_drops(self) -> dict[str, int]:
        """Dropped games a successful run must NOT contain.

        ⚑ This is the "gate that cannot fail" one level out from the verdict.
        Pre-gate drops — an unsupported record version, an empty game, a
        first-position legal set that does not match, a played_idx that is not
        legal — set no mismatch counter, so `ok` stayed True and a run that had
        silently discarded an INCOMPATIBLE game still printed PASS with exit 0.
        Measured: one crafted `version=7` game alongside 449 real ones produced
        `drop reasons {'version=7/...': 1}` and `VERDICT: PASS`. In a tool whose
        whole purpose is certifying data, a record it could not read must not
        vanish into the ledger.
        """
        return {
            reason: count for reason, count in self.drop_reasons.items()
            if reason not in self.BENIGN_DROP_REASONS and reason != EP_ALIAS_DROP_REASON
        }

    @property
    def ok(self) -> bool:
        return (
            self.rows > 0
            and self.planes_mismatch == 0
            and self.support_mismatch == 0
            and self.gather_disagrees == 0
            and self.argmax_illegal == 0
            and self.production_aux_mismatch == 0
            and self.value_range_mismatch == 0
            and not self.integrity_drops()
        )

    # ⚑ THE TWO COUNTER SCOPES, declared rather than assumed. Everything else
    # here is ATTEMPT-scoped: it records what the gates saw over `attempts` and
    # survives an abandoned game. These are ROW-scoped: they describe rows that
    # were EMITTED, so an abandoned game must give them back. A numerator from
    # one scope over a denominator from the other is exactly how a rate of
    # 110.4651% got reported.
    ROW_SCOPED: ClassVar[tuple[str, ...]] = ("rows", "best_idx_is_argmax", "played_promotions")

    def row_scoped_snapshot(self) -> dict[str, int | dict[str, int]]:
        """⚑ DRIVEN by ``ROW_SCOPED``, not merely documented by it. The tuple
        used to be declared and never read, and it disagreed with the hand-rolled
        snapshot beside it (which also restored ``played_promotions``) — a
        decorative constant next to the comment explaining why the scopes matter
        is precisely the shape this file exists to avoid. Adding a field to the
        tuple now changes the behaviour."""
        snapshot: dict[str, int | dict[str, int]] = {}
        for name in self.ROW_SCOPED:
            value = getattr(self, name)
            snapshot[name] = dict(value) if isinstance(value, dict) else int(value)
        return snapshot

    def restore_row_scoped(self, snapshot: dict[str, int | dict[str, int]]) -> None:
        for name, value in snapshot.items():
            current = getattr(self, name)
            if isinstance(current, dict) and isinstance(value, dict):
                current.clear()
                current.update(value)
            else:
                setattr(self, name, value)

    def consistency_problems(self) -> list[str]:
        """Internal-arithmetic violations. ⚑ Closes the CLASS, not the instance.

        Every rate this reports is a numerator over ``attempts`` or over
        ``rows``; if a numerator ever exceeds its denominator, or the drop
        ledger stops summing to the drop count, the report is describing
        something other than the run. Both failures are invisible on a clean
        run and appear on exactly the runs you would be reading these numbers
        for, so they are checked rather than eyeballed.
        """
        problems: list[str] = []
        for label, num in (
            ("planes_exact", self.planes_exact), ("support_exact", self.support_exact),
            ("gather_agrees", self.gather_agrees), ("argmax_legal", self.argmax_legal),
            ("production_aux_matches", self.production_aux_matches),
            ("rows_with_repetition", self.rows_with_repetition),
            ("rep_ep_alias_rows", self.rep_ep_alias_rows), ("rows", self.rows),
        ):
            if num > self.attempts:
                problems.append(f"{label}={num} exceeds attempts={self.attempts}")
        if self.best_idx_is_argmax > self.rows:
            problems.append(
                f"best_idx_is_argmax={self.best_idx_is_argmax} exceeds rows={self.rows}",
            )
        dropped_ledger = sum(
            count for reason, count in self.drop_reasons.items()
            if reason != EP_ALIAS_DROP_REASON
        )
        if dropped_ledger != self.games_dropped:
            problems.append(
                f"drop_reasons sums to {dropped_ledger} but games_dropped={self.games_dropped}",
            )
        return problems

    def note(self, what: str) -> None:
        self.drop_reasons[what] = self.drop_reasons.get(what, 0) + 1

    def note_promotion(self, symbol: str) -> None:
        self.played_promotions[symbol] = self.played_promotions.get(symbol, 0) + 1

    def chess960_problem(
        self, *, min_games: int = CHESS960_MIN_GAMES, low: float = 0.005, high: float = 0.12,
    ) -> str | None:
        """Why the chess960 detector's own rate is not believable, or None.

        ⚑ A detector that silently stops firing looks exactly like clean data.
        3.73% of lc0 T91 games start from a DFRC/chess960 position (112/3,000
        here; 36/900 in an independent reviewer's sample), so a run over enough
        games reporting far from that describes a detector that broke, not a
        corpus that changed. Below ``min_games`` the rate is too noisy to judge
        and the check ABSTAINS — which :meth:`overall_verdict` treats as
        inconclusive, never as a pass.

        ⚑ The bounds are sized so a CLEAN run does not trip them. At the true
        3.73% rate, ``min_games=150`` has mean 5.6 and P(<=1) ~ 2.4%: with the
        old ``low=0.01`` roughly 1 clean run in 40 would have FAILED. 400 games
        has mean 14.9 and P(<= 2 = 0.005) < 0.1%.
        """
        if self.games < min_games:
            return None  # see chess960_status(): this abstention is reported, not hidden
        rate = self.drop_reasons.get("chess960", 0) / self.games
        if low <= rate <= high:
            return None
        return f"chess960 drop rate {rate:.4f} outside the expected [{low}, {high}]"

    def chess960_status(self, *, min_games: int = CHESS960_MIN_GAMES) -> str:
        """Human-readable form of :meth:`chess960_problem`.

        ⚑ "abstained" and "within band" are DIFFERENT readings and are printed
        as such: a check that abstained has not passed.
        """
        if self.games < min_games:
            return f"ABSTAINED (only {self.games} games; need {min_games})"
        return self.chess960_problem() or (
            f"within expected band ({self.drop_reasons.get('chess960', 0)}/{self.games})"
        )

    def overall_verdict(self) -> tuple[str, int]:
        """``(verdict, exit code)`` — THREE states, because there are three.

        ⚑ An ABSTENTION IS NOT A PASS. The previous go/no-go was
        ``stats.ok and chess960_problem() is None``, and ``chess960_problem``
        returns None while abstaining — so a 40-game run printed ``PASS`` with
        the chess960 check having declined to answer. This module's own
        docstring warned against exactly that. Exit 0 = PASS, 1 = FAIL,
        2 = INCONCLUSIVE (nothing failed, but something declined to answer).
        """
        problems = self.consistency_problems()
        if problems:
            return f"FAIL (counter arithmetic: {problems[0]})", 1
        drops = self.integrity_drops()
        if drops:
            return f"FAIL (integrity drops: {drops})", 1
        if not self.ok:
            return "FAIL", 1
        abstained = [
            label for label, status in (("chess960", self.chess960_status()),)
            if status.startswith("ABSTAINED")
        ]
        if self.chess960_problem() is not None:
            return f"FAIL ({self.chess960_problem()})", 1
        if abstained:
            return f"INCONCLUSIVE (abstained: {', '.join(abstained)})", 2
        return "PASS", 0

    def divergence(self, text: str) -> None:
        if self.first_divergence is None:
            self.first_divergence = text

    def rate_lines(self) -> list[str]:
        def pct(num: int, den: int) -> str:
            return "n/a" if den == 0 else f"{100.0 * num / den:.4f}%"

        n = self.attempts
        return [
            f"games {self.games} (dropped {self.games_dropped}, "
            f"ep-repaired {self.games_ep_repaired}) positions attempted {n}, rows kept {self.rows}",
            f"planes bit-exact  {self.planes_exact}/{n} "
            f"({pct(self.planes_exact, n)}) "
            f"[rows exercising a repetition plane: {self.rows_with_repetition}; "
            f"known ep-alias false positives dropped: {self.rep_ep_alias_rows}]",
            f"legal-set exact   {self.support_exact}/{n} "
            f"({pct(self.support_exact, n)})",
            f"gather agrees     {self.gather_agrees}/{n} "
            f"({pct(self.gather_agrees, n)})",
            f"argmax ILLEGAL    {self.argmax_illegal}/{n} "
            f"({pct(self.argmax_illegal, n)})",
            f"prod aux planes   {self.production_aux_matches}/{n} "
            f"({pct(self.production_aux_matches, n)}) "
            f"[INTERNAL: our encoder vs our encoder, not refereed by lc0]",
            f"best_idx==argmax  {self.best_idx_is_argmax}/{self.rows} "
            f"({pct(self.best_idx_is_argmax, self.rows)}) [reported, not gated]",
            f"played promotions {self.played_promotions or '{}'} "
            "[each confirmed bit-exact by the NEXT record's planes]",
            f"drop reasons      {self.drop_reasons or '{}'}",
            f"integrity drops   {self.integrity_drops() or 'none'}"
            f"  [a PASS may carry only {sorted(self.BENIGN_DROP_REASONS)}"
            " plus the benign e.p.-alias class]",
            f"chess960 rate     {self.chess960_status()}",
            f"first divergence  {self.first_divergence or 'none'}",
            "⚑ The rates above are ONE NESTED measurement, not independent gates: the",
            "  gates run in sequence and a failure abandons the game, so every later",
            "  numerator is conditional on every earlier gate having passed and none of",
            "  them ever sees a row an earlier one rejected. The shortfall from 100% is",
            "  the SAME rows removed from all of them. Games dropped before any position",
            f"  is attempted ({self.games_dropped} here) contribute 0 to the denominator,",
            "  so it reads 'positions of games we could reconstruct', not 'positions of T91'.",
        ]


@dataclass
class ConvertOptions:
    input_history_encoding: str = "lc0_root_legacy_meta"
    input_extra_features: str = "v2_threats"
    moves_left_max_plies: float = 450.0
    history_rep_fix: bool = True


def _repetition_planes_set(rec: V6Record) -> bool:
    return any(
        int(rec.planes[frame * LC0_PLANES_PER_FRAME + 12])
        for frame in range(LC0_FULL.history_len)
    )


_REPETITION_PLANES = frozenset(
    frame * LC0_PLANES_PER_FRAME + 12 for frame in range(LC0_FULL.history_len)
)


def _pychess_repetition_flags(board: chess.Board, frames: int) -> list[bool]:
    """python-chess's verdict on "has this frame's position occurred before".

    The EXTERNAL referee for the repetition planes. ``is_repetition(2)``
    compares full transposition keys — which include the en-passant square when
    an en-passant capture is actually legal — and stops at the last irreversible
    move, which is the FIDE rule and what lc0 implements.
    """
    flags: list[bool] = []
    probe = board.copy()
    for _ in range(frames):
        flags.append(probe.is_repetition(2))
        if not probe.move_stack:
            break
        probe.pop()
    return flags


def known_repetition_ep_alias(
    board: chess.Board, ours: np.ndarray, reference: np.ndarray, bad: Sequence[int],
) -> bool:
    """True iff every differing plane is a repetition FALSE POSITIVE of ours that
    python-chess independently blames on the en-passant-blind repetition key.

    ⚑ This is a NAMED, pre-existing production behaviour, not a converter bug
    and not a blanket waiver. ``encoding/lc0.py::_check_repetitions`` keys a
    position on ``(pieces, occupancy, turn, castling)`` and deliberately OMITS
    ``ep_square`` — its own comment calls the resulting false positives "extremely
    rare and harmless". They are real: two positions four plies apart that differ
    only by a LEGAL en-passant right are treated as a repetition by us and are
    not one for lc0 or for python-chess. Measured here at 15 rows in 345,231
    (1 in 23,000). ⚑ An earlier read of "1 in 230,828" was an UNDERCOUNT by an
    order of magnitude: before this classifier existed the first such row
    abandoned its whole game, so the run could only ever see one per game.

    Three conditions, all required, so this can only ever excuse that one
    mechanism:

    * every differing plane is a repetition plane (any piece/aux plane
      difference still FAILS);
    * on every differing plane OURS says 1 and lc0 says 0 (a false NEGATIVE is
      a different bug and still FAILS);
    * python-chess, an independent implementation, agrees with lc0 on that frame
      — which is what makes this an explanation rather than an exemption.
    """
    if not bad or any(plane not in _REPETITION_PLANES for plane in bad):
        return False
    truth = _pychess_repetition_flags(board, LC0_FULL.history_len)
    for plane in bad:
        frame = plane // LC0_PLANES_PER_FRAME
        if not (float(ours[plane].max()) > 0.0 and float(reference[plane].max()) == 0.0):
            return False
        if frame >= len(truth) or truth[frame]:
            return False
    return True


def convert_game(
    name: str,
    records: Sequence[V6Record],
    stats: VerifyStats,
    options: ConvertOptions,
    *,
    game_id: int,
    collect: bool = True,
) -> list[ReplaySample]:
    """Replay one lc0 game, gate every position, and build our rows.

    A game is abandoned at the first gate failure: after a desynchronisation
    every later position is wrong, so counting those rows as separate failures
    would inflate the denominator rather than describe the defect.
    """
    stats.games += 1
    if not records:
        stats.games_dropped += 1
        stats.note("empty")
        return []
    for rec in records:
        if rec.version != V6_VERSION or rec.input_format != V6_INPUT_FORMAT_CLASSICAL:
            stats.games_dropped += 1
            stats.note(f"version={rec.version}/input_format={rec.input_format}")
            return []
    first = board_from_record(records[0])
    problem = castling_reconstruction_problem(records[0], first)
    if problem is not None:
        stats.games_dropped += 1
        stats.note(problem)
        return []
    repaired = repair_en_passant(first, records[0])
    if repaired is None:
        stats.games_dropped += 1
        stats.note("first-position legal set does not match the record")
        return []
    if repaired.ep_square is not None:
        stats.games_ep_repaired += 1
    ep_risk = first_record_en_passant_risk(repaired)
    if ep_risk is not None:
        stats.games_dropped += 1
        stats.note(ep_risk)
        return []
    board = repaired
    samples: list[ReplaySample] = []
    # An abandoned game emits NOTHING, so every ROW-SCOPED counter has to come
    # back off. ⚑ The two scopes are the whole point (see VerifyStats): the
    # attempt-scoped counters describe what the GATES SAW and must NOT be rolled
    # back, the row-scoped ones describe what was EMITTED and must be. Mixing a
    # numerator from one scope with a denominator from the other is what
    # produced a reported rate of 110.4651%.
    row_scope = stats.row_scoped_snapshot()
    # Set when the previous ply pushed a promotion: the decode of a promotion
    # slot is only CONFIRMED once this record's planes have been compared, so
    # the counter is credited here rather than at the push.
    pending_promotion: str | None = None
    for ply, rec in enumerate(records):
        gated = _gate_position(name, ply, board, rec, stats, options)
        if gated.verdict == "fail":
            stats.games_dropped += 1
            stats.restore_row_scoped(row_scope)
            return []
        # The move is decoded identically whether or not the row is kept, so the
        # chain and the promotion accounting do not fork on the verdict.
        move = board_leela_slots(board).get(rec.played_idx)
        if gated.verdict == "ok":
            if pending_promotion is not None:
                stats.note_promotion(pending_promotion)
            if collect:
                samples.append(_row_from(rec, options, gated, game_id=game_id, ply=ply))
            stats.rows += 1
        pending_promotion = None
        if move is None:
            if ply + 1 < len(records):
                stats.games_dropped += 1
                stats.note("played_idx is not a legal move")
                stats.restore_row_scoped(row_scope)
                return []
            break  # last record of the game: nothing left to chain onto
        if move.promotion is not None:
            pending_promotion = chess.piece_symbol(move.promotion)
        board.push(move)
    return samples


@dataclass(frozen=True)
class GatedPosition:
    """A gate verdict plus the artifacts it already computed.

    ``_row_from`` used to re-encode the board twice and rebuild the policy
    targets from scratch — four ``encode_position`` calls per row, most of the
    runtime over 345k rows. Handing the gate's own results forward also removes
    the possibility that the row is built from a DIFFERENT encode than the one
    that passed the gate.
    """

    verdict: str  # "ok" | "skip" (row unusable, chain sound) | "fail" (abandon game)
    production_planes: np.ndarray | None = None
    targets: PolicyTargets | None = None


def _gate_position(
    name: str,
    ply: int,
    board: chess.Board,
    rec: V6Record,
    stats: VerifyStats,
    options: ConvertOptions,
) -> GatedPosition:
    """Run every gate on one position."""
    stats.attempts += 1

    def fail(gate: str, detail: str) -> GatedPosition:
        # ⚑ Both halves, always: `divergence` records WHERE, `note` keeps
        # `drop_reasons` summing to `games_dropped`. The failure path used to
        # call only the first, so the drop ledger was empty on exactly the runs
        # you would be reading it for.
        stats.divergence(f"{name} ply {ply}: {detail}")
        stats.note(f"gate failure: {gate}")
        return GatedPosition("fail")

    value_problem = value_field_problem(rec)
    if value_problem is not None:
        stats.value_range_mismatch += 1
        return fail("value_fields_in_range", value_problem)

    planes_lc0_root = encode_position(
        board, add_features=False, input_history_encoding=LC0_HISTORY_ROOT,
    )
    reference = record_reference_planes(rec)
    if _repetition_planes_set(rec):
        stats.rows_with_repetition += 1
    bad = [
        plane for plane in range(LC0_FULL.num_planes)
        if not np.array_equal(planes_lc0_root[plane], reference[plane])
    ]
    if bad:
        if known_repetition_ep_alias(board, planes_lc0_root, reference, bad):
            # The board is right; only OUR repetition plane is. Drop the row and
            # let the game continue — the chain is unaffected.
            stats.rep_ep_alias_rows += 1
            stats.note(EP_ALIAS_DROP_REASON)
            return GatedPosition("skip")
        stats.planes_mismatch += 1
        return fail("planes_bit_exact", f"planes {bad[:6]} differ ({board.fen()})")
    stats.planes_exact += 1

    # ⚑ BEFORE the support gate, and decoded by `move_from_leela_slot`, which
    # never consults `board.legal_moves`. Ordered first because the support gate
    # would otherwise shadow it, and decoded independently because picking the
    # argmax out of the legal moves would make `is_legal` a tautology — which is
    # what this check used to be.
    argmax_move = move_from_leela_slot(board, int(np.argmax(np.asarray(rec.probabilities))))
    if argmax_move is not None and board.is_legal(argmax_move):
        stats.argmax_legal += 1
    else:
        stats.argmax_illegal += 1
        return fail(
            "argmax_is_legal",
            f"lc0's own policy argmax decodes to {argmax_move} which is illegal "
            f"({board.fen()})",
        )

    slots = board_leela_slots(board)
    if set(slots) != leela_support(rec):
        stats.support_mismatch += 1
        return fail(
            "legal_set_matches_policy_support",
            f"legal set != lc0 policy support ({board.fen()})",
        )
    stats.support_exact += 1

    targets = build_policy_targets(board, rec, planes_lc0_root)
    if not targets.gather_agrees:
        stats.gather_disagrees += 1
        return fail("gather_matches_reference", "gather map != per-move reference")
    stats.gather_agrees += 1

    if int(np.argmax(np.asarray(rec.probabilities))) == rec.best_idx:
        stats.best_idx_is_argmax += 1

    production = encode_position(
        board,
        add_features=True,
        input_history_encoding=options.input_history_encoding,
        input_extra_features=options.input_extra_features,
    )
    if not _production_aux_consistent(production, planes_lc0_root, board):
        stats.production_aux_mismatch += 1
        return fail("production_aux_planes", "production planes diverge from lc0_root")
    stats.production_aux_matches += 1
    return GatedPosition("ok", production_planes=production, targets=targets)


def _production_aux_consistent(
    production: np.ndarray, lc0_root: np.ndarray, board: chess.Board,
) -> bool:
    """The production encoding differs from lc0's ONLY where it is meant to.

    ``lc0_root_legacy_meta`` keeps lc0's layout except at planes 109 and 110,
    where it rescales rule50 to ``min(halfmove, 100) / 100`` and reuses the dead
    movecount plane as an en-passant file marker. Everything else — all 104
    history planes, castling, side to move, the ones plane — must be identical,
    so a divergence anywhere else means the production row is not the board the
    round trip just proved correct.
    """
    base = LC0_FULL.root_metadata_base
    rule50_plane, ep_plane = base + 5, base + 6
    for plane in range(LC0_FULL.num_planes):
        if plane in (rule50_plane, ep_plane):
            continue
        if not np.array_equal(production[plane], lc0_root[plane]):
            return False
    expected_rule50 = min(float(board.halfmove_clock), 100.0) / 100.0
    if not np.allclose(production[rule50_plane], expected_rule50):
        return False
    expected_ep = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        expected_ep[:, chess.square_file(board.ep_square)] = 1.0
    return np.array_equal(production[ep_plane], expected_ep)


def _row_from(
    rec: V6Record,
    options: ConvertOptions,
    gated: GatedPosition,
    *,
    game_id: int,
    ply: int,
) -> ReplaySample:
    """Build the row from the artifacts the gate ALREADY produced.

    Recomputing them here would both double the runtime and open a gap between
    the encode that passed the gate and the encode that was written.
    """
    if gated.production_planes is None or gated.targets is None:
        raise ValueError("_row_from needs a gate verdict of 'ok'")
    planes, targets = gated.production_planes, gated.targets
    outcome = _wdl_from_q_d(rec.result_q, rec.result_d)
    return ReplaySample(
        x=planes.astype(np.float32, copy=False),
        policy_target=targets.policy,
        wdl_target=int(np.argmax(outcome)),
        legal_mask=targets.legal_mask,
        search_wdl=_wdl_from_q_d(rec.best_q, rec.best_d),
        moves_left=max(0.0, float(rec.plies_left)) / max(1.0, options.moves_left_max_plies),
        has_policy=True,
        is_selfplay=True,
        is_network_turn=True,
        game_id=game_id,
        ply_index=ply,
        input_history_encoding=options.input_history_encoding,
        history_rep_fix=options.history_rep_fix,
    )


# ── the run-config gate ────────────────────────────────────────────────────────
#
# ⚑ This exists because a REQUIREMENT THAT LIVES ONLY IN PROSE IS NOT A GUARD.
# The rows carry no Stockfish label, and `losses.py` falls the SF component of
# the value blend back to the RAW ONE-HOT OUTCOME when `has_sf_wdl` is 0 — no
# error, no log line. So a control run launched at the production
# `sf_wdl_frac: 0.50` would train value on ~50% deep game outcome, which is the
# exact regime production avoids, and the positive control would be measuring
# something nobody chose. Nothing in the trainer refuses that today, so this
# command does, and it must be run before the launch.

VALUE_BLEND_KEYS = ("sf_wdl_frac", "sf_wdl_frac_floor")


def run_config_problems(
    config: Mapping[str, object],
    *,
    shards_have_sf_wdl: bool,
    shards_have_search_wdl: bool,
) -> list[str]:
    """Reasons this config must not be pointed at lc0-derived shards.

    ⚑ The collapse check is NOT short-circuited by ``shards_have_sf_wdl``.
    Returning ``[]`` on the first line whenever any SF label exists left a door
    open that PR #438's review walked through: ``sf_wdl_frac: 0`` with
    ``search_wdl_frac: 0`` and one SF-labelled shard in the list passes every
    guard and trains 100% of the value target on the raw one-hot outcome. The
    SF-label problems are label-dependent; "nothing but the outcome carries
    value weight" is not.

    ⚑ ``shards_have_search_wdl`` is REQUIRED, with no default, because the
    default that reads naturally (``True``) is the bug: ``compute_loss`` falls
    the SEARCH component back to the raw one-hot exactly the way it falls the
    SF component back, and the search share is the LARGER term here (0.70).
    A default would have re-created review F1 at the next call site.
    """
    problems: list[str] = []
    if not shards_have_sf_wdl:
        problems += [
            f"{key}={value!r} but the shards carry NO Stockfish label, so losses.py "
            f"silently redirects that share onto the raw game outcome; set {key}: 0.0"
            for key in VALUE_BLEND_KEYS
            if isinstance(value := config.get(key), (int, float)) and float(value) > 0.0
        ]
    sf = config.get("sf_wdl_frac")
    search = config.get("search_wdl_frac")
    if not shards_have_search_wdl and isinstance(search, (int, float)) and float(search) > 0.0:
        problems.append(
            f"search_wdl_frac={float(search)!r} but the shards carry NO search "
            "value label (has_search_wdl), so losses.py silently redirects that "
            "share onto the raw game outcome too — the same fallback as the SF "
            "component, on the larger term",
        )
    effective_sf = (
        float(sf) if shards_have_sf_wdl and isinstance(sf, (int, float)) else 0.0
    )
    effective_search = (
        float(search)
        if shards_have_search_wdl and isinstance(search, (int, float)) else 0.0
    )
    if effective_sf <= 0.0 and effective_search <= 0.0:
        problems.append(
            "search_wdl_frac is 0/absent and no SF share survives on these shards, "
            "so nothing carries a searched value and the whole value target "
            "collapses onto the raw game outcome",
        )
    return problems


def shard_dir_label_coverage(shard_dir: Path, flag: str) -> tuple[int, int]:
    """``(labelled_rows, total_rows)`` for one ``has_*`` column.

    ⚑ A FRACTION, not a boolean. ``any()`` over a mixed corpus reports "these
    shards carry the label" from a single labelled row out of millions, and
    every downstream decision then reasons about a corpus that does not exist.
    Reads the flag through the LAZY shard loader so the 175-plane inputs are
    never decoded.
    """
    from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays

    labelled = 0
    rows = 0
    for path in iter_shard_paths(shard_dir):
        arrs, _meta = load_shard_arrays(path, lazy=True)
        flags = arrs.get(flag)
        if flags is None:
  # ⚑ Row count off the INPUT array, not off `meta["positions"]`. That key is
  # optional and reads None on shards written without it, and an unlabelled
  # shard that contributed 0 to the denominator would make a mixed corpus read
  # as fully labelled — the exact failure this function replaces.
            rows += int(np.asarray(arrs["x"].shape[0]))
            continue
        values = np.asarray(flags)
        rows += int(values.shape[0])
        labelled += int((values > 0).sum())
    return labelled, rows


def shard_dir_sf_wdl_coverage(shard_dir: Path) -> tuple[int, int]:
    """``(labelled_rows, total_rows)`` for the Stockfish value label."""
    return shard_dir_label_coverage(shard_dir, "has_sf_wdl")


def shard_dir_search_wdl_coverage(shard_dir: Path) -> tuple[int, int]:
    """``(labelled_rows, total_rows)`` for the SEARCH value label.

    ⚑ The bigger term, and the one that had no reader until PR #438's review
    F1. ``compute_loss`` falls the SEARCH component back to the raw one-hot
    exactly the way it falls the SF component back, and on the lc0 control
    ``search_wdl_frac`` is 0.70 of the value target — so an unlabelled corpus
    here trains almost the whole value head on the game result while
    ``sf_wdl_frac: 0.0`` keeps every SF-side check clean.
    """
    return shard_dir_label_coverage(shard_dir, "has_search_wdl")


def shard_dir_has_sf_wdl(shard_dir: Path) -> bool:
    """Whether ANY shard under ``shard_dir`` carries a Stockfish value label."""
    labelled, _rows = shard_dir_sf_wdl_coverage(shard_dir)
    return labelled > 0


# ── policy-shape statistics ────────────────────────────────────────────────────

@dataclass
class PolicyShapeStats:
    """The four banked lc0-target shape numbers, recomputed."""

    records: int = 0
    entropy_p50: float = float("nan")
    max_prob_p50: float = float("nan")
    one_hot_frac: float = float("nan")
    support_p50: float = float("nan")
    full_support_frac: float = float("nan")


def policy_shape_stats(
    records: Iterable[V6Record], *, one_hot_threshold: float = 0.9999,
) -> PolicyShapeStats:
    """Entropy / max-prob / one-hot share / support over lc0's own targets.

    ``full_support_frac`` is not one of the banked four; it is here because it
    separates "the parse is aligned" from "the numbers look plausible": a
    misaligned read cannot produce a distribution whose non-zero entries land
    exactly on the legal moves.
    """
    entropies: list[float] = []
    max_probs: list[float] = []
    supports: list[int] = []
    one_hot = 0
    full_support = 0
    for rec in records:
        probabilities = np.asarray(rec.probabilities, dtype=np.float64)
        legal = probabilities[probabilities >= 0.0]
        if legal.size == 0:
            continue
        positive = legal[legal > 0.0]
        entropies.append(float(-(positive * np.log(positive)).sum()))
        max_probs.append(float(legal.max()))
        supports.append(int(positive.size))
        one_hot += int(legal.max() >= one_hot_threshold)
        full_support += int(positive.size == legal.size)
    if not entropies:
        return PolicyShapeStats()
    return PolicyShapeStats(
        records=len(entropies),
        entropy_p50=float(np.median(entropies)),
        max_prob_p50=float(np.median(max_probs)),
        one_hot_frac=one_hot / len(entropies),
        support_p50=float(np.median(supports)),
        full_support_frac=full_support / len(entropies),
    )


# ── the subtle-corruption harness ──────────────────────────────────────────────

def _with_bytes(blob: bytes, offset: int, payload: bytes) -> bytes:
    return blob[:offset] + payload + blob[offset + len(payload):]


def corrupt_shift_history(blob: bytes, record: int) -> bytes:
    """Rotate one record's history frames 1..7 back by one ply."""
    start = record * V6_RECORD_BYTES + _OFF_PLANES
    planes = bytearray(blob[start:start + V6_HISTORY_PLANES * 8])
    frame = LC0_PLANES_PER_FRAME * 8
    shifted = planes[2 * frame:8 * frame] + planes[frame:2 * frame]
    return _with_bytes(blob, start + frame, bytes(shifted))


def corrupt_castling_bit(blob: bytes, record: int) -> bytes:
    """Flip the side-to-move's kingside castling bit in one record."""
    offset = record * V6_RECORD_BYTES + _OFF_CASTLING + 1
    return _with_bytes(blob, offset, bytes([1 - blob[offset]]))


def corrupt_transpose_policy(blob: bytes, record: int) -> bytes:
    """Swap the two largest policy entries of one record.

    Both stay legal moves, so the distribution remains valid over the same
    support — the point of the test is to find out which gate, if any, can see
    a permutation that the data itself cannot distinguish from a real one.
    """
    start = record * V6_RECORD_BYTES + _OFF_PROBABILITIES
    probabilities = np.frombuffer(
        blob, dtype="<f4", count=LEELA_POLICY_SIZE, offset=start,
    ).copy()
    order = np.argsort(probabilities)[::-1]
    first, second = int(order[0]), int(order[1])
    probabilities[first], probabilities[second] = probabilities[second], probabilities[first]
    return _with_bytes(blob, start, probabilities.tobytes())


def corrupt_illegal_policy_slot(blob: bytes, record: int) -> bytes:
    """Move one legal move's probability onto an illegal slot."""
    start = record * V6_RECORD_BYTES + _OFF_PROBABILITIES
    probabilities = np.frombuffer(
        blob, dtype="<f4", count=LEELA_POLICY_SIZE, offset=start,
    ).copy()
    legal = int(np.argmax(probabilities))
    illegal = int(np.flatnonzero(probabilities < 0.0)[0])
    probabilities[illegal] = probabilities[legal]
    probabilities[legal] = -1.0
    return _with_bytes(blob, start, probabilities.tobytes())


def corrupt_mirror_board(blob: bytes, record: int) -> bytes:
    """Rank-mirror one record's live frame WITHOUT flipping the side-to-move byte."""
    start = record * V6_RECORD_BYTES + _OFF_PLANES
    planes = np.frombuffer(blob, dtype="<u8", count=V6_HISTORY_PLANES, offset=start).copy()
    for plane in range(12):
        mask = int(planes[plane])
        planes[plane] = np.uint64(
            int.from_bytes(mask.to_bytes(8, "little")[::-1], "little"),
        )
    return _with_bytes(blob, start, planes.tobytes())


CORRUPTIONS = {
    "history_shifted_one_ply": corrupt_shift_history,
    "castling_bit_flipped": corrupt_castling_bit,
    "policy_top_two_transposed": corrupt_transpose_policy,
    "policy_moved_to_illegal_slot": corrupt_illegal_policy_slot,
    "board_mirrored_stm_unchanged": corrupt_mirror_board,
}


def run_corruption_check(
    records: Sequence[V6Record], blob: bytes, options: ConvertOptions,
) -> dict[str, dict[str, object]]:
    """Apply each subtle corruption to a real game and report the gate that fires."""
    clean = VerifyStats()
    convert_game("clean", records, clean, options, game_id=0, collect=False)
    results: dict[str, dict[str, object]] = {
        # ⚑ OBSERVED, not asserted. Hardcoding `None` here would make the
        # negative control incapable of reporting that the clean run also trips
        # a gate — which is the one thing it exists to rule out.
        "clean": {"caught_by": _first_failing_gate(clean), "ok": clean.ok, "rows": clean.rows},
    }
    target = min(len(records) - 1, max(1, len(records) // 2))
    clean_agrees = _best_idx_is_argmax(records[target])
    for label, mutate in CORRUPTIONS.items():
        mutated_blob = mutate(blob, target)
        mutated = parse_v6_stream(mutated_blob)
        stats = VerifyStats()
        convert_game(label, mutated, stats, options, game_id=0, collect=False)
        results[label] = {
            "caught_by": _first_failing_gate(stats),
            "ok": stats.ok,
            "rows": stats.rows,
            "detail": stats.first_divergence,
            "drop_reasons": dict(stats.drop_reasons),
            # ⚑ Reported separately because it is a DIAGNOSTIC, not a gate:
            # lc0's stored best_idx is the search's best move and the policy is
            # its visit distribution, and those disagree on ~0.5% of clean rows,
            # so no per-row threshold on it can be a hard invariant.
            "best_idx_argmax_flipped_at_target": bool(
                clean_agrees and not _best_idx_is_argmax(mutated[target]),
            ),
        }
    return results


def _best_idx_is_argmax(rec: V6Record) -> bool:
    return int(np.argmax(np.asarray(rec.probabilities))) == rec.best_idx


def played_promotion_plies(records: Sequence[V6Record]) -> list[tuple[int, chess.Move]]:
    """Plies whose PLAYED move is a promotion and that have a following record."""
    first = board_from_record(records[0])
    if castling_reconstruction_problem(records[0], first) is not None:
        return []
    board = repair_en_passant(first, records[0])
    if board is None:
        return []
    found: list[tuple[int, chess.Move]] = []
    for ply, rec in enumerate(records):
        move = board_leela_slots(board).get(rec.played_idx)
        if move is None:
            break
        if move.promotion is not None and ply + 1 < len(records):
            found.append((ply, move))
        board.push(move)
    return found


def promotion_spelling_probe(
    records: Sequence[V6Record], blob: bytes, options: ConvertOptions,
) -> list[dict[str, object]]:
    """Rewrite a PLAYED promotion into Leela's OTHER spelling and re-run the gates.

    ⚑ This is the only test that can decide bare-slot semantics. Leela spells
    promotion-to-queen ``h7h8q`` and promotion-to-knight ``h7h8``; both slots
    are legal in a promoting position, so every mask/support check passes under
    either reading. Swapping the stored ``played_idx`` between the two spellings
    is exactly the mistake an inverted convention would make, and the chain must
    notice it — the pushed piece differs, so the NEXT record's piece planes do.
    """
    out: list[dict[str, object]] = []
    for ply, move in played_promotion_plies(records):
        spelled = LC0_1858_MOVE_STRS[records[ply].played_idx]
        alternative = spelled[:4] if len(spelled) == 5 and spelled[4] == "q" else spelled + "q"
        alternative_idx = LC0_1858_UCI_TO_IDX.get(alternative)
        if alternative_idx is None:
            continue
        mutated = _with_bytes(
            blob,
            ply * V6_RECORD_BYTES + _OFF_PLAYED_IDX,
            struct.pack("<H", int(alternative_idx)),
        )
        stats = VerifyStats()
        convert_game(
            "promotion_spelling", parse_v6_stream(mutated), stats, options,
            game_id=0, collect=False,
        )
        out.append({
            "ply": ply,
            "played": move.uci(),
            "leela_spelling": spelled,
            "rewritten_as": alternative,
            "caught_by": _first_failing_gate(stats),
            "rows_before_catch": stats.rows,
        })
    return out


def _first_failing_gate(stats: VerifyStats) -> str | None:
    for name, count in (
        ("planes_bit_exact", stats.planes_mismatch),
        ("legal_set_matches_policy_support", stats.support_mismatch),
        ("gather_matches_reference", stats.gather_disagrees),
        ("argmax_is_legal", stats.argmax_illegal),
        ("production_aux_planes", stats.production_aux_mismatch),
        ("value_fields_in_range", stats.value_range_mismatch),
    ):
        if count:
            return name
    # ⚑ The e.p.-alias drop is BENIGN — a row we chose not to emit, not a
    # corruption anyone detected. Crediting it would let a mutant that merely
    # provokes one read as "caught by a gate".
    chain = sorted(r for r in stats.drop_reasons if r != EP_ALIAS_DROP_REASON)
    if chain:
        return f"chain: {chain[0]}"
    return None


# ── CLI ────────────────────────────────────────────────────────────────────────

def _games(paths: Sequence[Path], limit: int) -> Iterator[tuple[str, list[V6Record]]]:
    for index, (name, records) in enumerate(iter_lc0_games(paths)):
        if limit and index >= limit:
            return
        yield name, records


def _raw_games(paths: Sequence[Path], limit: int) -> Iterator[tuple[str, bytes]]:
    """Same iteration as :func:`iter_lc0_games` but keeping the raw bytes.

    The corruption harness needs the undecoded record bytes so it can mutate a
    single field and re-parse; every other caller takes the decoded records.
    """
    for count, (name, blob) in enumerate(_walk_raw(paths)):
        if limit and count >= limit:
            return
        yield name, blob


def _walk_raw(paths: Sequence[Path]) -> Iterator[tuple[str, bytes]]:
    for path in paths:
        if path.is_dir():
            yield from _walk_raw(sorted(
                p for p in path.iterdir()
                if p.suffix in {".tar", ".gz"} and not p.name.endswith(".part")
            ))
        elif path.suffix == ".gz":
            yield path.name, gzip.decompress(path.read_bytes())
        elif path.suffix == ".tar":
            yield from _raw_tar_members(path)
        else:
            raise ValueError(f"unrecognised lc0 training path {path}")


def _raw_tar_members(path: Path) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(path) as tar:
        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue
                handle = tar.extractfile(member)
                if handle is not None:
                    yield member.name, gzip.decompress(handle.read())
        except (tarfile.ReadError, EOFError, gzip.BadGzipFile):
            return


def _options(args: argparse.Namespace) -> ConvertOptions:
    return ConvertOptions(
        input_history_encoding=args.history_encoding,
        input_extra_features=args.extra_features,
        moves_left_max_plies=float(args.moves_left_max_plies),
    )


def _manifest(options: ConvertOptions, stats: VerifyStats) -> dict[str, object]:
    return {
        "source": "lc0 v6 training records (input_format 1)",
        "input_history_encoding": options.input_history_encoding,
        "input_extra_features": options.input_extra_features,
        "policy_encoding": "lc0_1858",
        "moves_left_max_plies": options.moves_left_max_plies,
        "value_channels": {
            "wdl_target": "lc0 result_q/result_d (game outcome, side-to-move POV)",
            "search_wdl": "lc0 best_q/best_d (search-improved root value)",
            "sf_wdl": "ABSENT — lc0 data carries no Stockfish label",
        },
        "required_training_overrides": {
            "sf_wdl_frac": 0.0,
            "search_wdl_frac": "the whole non-outcome share (lc0's own recipe: 0.5)",
        },
        "why": (
            "losses.py falls the SF component back to the raw one-hot outcome "
            "when has_sf_wdl is 0, so a run left at the production sf_wdl_frac "
            "would silently train value on the deep game outcome."
        ),
        "verification": asdict(stats),
    }


def _print(lines: Sequence[str]) -> None:
    for line in lines:
        print(line)


def cmd_verify(args: argparse.Namespace) -> int:
    options = _options(args)
    stats = VerifyStats()
    for index, (name, records) in enumerate(_games(args.data, args.limit_games)):
        convert_game(name, records, stats, options, game_id=index, collect=False)
    _print(stats.rate_lines())
    if args.json_report:
        Path(args.json_report).write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    verdict, code = stats.overall_verdict()
    print("VERDICT:", verdict)
    return code


def cmd_stats(args: argparse.Namespace) -> int:
    def records() -> Iterator[V6Record]:
        for _, game in _games(args.data, args.limit_games):
            yield from game

    shape = policy_shape_stats(records())
    print(f"records            {shape.records}")
    print(f"entropy p50        {shape.entropy_p50:.4f} nats   (banked test80: 1.399)")
    print(f"max-prob p50       {shape.max_prob_p50:.4f}         (banked test80: 0.562)")
    print(f"one-hot            {100.0 * shape.one_hot_frac:.2f}%          (banked test80: 1.3%)")
    print(f"support p50        {shape.support_p50:.0f}              (banked test80: 30)")
    print(f"full-support frac  {shape.full_support_frac:.4f}  (parse-alignment check)")
    print(
        "⚑ The banked four were measured on test80 run1, a different lc0 run with a\n"
        "  different generating net. A difference here is NOT by itself evidence of a\n"
        "  parse defect; the parse is established by the structural checks (legal-set\n"
        "  agreement and full-support fraction), not by matching these numbers.",
    )
    return 0


def cmd_corrupt_check(args: argparse.Namespace) -> int:
    options = _options(args)
    games = _raw_games(args.data, args.limit_games or args.promotion_scan_games)
    first = next(games, None)
    if first is None:
        print("no games found")
        return 1
    name, blob = first
    records = parse_v6_stream(blob)
    results = run_corruption_check(records, blob, options)
    print(f"game {name} ({len(records)} records)")
    for label, outcome in results.items():
        print(f"  {label:34s} caught_by={outcome['caught_by']} rows={outcome['rows']}")
        if outcome.get("best_idx_argmax_flipped_at_target"):
            print("      + best_idx/argmax agreement FLIPPED at the mutated row "
                  "(diagnostic only — ~0.5% of clean rows also disagree)")
        if outcome.get("detail"):
            print(f"      {outcome['detail']}")
    undetected = [
        label for label, outcome in results.items()
        if label != "clean" and outcome["caught_by"] is None
    ]
    print(f"  UNDETECTED by a hard gate: {undetected or 'none'}")

    print("promotion-spelling probe (bare slot = KNIGHT for Leela, QUEEN for us):")
    probes: list[dict[str, object]] = []
    scanned = 0
    for _probe_name, probe_blob in _walk_raw(args.data):
        scanned += 1
        probes.extend(promotion_spelling_probe(
            parse_v6_stream(probe_blob), probe_blob, options,
        ))
        if len(probes) >= args.promotion_probes or scanned >= args.promotion_scan_games:
            break
    for probe in probes[:args.promotion_probes]:
        print(f"  ply {probe['ply']:>3} played {probe['played']} spelled "
              f"{probe['leela_spelling']!r} -> rewritten {probe['rewritten_as']!r}: "
              f"caught_by={probe['caught_by']}")
    missed = [p for p in probes[:args.promotion_probes] if p["caught_by"] is None]
    tested = min(len(probes), args.promotion_probes)
    print(f"  promotion positions tested {tested} over {scanned} games scanned; "
          f"undetected {len(missed)}")
    # ⚑ Each of these is a way the harness could report success while having
    # measured nothing, so each is its own failure rather than an implicit pass:
    # a clean control that itself trips a gate makes every "caught" reading
    # meaningless, and ZERO promotion probes found is an unrun test, not a
    # passed one.
    problems: list[str] = []
    if not bool(results["clean"]["ok"]) or results["clean"]["caught_by"] is not None:
        problems.append(f"clean control did not pass cleanly: {results['clean']}")
    if tested == 0:
        problems.append(f"no promotion probes found in {scanned} games — nothing was tested")
    problems.extend(f"undetected corruption: {label}" for label in undetected_hard(results))
    problems.extend(f"undetected promotion respelling at ply {p['ply']}" for p in missed)
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print("VERDICT:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


def undetected_hard(results: dict[str, dict[str, object]]) -> list[str]:
    """Corruptions no HARD gate caught. ``policy_top_two_transposed`` is here by
    construction: a permutation of two legal moves' visit counts leaves a valid
    distribution over the same support, and the record carries no redundancy
    that could contradict it beyond the soft ``best_idx`` pointer."""
    return [
        label for label, outcome in results.items()
        if label not in {"clean", "policy_top_two_transposed"}
        and outcome["caught_by"] is None
    ]


def cmd_check_run_config(args: argparse.Namespace) -> int:
    """Refuse a training config that would silently mis-target these rows."""
    import yaml

    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    config = flatten_run_config_defaults(yaml.safe_load(Path(args.config).read_text()))
    have_sf = shard_dir_has_sf_wdl(Path(args.shards))
  # ⚑ Coverage, not presence, on the SEARCH side: a share that is not carried
  # by EVERY row still lands on the raw outcome for the rest of them.
    search_labelled, search_rows = shard_dir_search_wdl_coverage(Path(args.shards))
    have_search = search_rows > 0 and search_labelled == search_rows
    problems = run_config_problems(
        config, shards_have_sf_wdl=have_sf, shards_have_search_wdl=have_search,
    )
    print(f"shards carry a Stockfish value label: {have_sf}")
    print(f"shards carry a search value label:    {have_search} "
          f"({search_labelled}/{search_rows} rows)")
    for key in (*VALUE_BLEND_KEYS, "search_wdl_frac"):
        print(f"  {key} = {config.get(key, '<absent>')!r}")
    for problem in problems:
        print(f"REFUSED: {problem}")
    print("VERDICT:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


STAGING_DIR_NAME = "_staging"
REJECTED_DIR_NAME = "_rejected"


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert to a STAGING dir, and publish only on a PASS.

    ⚑ Shards used to be written straight into ``--out`` inside the loop while
    the verdict was computed afterwards, so a 30-game run published
    ``shard_000000.zarr`` and then printed ``INCONCLUSIVE`` (exit 2). Writing
    the artifact while declaring the run inconclusive is the wrong pairing, and
    of the two ways to fix it — publish anyway, or don't publish — only one is
    defensible for a positive control: **rows whose provenance check declined
    to answer must not become training data.** An INCONCLUSIVE run is a smoke
    test, not a data-production run.

    Nothing is silently lost either: on a non-PASS the staged shards are moved
    to ``<out>/_rejected`` and named in the output, so a long run is not thrown
    away and cannot be mistaken for a published one. This also closes the
    partial-publish hole (a mid-run gate failure used to leave already-written
    shards in ``--out`` with no manifest).
    """
    options = _options(args)
    out = Path(args.out)
    staging = out / STAGING_DIR_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    stats = VerifyStats()
    pending: list[ReplaySample] = []
    shard_index = 0
    written = 0
    for index, (name, records) in enumerate(_games(args.data, args.limit_games)):
        pending.extend(convert_game(name, records, stats, options, game_id=index))
        if args.max_rows:
            keep = max(0, args.max_rows - written)
            if len(pending) > keep:
                del pending[keep:]
        while len(pending) >= args.rows_per_shard:
            _write_shard(staging, shard_index, pending[:args.rows_per_shard], options, stats)
            written += args.rows_per_shard
            del pending[:args.rows_per_shard]
            shard_index += 1
        if args.max_rows and written + len(pending) >= args.max_rows:
            break
    if pending:
        _write_shard(staging, shard_index, pending, options, stats)
        written += len(pending)
        shard_index += 1
    _print(stats.rate_lines())
    verdict, code = stats.overall_verdict()
    if code == 0:
        published = _publish(staging, out)
        print(f"published {written} rows in {published} shard(s) under {out}")
        if args.emit_manifest:
            (out / "lc0_rows_manifest.json").write_text(
                json.dumps(_manifest(options, stats), indent=2), encoding="utf-8",
            )
    else:
        rejected = out / REJECTED_DIR_NAME
        if rejected.exists():
            shutil.rmtree(rejected)
        staging.rename(rejected)
        print(
            f"NOT PUBLISHED: {written} rows in {shard_index} shard(s) held at {rejected} "
            "— a run that did not PASS does not produce training data",
        )
    print("VERDICT:", verdict)
    return code


def _publish(staging: Path, out: Path) -> int:
    """Move staged shards into ``out``. Refuses to overwrite an existing shard."""
    moved = 0
    for path in sorted(staging.iterdir()):
        destination = out / path.name
        if destination.exists():
            raise FileExistsError(
                f"{destination} already exists; refusing to mix this run's rows into a "
                "populated output directory",
            )
        path.rename(destination)
        moved += 1
    staging.rmdir()
    return moved


def _write_shard(
    out: Path,
    index: int,
    samples: Sequence[ReplaySample],
    options: ConvertOptions,
    stats: VerifyStats,
) -> None:
    if not stats.ok:
        raise RuntimeError(
            "refusing to write rows while a verification gate is failing: "
            f"{stats.first_divergence}",
        )
    save_local_shard_arrays(
        local_shard_path(out, index),
        arrs=samples_to_arrays(list(samples)),
        meta=ShardMeta(
            run_id="lc0_v6_import",
            input_history_encoding=options.input_history_encoding,
            history_rep_fix=options.history_rep_fix,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
            positions=len(samples),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("verify", cmd_verify),
        ("stats", cmd_stats),
        ("convert", cmd_convert),
        ("corrupt-check", cmd_corrupt_check),
        ("check-run-config", cmd_check_run_config),
    ):
        child = sub.add_parser(name)
        if name == "check-run-config":
            child.add_argument("--config", type=Path, required=True)
            child.add_argument("--shards", type=Path, required=True)
            child.set_defaults(handler=handler)
            continue
        child.add_argument("--data", type=Path, nargs="+", required=True)
        child.add_argument("--limit-games", type=int, default=0)
        child.add_argument("--history-encoding", default="lc0_root_legacy_meta")
        child.add_argument("--extra-features", default="v2_threats")
        child.add_argument("--moves-left-max-plies", type=float, default=450.0)
        child.set_defaults(handler=handler)
        if name == "verify":
            child.add_argument("--json-report", type=Path, default=None)
        if name == "corrupt-check":
            child.add_argument("--promotion-probes", type=int, default=12)
            child.add_argument("--promotion-scan-games", type=int, default=400)
        if name == "convert":
            child.add_argument("--out", type=Path, required=True)
            # 8192 rows x 175 planes x 64 squares is ~184 MB of float16 on disk
            # and ~370 MB of float32 buffered in RAM. The production 50,000-row
            # cap would be 1.1 GB / 2.2 GB here, because our x is 175 planes.
            child.add_argument("--rows-per-shard", type=int, default=8192)
            child.add_argument("--max-rows", type=int, default=0)
            child.add_argument(
                "--no-manifest", dest="emit_manifest", action="store_false", default=True,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
