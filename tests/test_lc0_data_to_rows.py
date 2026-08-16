"""Unit tests for the lc0 v6 -> our-row converter.

The synthetic records here are written from lc0's DOCUMENTED byte layout
(``ReverseBitsInBytes`` plane order, side-to-move orientation), deliberately
NOT by inverting the parser under test — otherwise a shared misreading would
cancel and every test would pass on a wrong convention.

That still leaves both sides of these tests written by the same author, which
is why they are the SECOND line of defence: the primary evidence is the
real-data run (``lc0_data_to_rows.py verify``), where lc0's own bytes are the
external referee. These tests pin the behaviour so a regression is caught
without the (uncommitted) corpus.
"""
from __future__ import annotations

import struct
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX
from chess_anti_engine.moves.leela_index import leela_index_for_move
from scripts.lc0_data_to_rows import (
    CHESS960_MIN_GAMES,
    LEELA_POLICY_SIZE,
    V6_HISTORY_PLANES,
    V6_RECORD_BYTES,
    ConvertOptions,
    VerifyStats,
    _OFF_CASTLING,
    _production_aux_consistent,
    board_from_record,
    board_leela_slots,
    castling_reconstruction_problem,
    convert_game,
    known_repetition_ep_alias,
    move_from_leela_slot,
    parse_v6_record,
    parse_v6_stream,
    policy_shape_stats,
    promotion_spelling_probe,
    record_reference_planes,
    repair_en_passant,
    run_config_problems,
    shard_dir_has_sf_wdl,
    _wdl_from_q_d,
)

_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def _lc0_mask(board: chess.Board, piece_type: int, color: chess.Color, *, mirror: bool) -> int:
    """One lc0 training plane mask, built from the documented bit order.

    lc0 stores a plane as ``ReverseBitsInBytes(bitboard)``, i.e. bit
    ``rank * 8 + (7 - file)`` of the SIDE-TO-MOVE-ORIENTED square.
    """
    mask = 0
    for square in board.pieces(piece_type, color):
        oriented = chess.square_mirror(square) if mirror else square
        mask |= 1 << (chess.square_rank(oriented) * 8 + (7 - chess.square_file(oriented)))
    return mask


def _history_boards(board: chess.Board, frames: int = 8) -> list[chess.Board]:
    out: list[chess.Board] = []
    walk = board.copy()
    for _ in range(frames):
        out.append(walk.copy())
        if not walk.move_stack:
            break
        walk.pop()
    return out


def make_v6_record(
    board: chess.Board,
    *,
    played_uci: str | None = None,
    probabilities: dict[str, float] | None = None,
    result_q: float = 0.0,
    result_d: float = 1.0,
    best_q: float = 0.0,
    best_d: float = 1.0,
    plies_left: float = 10.0,
    castling_override: tuple[int, int, int, int] | None = None,
) -> bytes:
    """Assemble one v6 record for ``board`` from lc0's documented layout."""
    us = board.turn
    mirror = us == chess.BLACK
    planes = np.zeros((V6_HISTORY_PLANES,), dtype="<u8")
    history = _history_boards(board)
    for frame, snapshot in enumerate(history):
        base = frame * 13
        for offset, color in ((0, us), (6, not us)):
            for i, piece_type in enumerate(_PIECES):
                planes[base + offset + i] = _lc0_mask(
                    snapshot, piece_type, color, mirror=mirror,
                )
        if snapshot.is_repetition(2):
            planes[base + 12] = np.uint64(0xFFFFFFFFFFFFFFFF)

    probs = np.full((LEELA_POLICY_SIZE,), -1.0, dtype="<f4")
    legal = list(board.legal_moves)
    weights = probabilities or {move.uci(): 1.0 for move in legal}
    total = 0.0
    for move in legal:
        total += float(weights.get(move.uci(), 0.0))
    for move in legal:
        slot = leela_index_for_move(board, move)
        probs[slot] = float(weights.get(move.uci(), 0.0)) / total if total else 0.0

    played = chess.Move.from_uci(played_uci) if played_uci else legal[0]
    played_idx = leela_index_for_move(board, played)
    best_idx = int(np.argmax(probs))

    if castling_override is not None:
        castling = castling_override
    else:
        castling = (
            int(board.has_queenside_castling_rights(us)),
            int(board.has_kingside_castling_rights(us)),
            int(board.has_queenside_castling_rights(not us)),
            int(board.has_kingside_castling_rights(not us)),
        )

    body = struct.pack("<II", 6, 1) + probs.tobytes() + planes.tobytes()
    body += struct.pack("<8B", *castling, int(mirror), int(board.halfmove_clock), 0, 0)
    body += struct.pack(
        "<15f",
        0.0, best_q, 0.0, best_d, 0.0, 0.0, plies_left,
        result_q, result_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    body += struct.pack("<IHHfI", 800, played_idx, best_idx, 0.0, 0)
    assert len(body) == V6_RECORD_BYTES
    return body


def make_game(moves: list[str], *, start: chess.Board | None = None) -> bytes:
    """A whole synthetic lc0 game blob: one record per ply, then the final one."""
    board = (start or chess.Board()).copy()
    blob = b""
    for uci in moves:
        blob += make_v6_record(board, played_uci=uci)
        board.push(chess.Move.from_uci(uci))
    blob += make_v6_record(board)
    return blob


def _options() -> ConvertOptions:
    return ConvertOptions()


# ── layout ─────────────────────────────────────────────────────────────────────

def test_record_size_matches_lc0_static_assert() -> None:
    assert V6_RECORD_BYTES == 8356
    assert len(make_v6_record(chess.Board())) == V6_RECORD_BYTES


def test_short_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be 8356 bytes"):
        parse_v6_record(make_v6_record(chess.Board())[:-1])


def test_stream_length_must_be_a_whole_number_of_records() -> None:
    with pytest.raises(ValueError, match="whole number of"):
        parse_v6_stream(make_v6_record(chess.Board()) + b"\x00")


# ── board reconstruction ───────────────────────────────────────────────────────

def test_startpos_round_trips_to_the_same_board() -> None:
    rec = parse_v6_record(make_v6_record(chess.Board()))
    assert board_from_record(rec).board_fen() == chess.Board().board_fen()


def test_black_to_move_record_rebuilds_the_TRUE_board_not_its_mirror() -> None:
    """v6 planes are side-to-move oriented; a black-to-move record must be
    un-mirrored AND recoloured, not read as a white-to-move position."""
    board = chess.Board()
    board.push_uci("e2e4")
    rec = parse_v6_record(make_v6_record(board))
    rebuilt = board_from_record(rec)
    assert rebuilt.turn == chess.BLACK
    assert rebuilt.board_fen() == board.board_fen()
    assert rebuilt.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_planes_are_bit_exact_against_the_record_over_a_real_line() -> None:
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"]
    stats = VerifyStats()
    convert_game("t", parse_v6_stream(make_game(moves)), stats, _options(),
                 game_id=0, collect=False)
    assert stats.first_divergence is None
    assert stats.rows == len(moves) + 1
    assert stats.planes_mismatch == 0
    assert stats.planes_exact == stats.rows


def test_repetition_planes_are_exercised_and_round_trip() -> None:
    """A shuffle that repeats a position must set the repetition plane in the
    record AND survive the bit-exact comparison — otherwise the plane check is
    vacuous on the one axis an 8-ply window cannot see."""
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]
    blob = make_game(moves)
    records = parse_v6_stream(blob)
    assert any(
        int(rec.planes[frame * 13 + 12])
        for rec in records for frame in range(8)
    ), "synthetic game did not produce a repetition; the test would be vacuous"
    stats = VerifyStats()
    convert_game("t", records, stats, _options(), game_id=0, collect=False)
    assert stats.planes_mismatch == 0
    assert stats.rows_with_repetition > 0


def _ep_alias_game() -> bytes:
    """A game whose final position aliases an earlier one modulo a LEGAL e.p.

    ``h2h4`` gives Black a legal ``g4xh3``; four reversible plies later the
    piece placement and side to move are identical but the e.p. right is gone.
    ``encoding/lc0.py::_check_repetitions`` keys without ``ep_square``, so OUR
    encoder calls that a repetition; lc0 and python-chess do not.
    """
    start = chess.Board("5R2/8/6k1/3p1p2/2bNp1p1/4P1P1/6KP/1r6 w - - 0 46")
    return make_game(["h2h4", "c4b5", "f8f7", "b5c4", "f7f8"], start=start)


def test_ep_alias_repetition_is_named_and_dropped_not_silently_kept() -> None:
    records = parse_v6_stream(_ep_alias_game())
    stats = VerifyStats()
    rows = convert_game("t", records, stats, _options(), game_id=0, collect=True)
    assert stats.rep_ep_alias_rows == 1, "fixture stopped producing the e.p. alias"
    assert stats.planes_mismatch == 0          # classified, not counted as a failure
    assert stats.rows == stats.attempts - 1    # the row is DROPPED, not emitted
    assert len(rows) == stats.rows
    assert stats.ok
    assert any("ep-alias" in reason for reason in stats.drop_reasons)


def test_an_abandoned_game_leaves_no_rows_on_the_counter() -> None:
    """``rows`` must equal the rows actually emitted. A game abandoned mid-way
    returns nothing, so the positions it already passed have to come back off
    the counter or the report over-states what was written."""
    clean = make_game(["e2e4", "e7e5", "g1f3", "b8c6"])
    flipped = bytearray(clean)
    flipped[2 * V6_RECORD_BYTES + _OFF_CASTLING + 1] = 0  # fails at ply 2
    stats = VerifyStats()
    rows = convert_game("t", parse_v6_stream(bytes(flipped)), stats, _options(),
                        game_id=0, collect=True)
    assert rows == []
    assert stats.rows == 0
    assert stats.attempts == 3  # plies 0 and 1 passed, ply 2 failed
    assert stats.planes_mismatch == 1


def test_ep_alias_classifier_refuses_a_false_NEGATIVE() -> None:
    """Ours=0 / lc0=1 is a different bug and must still fail the gate."""
    board = chess.Board()
    ours = np.zeros((112, 8, 8), dtype=np.float32)
    reference = np.zeros((112, 8, 8), dtype=np.float32)
    reference[12] = 1.0
    assert known_repetition_ep_alias(board, ours, reference, [12]) is False


def test_ep_alias_classifier_refuses_a_piece_plane_difference() -> None:
    board = chess.Board()
    ours = np.zeros((112, 8, 8), dtype=np.float32)
    ours[12] = 1.0
    ours[3] = 1.0
    reference = np.zeros((112, 8, 8), dtype=np.float32)
    assert known_repetition_ep_alias(board, ours, reference, [3, 12]) is False


def test_ep_alias_classifier_refuses_when_python_chess_agrees_with_us() -> None:
    """If the position REALLY is a repetition, lc0 disagreeing is unexplained."""
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert board.is_repetition(2)
    ours = np.zeros((112, 8, 8), dtype=np.float32)
    ours[12] = 1.0
    reference = np.zeros((112, 8, 8), dtype=np.float32)
    assert known_repetition_ep_alias(board, ours, reference, [12]) is False


# ── castling / chess960 ────────────────────────────────────────────────────────

def test_flipped_castling_bit_is_caught_by_the_plane_gate() -> None:
    clean = make_game(["e2e4", "e7e5", "g1f3", "b8c6"])
    stats = VerifyStats()
    convert_game("clean", parse_v6_stream(clean), stats, _options(),
                 game_id=0, collect=False)
    assert stats.ok

    # Flip the SECOND record's bit: from ply 1 on, castling comes from the
    # replayed board rather than the record, so only the plane gate can see it.
    flipped = bytearray(clean)
    flipped[V6_RECORD_BYTES + _OFF_CASTLING + 1] = 0
    dirty = VerifyStats()
    convert_game("dirty", parse_v6_stream(bytes(flipped)), dirty, _options(),
                 game_id=0, collect=False)
    assert dirty.planes_mismatch == 1
    assert dirty.first_divergence is not None
    assert "105" in dirty.first_divergence


def test_chess960_start_is_dropped_by_name() -> None:
    board = chess.Board("qbbnrknr/pppppppp/8/8/8/8/PPPPPPPP/QBBNRKNR w KQkq - 0 1", chess960=True)
    rec = parse_v6_record(make_v6_record(board, castling_override=(1, 1, 1, 1)))
    rebuilt = board_from_record(rec)
    assert castling_reconstruction_problem(rec, rebuilt) == "chess960"

    stats = VerifyStats()
    convert_game("frc", [rec], stats, _options(), game_id=0, collect=False)
    assert stats.drop_reasons == {"chess960": 1}
    assert stats.rows == 0


def test_chess960_with_STANDARD_rook_files_is_still_caught() -> None:
    """⚑ Separates the two halves of the detector. In ``qbbnrknr`` the rooks are
    on e1/h1, so the rook-file half fires and the king half is never exercised —
    a backstop that hollows out the test for the guard it backs. Here the rooks
    ARE on a1/h1 and only the displaced king can catch it."""
    board = chess.Board("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1", chess960=True)
    assert board.king(chess.WHITE) == chess.D1
    rec = parse_v6_record(make_v6_record(board, castling_override=(1, 1, 1, 1)))
    rebuilt = board_from_record(rec)
    assert rebuilt.castling_rights == (
        chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8
    )
    assert castling_reconstruction_problem(rec, rebuilt) == "chess960"


def test_chess960_with_king_on_e1_is_caught_by_the_rook_file_half() -> None:
    """The mirror of the test above: king on its home square, rooks on b/g."""
    board = chess.Board("nrbqkbrn/pppppppp/8/8/8/8/PPPPPPPP/NRBQKBRN w KQkq - 0 1", chess960=True)
    assert board.king(chess.WHITE) == chess.E1
    rec = parse_v6_record(make_v6_record(board, castling_override=(1, 1, 1, 1)))
    assert castling_reconstruction_problem(rec, board_from_record(rec)) == "chess960"


def test_castling_right_with_no_rook_is_named_separately() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    rec = parse_v6_record(make_v6_record(board, castling_override=(0, 1, 0, 0)))
    assert castling_reconstruction_problem(rec, board_from_record(rec)) == (
        "castling right with no backing rook"
    )


def test_chess960_rate_far_from_expected_is_flagged() -> None:
    """The detector must be observable: a run where it stops firing has to read
    as a number, not as clean output."""
    stats = VerifyStats(games=1000)
    assert stats.chess960_problem() is not None
    stats.drop_reasons["chess960"] = 40
    assert stats.chess960_problem() is None
    stats.drop_reasons["chess960"] = 900
    assert stats.chess960_problem() is not None
    assert VerifyStats(games=10).chess960_problem() is None  # abstains when noisy
    # ⚑ An abstention must READ as an abstention, not as a pass.
    assert VerifyStats(games=10).chess960_status().startswith("ABSTAINED")
    assert "within expected band" in stats_within_band().chess960_status()


def stats_within_band() -> VerifyStats:
    stats = VerifyStats(games=1000)
    stats.drop_reasons["chess960"] = 40
    return stats


# ── en passant ─────────────────────────────────────────────────────────────────

def test_en_passant_is_recovered_from_the_legal_move_set() -> None:
    """lc0's classical format cannot express en passant, so the first position
    of a game has to be repaired from lc0's own legal-move support."""
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert chess.Move.from_uci("e5f6") in board.legal_moves
    rec = parse_v6_record(make_v6_record(board))
    naive = board_from_record(rec)
    assert naive.ep_square is None
    repaired = repair_en_passant(naive, rec)
    assert repaired is not None
    assert repaired.ep_square == chess.F6
    assert chess.Move.from_uci("e5f6") in repaired.legal_moves


# ── policy index space ─────────────────────────────────────────────────────────

def test_bare_back_rank_slot_is_knight_for_leela_and_queen_for_us() -> None:
    board = chess.Board("4k3/7P/8/8/8/8/8/4K3 w - - 0 1")
    assert leela_index_for_move(board, chess.Move.from_uci("h7h8n")) == (
        LC0_1858_UCI_TO_IDX["h7h8"]
    )
    assert leela_index_for_move(board, chess.Move.from_uci("h7h8q")) == (
        LC0_1858_UCI_TO_IDX["h7h8q"]
    )


def test_swapping_a_played_promotion_spelling_breaks_the_chain() -> None:
    """The ONLY observation that can decide bare-slot semantics: a legal-mask
    check passes under either convention, the played move does not."""
    blob = make_game(["h7h8n", "e8d8", "h8g6"],
                     start=chess.Board("4k3/7P/8/8/8/8/8/4K3 w - - 0 1"))
    records = parse_v6_stream(blob)
    probes = promotion_spelling_probe(records, blob, _options())
    assert len(probes) == 1
    assert probes[0]["leela_spelling"] == "h7h8"  # bare == knight, for Leela
    assert probes[0]["rewritten_as"] == "h7h8q"
    assert probes[0]["caught_by"] == "planes_bit_exact"


def test_played_promotions_are_counted_only_once_confirmed() -> None:
    # Kd8 would still be on the 8th rank and so still in check from Qh8; the
    # king has to step off it. (The illegal line silently abandoned the game
    # and this test only noticed once `rows` stopped counting abandoned rows.)
    blob = make_game(["h7h8q", "e8e7", "h8h4"],
                     start=chess.Board("4k3/7P/8/8/8/8/8/4K3 w - - 0 1"))
    stats = VerifyStats()
    convert_game("t", parse_v6_stream(blob), stats, _options(), game_id=0, collect=False)
    assert stats.ok
    assert stats.played_promotions == {"q": 1}


def test_policy_target_lands_on_our_compact_slot_for_every_legal_move() -> None:
    # FEN-built: a single-record game has no history, so the record's frames
    # 1..7 must be empty for the bit-exact gate to be comparing like with like.
    board = chess.Board("rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 4")
    weights = {move.uci(): 1.0 for move in board.legal_moves}
    weights["c5d4"] = 97.0
    rec = parse_v6_record(make_v6_record(board, probabilities=weights, played_uci="c5d4"))
    stats = VerifyStats()
    rows = convert_game("t", [rec], stats, _options(), game_id=0, collect=True)
    assert stats.ok
    row = rows[0]
    assert row.legal_mask is not None
    assert int(row.legal_mask.sum()) == board.legal_moves.count()
    from chess_anti_engine.moves.leela_index import compact_index_for_move

    top = compact_index_for_move(board, chess.Move.from_uci("c5d4"))
    assert int(np.argmax(row.policy_target)) == top
    assert row.policy_target[~row.legal_mask.astype(bool)].sum() == 0.0
    assert float(row.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)


def _move_policy_mass_to_an_illegal_slot(blob: bytes, record: int, *, source: int) -> bytes:
    start = record * V6_RECORD_BYTES + 8
    probs = np.frombuffer(blob, dtype="<f4", count=LEELA_POLICY_SIZE, offset=start).copy()
    illegal = int(np.flatnonzero(probs < 0.0)[0])
    probs[illegal], probs[source] = probs[source], np.float32(-1.0)
    return blob[:start] + probs.tobytes() + blob[start + LEELA_POLICY_SIZE * 4:]


def _game_with_a_strict_argmax() -> bytes:
    """Two plies where record 1 has a STRICT policy argmax.

    The uniform-weight fixture ties every legal move, and ``np.argmax`` returns
    the first maximum — so relabelling any slot would move the argmax and both
    tests below would exercise the same gate.
    """
    board = chess.Board()
    blob = make_v6_record(board, played_uci="e2e4")
    board.push_uci("e2e4")
    weights = {move.uci(): 1.0 for move in board.legal_moves}
    weights["e7e5"] = 50.0
    blob += make_v6_record(board, played_uci="e7e5", probabilities=weights)
    board.push_uci("e7e5")
    return blob + make_v6_record(board)


def test_policy_moved_to_an_illegal_slot_is_caught_by_the_support_gate() -> None:
    """A NON-argmax legal move relabelled illegal: the argmax gate cannot see
    this one, so it is the support gate's to catch."""
    clean = _game_with_a_strict_argmax()
    probs = np.frombuffer(clean, dtype="<f4", count=LEELA_POLICY_SIZE,
                          offset=V6_RECORD_BYTES + 8)
    legal = np.flatnonzero(probs >= 0.0)
    non_argmax = int(legal[-1])
    assert non_argmax != int(np.argmax(probs))
    dirty = _move_policy_mass_to_an_illegal_slot(clean, 1, source=non_argmax)
    stats = VerifyStats()
    convert_game("t", parse_v6_stream(dirty), stats, _options(), game_id=0, collect=False)
    assert stats.support_mismatch == 1
    assert stats.argmax_illegal == 0
    assert stats.first_divergence is not None
    assert "legal set != lc0 policy support" in stats.first_divergence


def test_lc0_argmax_landing_on_an_illegal_slot_is_caught_BEFORE_the_support_gate() -> None:
    """⚑ The gate this replaced could not fail: it picked the argmax by
    iterating ``board.legal_moves`` and then asked whether it was legal. This
    one decodes lc0's OWN stored argmax slot through
    :func:`move_from_leela_slot`, which never consults legality — so it has an
    answer that can be No, and it is ordered ahead of the support gate that
    would otherwise shadow it."""
    clean = _game_with_a_strict_argmax()
    argmax = int(np.argmax(np.frombuffer(
        clean, dtype="<f4", count=LEELA_POLICY_SIZE, offset=V6_RECORD_BYTES + 8)))
    dirty = _move_policy_mass_to_an_illegal_slot(clean, 1, source=argmax)
    stats = VerifyStats()
    convert_game("t", parse_v6_stream(dirty), stats, _options(), game_id=0, collect=False)
    assert stats.argmax_illegal == 1
    assert stats.support_mismatch == 0, "the argmax gate must run first"
    assert stats.first_divergence is not None
    assert "policy argmax decodes to" in stats.first_divergence


def test_move_from_leela_slot_decodes_without_consulting_legal_moves() -> None:
    """The reverse decoder's three conventions, each on a board where getting it
    wrong yields a DIFFERENT move rather than an error."""
    promo = chess.Board("4k3/7P/8/8/8/8/8/4K3 w - - 0 1")
    assert move_from_leela_slot(promo, LC0_1858_UCI_TO_IDX["h7h8"]) == (
        chess.Move.from_uci("h7h8n")          # bare slot == KNIGHT for Leela
    )
    assert move_from_leela_slot(promo, LC0_1858_UCI_TO_IDX["h7h8q"]) == (
        chess.Move.from_uci("h7h8q")
    )
    castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert move_from_leela_slot(castle, LC0_1858_UCI_TO_IDX["e1h1"]) == (
        chess.Move.from_uci("e1g1")           # Leela spells O-O king-takes-rook
    )
    # Black to move: the slot is in ORIENTED coordinates and must be un-mirrored.
    castle.turn = chess.BLACK
    assert move_from_leela_slot(castle, LC0_1858_UCI_TO_IDX["e1h1"]) == (
        chess.Move.from_uci("e8g8")
    )
    # It really does not filter by legality — that is what gives the gate content.
    empty = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    decoded = move_from_leela_slot(empty, LC0_1858_UCI_TO_IDX["a1a8"])
    assert decoded is not None
    assert decoded == chess.Move.from_uci("a1a8")
    assert not empty.is_legal(decoded)


def test_first_position_support_mismatch_is_named_separately() -> None:
    """A corrupt FIRST record cannot reach the in-loop gate: the game's opening
    board is built from it, so the mismatch surfaces as a drop, not a gate."""
    clean = make_v6_record(chess.Board())
    probs = np.frombuffer(clean, dtype="<f4", count=LEELA_POLICY_SIZE, offset=8).copy()
    probs[int(np.argmax(probs))] = -1.0
    dirty = clean[:8] + probs.tobytes() + clean[8 + LEELA_POLICY_SIZE * 4:]
    stats = VerifyStats()
    convert_game("t", parse_v6_stream(dirty), stats, _options(), game_id=0, collect=False)
    assert stats.drop_reasons == {"first-position legal set does not match the record": 1}


# ── value targets ──────────────────────────────────────────────────────────────

def test_wdl_from_q_d_is_side_to_move_pov() -> None:
    assert list(_wdl_from_q_d(1.0, 0.0)) == [1.0, 0.0, 0.0]
    assert list(_wdl_from_q_d(-1.0, 0.0)) == [0.0, 0.0, 1.0]
    assert list(_wdl_from_q_d(0.0, 1.0)) == [0.0, 1.0, 0.0]
    blend = _wdl_from_q_d(0.4, 0.2)
    assert blend[0] == pytest.approx(0.6)
    assert blend[2] == pytest.approx(0.2)


def test_row_takes_outcome_for_wdl_and_lc0_best_q_for_search_wdl() -> None:
    """⚑ The value-target decision, pinned. ``sf_wdl`` stays ABSENT: writing
    lc0's search estimate into a Stockfish-labelled field is exactly the
    silent-mislabel failure this converter exists to avoid."""
    board = chess.Board()
    rec = parse_v6_record(make_v6_record(
        board, result_q=-1.0, result_d=0.0, best_q=0.5, best_d=0.4, plies_left=90.0,
    ))
    stats = VerifyStats()
    rows = convert_game("t", [rec], stats, _options(), game_id=7, collect=True)
    row = rows[0]
    assert row.wdl_target == 2  # side to move lost
    assert row.sf_wdl is None
    assert row.search_wdl is not None
    assert list(np.round(row.search_wdl, 4)) == [0.55, 0.4, 0.05]
    assert row.moves_left == pytest.approx(90.0 / 450.0)
    assert row.game_id == 7


def test_row_carries_the_production_encoding_identity() -> None:
    rec = parse_v6_record(make_v6_record(chess.Board()))
    stats = VerifyStats()
    rows = convert_game("t", [rec], stats, _options(), game_id=0, collect=True)
    row = rows[0]
    assert row.x.shape == (175, 8, 8)
    assert row.policy_target.shape == (1858,)
    assert row.input_history_encoding == "lc0_root_legacy_meta"
    assert row.history_rep_fix is True


# ── statistics ─────────────────────────────────────────────────────────────────

def test_policy_shape_stats_read_the_legal_entries_only() -> None:
    board = chess.Board()
    weights = {move.uci(): 0.0 for move in board.legal_moves}
    weights["e2e4"] = 1.0
    rec = parse_v6_record(make_v6_record(board, probabilities=weights, played_uci="e2e4"))
    shape = policy_shape_stats([rec])
    assert shape.records == 1
    assert shape.max_prob_p50 == pytest.approx(1.0)
    assert shape.one_hot_frac == 1.0
    assert shape.support_p50 == 1.0
    assert shape.full_support_frac == 0.0


# ── refusal to write ───────────────────────────────────────────────────────────

def test_convert_refuses_to_write_a_shard_while_a_gate_is_failing(tmp_path: Path) -> None:
    from scripts.lc0_data_to_rows import _write_shard

    rec = parse_v6_record(make_v6_record(chess.Board()))
    stats = VerifyStats()
    rows = convert_game("t", [rec], stats, _options(), game_id=0, collect=True)
    stats.planes_mismatch = 1
    stats.first_divergence = "synthetic"
    with pytest.raises(RuntimeError, match="refusing to write rows"):
        _write_shard(tmp_path, 0, rows, _options(), stats)


def test_reference_planes_carry_lc0s_own_aux_block() -> None:
    board = chess.Board()
    board.halfmove_clock = 7
    rec = parse_v6_record(make_v6_record(board))
    ref = record_reference_planes(rec)
    assert ref.shape == (112, 8, 8)
    assert float(ref[104].max()) == 1.0   # us queenside
    assert float(ref[108].max()) == 0.0   # white to move
    assert float(ref[109].max()) == 7.0   # raw rule50
    assert float(ref[111].min()) == 1.0   # ones plane


# ── the checks that had no test at all ─────────────────────────────────────────

def test_gather_cross_check_can_actually_disagree(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚑ Replacing the comparison with ``x != x`` left 31/31 passing: the third
    headline rate was a real-data measurement whose guard nothing exercised.

    Permuting what the VECTORISED path returns must make the scalar per-move
    reference contradict it. If the comparison ever degenerates into comparing
    the gather with itself, no permutation can be seen and this fails.
    """
    import scripts.lc0_data_to_rows as module
    from chess_anti_engine.moves.leela_index import compact_index_for_move

    board = chess.Board()
    # Two slots the fixture's legal moves actually occupy — permuting unreachable
    # slots would be invisible and would make this test vacuous.
    a = compact_index_for_move(board, chess.Move.from_uci("e2e4"))
    b = compact_index_for_move(board, chess.Move.from_uci("d2d4"))
    real = module.leela_gather_indices

    def permuted(*args: object, **kwargs: object) -> np.ndarray:
        out = real(*args, **kwargs).copy()  # pyright: ignore[reportArgumentType]
        out[:, [a, b]] = out[:, [b, a]]
        return out

    monkeypatch.setattr(module, "leela_gather_indices", permuted)
    stats = VerifyStats()
    convert_game("t", [parse_v6_record(make_v6_record(board))], stats, _options(),
                 game_id=0, collect=False)
    assert stats.gather_disagrees == 1
    assert stats.gather_agrees == 0
    assert stats.first_divergence is not None
    assert "gather map != per-move reference" in stats.first_divergence


def test_gather_cross_check_passes_unpermuted() -> None:
    """The negative control for the test above: without the permutation the
    same fixture must agree, so the assertion above is about the permutation
    and not about the fixture."""
    stats = VerifyStats()
    convert_game("t", [parse_v6_record(make_v6_record(chess.Board()))], stats, _options(),
                 game_id=0, collect=False)
    assert stats.gather_disagrees == 0
    assert stats.gather_agrees == stats.attempts


def _aux_pair(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    from chess_anti_engine.encoding.encode import encode_position

    production = encode_position(
        board, add_features=True, input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    lc0_root = encode_position(
        board, add_features=False, input_history_encoding="lc0_root",
    )
    return production, lc0_root


def test_production_aux_check_rejects_a_shared_plane_difference() -> None:
    """The two encodings must agree EVERYWHERE except planes 109/110."""
    board = chess.Board()
    production, lc0_root = _aux_pair(board)
    assert _production_aux_consistent(production, lc0_root, board)
    production[0][0][0] = 1.0 - production[0][0][0]
    assert not _production_aux_consistent(production, lc0_root, board)


def test_production_aux_check_rejects_a_wrong_rule50_rescale() -> None:
    """Plane 109 is the one plane that is allowed to DIFFER, which is exactly
    why its expected value has to be pinned rather than skipped."""
    board = chess.Board()
    board.halfmove_clock = 40
    production, lc0_root = _aux_pair(board)
    assert _production_aux_consistent(production, lc0_root, board)
    production[109][:, :] = 40.0  # lc0's raw scale instead of our /100 rescale
    assert not _production_aux_consistent(production, lc0_root, board)


def test_production_aux_check_rejects_a_wrong_en_passant_plane() -> None:
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    production, lc0_root = _aux_pair(board)
    assert _production_aux_consistent(production, lc0_root, board)
    production[110][:, :] = 0.0
    assert not _production_aux_consistent(production, lc0_root, board)


def test_en_passant_repair_requires_a_UNIQUE_witness() -> None:
    """⚑ The documented property is uniqueness, and accepting the first of
    several matches passed the old suite. The branch never fires on real data
    (``ep-repaired 0`` over 3,000 games), so a test is the only thing holding
    it up."""
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    rec = parse_v6_record(make_v6_record(board))
    naive = board_from_record(rec)

    # A record whose support is the EP-free move set: no candidate should match
    # a legal set that already agrees, so the untouched board comes straight back.
    plain = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    plain_rec = parse_v6_record(make_v6_record(plain))
    plain_repaired = repair_en_passant(board_from_record(plain_rec), plain_rec)
    assert plain_repaired is not None
    assert plain_repaired.ep_square is None

    # A support set no ep square can reproduce must be refused, not guessed at.
    probs = np.frombuffer(
        make_v6_record(board), dtype="<f4", count=LEELA_POLICY_SIZE, offset=8,
    ).copy()
    probs[probs >= 0.0] = -1.0
    probs[7] = 1.0
    forged = make_v6_record(board)
    forged = forged[:8] + probs.tobytes() + forged[8 + LEELA_POLICY_SIZE * 4:]
    assert repair_en_passant(naive, parse_v6_record(forged)) is None


# ── counter arithmetic and the verdict ─────────────────────────────────────────

def test_counter_consistency_catches_a_numerator_over_its_denominator() -> None:
    """⚑ Closes the CLASS. A rate of 110.4651% was reported once because a
    row-scoped numerator was divided by a rolled-back denominator; this makes
    any such pairing fail the run instead of printing."""
    stats = VerifyStats(games=1, attempts=10, rows=5)
    assert stats.consistency_problems() == []
    stats.best_idx_is_argmax = 6
    assert any("best_idx_is_argmax" in p for p in stats.consistency_problems())
    stats.best_idx_is_argmax = 5
    stats.planes_exact = 11
    assert any("planes_exact" in p for p in stats.consistency_problems())


def test_drop_ledger_must_sum_to_the_drop_count() -> None:
    stats = VerifyStats(games=1, games_dropped=1)
    assert any("drop_reasons sums to 0" in p for p in stats.consistency_problems())
    stats.note("gate failure: planes_bit_exact")
    assert stats.consistency_problems() == []


def test_a_gate_failure_records_BOTH_a_divergence_and_a_drop_reason() -> None:
    clean = make_game(["e2e4", "e7e5", "g1f3", "b8c6"])
    flipped = bytearray(clean)
    flipped[2 * V6_RECORD_BYTES + _OFF_CASTLING + 1] = 0
    stats = VerifyStats()
    convert_game("t", parse_v6_stream(bytes(flipped)), stats, _options(),
                 game_id=0, collect=False)
    assert stats.first_divergence is not None
    assert stats.drop_reasons == {"gate failure: planes_bit_exact": 1}
    assert stats.consistency_problems() == []


def test_an_abstention_is_INCONCLUSIVE_and_never_a_pass() -> None:
    """⚑ The go/no-go used to map an abstention to PASS with exit 0, which this
    module's own docstring warns against."""
    small = VerifyStats(games=10, attempts=5, rows=5)
    verdict, code = small.overall_verdict()
    assert verdict.startswith("INCONCLUSIVE")
    assert code == 2

    big = VerifyStats(games=CHESS960_MIN_GAMES, attempts=5, rows=5)
    big.drop_reasons["chess960"] = int(0.037 * CHESS960_MIN_GAMES)
    big.games_dropped = big.drop_reasons["chess960"]
    assert big.overall_verdict() == ("PASS", 0)

    broken = VerifyStats(games=CHESS960_MIN_GAMES, attempts=5, rows=5, planes_mismatch=1)
    assert broken.overall_verdict()[1] == 1


def test_chess960_band_does_not_fail_a_clean_run_at_the_minimum_sample() -> None:
    """⚑ At the true 3.73% rate the old ``low=0.01`` with ``min_games=150``
    failed roughly 1 clean run in 40. The band must pass the low tail."""
    stats = VerifyStats(games=CHESS960_MIN_GAMES, attempts=1, rows=1)
    for count in (2, 6, 15, 30):
        stats.drop_reasons["chess960"] = count
        assert stats.chess960_problem() is None, count
    stats.drop_reasons["chess960"] = 0
    assert stats.chess960_problem() is not None


# ── the run-config gate ────────────────────────────────────────────────────────

def test_run_config_gate_refuses_the_production_value_blend_on_sf_less_shards() -> None:
    """⚑ The requirement that used to be prose. `losses.py` redirects the SF
    share onto the raw game outcome when `has_sf_wdl` is 0, so this is the
    difference between a positive control and a value-collapse run."""
    production_like: dict[str, object] = {"sf_wdl_frac": 0.50, "sf_wdl_frac_floor": 0.45, "search_wdl_frac": 0.20}
    problems = run_config_problems(production_like, shards_have_sf_wdl=False)
    assert len(problems) == 2
    assert all("set sf_wdl_frac" in p or "set sf_wdl_frac_floor" in p for p in problems)


def test_run_config_gate_accepts_the_prescribed_override() -> None:
    prescribed: dict[str, object] = {"sf_wdl_frac": 0.0, "sf_wdl_frac_floor": 0.0, "search_wdl_frac": 0.50}
    assert run_config_problems(prescribed, shards_have_sf_wdl=False) == []


def test_run_config_gate_refuses_a_target_with_nothing_but_the_outcome() -> None:
    """Zeroing the SF share is necessary but not sufficient: with
    `search_wdl_frac` also 0 the blend is 100% game outcome, which is the same
    collapse by a different route."""
    collapsed: dict[str, object] = {
        "sf_wdl_frac": 0.0, "sf_wdl_frac_floor": 0.0, "search_wdl_frac": 0.0,
    }
    problems = run_config_problems(collapsed, shards_have_sf_wdl=False)
    assert any("collapses onto the game outcome" in p for p in problems)


def test_run_config_gate_stays_out_of_the_way_when_shards_DO_carry_sf() -> None:
    """Negative control: on our own selfplay shards the production blend is
    correct and this must not fire."""
    production_like: dict[str, object] = {"sf_wdl_frac": 0.50, "sf_wdl_frac_floor": 0.45, "search_wdl_frac": 0.20}
    assert run_config_problems(production_like, shards_have_sf_wdl=True) == []


def test_the_rows_this_converter_writes_really_carry_no_sf_label(tmp_path: Path) -> None:
    """Ties the gate to the artifact: the refusal above is only meaningful if
    the shards genuinely lack `has_sf_wdl`."""
    from chess_anti_engine.replay.shard import ShardMeta, local_shard_path
    from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays

    stats = VerifyStats()
    rows = convert_game("t", parse_v6_stream(make_game(["e2e4", "e7e5"])), stats,
                        _options(), game_id=0, collect=True)
    assert rows
    assert all(row.sf_wdl is None for row in rows)
    save_local_shard_arrays(
        local_shard_path(tmp_path, 0), arrs=samples_to_arrays(rows),
        meta=ShardMeta(policy_encoding="lc0_1858", policy_size=1858),
    )
    assert shard_dir_has_sf_wdl(tmp_path) is False


def test_at_most_one_en_passant_witness_is_REACHABLE() -> None:
    """⚑ EQUIVALENT-MUTANT PROOF, not a coverage patch.

    A mutant weakening ``len(matches) == 1`` to "take the first" survives the
    suite, and no fixture can kill it — because the ambiguous state is
    unreachable, not untested. Whenever the early return does NOT fire, the
    target legal set contains an en-passant capture, and distinct ep squares
    add distinct capture moves, so exactly one candidate can reproduce it.

    This brute-forces that claim over every ep file on positions chosen to
    stress it (two pawns able to take the same square, edge files, both
    colours). If python-chess ever makes two witnesses possible, this fails and
    the "unreachable" comment in ``repair_en_passant`` stops being true.
    """
    fens = (
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "8/8/8/pP6/8/8/8/4K1k1 w - a6 0 1",
        "4k3/8/8/PpP5/8/8/8/4K3 w - b6 0 1",
        "4k3/8/8/1PpP4/8/8/8/4K3 w - c6 0 1",
        "4k3/8/8/8/2pP4/8/8/4K3 b - d3 0 1",
    )
    exercised = 0
    for fen in fens:
        truth = chess.Board(fen)
        want = set(board_leela_slots(truth))
        base = truth.copy(stack=False)
        base.ep_square = None
        if set(board_leela_slots(base)) == want:
            continue  # early return fires; the ambiguous branch is not entered
        rank = 5 if truth.turn == chess.WHITE else 2
        matches = []
        for file_index in range(8):
            candidate = base.copy(stack=False)
            candidate.ep_square = chess.square(file_index, rank)
            if set(board_leela_slots(candidate)) == want:
                matches.append(candidate.ep_square)
        assert matches == [truth.ep_square], f"{fen}: witnesses {matches}"
        exercised += 1
    assert exercised >= 5, "fixtures stopped reaching the branch this proves"
