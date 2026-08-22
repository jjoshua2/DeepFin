"""The BT4 policy dump's gather + renormalisation, without an ONNX session.

The pieces under test are the ones that can be wrong SILENTLY: which 1858 slot
each legal move reads, and whether the emitted dict is a distribution over the
legal moves. A synthetic policy row whose value at slot ``i`` IS ``i`` makes a
mis-gather visible as a wrong number rather than a wrong-looking one.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX
from scripts.bt4_policy_dump import (
    castling_probability,
    entropy_nats,
    iter_rows,
    legal_move_policy,
    load_done_keys,
    remap_provenance,
)

# Both castles legal for white, plus a 7th-rank pawn and pieces that can slide
# onto the back rank -- every family the static remap got wrong, in one board.
KITCHEN_SINK = "r3k2r/P6P/8/8/8/8/8/R3K2R w KQkq - 0 1"


def _identity_row() -> np.ndarray:
    """policy[i] == i, so a gathered logit names the slot it came from."""
    return np.arange(COMPACT_POLICY_SIZE, dtype=np.float32)


def test_each_legal_move_reads_its_own_leela_slot() -> None:
    """Softmax is strictly monotone, so with policy[i] == i the probability
    ranking must be the ranking of the slots the moves SHOULD have read."""
    board = chess.Board(KITCHEN_SINK)
    ucis, probs = legal_move_policy(board, _identity_row())
    assert len(ucis) == board.legal_moves.count()
    want = [
        LC0_1858_UCI_TO_IDX[_expected_leela_uci(board, chess.Move.from_uci(u))]
        for u in ucis
    ]
    assert len(set(want)) == len(want), "two legal moves cannot share a slot"
    by_prob = [u for _p, u in sorted(zip(probs.tolist(), ucis, strict=True))]
    by_slot = [u for _s, u in sorted(zip(want, ucis, strict=True))]
    assert by_prob == by_slot


def _expected_leela_uci(board: chess.Board, move: chess.Move) -> str:
    """LC0's spelling of ``move``, written out here rather than imported.

    A test that asked the remap what the remap thinks would pass with the
    remap broken; this restates the two conventions independently.
    """
    to_sq = move.to_square
    if board.is_castling(move):
        back = chess.BB_RANK_1 if board.turn == chess.WHITE else chess.BB_RANK_8
        rooks = board.castling_rights & back
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        to_sq = chess.msb(rooks) if kingside else chess.lsb(rooks)
    flip = board.turn == chess.BLACK
    f = chess.square_mirror(move.from_square) if flip else move.from_square
    t = chess.square_mirror(to_sq) if flip else to_sq
    suffix = {chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b", chess.KNIGHT: ""}
    return (chess.square_name(f) + chess.square_name(t)
            + ("" if move.promotion is None else suffix[int(move.promotion)]))


def test_castling_reads_the_king_takes_rook_slot_not_the_slide() -> None:
    """The fingerprint of the fix: O-O must read ``e1h1``, not ``e1g1``.

    With the static remap this gathered LC0's ordinary ``e1g1`` slide logit --
    a real number for an unrelated move, which is why the defect never raised.
    """
    board = chess.Board(KITCHEN_SINK)
    row = np.full((COMPACT_POLICY_SIZE,), -20.0, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e1h1"]] = 10.0  # the castling slot
    row[LC0_1858_UCI_TO_IDX["e1g1"]] = -20.0  # the decoy slide slot
    ucis, probs = legal_move_policy(board, row)
    best = ucis[int(np.argmax(probs))]
    assert best == "e1g1", "O-O should dominate when its LC0 slot holds the mass"
    assert probs.max() > 0.9
    castles = castling_probability(board, ucis, probs)
    assert set(castles) == {"e1g1", "e1c1"}
    assert castles["e1g1"] > 0.9


def test_black_to_move_moves_are_oriented_before_lookup() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    row = np.full((COMPACT_POLICY_SIZE,), -20.0, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e1h1"]] = 10.0  # oriented: black's O-O
    ucis, probs = legal_move_policy(board, row)
    assert ucis[int(np.argmax(probs))] == "e8g8"


def test_promotion_pieces_do_not_collide() -> None:
    """=Q/=R/=B have their own slots and =N aliases the bare entry; all four
    must come back as four DIFFERENT probabilities."""
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    ucis, probs = legal_move_policy(board, _identity_row())
    promos = {u: p for u, p in zip(ucis, probs.tolist(), strict=True) if len(u) == 5}
    assert set(promos) == {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}
    assert len(set(promos.values())) == 4


def test_output_is_a_distribution_over_legal_moves_only() -> None:
    for fen in (KITCHEN_SINK, chess.STARTING_FEN,
                "8/8/6r1/1np5/p2k4/P7/8/2K5 w - - 0 1"):
        board = chess.Board(fen)
        ucis, probs = legal_move_policy(board, _identity_row())
        assert len(probs) == board.legal_moves.count()
        assert probs.min() > 0.0
        assert float(probs.sum()) == pytest.approx(1.0)
        assert chess.Move.from_uci(ucis[int(np.argmax(probs))]) in board.legal_moves


def test_masked_slots_do_not_poison_the_softmax() -> None:
    board = chess.Board(chess.STARTING_FEN)
    row = np.full((COMPACT_POLICY_SIZE,), -np.inf, dtype=np.float32)
    row[LC0_1858_UCI_TO_IDX["e2e4"]] = 1.0
    ucis, probs = legal_move_policy(board, row)
    assert np.isfinite(probs).all()
    assert ucis[int(np.argmax(probs))] == "e2e4"
    assert float(probs.sum()) == pytest.approx(1.0)


def test_terminal_position_yields_no_moves() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # checkmate
    ucis, probs = legal_move_policy(board, _identity_row())
    assert ucis == []
    assert probs.shape == (0,)


def test_entropy_matches_a_hand_computed_uniform() -> None:
    assert entropy_nats(np.full(4, 0.25)) == pytest.approx(np.log(4.0))
    assert entropy_nats(np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_iter_rows_keeps_the_key_but_builds_from_the_full_fen(tmp_path: Path) -> None:
    """rule50 is plane 109 and the 4-field key does not carry it, so the board
    must come from `fen` while the output stays keyed by `key`."""
    key = "8/8/6r1/1np5/p2k4/P7/8/2K5 w - -"
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        json.dumps({"key": key, "fen": key + " 89 1"}) + "\n"
        + json.dumps({"fen": chess.STARTING_FEN}) + "\n",
        encoding="utf-8",
    )
    assert list(iter_rows(jsonl)) == [
        (key, key + " 89 1"), (chess.STARTING_FEN, chess.STARTING_FEN),
    ]
    assert chess.Board(key + " 89 1").halfmove_clock == 89
    assert chess.Board(key).halfmove_clock == 0  # what keying off `key` would feed
    plain = tmp_path / "rows.txt"
    plain.write_text(f"{chess.STARTING_FEN}\n\n{chess.STARTING_FEN}\n", encoding="utf-8")
    assert list(iter_rows(plain)) == [(chess.STARTING_FEN, chess.STARTING_FEN)] * 2


def test_resume_ignores_a_torn_trailing_line(tmp_path: Path) -> None:
    out = tmp_path / "dump.jsonl"
    out.write_text(
        json.dumps({"record": "header"}) + "\n"
        + json.dumps({"key": "a", "n_legal": 1, "policy": {}}) + "\n"
        + '{"key": "b", "n_leg',  # killed mid-write
        encoding="utf-8",
    )
    assert load_done_keys(out) == {"a"}


def test_provenance_pins_the_remap_sources() -> None:
    prov = remap_provenance()
    assert set(prov["blobs"]) == {
        "chess_anti_engine/moves/leela_index.py",
        "chess_anti_engine/moves/lc0_1858_movestrs.py",
        "chess_anti_engine/encoding/lc0.py",
    }
    # A dump must be attributable even from a dirty tree, so the blob hashes
    # (not just the commit) have to be real.
    assert all(len(v) == 40 for v in prov["blobs"].values())
