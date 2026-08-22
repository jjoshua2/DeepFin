"""Our stored replay tensor -> the 112-plane input an LC0/BT4 net reads.

The conversion is one plane-slice plus two metadata fixups, and both fixups
fail SILENTLY if dropped: a rule50 left at ``clock/100`` is a valid float that
tells BT4 a 77-ply-stale position is 0.77 plies old, and our en-passant file
left in plane 110 is a valid plane that LC0 reads as an unused movecount.
Neither raises, so each is pinned by an equivalence a wrong version cannot
satisfy.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import LC0_FULL, x_to_lc0_planes
from scripts.bt4_policy_dump import board_from_stored_x, shard_castling_rights

# Positions chosen so the two fixups are actually exercised: a large halfmove
# clock, a live en-passant square, and every castling-rights shape.
CASES = (
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 77 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 99 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b Qk - 100 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
    chess.STARTING_FEN,
)


@pytest.mark.parametrize("fen", CASES)
def test_conversion_reproduces_the_canonical_lc0_encoding(fen: str) -> None:
    """The load-bearing equivalence.

    ``lc0_root`` IS the LC0-canonical layout, so converting a
    ``lc0_root_legacy_meta`` tensor must land on exactly what ``lc0_root``
    would have written for the same board. This is what a dropped rule50
    rescale or a leaked en-passant plane breaks.
    """
    board = chess.Board(fen)
    stored = encode_position(board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    want = encode_position(board, add_features=False,
                           input_history_encoding="lc0_root")
    got = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    assert got.shape == (LC0_FULL.num_planes, 8, 8)
    assert np.array_equal(got, want)


def test_rule50_is_rescaled_to_raw_plies() -> None:
    """Pinned as a VALUE, not just via the equivalence: 77 plies must arrive as
    77.0, because LC0 nets are trained on the raw count."""
    base = LC0_FULL.root_metadata_base
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 77 1")
    stored = encode_position(board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    assert float(stored[base + 5][0, 0]) == pytest.approx(0.77)  # what we store
    got = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    assert float(got[base + 5][0, 0]) == 77.0


def test_en_passant_plane_is_dropped() -> None:
    """LC0's classical input has no EP plane; ours packs the EP file into 110."""
    base = LC0_FULL.root_metadata_base
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
    stored = encode_position(board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    assert stored[base + 6].any(), "this position should carry an EP file plane"
    got = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    assert not got[base + 6].any()
    assert got[LC0_FULL.ones_plane].all()


def test_history_planes_pass_through_untouched() -> None:
    """History is the entire reason for this path -- it must not be rebuilt."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        board.push(chess.Move.from_uci(uci))
    stored = encode_position(board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    got = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    assert np.array_equal(got[:LC0_FULL.root_metadata_base],
                          stored[:LC0_FULL.root_metadata_base])
    # and the older slots are genuinely populated, or the assertion above is vacuous
    assert stored[13:26].any()
    assert stored[78:91].any()


def test_batched_and_single_agree() -> None:
    boards = [chess.Board(f) for f in CASES]
    stored = np.stack([
        encode_position(b, add_features=True,
                        input_history_encoding="lc0_root_legacy_meta")
        for b in boards
    ])
    batch = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    assert batch.shape == (len(boards), LC0_FULL.num_planes, 8, 8)
    for i in range(len(boards)):
        single = x_to_lc0_planes(stored[i], input_history_encoding="lc0_root_legacy_meta")
        assert np.array_equal(batch[i], single)


def test_source_tensor_is_not_mutated() -> None:
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
    stored = encode_position(board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    before = stored.copy()
    x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    assert np.array_equal(stored, before)


def test_legacy_layout_is_refused() -> None:
    """The `legacy` layout APPENDS its repetition planes instead of interleaving
    them, so planes 0..103 are not LC0's history block. Converting it would
    produce a plausible tensor that is wrong everywhere."""
    board = chess.Board()
    stored = encode_position(board, add_features=True,
                             input_history_encoding="legacy")
    with pytest.raises(ValueError, match="LC0-root layout"):
        x_to_lc0_planes(stored, input_history_encoding="legacy")


@pytest.mark.parametrize("fen", CASES)
def test_board_decodes_back_out_of_the_stored_tensor(fen: str) -> None:
    """Shards hold planes, not FENs, so the dump's shard path has to recover the
    position from the tensor. Pieces, castling rights and the side-to-move frame
    must all come back."""
    board = chess.Board(fen)
    stored = encode_position(board, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta")
    planes = x_to_lc0_planes(stored, input_history_encoding="lc0_root_legacy_meta")
    got = board_from_stored_x(stored, planes)
    # The decode is side-to-move canonical, so compare against the oriented board.
    want = board if board.turn == chess.WHITE else board.mirror()
    assert got.board_fen() == want.board_fen()
    assert got.turn == chess.WHITE
    assert got.castling_rights == want.castling_rights
    assert {m.uci() for m in got.legal_moves} == {m.uci() for m in want.legal_moves}


def test_castling_planes_are_read_in_lc0_order() -> None:
    """(us_Q, us_K, them_Q, them_K). Swapping the pair is the silent failure:
    every plane is still a valid 0/1 and only the rights come out mirrored."""
    base = LC0_FULL.root_metadata_base
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1")
    planes = x_to_lc0_planes(
        encode_position(board, add_features=True,
                        input_history_encoding="lc0_root_legacy_meta"),
        input_history_encoding="lc0_root_legacy_meta",
    )
    assert float(planes[base + 1][0, 0]) == 1.0  # us_K
    assert float(planes[base + 0][0, 0]) == 0.0  # us_Q
    assert shard_castling_rights(planes, board) == chess.BB_H1
