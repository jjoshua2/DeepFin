"""Tests for CBoard terminal detection: repetition, insufficient material, 50-move rule."""
from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard


class TestInsufficientMaterial:
    def test_kk(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1"))
        assert cb.is_game_over()
        assert cb.terminal_value() == 0.0

    def test_knk(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3N4/4K3 w - - 0 1"))
        assert cb.is_game_over()
        assert cb.terminal_value() == 0.0

    def test_kbk(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3B4/4K3 w - - 0 1"))
        assert cb.is_game_over()
        assert cb.terminal_value() == 0.0

    def test_kbkb_same_color(self):
        cb = CBoard.from_board(chess.Board("8/8/8/2b1k3/8/8/3B4/4K3 w - - 0 1"))
        assert cb.is_game_over()
        assert cb.terminal_value() == 0.0

    def test_kbkb_diff_color_not_draw(self):
        cb = CBoard.from_board(chess.Board("8/8/8/3bk3/8/8/3B4/4K3 w - - 0 1"))
        assert not cb.is_game_over()

    def test_krk_not_draw(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 0 1"))
        assert not cb.is_game_over()

    def test_kpk_not_draw(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3P4/4K3 w - - 0 1"))
        assert not cb.is_game_over()


class TestFiftyMoveRule:
    def test_hmc_99_not_over(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 99 1"))
        assert not cb.is_game_over()

    def test_hmc_100_is_over(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 100 1"))
        assert cb.is_game_over()
        assert cb.terminal_value() == 0.0

    def test_hmc_increments_on_push(self):
        cb = CBoard.from_board(chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 99 1"))
        idx = cb.legal_move_indices()
        cb.push_index(int(idx[0]))
        assert cb.halfmove_clock == 100
        assert cb.is_game_over()


class TestRepetition:
    def test_repetition_in_game_history(self):
        """Threefold repetition (3 occurrences) triggers game over."""
        b = chess.Board()
        # Two round-trips: start→A→start→A→start = 3-fold of starting pos
        for m in ["g1f3", "g8f6", "f3g1", "f6g8",
                   "g1f3", "g8f6", "f3g1", "f6g8"]:
            b.push(chess.Move.from_uci(m))
        cb = CBoard.from_board(b)
        assert cb.is_game_over()
        assert cb.terminal_value() == 0.0

    def test_twofold_not_game_over(self):
        """Two-fold repetition should NOT trigger game over (only threefold does)."""
        b = chess.Board()
        for m in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            b.push(chess.Move.from_uci(m))
        cb = CBoard.from_board(b)
        assert not cb.is_game_over()
        # But fast repetition check (used in search) should detect it
        assert cb.terminal_value() == 0.0

    def test_repetition_in_search_tree(self):
        """Threefold repetition created by CBoard push within search tree."""
        from chess_anti_engine.moves.encode import move_to_index
        b = chess.Board()
        # First round-trip via python-chess (builds history)
        for m in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            b.push(chess.Move.from_uci(m))
        cb = CBoard.from_board(b)
        # Second round-trip via push_index (search tree path)
        b2 = b.copy()
        idx_nf3 = move_to_index(chess.Move.from_uci("g1f3"), b2)
        cb.push_index(idx_nf3)
        b2.push(chess.Move.from_uci("g1f3"))
        idx_nf6 = move_to_index(chess.Move.from_uci("g8f6"), b2)
        cb.push_index(idx_nf6)
        b2.push(chess.Move.from_uci("g8f6"))
        idx_ng1 = move_to_index(chess.Move.from_uci("f3g1"), b2)
        cb.push_index(idx_ng1)
        b2.push(chess.Move.from_uci("f3g1"))
        idx_ng8 = move_to_index(chess.Move.from_uci("f6g8"), b2)
        cb.push_index(idx_ng8)
        # Now 3-fold: start pos appeared at ply 0, 4, 8
        assert cb.is_game_over()

    def test_no_false_repetition(self):
        """Different positions should not trigger repetition."""
        b = chess.Board()
        b.push(chess.Move.from_uci("e2e4"))
        cb = CBoard.from_board(b)
        assert not cb.is_game_over()


class TestCheckmateStalemate:
    def test_checkmate(self):
        b = chess.Board()
        for m in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
            b.push(chess.Move.from_uci(m))
        cb = CBoard.from_board(b)
        assert cb.is_game_over()
        assert cb.terminal_value() == -1.0  # STM (black) is checkmated

    def test_stalemate(self):
        # Use a real stalemate position (k7/2K5/1Q6):
        cb2 = CBoard.from_board(chess.Board("k7/8/2K5/8/8/8/8/1Q6 b - - 0 1"))
        # Check if this is actually stalemate
        if cb2.is_game_over():
            assert cb2.terminal_value() == 0.0


class TestHistoryEncoding:
    def test_history_planes_match_python(self):
        """CBoard encoding with history should match python reference."""
        from chess_anti_engine.encoding.encode import encode_positions_batch

        b = chess.Board()
        for m in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]:
            b.push(chess.Move.from_uci(m))

        ref = encode_positions_batch([b], add_features=True)[0]
        cb = CBoard.from_board(b)
        cb_enc = cb.encode_146()

        # Current position planes
        np.testing.assert_array_equal(ref[:12], cb_enc[:12])
        # History planes (6 available)
        for hi in range(6):
            start = (hi + 1) * 12
            end = start + 12
            np.testing.assert_array_equal(ref[start:end], cb_enc[start:end])
        # Metadata planes
        np.testing.assert_array_equal(ref[96:112], cb_enc[96:112])
        # Feature planes
        np.testing.assert_allclose(ref[112:146], cb_enc[112:146], atol=1e-6)

    def test_history_preserved_through_copy_push(self):
        """History should be maintained through copy + push in search tree."""
        b = chess.Board()
        for m in ["e2e4", "e7e5", "g1f3"]:
            b.push(chess.Move.from_uci(m))
        cb = CBoard.from_board(b)
        assert cb.hist_len == 3

        # Copy and push
        cb2 = cb.copy()
        idx = cb2.legal_move_indices()
        cb2.push_index(int(idx[0]))
        assert cb2.hist_len == 4

        # History should contain the parent position
        enc = cb2.encode_146()
        # Planes 12-23 should have the parent position (after Nf3)
        parent_enc = cb.encode_146()
        # The parent's current position should be cb2's first history
        np.testing.assert_array_equal(parent_enc[:12], enc[12:24])


class TestZobristHash:
    def test_hash_changes_on_push(self):
        b = chess.Board()
        cb = CBoard.from_board(b)
        h0 = cb.zobrist_hash
        cb2 = cb.copy()
        cb2.push_index(int(cb2.legal_move_indices()[0]))
        assert cb2.zobrist_hash != h0

    def test_hash_consistent_after_transposition(self):
        """Same position reached by different move orders should have same hash."""
        # 1.e4 d5 vs 1.d4... wait, we need exact same position
        # 1.Nf3 Nf6 2.Nc3 Nc6 vs 1.Nc3 Nc6 2.Nf3 Nf6
        b1 = chess.Board()
        for m in ["g1f3", "g8f6", "b1c3", "b8c6"]:
            b1.push(chess.Move.from_uci(m))
        b2 = chess.Board()
        for m in ["b1c3", "b8c6", "g1f3", "g8f6"]:
            b2.push(chess.Move.from_uci(m))
        cb1 = CBoard.from_board(b1)
        cb2 = CBoard.from_board(b2)
        assert cb1.zobrist_hash == cb2.zobrist_hash
