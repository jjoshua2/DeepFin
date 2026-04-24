"""Regression tests: verify optimized encoding paths produce identical output."""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.encode import encode_position

# ---------------------------------------------------------------------------
# Reference boards with varied state
# ---------------------------------------------------------------------------

_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    # Position with EP, castling rights mixed
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    # Near endgame
    "8/8/4k3/8/2p5/8/B2K4/8 w - - 0 1",
    # Black to move
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
]


def _make_boards() -> list[chess.Board]:
    boards = []
    for fen in _FENS:
        b = chess.Board(fen)
        boards.append(b)
    # Also make some boards with history (for repetition detection)
    b = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]:
        b.push(chess.Move.from_uci(uci))
    boards.append(b.copy())
    return boards


class TestEncodePositionSnapshot:
    """Snapshot: capture reference outputs, verify they don't change."""

    @pytest.fixture(autouse=True)
    def _boards(self):
        self.boards = _make_boards()

    def test_encode_position_deterministic(self):
        """Two calls with same board produce identical output."""
        for b in self.boards:
            a = encode_position(b, add_features=True)
            b2 = b.copy()
            c = encode_position(b2, add_features=True)
            np.testing.assert_array_equal(a, c, err_msg=f"Non-deterministic for {b.fen()}")

    def test_encode_shape_and_dtype(self):
        for b in self.boards:
            x = encode_position(b, add_features=True)
            assert x.shape == (146, 8, 8), f"Wrong shape for {b.fen()}"
            assert x.dtype == np.float32, f"Wrong dtype for {b.fen()}"

    def test_encode_no_features(self):
        for b in self.boards:
            x = encode_position(b, add_features=False)
            assert x.shape == (112, 8, 8)

    def test_encode_batch_matches_individual(self):
        """encode_positions_batch must match per-board encode_position."""
        try:
            from chess_anti_engine.encoding.encode import encode_positions_batch
        except ImportError:
            pytest.skip("encode_positions_batch not yet implemented")

        boards = self.boards
        batch = encode_positions_batch(boards, add_features=True)
        assert batch.shape == (len(boards), 146, 8, 8)
        assert batch.dtype == np.float32

        for i, b in enumerate(boards):
            ref = encode_position(b, add_features=True)
            np.testing.assert_array_equal(
                batch[i], ref,
                err_msg=f"Batch[{i}] mismatch for {b.fen()}",
            )

    def test_encode_batch_no_features_matches_individual(self):
        """encode_positions_batch(add_features=False) must match per-board encoding."""
        try:
            from chess_anti_engine.encoding.encode import encode_positions_batch
        except ImportError:
            pytest.skip("encode_positions_batch not yet implemented")

        boards = self.boards
        batch = encode_positions_batch(boards, add_features=False)
        assert batch.shape == (len(boards), 112, 8, 8)
        assert batch.dtype == np.float32

        for i, b in enumerate(boards):
            ref = encode_position(b, add_features=False)
            np.testing.assert_array_equal(
                batch[i], ref,
                err_msg=f"No-feature batch[{i}] mismatch for {b.fen()}",
            )

    def test_encode_fused_matches_original(self):
        """encode_position_fused must match encode_position exactly."""
        try:
            from chess_anti_engine.encoding.encode import encode_position_fused
        except ImportError:
            pytest.skip("encode_position_fused not yet implemented")

        for b in self.boards:
            ref = encode_position(b, add_features=True)
            fused = encode_position_fused(b)
            np.testing.assert_array_equal(
                fused, ref,
                err_msg=f"Fused mismatch for {b.fen()}",
            )

    def test_encode_into_buffer(self):
        """encode_position_into writes correct data into pre-allocated buffer."""
        try:
            from chess_anti_engine.encoding.encode import encode_position_into
        except ImportError:
            pytest.skip("encode_position_into not yet implemented")

        buf = np.zeros((146, 8, 8), dtype=np.float32)
        for b in self.boards:
            buf[:] = 0.0
            encode_position_into(b, buf)
            ref = encode_position(b, add_features=True)
            np.testing.assert_array_equal(
                buf, ref,
                err_msg=f"Into-buffer mismatch for {b.fen()}",
            )

    def test_encode_into_buffer_reuse_without_manual_clear(self):
        """encode_position_into must fully overwrite a reused buffer."""
        try:
            from chess_anti_engine.encoding.encode import encode_position_into
        except ImportError:
            pytest.skip("encode_position_into not yet implemented")

        buf = np.empty((146, 8, 8), dtype=np.float32)
        for b in self.boards:
            encode_position_into(b, buf)
            ref = encode_position(b, add_features=True)
            np.testing.assert_array_equal(
                buf, ref,
                err_msg=f"Into-buffer reuse mismatch for {b.fen()}",
            )

    def test_encode_into_buffer_reduced_lc0(self):
        """Reduced LC0 path should populate only its planes and clear the rest."""
        try:
            from chess_anti_engine.encoding.encode import encode_position_into
        except ImportError:
            pytest.skip("encode_position_into not yet implemented")

        buf = np.empty((146, 8, 8), dtype=np.float32)
        for b in self.boards:
            encode_position_into(b, buf, add_features=False, use_full_lc0=False)
            ref = encode_position(b, add_features=False, use_full_lc0=False)
            np.testing.assert_array_equal(
                buf[:ref.shape[0]], ref,
                err_msg=f"Reduced LC0 into-buffer mismatch for {b.fen()}",
            )
            assert not np.any(buf[ref.shape[0]:]), f"Trailing planes not cleared for {b.fen()}"

    def test_cboard_encode_146_matches_python(self):
        """CBoard.encode_146() must match encode_position for current-position planes."""
        try:
            from chess_anti_engine.encoding._lc0_ext import CBoard
        except ImportError:
            pytest.skip("CBoard C extension not available")

        from chess_anti_engine.encoding.cboard_encode import encode_cboard

        for b in self.boards:
            cb = CBoard.from_board(b)
            cb_enc = encode_cboard(cb)
            assert cb_enc.shape == (146, 8, 8), f"Wrong shape for {b.fen()}"
            assert cb_enc.dtype == np.float32

            # CBoard has no history, so compare only non-history planes:
            # planes 0-11 (current pieces), 96-111 (metadata), 112-145 (features)
            ref = encode_position(b, add_features=True)

            # Piece planes for current position (planes 0-11)
            np.testing.assert_allclose(
                cb_enc[:12], ref[:12], atol=1e-6,
                err_msg=f"CBoard piece planes mismatch for {b.fen()}",
            )
            # Metadata planes (96-102, 111)
            np.testing.assert_allclose(
                cb_enc[96:103], ref[96:103], atol=1e-6,
                err_msg=f"CBoard metadata planes mismatch for {b.fen()}",
            )
            np.testing.assert_allclose(
                cb_enc[111], ref[111], atol=1e-6,
                err_msg=f"CBoard bias plane mismatch for {b.fen()}",
            )
            # Feature planes (112-145): should match exactly
            np.testing.assert_allclose(
                cb_enc[112:], ref[112:], atol=1e-6,
                err_msg=f"CBoard feature planes mismatch for {b.fen()}",
            )
