"""Regression: ponderhit must derive real-phase clock from OUR side,
not the opponent's (which is the side-to-move on the pre-predicted-move
board that ponder is searching).

Codex adversarial review flagged that with asymmetric clocks
(e.g. 5000ms vs 300000ms) the pre-fix code would cause the engine to
undersearch or flag when ponder transitioned to the real deadline.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import chess

from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdGo, GoArgs


def _make_engine() -> Engine:
    worker = MagicMock()
    worker.advance_root.return_value = True
    engine = Engine(worker=worker)
    # Swap the search-thread factory so _handle_go doesn't actually launch
    # a search — we only want to inspect what limits it computes.
    engine._search_thread = None  # type: ignore[attr-defined]
    engine._state_lock = threading.Lock()  # type: ignore[attr-defined]
    return engine


def test_ponderhit_real_limits_use_our_clock_not_opponent() -> None:
    """We're white, we expect opponent (black) to reply ...e5, and we
    ponder the position AFTER that predicted reply so it's us to move.

    Ponder search_board therefore has side-to-move = WHITE (us), with
    the popped move (...e5) = black's reply. When ponderhit fires and
    we push ...e5, the real board flips to black... wait, that's wrong.

    Correct setup: we search for opponent's reply, so:
      - search_board has BLACK to move (opponent),
      - popped = black's predicted reply,
      - real board after push has WHITE to move (us).
    Real-phase limits must use our (white) clock.
    """
    engine = _make_engine()

    # Mimic `position startpos moves e2e4` then `go ponder ...`
    # where our engine is pondering black's reply.
    engine._board = chess.Board()
    # After 1.e4, it's black to move. We're pondering their reply,
    # which means we project one opponent move ahead.
    # For the ponder protocol in the engine, _pending_moves is the
    # sequence ending with the predicted ponder move (their ...e5).
    engine._pending_fen = None
    engine._pending_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),  # predicted opponent reply
    ]

    # Asymmetric clocks: we (white) have only 300ms, opponent (black)
    # has 5000ms. Pre-fix code used opponent's clock → would budget
    # ~5000/30 = 166ms. Post-fix should budget ~300/30 = 10ms floor
    # (or at least much less than 166ms).
    args = GoArgs(
        ponder=True,
        wtime_ms=300,
        btime_ms=5000,
        winc_ms=0,
        binc_ms=0,
    )
    engine._handle_go(CmdGo(args=args))

    real_lim = engine._pending_real_limits  # type: ignore[attr-defined]
    assert real_lim is not None, "ponderhit should have pre-computed real limits"
    # Our side (white) after the ponder move is pushed.
    # Deadline should come from wtime (300), not btime (5000).
    assert real_lim.deadline_ms is not None
    # wtime=300 → ~300/30 + 0 = 10; btime=5000 → ~5000/30 + 0 = 166.
    # The fix should give us <100; the bug would give us >100.
    assert real_lim.deadline_ms < 100, (
        f"real-phase deadline {real_lim.deadline_ms}ms suggests the engine "
        f"picked black's 5000ms clock instead of white's 300ms (fix regression)"
    )


def test_ponderhit_real_limits_use_our_clock_as_black() -> None:
    """Mirror: we're black pondering white's reply after e4."""
    engine = _make_engine()

    engine._board = chess.Board()
    # Ponder is from black's perspective pondering white's reply.
    # Sequence: [white's e4, black's c5, white's predicted Nf3].
    engine._pending_fen = None
    engine._pending_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("c7c5"),
        chess.Move.from_uci("g1f3"),  # predicted white reply — we expect Nf3
    ]

    # Asymmetric the other way: our (black) clock is huge, white's is tiny.
    # Pre-fix would use white's 200ms; post-fix uses our 10000ms.
    args = GoArgs(
        ponder=True,
        wtime_ms=200,
        btime_ms=10000,
        winc_ms=0,
        binc_ms=0,
    )
    engine._handle_go(CmdGo(args=args))

    real_lim = engine._pending_real_limits  # type: ignore[attr-defined]
    assert real_lim is not None
    assert real_lim.deadline_ms is not None
    # btime=10000 → ~10000/30 + 0 = 333; wtime=200 → ~200/30 + 0 = 20 (floored).
    assert real_lim.deadline_ms > 100, (
        f"real-phase deadline {real_lim.deadline_ms}ms suggests the engine "
        f"picked white's 200ms clock instead of black's 10000ms (fix regression)"
    )
