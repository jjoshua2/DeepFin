from __future__ import annotations

from chess_anti_engine.stockfish.uci import _stockfish_child_nice


def test_stockfish_child_nice_uses_absolute_target() -> None:
    assert _stockfish_child_nice(current_nice=0, configured_nice=19) == 19
    assert _stockfish_child_nice(current_nice=15, configured_nice=19) == 19


def test_stockfish_child_nice_never_raises_priority() -> None:
    assert _stockfish_child_nice(current_nice=15, configured_nice=5) == 15


def test_stockfish_child_nice_clamps_config() -> None:
    assert _stockfish_child_nice(current_nice=0, configured_nice=50) == 19
    assert _stockfish_child_nice(current_nice=7, configured_nice=-5) == 7
