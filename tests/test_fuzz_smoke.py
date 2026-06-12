"""Smoke-run the CBoard differential fuzzer (scripts/fuzz_cboard_diff.py).

A handful of lockstep games asserting state parity — enough to catch gross
C-vs-Python divergence regressions in CI without fuzz-scale runtime. The
encode oracle stays off (see the test docstring). The real budgets live in
scripts/fuzz/run_fuzz.sh.
"""
from __future__ import annotations

from tests.script_loading import load_script_module

_MOD = load_script_module("fuzz_cboard_diff.py")


def test_differential_fuzz_state_parity_smoke() -> None:
    """State parity (legal moves, bitboards, clocks, terminal flags) is a hard
    invariant between the C board and python-chess — assert it over many plies.

    The encode-plane oracle is deliberately disabled here (``encode_every=0``):
    it surfaced a real, history-dependent repetition-plane divergence between
    the C (bounded hash-stack) and Python (full move-stack) encoders that is an
    open semantics question, not yet a settled invariant. Run the encode oracle
    explicitly via ``scripts/fuzz_cboard_diff.py --encode-every N`` to probe it.
    """
    failure = _MOD.run(games=8, seed=1234, max_plies=200, encode_every=0)
    assert failure is None, str(failure)


def test_root_encode_parity_smoke() -> None:
    """The encoders must agree at the root (no history) for production modes."""
    import chess
    import numpy as np

    from chess_anti_engine.encoding import encode_position
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard

    b = chess.Board()
    cb = CBoard.from_board(b)
    for mode in ("lc0_root", "lc0_root_legacy_meta"):
        c_planes = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
        py_planes = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
        assert np.array_equal(c_planes, py_planes), f"root encode diverged for {mode!r}"
