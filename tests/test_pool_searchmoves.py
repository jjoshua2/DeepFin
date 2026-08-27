"""`searchmoves` must survive the StockfishPool boundary, proven END-TO-END.

⚑ The failure this file exists for is this repo's signature defect: an option
that is accepted at the API and then silently ignored on the path that matters.
`StockfishUCI.search` already had `searchmoves` and its own protocol-level
tests — but every one of those drives a FAKE engine and asserts on the `go`
line. A pool that dropped the argument would pass all of them.

So the assertions here are deliberately NOT "the pool forwards the kwarg".
They are "a real Stockfish, reached through the pool, returned the move we
permitted and not the one it actually wanted". That is the only observation
that distinguishes a wired knob from a decorative one.

The precondition (the unrestricted best move differs from the permitted move)
is MEASURED inside each test rather than hardcoded, so the test cannot quietly
decay into a tautology when a new Stockfish build changes its mind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import chess
import pytest

from chess_anti_engine.stockfish.pool import StockfishPool
from tests.stockfish_binary import find_stockfish

if TYPE_CHECKING:
    from collections.abc import Sequence

SF_PATH = find_stockfish()
pytestmark = pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")

# A quiet middlegame with a wide root (30+ legal moves), so restricting to one
# move is a real restriction and not a near-no-op.
WIDE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

NODES = 60_000


def _pool(**kw: Any) -> StockfishPool:
    assert SF_PATH is not None
    return StockfishPool(path=SF_PATH, nodes=NODES, num_workers=1, **kw)


def _bestmove(pool: StockfishPool, fen: str, **kw: Any) -> str:
    return str(pool.submit(fen, **kw).result().bestmove_uci)


def _a_legal_move_other_than(fen: str, exclude: str) -> str:
    """A deterministic legal move that is NOT `exclude`."""
    board = chess.Board(fen)
    for move in sorted(board.legal_moves, key=str):
        if str(move) != exclude:
            return str(move)
    raise AssertionError("position has only one legal move; unusable here")


def test_pooled_searchmoves_returns_the_permitted_move_not_the_wanted_one() -> None:
    """THE ACCEPTANCE TEST: the pool must be able to overrule the engine.

    Unrestricted, Stockfish tells us what it wants. Restricted to a single
    DIFFERENT legal move, it must return that move instead. If the pool drops
    the argument the engine searches the full root and hands back its own
    preference, which is exactly the move this test excluded.
    """
    pool = _pool()
    try:
        wanted = _bestmove(pool, WIDE_FEN)
        permitted = _a_legal_move_other_than(WIDE_FEN, wanted)
        # The precondition, asserted rather than assumed.
        assert permitted != wanted

        got = _bestmove(pool, WIDE_FEN, searchmoves=[permitted])

        assert got == permitted, (
            f"pool returned {got!r}; permitted only {permitted!r}. "
            f"{wanted!r} is what the engine wanted unrestricted — getting it "
            f"back means `searchmoves` never reached the process."
        )
    finally:
        pool.close()


def test_pooled_searchmoves_constrains_the_pv_too() -> None:
    """The PV must START with the permitted move, not merely the bestmove field.

    A bestmove-only assertion could in principle be satisfied while the engine
    reported a line it never searched.
    """
    pool = _pool(multipv=3)
    try:
        wanted = _bestmove(pool, WIDE_FEN)
        permitted = _a_legal_move_other_than(WIDE_FEN, wanted)

        res = pool.submit(WIDE_FEN, searchmoves=[permitted]).result()

        assert res.bestmove_uci == permitted
        # At most len(searchmoves) PV lines come back, and every one of them
        # must be inside the permitted set -- with one move permitted, that
        # means exactly one PV, headed by that move.
        assert res.pvs, "restricted search returned no PV lines at all"
        for rank, pv in enumerate(res.pvs, start=1):
            assert pv.move_uci == permitted, (
                f"PV rank {rank} heads with {pv.move_uci!r}, "
                f"outside the permitted set {{{permitted!r}}}"
            )
    finally:
        pool.close()


def test_unrestricted_pool_search_is_unchanged() -> None:
    """The production label path calls submit() with no searchmoves.

    It must keep getting a full-width search — the restriction must not leak
    into calls that never asked for it.
    """
    pool = _pool()
    try:
        a = _bestmove(pool, WIDE_FEN)
        b = _bestmove(pool, WIDE_FEN, searchmoves=None)
        c = _bestmove(pool, WIDE_FEN, searchmoves=[])
        assert a == b == c
    finally:
        pool.close()


def test_dropping_the_forwarding_at_the_pool_boundary_breaks_it() -> None:
    """MUTATION: re-create the pre-fix `_search` and prove the test above fails.

    This is the guard on the guard. Without it, the acceptance test could pass
    for a reason unrelated to the pool — e.g. if the restriction happened to
    coincide with the engine's own choice.
    """
    pool = _pool()
    try:
        wanted = _bestmove(pool, WIDE_FEN)
        permitted = _a_legal_move_other_than(WIDE_FEN, wanted)

        def _search_without_forwarding(
            fen: str,
            nodes: int | None,
            syzygy_path: str | None,
            fresh: bool,
            searchmoves: Sequence[str] | None = None,  # accepted, then IGNORED
        ) -> Any:
            del fresh, searchmoves  # the mutation: accepted, then dropped
            engine = pool._worker_state.engine
            return engine.search(fen, nodes=nodes, syzygy_path=syzygy_path)

        pool._search = _search_without_forwarding

        got = _bestmove(pool, WIDE_FEN, searchmoves=[permitted])

        assert got == wanted, (
            "the mutant still honoured searchmoves — the acceptance test is "
            "not actually sensitive to the pool-boundary forwarding"
        )
        assert got != permitted
    finally:
        pool.close()
