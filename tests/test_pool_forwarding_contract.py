"""The pool→engine call contract, checkable WITHOUT a Stockfish binary.

⚑ WHY THIS FILE EXISTS SEPARATELY FROM `test_pool_searchmoves.py`.
That file proves the real thing — a real engine, reached through the pool,
returns the permitted move. But it is `skipif(SF_PATH is None)`, and **CI
installs no Stockfish**, so in CI every one of those tests is skipped. A
one-character typo in the kwarg name at the pool boundary would therefore be
invisible to CI: it type-checks clean if the call is built as a `**kwargs`
dict, passes the entire SF-free suite, and raises `TypeError` on the first
restricted query in production.

So these tests drive `StockfishPool._search` against a RECORDING fake engine.
They assert on the arguments the engine actually received — a value read, not a
presence check — and they run everywhere, including CI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class _RecordingEngine:
    """Accepts exactly StockfishUCI.search's keyword contract, and records it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.desynced = False

    def search(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None, fresh: bool = False,
        searchmoves: Any = None,
    ) -> StockfishResult:
        self.calls.append({
            "fen": fen, "nodes": nodes, "syzygy_path": syzygy_path,
            "fresh": fresh,
            "searchmoves": None if searchmoves is None else list(searchmoves),
        })
        return StockfishResult(
            bestmove_uci="e2e4", wdl=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            pvs=[],
        )


def _pool_with(engine: _RecordingEngine) -> StockfishPool:
    """A StockfishPool that spawns no process: __init__ is deliberately skipped."""
    pool = StockfishPool.__new__(StockfishPool)

    class _State:
        pass

    state = _State()
    state.engine = engine  # pyright: ignore[reportAttributeAccessIssue]
    pool._worker_state = state  # pyright: ignore[reportAttributeAccessIssue]
    return pool


def test_searchmoves_reaches_the_engine_verbatim() -> None:
    engine = _RecordingEngine()
    pool = _pool_with(engine)

    pool._search(FEN, None, None, False, ["e2e4", "d2d4"])

    assert len(engine.calls) == 1
    # The VALUE, not merely the key's presence.
    assert engine.calls[0]["searchmoves"] == ["e2e4", "d2d4"]


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("searchmoves", [None, [], ["e2e4"]])
def test_every_argument_combination_is_forwarded_faithfully(
    fresh: bool, searchmoves: list[str] | None,
) -> None:
    engine = _RecordingEngine()
    pool = _pool_with(engine)

    pool._search(FEN, 1234, "/tmp/syz", fresh, searchmoves)

    call = engine.calls[0]
    assert call["fen"] == FEN
    assert call["nodes"] == 1234
    assert call["syzygy_path"] == "/tmp/syz"
    assert call["fresh"] is fresh
    assert call["searchmoves"] == (
        None if searchmoves is None else list(searchmoves)
    )


def test_the_default_call_still_looks_like_the_production_label_path() -> None:
    """No searchmoves, no fresh — the shape selfplay has always sent."""
    engine = _RecordingEngine()
    pool = _pool_with(engine)

    pool._search(FEN, None, None, False)

    assert engine.calls[0] == {
        "fen": FEN, "nodes": None, "syzygy_path": None,
        "fresh": False, "searchmoves": None,
    }


def test_a_renamed_kwarg_at_the_boundary_is_caught_here() -> None:
    """MUTATION GUARD: an engine that does NOT accept `searchmoves` must fail.

    This is the CI-visible half of the acceptance test. If `_search` is ever
    rewritten to pass the argument under a different name, the real engine will
    reject it — and so does this fake, without needing a binary.
    """

    class _EngineWithoutSearchmoves:
        desynced = False

        def search(
            self, fen: str, *, nodes: int | None = None,
            syzygy_path: str | None = None, fresh: bool = False,
        ) -> StockfishResult:
            del fen, nodes, syzygy_path, fresh
            raise AssertionError("should not be reached")

    pool = _pool_with(_EngineWithoutSearchmoves())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="searchmoves"):
        pool._search(FEN, None, None, False, ["e2e4"])
