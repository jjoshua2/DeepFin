"""Optional history must reach the real UCI writer and survive pool custody.

Only fake transports/engines are used: no Stockfish process, model or CUDA.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, ClassVar, cast

import chess
import pytest

from chess_anti_engine.stockfish import pool as pool_module
from chess_anti_engine.stockfish.uci import (
    StockfishDesyncError,
    StockfishResult,
    StockfishTimeoutError,
    StockfishUCI,
)


ROOT = chess.STARTING_FEN
MOVES = ("g1f3", "g8f6", "f3g1", "f6g8") * 2


def _board(moves: tuple[str, ...] = MOVES) -> chess.Board:
    board = chess.Board()
    for token in moves:
        board.push_uci(token)
    return board


def _driver() -> tuple[Any, list[str]]:
    sf = cast(Any, object.__new__(StockfishUCI))
    commands: list[str] = []
    sf.nodes = 2000
    sf.multipv = 1
    sf.read_timeout_s = 1.0
    sf._lock = threading.Lock()
    sf._send = commands.append
    sf._new_game_locked = lambda: commands.append("ucinewgame")
    replies = iter([
        "info depth 3 score cp 10 wdl 500 400 100 nodes 17 pv c2c4\n",
        "bestmove c2c4\n",
    ])
    sf._readline_with_deadline = lambda _deadline: next(replies)
    return sf, commands


def test_history_reaches_position_writer_with_repetitions_and_limits() -> None:
    board = _board()
    assert board.is_repetition(3)
    sf, commands = _driver()
    result = sf.search(
        board.fen(), root_fen=ROOT, root_moves=MOVES,
        nodes=17, fresh=True, searchmoves=("c2c4",),
    )
    assert commands == [
        "ucinewgame",
        f"position fen {ROOT} moves {' '.join(MOVES)}",
        "go nodes 17 searchmoves c2c4",
    ]
    assert result.nodes == 17
    assert result.bestmove_uci == "c2c4"


@pytest.mark.parametrize("fen", [ROOT, "unparsed legacy FEN"])
def test_omitted_history_keeps_legacy_commands(fen: str) -> None:
    sf, commands = _driver()
    sf.search(fen)
    assert commands == [f"position fen {fen}", "go nodes 2000"]


def test_explicit_empty_history_keeps_root_and_depth_limit() -> None:
    sf, commands = _driver()
    sf.search(ROOT, root_fen=ROOT, root_moves=(), depth=3)
    assert commands == [f"position fen {ROOT}", "go depth 3"]


def test_history_accepts_live_default_fen_without_phantom_ep_square() -> None:
    board = _board(("e2e4",))
    assert board.ep_square is not None
    assert board.fen().split()[3] == "-"
    sf, commands = _driver()
    sf.search(board.fen(), root_fen=ROOT, root_moves=("e2e4",))
    assert commands[0] == f"position fen {ROOT} moves e2e4"


def test_history_root_is_canonicalized_before_protocol_write() -> None:
    sf, commands = _driver()
    sf.search(ROOT, root_fen=ROOT.replace(" w ", "\n w "), root_moves=())
    assert commands == [f"position fen {ROOT}", "go nodes 2000"]


@pytest.mark.parametrize("kwargs", [
    {"root_fen": ROOT},
    {"root_moves": ()},
    {"root_fen": ROOT, "root_moves": "e2e4"},
    {"root_fen": False, "root_moves": ()},
    {"root_fen": ROOT, "root_moves": (1,)},
    {"root_fen": ROOT, "root_moves": {"e2e4"}},
    {"root_fen": "not a FEN", "root_moves": ()},
    {"root_fen": "8/8/8/8/8/8/8/8 w - - 0 1", "root_moves": ()},
    {"root_fen": ROOT, "root_moves": ("e2e5",)},
    {"root_fen": ROOT, "root_moves": ("e2e4\ngo infinite",)},
    {"root_fen": ROOT, "root_moves": ("0000",)},
    {"root_fen": ROOT, "root_moves": ("e2e4",)},
])
def test_bad_history_refuses_before_any_protocol_io(kwargs: dict[str, Any]) -> None:
    sf, commands = _driver()
    with pytest.raises(ValueError, match="history"):
        sf.search(ROOT, fresh=True, **kwargs)
    assert commands == []
    assert not sf.desynced


def test_history_final_fen_comparison_includes_clocks() -> None:
    sf, commands = _driver()
    wrong = _board().fen().replace(" 8 5", " 0 1")
    with pytest.raises(ValueError, match="final FEN"):
        sf.search(wrong, root_fen=ROOT, root_moves=MOVES)
    assert commands == []


def test_history_search_timeout_still_poisons_the_stream() -> None:
    sf, commands = _driver()

    def timeout(_deadline: float) -> str:
        raise StockfishTimeoutError("fake transport expired")

    sf._readline_with_deadline = timeout
    with pytest.raises(StockfishTimeoutError):
        sf.search(_board().fen(), root_fen=ROOT, root_moves=MOVES, nodes=17)
    assert sf.desynced
    assert commands[-1] == "go nodes 17"
    previous = commands[:]
    with pytest.raises(StockfishDesyncError):
        sf.search(_board().fen(), root_fen=ROOT, root_moves=MOVES)
    assert commands == previous


class _CapturingEngine:
    instances: ClassVar[list[_CapturingEngine]] = []

    def __init__(self, *_args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.constructor_kwargs = kwargs
        self.desynced = False
        self.closed = False
        self.instances.append(self)

    def search(self, fen: str, **kwargs: Any) -> StockfishResult:
        self.calls.append((fen, kwargs))
        if self.desynced:
            raise StockfishDesyncError("fake poisoned stream")
        return StockfishResult(bestmove_uci="c2c4", wdl=None, pvs=[])

    def close(self) -> None:
        self.closed = True


class _QueuedExecutor:
    """Keep the actual pool worker call queued until caller mutation is done."""

    def __init__(self) -> None:
        self.call: tuple[Any, tuple[Any, ...]] | None = None

    def submit(self, function: Any, *args: Any) -> Future[StockfishResult]:
        self.call = function, args
        return Future()


def test_pool_validates_and_copies_history_before_queueing() -> None:
    pool = cast(Any, object.__new__(pool_module.StockfishPool))
    queued = _QueuedExecutor()
    pool._exec = queued
    moves = list(MOVES)
    pool.submit(_board().fen(), root_fen=ROOT, root_moves=moves, nodes=17)
    moves.clear()
    assert queued.call is not None
    function, args = queued.call
    engine = _CapturingEngine()
    pool._worker_state = threading.local()
    pool._worker_state.engine = engine
    function(*args)
    assert engine.calls == [(_board().fen(), {
        "nodes": 17, "syzygy_path": None, "fresh": False,
        "searchmoves": None, "root_fen": ROOT, "root_moves": MOVES,
    })]
    queued.call = None
    with pytest.raises(ValueError, match="illegal move"):
        pool.submit(ROOT, root_fen=ROOT, root_moves=("e2e5",))
    assert queued.call is None


def test_actual_pool_replacement_preserves_history_and_query_options(monkeypatch) -> None:
    _CapturingEngine.instances = []
    monkeypatch.setattr(pool_module, "StockfishUCI", _CapturingEngine)
    pool = pool_module.StockfishPool(
        path="unused", nodes=2000, num_workers=1, read_timeout_s=0.25,
    )
    first = _CapturingEngine.instances[0]
    first.desynced = True
    kwargs: dict[str, Any] = {
        "nodes": 17, "syzygy_path": "test-tables", "fresh": True,
        "searchmoves": ("c2c4",), "root_fen": ROOT, "root_moves": MOVES,
    }
    try:
        with pytest.raises(StockfishDesyncError):
            pool.submit(_board().fen(), **kwargs).result(timeout=1)
        assert pool.replacements == 1
        assert first.closed
        # Replacement is for later queries: the failed query was not retried.
        replacement = _CapturingEngine.instances[1]
        assert replacement.calls == []
        result = pool.submit(_board().fen(), **kwargs).result(timeout=1)
        assert result.bestmove_uci == "c2c4"
        assert replacement.calls == first.calls == [(_board().fen(), kwargs)]
        assert replacement.constructor_kwargs["read_timeout_s"] == 0.25
    finally:
        pool.close()
    assert all(engine.closed for engine in _CapturingEngine.instances)
