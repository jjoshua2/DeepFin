from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import chess

from chess_anti_engine.uci.engine import Engine, EngineOptions, emit_handshake
from chess_anti_engine.uci.protocol import CmdPosition, CmdSetOption
from chess_anti_engine.uci.search import SearchResult
from chess_anti_engine.uci.time_manager import SearchLimits


def test_run_one_phase_falls_back_to_legal_move_on_search_exception(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())
    result = engine._run_one_phase(limits, is_ponder=False, board=board)  # noqa: SLF001
    assert result.bestmove_uci in {m.uci() for m in board.legal_moves}
    assert result.ponder_uci is None
    assert "boom" in capsys.readouterr().out


def test_run_one_phase_fallback_respects_searchmoves(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=("g1f3",))
    result = engine._run_one_phase(limits, is_ponder=False, board=board)  # noqa: SLF001
    assert result.bestmove_uci == "g1f3"
    assert "boom" in capsys.readouterr().out


def test_run_one_phase_fallback_returns_null_when_searchmoves_have_no_legal_move(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=("a1a8",))
    result = engine._run_one_phase(limits, is_ponder=False, board=board)  # noqa: SLF001
    assert result.bestmove_uci == "0000"
    assert "boom" in capsys.readouterr().out


def test_run_one_phase_does_not_duplicate_worker_info(capsys) -> None:
    worker = MagicMock()

    def _run(*_args: Any, **kwargs: Any) -> SearchResult:
        kwargs["info_cb"](
            nodes=7,
            elapsed_ms=5,
            score_cp=12,
            pv=("e2e4",),
            tbhits=0,
            score_mate=None,
            multipv=2,
            wdl=(500, 0, 500),
        )
        return SearchResult(
            bestmove_uci="e2e4",
            ponder_uci=None,
            nodes=7,
            pv=("e2e4",),
            score_cp=12,
            tbhits=0,
        )

    worker.run = _run
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())

    engine._run_one_phase(limits, is_ponder=False, board=board)  # noqa: SLF001

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("info ")]
    assert len(lines) == 1
    assert "multipv 2" in lines[0]
    assert "wdl 500 0 500" in lines[0]


def test_run_one_phase_emits_final_info_when_worker_is_silent(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(
        return_value=SearchResult(
            bestmove_uci="e2e4",
            ponder_uci=None,
            nodes=3,
            pv=("e2e4",),
            score_cp=8,
            tbhits=0,
        ),
    )
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())

    engine._run_one_phase(limits, is_ponder=False, board=board)  # noqa: SLF001

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("info ")]
    assert len(lines) == 1
    assert "nodes 3" in lines[0]
    assert "pv e2e4" in lines[0]


def test_emit_bestmove_omits_ponder_when_option_false(capsys) -> None:
    engine = Engine(worker=MagicMock())
    engine._options.ponder = False  # noqa: SLF001
    result = SearchResult(
        bestmove_uci="e2e4", ponder_uci="e7e5",
        nodes=0, pv=(), score_cp=0, tbhits=0,
    )
    engine._emit_bestmove(result)  # noqa: SLF001
    out = capsys.readouterr().out
    assert "bestmove e2e4" in out
    assert "ponder" not in out


def test_emit_bestmove_includes_ponder_when_option_true(capsys) -> None:
    engine = Engine(worker=MagicMock())
    engine._options.ponder = True  # noqa: SLF001
    result = SearchResult(
        bestmove_uci="e2e4", ponder_uci="e7e5",
        nodes=0, pv=(), score_cp=0, tbhits=0,
    )
    engine._emit_bestmove(result)  # noqa: SLF001
    out = capsys.readouterr().out
    assert "bestmove e2e4 ponder e7e5" in out


def test_invalid_position_fen_clears_pending_state() -> None:
    engine = Engine(worker=MagicMock())
    engine._handle_position(CmdPosition(fen=None, moves=("e2e4",)))  # noqa: SLF001

    assert engine._pending_moves == [chess.Move.from_uci("e2e4")]  # noqa: SLF001

    engine._handle_position(CmdPosition(fen="not a valid fen", moves=()))  # noqa: SLF001

    assert engine._board == chess.Board()  # noqa: SLF001
    assert engine._pending_fen is None  # noqa: SLF001
    assert engine._pending_moves == []  # noqa: SLF001
    assert engine._applied_fen is None  # noqa: SLF001
    assert engine._applied_moves == ()  # noqa: SLF001
    assert engine._popped_ponder_move is None  # noqa: SLF001


def test_isready_does_not_stop_active_search(capsys) -> None:
    engine = Engine(worker=MagicMock())
    thread = MagicMock()
    thread.is_alive.return_value = True
    engine._search_thread = thread  # noqa: SLF001

    engine._handle_isready()  # noqa: SLF001

    assert capsys.readouterr().out == "readyok\n"
    assert not engine._stop_event.is_set()  # noqa: SLF001
    thread.join.assert_not_called()


def test_handshake_exposes_multi_gpu_pucv_option(capsys) -> None:
    emit_handshake(
        EngineOptions(
            use_multi_gpu_pucv=True,
            pucv_pending_mode="virtual-mean",
            eval_cache_entries=1234,
        ),
    )

    out = capsys.readouterr().out
    assert "option name UseMultiGpuPUCV type check default true" in out
    assert "option name PUCVPendingMode type combo default virtual-mean" in out
    assert "option name EvalCacheEntries type spin default 1234" in out


def test_use_multi_gpu_pucv_setoption_installs_factories(capsys) -> None:
    worker = MagicMock()
    factories: list[Callable[[], Any]] = [lambda: object(), lambda: object()]
    engine = Engine(
        worker=worker,
        rebuild_multi_gpu_pucv_factories=lambda max_batch, gather: factories,
    )

    engine._handle_setoption(  # noqa: SLF001
        CmdSetOption(name="UseMultiGpuPUCV", value="true"),
    )

    worker.install_multi_gpu_pucv.assert_called_once_with(
        factories, gather=512, as_factories=True,
    )
    assert engine._options.use_multi_gpu_pucv is True  # noqa: SLF001
    assert "UseMultiGpuPUCV on" in capsys.readouterr().out


def test_pucv_pending_mode_setoption_updates_worker(capsys) -> None:
    worker = MagicMock()
    engine = Engine(worker=worker)

    engine._handle_setoption(  # noqa: SLF001
        CmdSetOption(name="PUCVPendingMode", value="virtual-mean"),
    )

    worker.set_pucv_vloss_mode.assert_called_once_with(1)
    assert engine._options.pucv_pending_mode == "virtual-mean"  # noqa: SLF001
    assert "PUCVPendingMode set to virtual-mean" in capsys.readouterr().out


def test_eval_cache_entries_setoption_rebuilds_evaluator(capsys) -> None:
    worker = MagicMock()
    class _DummyEval:
        def evaluate_encoded(self, x: Any) -> tuple[Any, Any]:
            return x, x

    rebuilt = _DummyEval()
    calls: list[tuple[int, int]] = []

    def rebuild(max_batch: int, eval_cache_entries: int) -> _DummyEval:
        calls.append((max_batch, eval_cache_entries))
        return rebuilt

    engine = Engine(worker=worker, rebuild_evaluator=rebuild)

    engine._handle_setoption(  # noqa: SLF001
        CmdSetOption(name="EvalCacheEntries", value="256"),
    )

    assert calls == [(1024, 256)]
    assert engine._options.eval_cache_entries == 256  # noqa: SLF001
    worker.set_eval_cache_entries.assert_called_once_with(256)
    worker.set_evaluator.assert_called_once_with(rebuilt)
    assert "EvalCacheEntries set to 256" in capsys.readouterr().out


def test_eval_cache_entries_reinstalls_multi_gpu_pucv(capsys) -> None:
    worker = MagicMock()
    factories: list[Callable[[], Any]] = [lambda: object(), lambda: object()]

    class _DummyEval:
        def evaluate_encoded(self, x: Any) -> tuple[Any, Any]:
            return x, x

    rebuilt = _DummyEval()
    engine = Engine(
        worker=worker,
        rebuild_evaluator=lambda max_batch, eval_cache_entries: rebuilt,
        rebuild_multi_gpu_pucv_factories=lambda max_batch, gather: factories,
        options=EngineOptions(use_multi_gpu_pucv=True),
    )

    engine._handle_setoption(  # noqa: SLF001
        CmdSetOption(name="EvalCacheEntries", value="256"),
    )

    worker.set_evaluator.assert_called_once_with(rebuilt)
    worker.install_multi_gpu_pucv.assert_called_once_with(
        factories, gather=512, as_factories=True,
    )
    assert engine._options.use_multi_gpu_pucv is True  # noqa: SLF001
    out = capsys.readouterr().out
    assert "UseMultiGpuPUCV on" in out
    assert "EvalCacheEntries set to 256" in out
