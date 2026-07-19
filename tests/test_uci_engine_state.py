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
    result = engine._run_one_phase(limits, is_ponder=False, board=board)
    assert result.bestmove_uci in {m.uci() for m in board.legal_moves}
    assert result.ponder_uci is None
    assert "boom" in capsys.readouterr().out


def test_run_one_phase_fallback_respects_searchmoves(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=("g1f3",))
    result = engine._run_one_phase(limits, is_ponder=False, board=board)
    assert result.bestmove_uci == "g1f3"
    assert "boom" in capsys.readouterr().out


def test_run_one_phase_fallback_returns_null_when_searchmoves_have_no_legal_move(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=("a1a8",))
    result = engine._run_one_phase(limits, is_ponder=False, board=board)
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

    engine._run_one_phase(limits, is_ponder=False, board=board)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("info ")]
    assert len(lines) == 1
    assert "multipv 2" in lines[0]
    assert "wdl 500 0 500" in lines[0]


def test_run_one_phase_refreshes_stale_periodic_node_count(capsys) -> None:
    worker = MagicMock()

    def _run(*_args: Any, **kwargs: Any) -> SearchResult:
        kwargs["info_cb"](
            nodes=7680,
            elapsed_ms=100,
            score_cp=12,
            pv=("e2e4",),
            tbhits=0,
            score_mate=None,
            multipv=None,
            wdl=None,
        )
        return SearchResult(
            bestmove_uci="e2e4",
            ponder_uci=None,
            nodes=8192,
            pv=("e2e4",),
            score_cp=12,
            tbhits=0,
        )

    worker.run = _run
    engine = Engine(worker=worker)
    engine._run_one_phase(
        SearchLimits(deadline_ms=None, max_nodes=8192, searchmoves=()),
        is_ponder=False,
        board=chess.Board(),
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("info ")]
    assert len(lines) == 2
    assert "nodes 7680" in lines[0]
    assert "nodes 8192" in lines[1]


def test_gil_profile_emits_uci_safe_search_metadata(capsys) -> None:
    worker = MagicMock()
    worker.concurrency_profile.return_value = ("walker_puct", 4)
    engine = Engine(worker=worker, search_devices=("cuda:0", "cuda:1"), gil_profile=True)
    assert engine._gil_probe is not None
    engine._gil_probe.reset()
    engine._gil_probe._record_delay(0.002)

    engine._emit_gil_profile(nodes=123, elapsed_s=0.5, is_ponder=False)
    engine.close()

    output = capsys.readouterr().out
    assert output.startswith("info string gil_profile ")
    assert "phase=main mode=walker_puct threads=4 devices=2 nodes=123" in output
    assert "samples=1" in output


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

    engine._run_one_phase(limits, is_ponder=False, board=board)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("info ")]
    assert len(lines) == 1
    assert "nodes 3" in lines[0]
    assert "pv e2e4" in lines[0]


def test_run_one_phase_disables_terminal_shortcuts_during_ponder() -> None:
    worker = MagicMock()
    worker.run = MagicMock(
        return_value=SearchResult(
            bestmove_uci="e2e4",
            ponder_uci=None,
            nodes=1,
            pv=("e2e4",),
            score_cp=0,
            tbhits=0,
        ),
    )
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, ponder=True, searchmoves=())

    engine._run_one_phase(limits, is_ponder=True, board=board)

    assert worker.run.call_args.kwargs["allow_terminal_shortcuts"] is False


def test_run_one_phase_allows_terminal_shortcuts_for_normal_search() -> None:
    worker = MagicMock()
    worker.run = MagicMock(
        return_value=SearchResult(
            bestmove_uci="e2e4",
            ponder_uci=None,
            nodes=1,
            pv=("e2e4",),
            score_cp=0,
            tbhits=0,
        ),
    )
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=1000, max_nodes=None, searchmoves=())

    engine._run_one_phase(limits, is_ponder=False, board=board)

    assert worker.run.call_args.kwargs["allow_terminal_shortcuts"] is True


def test_run_one_phase_disables_terminal_shortcuts_for_open_ended_search() -> None:
    worker = MagicMock()
    worker.run = MagicMock(
        return_value=SearchResult(
            bestmove_uci="e2e4",
            ponder_uci=None,
            nodes=1,
            pv=("e2e4",),
            score_cp=0,
            tbhits=0,
        ),
    )
    engine = Engine(worker=worker)
    board = chess.Board()
    limits = SearchLimits(deadline_ms=None, max_nodes=None, max_depth=None, searchmoves=())

    engine._run_one_phase(limits, is_ponder=False, board=board)

    assert limits.is_open_ended()
    assert worker.run.call_args.kwargs["allow_terminal_shortcuts"] is False


def test_emit_bestmove_omits_ponder_when_option_false(capsys) -> None:
    engine = Engine(worker=MagicMock())
    engine._options.ponder = False
    result = SearchResult(
        bestmove_uci="e2e4", ponder_uci="e7e5",
        nodes=0, pv=(), score_cp=0, tbhits=0,
    )
    engine._emit_bestmove(result)
    out = capsys.readouterr().out
    assert "bestmove e2e4" in out
    assert "ponder" not in out


def test_emit_bestmove_includes_ponder_when_option_true(capsys) -> None:
    engine = Engine(worker=MagicMock())
    engine._options.ponder = True
    result = SearchResult(
        bestmove_uci="e2e4", ponder_uci="e7e5",
        nodes=0, pv=(), score_cp=0, tbhits=0,
    )
    engine._emit_bestmove(result)
    out = capsys.readouterr().out
    assert "bestmove e2e4 ponder e7e5" in out


def test_invalid_position_fen_clears_pending_state() -> None:
    engine = Engine(worker=MagicMock())
    engine._handle_position(CmdPosition(fen=None, moves=("e2e4",)))

    assert engine._pending_moves == [chess.Move.from_uci("e2e4")]

    engine._handle_position(CmdPosition(fen="not a valid fen", moves=()))

    assert engine._board == chess.Board()
    assert engine._pending_fen is None
    assert engine._pending_moves == []
    assert engine._applied_fen is None
    assert engine._applied_moves == ()
    assert engine._popped_ponder_move is None


def test_isready_does_not_stop_active_search(capsys) -> None:
    engine = Engine(worker=MagicMock())
    thread = MagicMock()
    thread.is_alive.return_value = True
    engine._search_thread = thread

    engine._handle_isready()

    assert capsys.readouterr().out == "readyok\n"
    assert not engine._stop_event.is_set()
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
    factories: list[Callable[[], Any]] = [object, object]
    engine = Engine(
        worker=worker,
        rebuild_multi_gpu_pucv_factories=lambda max_batch, gather: factories,
    )

    engine._handle_setoption(
        CmdSetOption(name="UseMultiGpuPUCV", value="true"),
    )

    worker.install_multi_gpu_pucv.assert_called_once_with(
        factories, gather=512, as_factories=True,
    )
    assert engine._options.use_multi_gpu_pucv is True
    out = capsys.readouterr().out
    assert "UseMultiGpuPUCV on" in out
    # Inert-tuning parity with the Threads>1 warning: the pool is plain PUCT,
    # so installing it must surface that c_scale/c_visit stop applying.
    assert "c_scale/c_visit are inactive on the multi-GPU PUCV pool" in out


def test_pucv_pending_mode_setoption_updates_worker(capsys) -> None:
    worker = MagicMock()
    engine = Engine(worker=worker)

    engine._handle_setoption(
        CmdSetOption(name="PUCVPendingMode", value="virtual-mean"),
    )

    worker.set_pucv_vloss_mode.assert_called_once_with(1)
    assert engine._options.pucv_pending_mode == "virtual-mean"
    assert "PUCVPendingMode set to virtual-mean" in capsys.readouterr().out


def test_eval_cache_entries_setoption_rebuilds_evaluator(capsys) -> None:
    worker = MagicMock()
    class _DummyEval:
        def evaluate_encoded(self, x: Any, relations: Any = None) -> tuple[Any, Any]:
            del relations  # interface conformance for BatchEvaluator
            return x, x

    rebuilt = _DummyEval()
    calls: list[tuple[int, int]] = []

    def rebuild(max_batch: int, eval_cache_entries: int) -> _DummyEval:
        calls.append((max_batch, eval_cache_entries))
        return rebuilt

    engine = Engine(worker=worker, rebuild_evaluator=rebuild)

    engine._handle_setoption(
        CmdSetOption(name="EvalCacheEntries", value="256"),
    )

    assert calls == [(1024, 256)]
    assert engine._options.eval_cache_entries == 256
    worker.set_eval_cache_entries.assert_called_once_with(256)
    worker.set_evaluator.assert_called_once_with(rebuilt)
    assert "EvalCacheEntries set to 256" in capsys.readouterr().out


def test_eval_cache_entries_reinstalls_multi_gpu_pucv(capsys) -> None:
    worker = MagicMock()
    factories: list[Callable[[], Any]] = [object, object]

    class _DummyEval:
        def evaluate_encoded(self, x: Any, relations: Any = None) -> tuple[Any, Any]:
            del relations  # interface conformance for BatchEvaluator
            return x, x

    rebuilt = _DummyEval()
    engine = Engine(
        worker=worker,
        rebuild_evaluator=lambda max_batch, eval_cache_entries: rebuilt,
        rebuild_multi_gpu_pucv_factories=lambda max_batch, gather: factories,
        options=EngineOptions(use_multi_gpu_pucv=True),
    )

    engine._handle_setoption(
        CmdSetOption(name="EvalCacheEntries", value="256"),
    )

    worker.set_evaluator.assert_called_once_with(rebuilt)
    worker.install_multi_gpu_pucv.assert_called_once_with(
        factories, gather=512, as_factories=True,
    )
    assert engine._options.use_multi_gpu_pucv is True
    out = capsys.readouterr().out
    # Reinstall is silent (no second "UseMultiGpuPUCV on"); primary already
    # rebuilt via set_evaluator under the still-on option.
    assert "EvalCacheEntries set to 256" in out
