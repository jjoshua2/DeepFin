from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import chess

from chess_anti_engine.uci.engine import Engine, EngineOptions, emit_handshake
from chess_anti_engine.uci.protocol import CmdPosition, CmdSetOption


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
