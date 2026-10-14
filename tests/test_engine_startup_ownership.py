"""Failed startup and independent teardown retain responsibility for owners."""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import match_vs_uci as match
from chess_anti_engine.stockfish import uci as stockfish


@pytest.mark.parametrize("stage", ["configure", "warmup", "interrupt"])
def test_warm_helper_closes_failed_owner_and_preserves_startup_error(monkeypatch, stage):
    engine = MagicMock()
    engine.id = {"name": "fake"}
    engine.options = {"Hash": SimpleNamespace(type="spin")}
    error = KeyboardInterrupt() if stage == "interrupt" else TimeoutError(stage)
    engine.quit.side_effect = TimeoutError("cleanup also failed")
    monkeypatch.setattr(match, "_open_engine", lambda *_a, **_kw: engine)

    def fail(*_a, **_kw):
        raise error

    if stage == "configure":
        engine.configure.side_effect = error
    else:
        monkeypatch.setattr(match, "_warmup_engine", fail)
    with pytest.raises(type(error)) as caught:
        match._open_warm_engine("fake", {"Hash": "16"}, nodes=1,
                                label="fake", warmup_timeout_s=120)
    assert caught.value is error
    assert engine.timeout == match._ENGINE_COMMAND_TIMEOUT_S
    engine.quit.assert_called_once()
    engine.close.assert_called_once()


def test_warm_helper_transfers_successful_owner_without_closing(monkeypatch):
    engine = MagicMock()
    monkeypatch.setattr(match, "_open_engine", lambda *_a, **_kw: engine)
    monkeypatch.setattr(match, "_warmup_engine", lambda *_a, **_kw: None)
    result = match._open_warm_engine("fake", {}, nodes=1, label="fake", warmup_timeout_s=120)
    assert result is engine
    assert engine.timeout == match._ENGINE_COMMAND_TIMEOUT_S
    engine.quit.assert_not_called()
    engine.close.assert_not_called()


@pytest.mark.parametrize("failed_owner", ["tablebase", "engine_a"])
def test_actual_match_finalizer_attempts_every_owner(failed_owner):
    # Execute the real main finalizer without opening matches or engine processes.
    tree = ast.parse(Path(match.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    finalizer = next(n for n in ast.walk(main) if isinstance(n, ast.Try) and
                     {"tb_adjudicator", "eng_a", "eng_b"}.issubset(
                         {v.id for s in n.finalbody for v in ast.walk(s) if isinstance(v, ast.Name)}))
    owners = {name: MagicMock() for name in ("tb_adjudicator", "eng_a", "eng_b")}
    error = TimeoutError(failed_owner)
    if failed_owner == "tablebase":
        owners["tb_adjudicator"].close.side_effect = error
    else:
        owners["eng_a"].quit.side_effect = error
    code = ast.Module(body=finalizer.finalbody, type_ignores=[])
    with pytest.raises(TimeoutError) as caught:
        exec(compile(code, match.__file__, "exec"), {**vars(match), **owners})
    assert caught.value is error
    owners["tb_adjudicator"].close.assert_called_once()
    for name in ("eng_a", "eng_b"):
        owners[name].quit.assert_called_once()
        owners[name].close.assert_called_once()


@pytest.mark.parametrize("stage", ["tcgetattr", "tcsetattr", "popen", "getpriority", "setpriority", "handshake", "success"])
def test_stockfish_constructor_releases_real_ptys_and_owned_fake_child(monkeypatch, stage):
    # Real PTYs exercise descriptor ownership; process launch and signals are fakes.
    opened = []
    real_openpty = stockfish.pty.openpty

    def openpty():
        pair = real_openpty()
        opened.extend(pair)
        return pair

    error = OSError(stage)
    proc = MagicMock()
    proc.pid = 99999999
    proc.poll.return_value = None
    proc.wait.return_value = 0
    monkeypatch.setattr(stockfish.pty, "openpty", openpty)
    monkeypatch.setattr(stockfish.subprocess, "Popen", lambda *_a, **_kw: proc)
    kill = MagicMock()
    monkeypatch.setattr(stockfish.os, "killpg", kill)
    monkeypatch.setattr(stockfish.os, "getpriority", lambda *_a: 0)
    monkeypatch.setattr(stockfish.os, "setpriority", lambda *_a: None)
    monkeypatch.setattr(stockfish.StockfishUCI, "_send", lambda *_a: None)
    monkeypatch.setattr(stockfish.StockfishUCI, "_read_uci_capabilities", lambda *_a: None)
    monkeypatch.setattr(stockfish.StockfishUCI, "_wait_for", lambda *_a: None)

    def fail(*_a, **_kw):
        raise error

    if stage in ("tcgetattr", "tcsetattr"):
        monkeypatch.setattr(stockfish.termios, stage, fail)
    elif stage == "popen":
        monkeypatch.setattr(stockfish.subprocess, "Popen", fail)
    elif stage in ("getpriority", "setpriority"):
        monkeypatch.setattr(stockfish.os, stage, fail)
    elif stage == "handshake":
        monkeypatch.setattr(stockfish.StockfishUCI, "_read_uci_capabilities", fail)
    if stage == "success":
        engine = stockfish.StockfishUCI("fake", nice=1)
        os.fstat(opened[0])
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(opened[1])
        engine.close()
    else:
        with pytest.raises(OSError, match=stage) as caught:
            stockfish.StockfishUCI("fake", nice=1)
        assert caught.value is error
    for fd in opened:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)
    if stage in ("getpriority", "setpriority", "handshake", "success"):
        kill.assert_called_once_with(proc.pid, stockfish.signal.SIGKILL)
        proc.wait.assert_called_once()
    else:
        kill.assert_not_called()
        proc.wait.assert_not_called()
