"""A command handler error never kills the engine; every ``go`` gets one bestmove.

A crash or a missing ``bestmove`` loses the game, so the properties here are
about the PROCESS and the PROTOCOL, not about move quality:

  (a) an exception inside any handler is reported on one ``info string`` and
      the engine keeps answering; a ``go`` whose handler raised before the
      search thread took over still gets exactly one legal ``bestmove``;
  (b) every declared ``type spin`` option enforces both advertised bounds;
  (c) a failed evaluator rebuild keeps the previous evaluator and settings;
  (d) ``quit`` during startup joins the builder (bounded) and exits 0.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import chess
import pytest

from chess_anti_engine.mcts.search_options import OPTIONS_BY_NAME
from chess_anti_engine.uci import engine as engine_mod
from chess_anti_engine.uci.engine import Engine, EngineOptions, emit_handshake
from chess_anti_engine.uci.protocol import (
    CmdGo,
    CmdIsReady,
    CmdPonderHit,
    CmdPosition,
    CmdSetOption,
    CmdStop,
    GoArgs,
    parse_command,
)
from chess_anti_engine.uci.search import SearchResult

# Read off the module (not imported by name) so this file still collects
# against a tree without the table and fails on behaviour, not on import.
_SPIN_BOUNDS: dict[str, tuple[int, int]] = getattr(engine_mod, "_SPIN_BOUNDS", {})


def _bestmoves(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if ln.startswith("bestmove ")]


def _join_search(engine: Engine) -> None:
    thread = engine._search_thread
    if thread is not None:
        thread.join(5)
        assert not thread.is_alive()


# --- (a) handler errors are contained ---------------------------------------


def test_handler_exception_is_one_info_line_and_the_engine_answers_on(capsys) -> None:
    worker = MagicMock()
    worker.set_multi_pv.side_effect = RuntimeError("CUDA error: out of memory\nsecond line")
    engine = Engine(worker, options=EngineOptions())

    engine.dispatch(CmdSetOption("MultiPV", "3"))
    engine.dispatch(CmdIsReady())

    out = capsys.readouterr()
    errors = [ln for ln in out.out.splitlines() if "error handling CmdSetOption" in ln]
    assert len(errors) == 1, out.out
    assert errors[0].startswith("info string ")
    assert "CUDA error: out of memory second line" in errors[0]
    assert "Traceback" not in out.out
    assert out.out.splitlines()[-1] == "readyok"


def test_isready_whose_rewarm_raises_still_answers_readyok(capsys) -> None:
    worker = MagicMock()
    engine = Engine(worker, options=EngineOptions())
    engine._warmup_dirty = True
    worker.run.side_effect = RuntimeError("rewarm failed")
    # Multi-GPU warmup re-raises (single-GPU swallows), so force that branch.
    engine._options.search_parallel = "gumbel"

    engine.dispatch(CmdIsReady())

    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "readyok"
    assert any("rewarm failed" in ln for ln in lines)


def _assert_reply_is_held(engine: Engine) -> None:
    """The reply thread exists and is still waiting for UCI's permission.

    A bounded join that is EXPECTED to time out: correct code never finishes it
    (the thread blocks on an Event), so this cannot false-fail; a reply that
    does not wait would have exited and printed within the bound.
    """
    thread = engine._search_thread
    assert thread is not None
    thread.join(0.2)
    assert thread.is_alive()


def _failing_go_engine() -> tuple[Engine, MagicMock]:
    worker = MagicMock()
    # `_sync_tree_root` -> advance_root / reset_tree raise: the handler dies
    # BEFORE it starts the search thread, so on main nothing ever answered.
    worker.advance_root.side_effect = RuntimeError("tree allocation failed")
    worker.reset_tree.side_effect = RuntimeError("tree allocation failed")
    worker.root_policy_bestmove.return_value = None
    return Engine(worker, options=EngineOptions()), worker


def test_go_whose_handler_raises_gets_exactly_one_legal_bestmove(capsys) -> None:
    engine, worker = _failing_go_engine()
    engine.dispatch(CmdPosition(None, ("e2e4",)))

    engine.dispatch(CmdGo(GoArgs(nodes=64)))
    _join_search(engine)

    out = capsys.readouterr().out
    moves = _bestmoves(out)
    board = chess.Board()
    board.push_uci("e2e4")
    assert len(moves) == 1, out
    assert chess.Move.from_uci(moves[0]) in board.legal_moves
    assert engine.bestmove_fallback_used == 1
    worker.run.assert_not_called()

    # The engine is still in service: the next go searches normally.
    worker.advance_root.side_effect = None
    worker.advance_root.return_value = True
    worker.reset_tree.side_effect = None
    worker.run.side_effect = lambda b, **_kw: SearchResult(
        next(iter(b.legal_moves)).uci(), None, 1, (), 0,
    )
    engine.dispatch(CmdGo(GoArgs(nodes=64)))
    _join_search(engine)
    assert len(_bestmoves(capsys.readouterr().out)) == 1
    worker.run.assert_called_once()


def test_failed_infinite_go_waits_for_stop_then_answers_once(capsys) -> None:
    engine, _worker = _failing_go_engine()

    engine.dispatch(CmdGo(GoArgs(infinite=True)))
    _assert_reply_is_held(engine)
    assert _bestmoves(capsys.readouterr().out) == []
    engine.dispatch(CmdStop())

    moves = _bestmoves(capsys.readouterr().out)
    assert len(moves) == 1
    assert chess.Move.from_uci(moves[0]) in chess.Board().legal_moves


@pytest.mark.parametrize("release", ["ponderhit", "stop"])
def test_failed_ponder_go_answers_once_on_release(release: str, capsys) -> None:
    engine, _worker = _failing_go_engine()
    engine.dispatch(CmdPosition(None, ("e2e4", "e7e5")))

    engine.dispatch(CmdGo(GoArgs(ponder=True, wtime_ms=10_000, btime_ms=10_000)))
    _assert_reply_is_held(engine)
    assert _bestmoves(capsys.readouterr().out) == []
    engine.dispatch(CmdPonderHit() if release == "ponderhit" else CmdStop())
    _join_search(engine)

    moves = _bestmoves(capsys.readouterr().out)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert len(moves) == 1
    assert chess.Move.from_uci(moves[0]) in board.legal_moves


def test_handler_failure_after_the_search_started_leaves_the_reply_to_it(monkeypatch, capsys) -> None:
    worker = MagicMock()
    worker.advance_root.return_value = True
    release = threading.Event()

    def run(_board, **_kw):
        assert release.wait(5)
        return SearchResult("e2e4", None, 64, ("e2e4",), 0)

    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions())
    original = engine._handle_go

    def go_then_fail(cmd):
        original(cmd)
        raise RuntimeError("failure after the search thread took ownership")

    monkeypatch.setattr(engine, "_handle_go", go_then_fail)
    engine.dispatch(CmdGo(GoArgs(nodes=64)))
    release.set()
    _join_search(engine)

    # The search's own answer, once -- not a fallback racing or replacing it.
    assert _bestmoves(capsys.readouterr().out) == ["e2e4"]
    assert engine.bestmove_fallback_used == 0


def test_quit_supersedes_a_pending_failed_go_reply(capsys) -> None:
    engine, _worker = _failing_go_engine()
    engine.dispatch(CmdGo(GoArgs(infinite=True)))
    engine.close()
    assert _bestmoves(capsys.readouterr().out) == []


# --- (b) every declared spin option enforces both bounds ---------------------

_SPIN_RE = re.compile(
    r"^option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)$"
)


def _declared_spins(options: EngineOptions, capsys) -> dict[str, tuple[int, int, int]]:
    capsys.readouterr()
    emit_handshake(options)
    spins = {}
    for line in capsys.readouterr().out.splitlines():
        m = _SPIN_RE.match(line)
        if m:
            spins[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        else:
            assert " type spin " not in line, f"unparsed spin declaration: {line}"
    return spins


def test_every_declared_spin_option_has_an_enforced_bound_table_entry(capsys) -> None:
    spins = _declared_spins(EngineOptions(), capsys)
    assert {"Hash", "Threads", "MaxBatch", "MultiPV", "LeafGather"} <= set(spins)
    for name, (_default, lo, hi) in spins.items():
        registry = OPTIONS_BY_NAME.get(name.lower())
        if registry is not None:
            assert (registry.lo, registry.hi) == (lo, hi), name
        else:
            assert _SPIN_BOUNDS.get(name) == (lo, hi), (
                f"{name} is declared as a spin but has no _SPIN_BOUNDS entry, "
                "so its advertised range is not what setoption enforces"
            )


@pytest.mark.parametrize("which", ["below", "above", "huge"])
def test_every_declared_spin_option_stays_inside_its_advertised_range(which, capsys) -> None:
    worker = MagicMock()
    options = EngineOptions()
    engine = Engine(
        worker, options=options,
        rebuild_evaluator=lambda *_a, **_kw: MagicMock(name="evaluator"),
    )
    declared = _declared_spins(options, capsys)
    for name, (_default, lo, hi) in declared.items():
        value = {"below": lo - 1, "above": hi + 1, "huge": 50_000_000_000}[which]
        engine.dispatch(parse_command(f"setoption name {name} value {value}"))
        out = capsys.readouterr().out
        assert "error handling" not in out, out
    realized = _declared_spins(options, capsys)
    for name, (default, lo, hi) in realized.items():
        assert lo <= default <= hi, f"{name}={default} outside advertised [{lo}, {hi}]"
        if which != "below" and name in _SPIN_BOUNDS:
            # Hand-written spins clamp (announced) rather than reject.
            assert default == hi, name


def test_spin_clamp_is_announced(capsys) -> None:
    engine = Engine(MagicMock(), options=EngineOptions())
    engine.dispatch(parse_command("setoption name MultiPV value 1000"))
    out = capsys.readouterr().out
    assert "MultiPV 1000 is above the advertised maximum 256; using 256" in out
    assert engine._options.multi_pv == 256


def test_clamped_max_batch_is_what_the_evaluator_factory_receives(capsys) -> None:
    """Observed at the CONSUMER: the factory's own max_batch argument."""
    built: list[int] = []

    def factory(max_batch, _cache=0, /, **_kw):
        built.append(max_batch)
        return MagicMock(name="evaluator")

    worker = MagicMock()
    engine = Engine(worker, options=EngineOptions(), rebuild_evaluator=factory)
    engine.dispatch(parse_command("setoption name MaxBatch value 50000000"))
    assert built == [8192]
    assert engine._options.max_batch == 8192
    worker.set_evaluator.assert_called_once()
    assert "MaxBatch set to 8192" in capsys.readouterr().out


def test_startup_cli_spin_values_are_clamped_too(capsys) -> None:
    from chess_anti_engine.uci import __main__ as main

    args = main._build_parser().parse_args(
        ["--checkpoint", "x.pt", "--max-batch", "50000000", "--walkers", "1"],
    )
    options = main._startup_engine_options(
        args, search_parallel="pucv", restore_multi_gpu_pucv=False,
    )
    assert options.max_batch == _SPIN_BOUNDS["MaxBatch"][1]
    assert "--max-batch 50000000" in capsys.readouterr().err


# --- (c) a failed evaluator rebuild keeps the previous evaluator ------------


def _oom(*_a, **_kw):
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 TiB")


@pytest.mark.parametrize(
    ("option", "value", "attr"),
    [("MaxBatch", "4096", "max_batch"),
     ("EvalCacheEntries", "1000", "eval_cache_entries"),
     ("Threads", "1", "threads")],
)
def test_failed_rebuild_keeps_previous_evaluator_and_settings(option, value, attr, capsys) -> None:
    worker = MagicMock()
    options = EngineOptions(threads=2)
    engine = Engine(worker, options=options, rebuild_evaluator=_oom)
    before = getattr(options, attr)

    engine.dispatch(CmdSetOption(option, value))

    assert getattr(options, attr) == before
    worker.set_evaluator.assert_not_called()
    worker.set_num_threads.assert_not_called()
    worker.set_eval_cache_entries.assert_not_called()
    out = capsys.readouterr().out
    assert "evaluator rebuild failed" in out
    assert "keeping the previous evaluator and settings" in out
    assert "error handling" not in out  # handled at the rebuild, not by the guard


def test_factory_keeps_its_walker_tier_after_a_failed_rebuild(monkeypatch) -> None:
    from chess_anti_engine.uci import __main__ as main

    fail = [False]

    class _Direct:
        def __init__(self, *_a, **_kw):
            if fail[0]:
                raise RuntimeError("out of memory")

    class _Coalescing:
        def __init__(self, inner, **_kw):
            self.inner = inner

    monkeypatch.setattr(main, "DirectGPUEvaluator", _Direct)
    monkeypatch.setattr(main, "ThreadSafeGPUDispatcher", lambda e: e)
    monkeypatch.setattr(main, "BatchCoalescingDispatcher", _Coalescing)
    monkeypatch.setattr(main, "_warmup_evaluator", lambda *_a, **_kw: None)
    build = main._make_evaluator_factory(
        [SimpleNamespace()], ["cpu"], coalesce=True, n_walkers=1, walker_gather=1,
    )
    assert isinstance(build(64), _Direct)
    fail[0] = True
    with pytest.raises(RuntimeError):
        build(64, n_walkers=4)
    fail[0] = False
    # The failed 1 -> 4 rebuild must not leave the factory believing in 4.
    assert isinstance(build(64), _Direct)


# --- (d) quit during startup is deterministic --------------------------------


def _patch_slow_builder(monkeypatch, entered: threading.Event, release: threading.Event):
    from chess_anti_engine.uci import __main__ as main

    later_phase = MagicMock(side_effect=AssertionError("builder ran past shutdown"))

    def slow_load(*_a, **_kw):
        entered.set()
        assert release.wait(5), "test never released the builder"
        return [SimpleNamespace()]

    def stdin():
        assert entered.wait(5)
        yield "quit\n"

    monkeypatch.setattr(main, "_load_models", slow_load)
    monkeypatch.setattr(main, "_make_evaluator_factory", later_phase)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])
    monkeypatch.setattr(main.sys, "stdin", stdin())
    return main, later_phase


def _builder_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "deepfin-build"]


def test_quit_during_startup_signals_and_joins_the_builder(monkeypatch) -> None:
    entered, release = threading.Event(), threading.Event()
    main, later_phase = _patch_slow_builder(monkeypatch, entered, release)
    original = main._request_builder_shutdown if hasattr(main, "_request_builder_shutdown") else None

    def request(event):
        if original is not None:
            original(event)
        release.set()  # the stubbed builder leaves torch once signalled

    monkeypatch.setattr(main, "_request_builder_shutdown", request, raising=False)
    hard_exits: list[int] = []
    monkeypatch.setattr(main, "_hard_exit", hard_exits.append, raising=False)
    try:
        assert main.main() == 0
        # Joined, not abandoned to interpreter shutdown...
        assert _builder_threads() == []
        # ...and it stopped at the next phase boundary instead of building on.
        later_phase.assert_not_called()
        assert hard_exits == []
    finally:
        release.set()
        for t in _builder_threads():
            t.join(5)


def test_builder_stuck_past_the_bound_hard_exits_zero_after_flushing(monkeypatch, capsys) -> None:
    entered, release = threading.Event(), threading.Event()
    main, _later = _patch_slow_builder(monkeypatch, entered, release)
    hard_exits: list[int] = []
    monkeypatch.setattr(main, "_hard_exit", hard_exits.append, raising=False)
    monkeypatch.setattr(main, "_BUILDER_JOIN_TIMEOUT_S", 0.0, raising=False)
    try:
        assert main.main() == 0
        assert hard_exits == [0]
        assert "startup build still running" in capsys.readouterr().err
    finally:
        release.set()
        for t in _builder_threads():
            t.join(5)


def test_hard_exit_flushes_stdout_before_exiting() -> None:
    script = (
        "import sys\n"
        "from chess_anti_engine.uci import __main__ as m\n"
        "sys.stdout.write('last line')\n"
        "m._hard_exit(0)\n"
        "print('unreachable')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False,
        env={**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode == 0
    assert proc.stdout == "last line"


# --- pipe level: bad option, then the engine still plays ---------------------


def test_pipe_bad_option_then_isready_and_go(tmp_path: Path) -> None:
    from tests.test_uci_smoke import _LineReader, _make_tiny_checkpoint, _send

    ckpt = _make_tiny_checkpoint(tmp_path)
    env = {**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "chess_anti_engine.uci", "--checkpoint", str(ckpt),
         "--device", "cpu", "--no-compile", "--walkers", "1", "--max-batch", "8192"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        # On main this reached the evaluator factory as 50,000,000 rows (a
        # 2.2 TB allocation) and the process exited 1.
        _send(proc, "setoption name MaxBatch value 50000000")
        _send(proc, "setoption name MultiPV value notanumber")
        _send(proc, "isready")
        lines = reader.read_until("readyok", timeout_s=60.0)
        assert any("MaxBatch 50000000 is above the advertised maximum" in ln for ln in lines)
        _send(proc, "position startpos moves e2e4")
        _send(proc, "go nodes 16")
        lines = reader.read_until("bestmove", timeout_s=60.0)
        move = next(ln for ln in lines if ln.startswith("bestmove ")).split()[1]
        board = chess.Board()
        board.push_uci("e2e4")
        assert chess.Move.from_uci(move) in board.legal_moves
        _send(proc, "quit")
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
