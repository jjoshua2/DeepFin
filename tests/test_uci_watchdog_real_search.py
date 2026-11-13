"""The deadline watchdog against REAL searches (CPU, tiny net, 8-sim chunks).

A healthy search must not be pre-empted: every watchdog answer in these runs
must be one where the search itself would have answered after the move's
budget, i.e. too late on main as well. The "no progress" rule must never fire
on a healthy search at all. The search's own finish time is recorded for every
move (a silenced search still returns from `run`), so a pre-emption of a search
that would have made it is visible as such.

Also: a hang is still answered after a recovered hang, at the healthy chunk
scale (the hung search's stall is not learned as a chunk time).
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import chess
import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker


class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._cond = threading.Condition()

    def write(self, s: str) -> int:
        with self._cond:
            self.lines.extend(ln for ln in s.splitlines() if ln)
            self._cond.notify_all()
        return len(s)

    def flush(self) -> None:
        pass

    def count(self, prefix: str) -> int:
        with self._cond:
            return sum(ln.startswith(prefix) for ln in self.lines)

    def wait_bestmoves(self, n: int, timeout: float = 10.0) -> None:
        with self._cond:
            self._cond.wait_for(
                lambda: sum(ln.startswith("bestmove ") for ln in self.lines) >= n, timeout,
            )

    def last_bestmove(self) -> str:
        with self._cond:
            return [ln for ln in self.lines if ln.startswith("bestmove ")][-1].split()[1]

    def watchdog_lines(self) -> list[str]:
        with self._cond:
            return [ln for ln in self.lines if "deadline watchdog" in ln]


class _Switchable:
    """Evaluator proxy whose batch calls block while ``wedged`` is set."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.wedged = threading.Event()
        self.release = threading.Event()
        self.entered = threading.Event()

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr) or not name.startswith("evaluate"):
            return attr

        def call(*args, **kwargs):
            if self.wedged.is_set():
                self.entered.set()
                assert self.release.wait(30)
            return attr(*args, **kwargs)

        return call


@pytest.fixture
def real(monkeypatch):
    out = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=out, stderr=sys.stderr))
    cfg = ModelConfig(input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2,
                      ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    evaluator = _Switchable(ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        model, device="cpu", max_batch=64, use_amp=False, n_slots=2,
    )))
    worker = SearchWorker(
        evaluator,  # pyright: ignore[reportArgumentType]
        device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=8, n_walkers=1,
    )
    ends: list[float] = []  # each run()'s own finish time, absolute
    real_run = worker.run

    def timed_run(board, **kw):
        result = real_run(board, **kw)
        ends.append(time.monotonic())
        return result

    monkeypatch.setattr(worker, "run", timed_run)
    engine = Engine(worker, options=EngineOptions(ponder=True))
    yield SimpleNamespace(out=out, engine=engine, evaluator=evaluator, ends=ends)
    evaluator.release.set()
    engine.close()
    worker.close()


def _drain(engine: Engine) -> None:
    for thread in [engine._search_thread, *engine._orphan_threads]:
        if thread is not None:
            thread.join(30)
    engine._wait_for_search(timeout_s=30)


def _play(r, go: str, n: int, *, budget_ms: float, ponderhit_after_s: float | None = None,
          allow_in_time: int | None = None):
    """Play ``n`` moves of a real game with ``go``; return per-move
    (fired_line_or_None, search's own finish ms after its clock started)."""
    board = chess.Board()
    results = []
    base = r.out.count("bestmove ")
    for i in range(n):
        before = len(r.out.watchdog_lines())
        r.engine.dispatch(parse_command(f"position fen {board.fen()}"))
        clock_start = time.monotonic()
        r.engine.dispatch(parse_command(go))
        if ponderhit_after_s is not None:
            time.sleep(ponderhit_after_s)  # pondering; the clock starts at ponderhit
            clock_start = time.monotonic()
            r.engine.dispatch(parse_command("ponderhit"))
        r.out.wait_bestmoves(base + i + 1)
        _drain(r.engine)
        assert r.out.count("bestmove ") == base + i + 1  # exactly one per go
        move = r.out.last_bestmove()
        assert chess.Move.from_uci(move) in board.legal_moves
        lines = r.out.watchdog_lines()[before:]
        results.append((lines[0] if lines else None, (r.ends[-1] - clock_start) * 1000.0))
        board.push_uci(move)
        if board.is_game_over():
            board = chess.Board()
    fired = [(line, own_ms) for line, own_ms in results if line is not None]
    # Never the progress rule on a healthy search.
    assert not [f for f in fired if "no search progress" in f[0]], results
    # The last-moment rule answers where the search would itself have missed
    # the budget. A search finishing inside the last-moment margin (the final
    # 3 ms) is pre-empted by design: that margin is the watchdog's own
    # measured wake + answer latency. Allow that window at most one move in ten.
    in_time = [f for f in fired if f[1] <= budget_ms]
    allowed = max(1, n // 10) if allow_in_time is None else allow_in_time
    assert len(in_time) <= allowed, (in_time, results)
    return results


def test_healthy_movetime_100(real) -> None:
    _play(real, "go movetime 100", 15, budget_ms=100)


def test_healthy_movetime_1000(real) -> None:
    _play(real, "go movetime 1000", 3, budget_ms=1000)


def test_healthy_increment_clock(real) -> None:
    _play(real, "go wtime 10000 btime 10000 winc 100 binc 100", 6, budget_ms=10000)


def test_healthy_ponder_then_ponderhit(real) -> None:
    _play(real, "go ponder movetime 300", 4, budget_ms=300, ponderhit_after_s=0.1)


def test_healthy_movetime_100_beside_a_cpu_hog_thread(real) -> None:
    stop = threading.Event()

    def hog() -> None:
        # Half duty cycle: holds the GIL in pure Python for ~2 ms, then yields.
        # (A thread that never sleeps convoys every torch op through the 5 ms
        # switch interval and measures nothing but a 40x slower search.)
        x = 0
        while not stop.is_set():
            end = time.perf_counter() + 0.002
            while time.perf_counter() < end:
                x += 1
            time.sleep(0.002)

    thread = threading.Thread(target=hog, daemon=True)
    thread.start()
    try:
        _play(real, "go movetime 100", 8, budget_ms=100)
    finally:
        stop.set()
        thread.join(5)


def test_after_a_recovered_hang_healthy_moves_do_not_fire_and_a_hang_is_still_quick(real) -> None:
    def hang_once(budget_go: str) -> float:
        before = real.out.count("bestmove ")
        real.evaluator.release.clear()
        real.evaluator.entered.clear()
        real.evaluator.wedged.set()
        start = time.monotonic()
        real.engine.dispatch(parse_command("position startpos moves e2e4"))
        real.engine.dispatch(parse_command(budget_go))
        real.out.wait_bestmoves(before + 1)
        answered_ms = (time.monotonic() - start) * 1000.0
        assert real.evaluator.entered.is_set()
        real.evaluator.wedged.clear()
        real.evaluator.release.set()
        _drain(real.engine)
        assert real.out.count("bestmove ") == before + 1
        return answered_ms

    # Healthy moves first, so a chunk time is known.
    _play(real, "go movetime 300", 2, budget_ms=300)
    fires = len(real.out.watchdog_lines())
    first = hang_once("go wtime 5000 btime 5000")
    assert len(real.out.watchdog_lines()) == fires + 1
    # A second hang right after the first is answered at the healthy scale
    # again: the hung search, once released, finished a (silenced) chunk whose
    # "time to first chunk" was the whole hang; learning it would multiply the
    # stall limit by 4 x the hang.
    second = hang_once("go wtime 5000 btime 5000")
    assert second < 1.5 * first + 50, (first, second)
    # Healthy moves after the recovered hangs: no progress-rule fire and no
    # pre-emption of a search that would have answered in time. (A move whose
    # own search runs past the budget is answered by the last-moment rule, as
    # it would be at any time.)
    _play(real, "go movetime 300", 3, budget_ms=300, allow_in_time=0)
