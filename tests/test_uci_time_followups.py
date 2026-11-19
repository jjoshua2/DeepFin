"""Follow-ups to the time-forfeit fixes (#1402): the gaps its final review left.

- The chunk-time model clamps a negative fitted fixed cost to zero.
- An at-once answer's eval is finished by the watchdog's last moment.
- ``go movetime N`` with N too small for a search is answered at once, and a
  searched movetime's deadline is never past the watchdog's last moment.
- A `go` / `ponderhit` starts the move's clock when the stdin reader read it.

No test here reads the host's wall-clock speed: times are injected or the
assertions hold for any elapsed time.
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import chess_anti_engine.uci.__main__ as main_module
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import (
    _MOVETIME_FINISH_SLACK_MS,
    _WATCHDOG_LAST_MOMENT_MS,
    Engine,
    EngineOptions,
    _at_once_answer_by_ms,
    _fit_movetime_limits,
)
from chess_anti_engine.uci.protocol import (
    CmdGo,
    CmdIsReady,
    CmdPonderHit,
    GoArgs,
    parse_command,
    stamp_received,
)
from chess_anti_engine.uci.search import SearchResult
from chess_anti_engine.uci.time_manager import _MIN_DEADLINE_MS, Deadline, limits_from_go
from tests.test_uci_time_forfeits import _bare_worker, _feed, _mock_worker, _Out


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


# -- the chunk-time model ---------------------------------------------------------


def test_a_negative_fitted_fixed_cost_is_clamped_to_zero() -> None:
    """Chunks timed at exactly ``sims - 40`` ms fit a fixed cost of -40 ms: noise.
    The model then prices every sim at the mean rate (no fixed cost), so a
    small chunk is never predicted faster than its share of the mean."""
    w = _bare_worker(2048)
    _feed(w, [(128, 88.0), (256, 216.0), (512, 472.0), (64, 24.0), (384, 344.0)])
    fitted, fixed, per_sim, mean_s, mean_t = w._chunk_model()
    assert fitted
    assert fixed == 0.0
    assert per_sim == pytest.approx(mean_t / mean_s)
    assert per_sim < 1.0  # the raw fit's slope was exactly 1 ms per sim
    # A 16-sim chunk: 16 x the mean rate (the raw fit said 16 - 40 < 0 ms).
    assert w._predict_chunk_ms(16) == pytest.approx(16 * mean_t / mean_s)
    assert w._sims_for_ms(100.0) == int(100.0 / per_sim)
    # The first chunk of a 300 ms search: half of 300 - 6 (reserve) = 147 ms,
    # ~171 sims at the mean rate (the raw fit: 147 + 40 = 187 sims).
    d = Deadline(deadline_ms=300, now=time.monotonic())
    got = w._time_capped_chunk(8192, d, 0)
    assert abs(got - 147.0 / per_sim) <= 2.0


# -- the at-once answer's eval budget ---------------------------------------------


@pytest.mark.parametrize("clock", range(30, 46))
@pytest.mark.parametrize("elapsed_ms", [0.0, 1.0, 4.0])
def test_an_at_once_answer_is_due_by_the_watchdogs_last_moment(clock, elapsed_ms) -> None:
    overhead = 30
    clock_left = clock - overhead  # what `_move_budget_ms` hands it
    answer_by = _at_once_answer_by_ms(clock_left, elapsed_ms)
    assert answer_by <= clock - overhead - _WATCHDOG_LAST_MOMENT_MS
    # The share rule still binds on a larger clock: at most the floor / 5 %.
    assert _at_once_answer_by_ms(900.0, elapsed_ms) == pytest.approx(elapsed_ms + 45.0)


def test_the_at_once_answer_uses_that_bound_for_its_eval(out, monkeypatch) -> None:
    """`go wtime 45` with MoveOverheadMs 30: 15 ms left, so the answer (and any
    eval behind it) is due 7 ms after `go` whatever the host's speed: the 10 ms
    eval floor would put it at 10 ms or later."""
    worker = _mock_worker()
    engine = Engine(worker, options=EngineOptions())
    due: list[float] = []
    real = engine._clock_exhausted_move

    def spy(board, searchmoves, answer_by_ms, go_received_at):
        due.append(answer_by_ms)
        return real(board, searchmoves, answer_by_ms, go_received_at)

    monkeypatch.setattr(engine, "_clock_exhausted_move", spy)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go wtime 45 btime 45"))
        assert len(out.wait_bestmoves(1)) == 1
        assert "clock too short to search" in out.text()
        assert due == [pytest.approx(45 - 30 - _WATCHDOG_LAST_MOMENT_MS)]
        assert worker.run.call_count == 0
    finally:
        engine.close()


# -- go movetime N, N too small to search -------------------------------------------


def _mt_limits(n: int, overhead: int = 30) -> Any:
    args = GoArgs(movetime_ms=n)
    return _fit_movetime_limits(
        args, limits_from_go(args, side_to_move_is_white=True, move_overhead_ms=overhead),
    )


_SLACK = _WATCHDOG_LAST_MOMENT_MS + _MOVETIME_FINISH_SLACK_MS  # 8 + 20 ms


@pytest.mark.parametrize("n", [1, 5, 10, 20, 27, 28, 40, 47])
def test_a_movetime_too_short_to_search_is_answered_at_once(n) -> None:
    limits = _mt_limits(n)
    assert limits.clock_exhausted
    assert limits.deadline_ms == max(0, n - int(_SLACK))
    assert n - _SLACK < _MIN_DEADLINE_MS


@pytest.mark.parametrize("overhead", [0, 5, 30, 100])
@pytest.mark.parametrize("n", [48, 49, 50, 100, 300, 1000])
def test_a_searched_movetime_ends_well_before_the_watchdogs_last_moment(n, overhead) -> None:
    """The watchdog's last moment is ``max(deadline, N - 8)``, and a search
    answers up to a small chunk after its deadline: a deadline at or past
    ``N - 8`` (the 20 ms floor at N < 28, or MoveOverheadMs under 8) made the
    answer land after ``N``, or raced the watchdog for it."""
    limits = _mt_limits(n, overhead)
    assert not limits.clock_exhausted
    assert _MIN_DEADLINE_MS <= limits.deadline_ms <= n - _SLACK
    assert limits.deadline_ms == min(max(_MIN_DEADLINE_MS, n - overhead), int(n - _SLACK))
    # The default MoveOverheadMs (30) already keeps that slack from 50 ms up.
    if overhead == 30 and n >= 50:
        assert limits.deadline_ms == max(_MIN_DEADLINE_MS, n - 30)


@pytest.mark.parametrize(("go", "overhead", "fitted"), [
    ("go movetime 40", 30, True), ("go movetime 100", 0, True), ("go movetime 100", 30, False),
])
def test_the_go_wait_is_sized_from_the_fitted_movetime_deadline(go, overhead, fitted) -> None:
    """The bounded wait for a still-running search is a share of the deadline
    the `go` will really get: 12 ms for `movetime 40` (answered at once), not
    the unfitted 20 ms floor; 72 ms, not 100, for `movetime 100` with no
    overhead. At the default overhead from 48 ms up the fit changes nothing."""
    engine = Engine(_mock_worker(), options=EngineOptions(move_overhead_ms=overhead))
    try:
        cmd = parse_command(go)
        assert isinstance(cmd, CmdGo)
        deadline_ms = _mt_limits(cmd.args.movetime_ms or 0, overhead).deadline_ms
        deadline_s = deadline_ms / 1000.0
        floor_s = min(engine_module._GO_WAIT_MIN_S, engine_module._GO_WAIT_MAX_SHARE * deadline_s)
        expect = min(engine_module._JOIN_TIMEOUT_S,
                     max(floor_s, engine_module._GO_WAIT_FRACTION * deadline_s))
        assert engine._go_wait_budget_s(cmd) == pytest.approx(expect)
        unfitted = limits_from_go(cmd.args, side_to_move_is_white=True, move_overhead_ms=overhead)
        assert (unfitted.deadline_ms != deadline_ms) is fitted
    finally:
        engine.close()


def test_other_bounds_are_not_movetime_fitted() -> None:
    for args in (GoArgs(wtime_ms=40, btime_ms=40), GoArgs(nodes=64), GoArgs(infinite=True),
                 GoArgs(movetime_ms=5, ponder=True)):
        limits = limits_from_go(args, side_to_move_is_white=True, move_overhead_ms=30)
        assert _fit_movetime_limits(args, limits) is limits


@pytest.mark.parametrize("n", [5, 47])
def test_the_engine_answers_a_tiny_movetime_without_searching(out, n) -> None:
    worker = _mock_worker()
    worker.tree_root_bestmove.return_value = "e7e5"
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command(f"go movetime {n}"))
        assert out.wait_bestmoves(1) == ["e7e5"], out.text()
        assert f"movetime too short to search: movetime {n} ms" in out.text()
        assert "source=tree" in out.text()
        assert worker.run.call_count == 0
        assert engine._watchdog.armed is None
        assert engine.bestmove_fallback_used == 0
        assert worker.fresh_root_policy_bestmove.call_count == 0
    finally:
        engine.close()


def test_a_movetime_just_long_enough_is_searched_to_its_fitted_deadline(out) -> None:
    worker = _mock_worker()
    seen: list[int | None] = []

    def run(_board, **kw):
        d = kw["deadline"]
        seen.append(round((d._deadline_s - d._start) * 1000.0))
        return SearchResult("e7e5", None, 16, ("e7e5",), 0)

    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions(move_overhead_ms=0))
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 60"))
        assert out.wait_bestmoves(1) == ["e7e5"], out.text()
        # MoveOverheadMs 0 asked for a 60 ms search, past the 52 ms last moment.
        assert seen == [60 - int(_SLACK)]
    finally:
        engine.close()


# -- the clock starts when the line is read -------------------------------------------


def test_go_and_ponderhit_are_stamped_and_nothing_else() -> None:
    go = parse_command("go movetime 100")
    assert isinstance(go, CmdGo)
    assert go.received_at is None
    stamped = stamp_received(go, 12.5)
    assert isinstance(stamped, CmdGo)
    assert stamped.received_at == 12.5
    assert stamped == go  # the stamp is not part of the command's identity
    hit = stamp_received(parse_command("ponderhit"), 3.0)
    assert isinstance(hit, CmdPonderHit)
    assert hit.received_at == 3.0
    ready = parse_command("isready")
    assert stamp_received(ready, 3.0) is ready
    assert isinstance(ready, CmdIsReady)


def test_the_stdin_reader_stamps_each_line_when_read(monkeypatch) -> None:
    ticks = iter([10.0, 20.0, 30.0])
    monkeypatch.setattr(main_module, "_receipt_clock", lambda: next(ticks))
    got = list(main_module._received_lines(iter(["isready\n", "go nodes 1\n", "ponderhit\n"])))
    assert got == [("isready\n", 10.0), ("go nodes 1\n", 20.0), ("ponderhit\n", 30.0)]


def test_the_main_loop_dispatches_stamped_commands(monkeypatch) -> None:
    """Through the real `main` loop: the `go` and `ponderhit` the engine
    receives carry the time `_received_lines` read them at."""
    got: list[Any] = []

    class _Engine:
        quit_requested = False
        exit_code = 0

        def warmup_search(self) -> None:
            pass

        def dispatch(self, cmd: Any) -> None:
            got.append(cmd)

        def close(self) -> None:
            pass

    ticks = iter(float(i) for i in range(100, 200))
    monkeypatch.setattr(main_module, "_receipt_clock", lambda: next(ticks))
    monkeypatch.setattr(main_module, "_load_models", lambda *_a, **_kw: [SimpleNamespace()])
    monkeypatch.setattr(main_module, "_make_evaluator_factory",
                        lambda *_a, **_kw: lambda *_a, **_kw: None)
    monkeypatch.setattr(main_module, "_build_engine", lambda **_kw: _Engine())
    monkeypatch.setattr(main_module.sys, "argv",
                        ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])
    monkeypatch.setattr(main_module.sys, "stdin",
                        iter(["isready\n", "go ponder movetime 50\n", "ponderhit\n", "quit\n"]))
    assert main_module.main() == 0
    gos = [c for c in got if isinstance(c, CmdGo)]
    hits = [c for c in got if isinstance(c, CmdPonderHit)]
    assert [c.received_at for c in gos] == [101.0]
    assert [c.received_at for c in hits] == [102.0]


def test_a_ponderhit_clock_starts_when_it_was_read(out) -> None:
    """A ponderhit read 30 s before its handler ran. The real phase's clock
    started then: with the search held inside its ponder chunk, the deadline
    watchdog answers at once, 30 s into the move. Counted from the handler,
    a 60 s clock's deadline would be seconds away and nothing would answer."""
    gate_entered, gate_release = threading.Event(), threading.Event()

    def ponder_run(*_args: Any, **_kw: Any) -> SearchResult:
        gate_entered.set()
        assert gate_release.wait(30)
        return SearchResult("e7e5", None, 16, ("e7e5",), 0)

    worker = _mock_worker()
    worker.run.side_effect = ponder_run
    engine = Engine(worker, options=EngineOptions(ponder=True))
    armed: list[float] = []
    real_arm = engine._arm_watchdog

    def spy_arm(gen, **kw):
        armed.append(kw["started_at"])
        return real_arm(gen, **kw)

    engine._arm_watchdog = spy_arm  # type: ignore[method-assign]
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 60000 btime 60000 winc 1000 binc 1000"))
        assert gate_entered.wait(5)
        read_at = time.monotonic() - 30.0
        engine.dispatch(stamp_received(parse_command("ponderhit"), read_at))
        assert engine._ponderhit_at == read_at
        assert armed == [read_at]
        # The search is still held: only the watchdog can answer.
        assert len(out.wait_bestmoves(1, timeout=5)) == 1, out.text()
        line = next(ln for ln in out.text().splitlines() if "deadline watchdog" in ln)
        assert int(line.split(" ms into a clocked move")[0].rsplit(" ", 1)[1]) >= 30000
    finally:
        gate_release.set()
        engine.close()


def test_a_go_clock_starts_when_it_was_read(out) -> None:
    worker = _mock_worker()
    seen: list[float] = []

    def run(_board, **kw):
        seen.append(kw["deadline"]._start)
        return SearchResult("e7e5", None, 16, ("e7e5",), 0)

    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions())
    armed: list[float] = []
    real_arm = engine._arm_watchdog

    def spy_arm(gen, **kw):
        armed.append(kw["started_at"])
        return real_arm(gen, **kw)

    engine._arm_watchdog = spy_arm  # type: ignore[method-assign]
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        read_at = time.monotonic() - 0.05
        engine.dispatch(stamp_received(parse_command("go movetime 60000"), read_at))
        assert out.wait_bestmoves(1) == ["e7e5"], out.text()
        assert engine._go_received_at == read_at
        assert armed == [read_at]
        assert seen == [read_at]  # the search's own deadline counts from it too
        # A stamp from the future (a clock mix-up) never moves the origin past now.
        later = time.monotonic() + 60.0
        engine.dispatch(stamp_received(parse_command("go movetime 60000"), later))
        assert out.wait_bestmoves(2)[1] == "e7e5"
        assert engine._go_received_at is not None
        assert engine._go_received_at < later
    finally:
        engine.close()
