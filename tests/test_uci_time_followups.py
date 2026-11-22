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

import chess
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

    def spy(board, searchmoves, answer_by_ms, go_received_at, **kw):
        due.append(answer_by_ms)
        assert kw == {"min_eval_s": 0.0}  # a binding clock gets no movetime floor
        return real(board, searchmoves, answer_by_ms, go_received_at, **kw)

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
    receives carry their clock origin -- when the reader read them, or when
    the engine became ready if that was later (`_clock_origin`). The reader and
    the builder both read the clock, from their own threads, so the exact ticks
    are not fixed; the origin is one of them, after readiness, in order."""
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
    tick_lock = threading.Lock()
    drawn: list[float] = []

    def clock() -> float:
        with tick_lock:
            drawn.append(next(ticks))
            return drawn[-1]

    monkeypatch.setattr(main_module, "_receipt_clock", clock)
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
    assert len(gos) == 1
    assert len(hits) == 1
    go_at, hit_at = gos[0].received_at, hits[0].received_at
    assert go_at is not None
    assert hit_at is not None
    assert go_at in drawn
    assert hit_at in drawn
    assert go_at < hit_at


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



# -- the clock origin: arrival, or readiness for a line that came before it -------


def test_the_clock_origin_is_arrival_or_readiness_whichever_is_later() -> None:
    assert main_module._clock_origin(5.0, None) == 5.0
    assert main_module._clock_origin(5.0, 9.0) == 9.0  # arrived during the model load
    assert main_module._clock_origin(12.0, 9.0) == 12.0


def test_a_go_whose_wait_left_no_time_to_search_is_answered_at_once(out) -> None:
    """A `go movetime 1000` that arrived 0.99 s before its handler ran (queued
    behind earlier commands): less than the minimum search is left of its
    deadline, so it is answered at once, without starting a search the
    watchdog would answer unsearched."""
    worker = _mock_worker()
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos"))
        go = stamp_received(parse_command("go movetime 1000"), time.monotonic() - 0.99)
        engine.dispatch(go)
        assert len(out.wait_bestmoves(1)) == 1
        assert "behind earlier commands" in out.text()
        assert "deadline watchdog answers unsearched" not in out.text()
        assert worker.run.call_count == 0
    finally:
        engine.close()


def test_a_go_that_waited_less_than_its_budget_still_searches(out) -> None:
    worker = _mock_worker()
    worker.run.return_value = SearchResult("e2e4", None, 16, ("e2e4",), 0)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos"))
        go = stamp_received(parse_command("go movetime 1000"), time.monotonic() - 0.3)
        engine.dispatch(go)
        assert len(out.wait_bestmoves(1)) == 1
        assert "behind earlier commands" not in out.text()
        assert worker.run.call_count >= 1
    finally:
        engine.close()



class _TimedOut:
    """Captured stdout that also records when each line was written."""

    def __init__(self) -> None:
        self.stamps: list[tuple[float, str]] = []
        self._cond = threading.Condition()

    def write(self, s: str) -> int:
        now = time.monotonic()
        with self._cond:
            self.stamps.extend((now, ln) for ln in s.splitlines() if ln)
            self._cond.notify_all()
        return len(s)

    def flush(self) -> None:
        pass

    def wait_bestmove_after(self, n0: int, timeout: float = 10.0) -> tuple[float, str] | None:
        with self._cond:
            self._cond.wait_for(
                lambda: any(ln.startswith("bestmove ") for _, ln in self.stamps[n0:]), timeout,
            )
            hits = [(t, ln) for t, ln in self.stamps[n0:] if ln.startswith("bestmove ")]
            return hits[0] if hits else None

    def text(self) -> str:
        with self._cond:
            return "\n".join(ln for _, ln in self.stamps)


@pytest.fixture
def timed_out(monkeypatch) -> _TimedOut:
    capture = _TimedOut()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


@pytest.mark.parametrize("age_ms", [900, 930, 945])
@pytest.mark.parametrize("stop_lag_ms", [50, 80, 150])
def test_an_aged_go_behind_a_late_stopping_search_answers_within_its_movetime(
    timed_out, age_ms, stop_lag_ms,
) -> None:
    """A live `go infinite` whose search honours stop late, then a `go movetime
    1000` stamped 900-945 ms before its handler ran. The wait for the old
    search is sized from what is LEFT of the 970 ms deadline, so it ends at
    least the 20 ms minimum search before that deadline (950 ms), or is not
    taken at all when nothing would be left; the old search is parked
    (silenced) and this `go` answered from the busy-worker fallback: never
    the watchdog, never the old search's move, no new search started. Before,
    the wait was sized from the full deadline (100 ms) and these answered at
    0.997-1.046 s."""
    worker = _mock_worker()
    entered = threading.Event()
    calls = [0]

    def run(_board, *, stop_event, **_kw):
        calls[0] += 1
        if calls[0] == 1:
            entered.set()
            stop_event.wait(30)
            time.sleep(stop_lag_ms / 1000.0)
            return SearchResult("e2e4", None, 16, ("e2e4",), 0)
        return SearchResult("e7e5", None, 16, ("e7e5",), 0)

    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos"))
        engine.dispatch(parse_command("go infinite"))
        assert entered.wait(5)
        engine.dispatch(parse_command("position startpos moves e2e4"))
        n0 = len(timed_out.stamps)
        origin = time.monotonic() - age_ms / 1000.0
        engine.dispatch(stamp_received(parse_command("go movetime 1000"), origin))
        hit = timed_out.wait_bestmove_after(n0)
        assert hit is not None
        at, line = hit
        assert (at - origin) * 1000.0 <= 1000.0, (at - origin)
        move = line.split()[1]
        board = chess.Board()
        board.push_uci("e2e4")
        assert chess.Move.from_uci(move) in board.legal_moves
        time.sleep(stop_lag_ms / 1000.0 + 0.05)  # the parked search returns
        text = timed_out.text()
        assert "deadline watchdog answers unsearched" not in text
        assert sum(ln.startswith("bestmove ") for _, ln in timed_out.stamps[n0:]) == 1
        assert calls[0] == 1  # no search started for the aged go
    finally:
        engine.close()


@pytest.mark.parametrize(("age_s", "expect_s"), [
    (0.3, 0.1), (0.9, 0.0175), (0.93, 0.01), (0.935, 0.01), (0.945, 0.0), (0.96, 0.0),
])
def test_the_go_wait_is_sized_from_what_is_left_after_the_gos_age(age_s, expect_s) -> None:
    """`go movetime 1000` (970 ms deadline at the default overhead): the wait
    for a live search is a share of the deadline less the go's age, floored at
    10 ms (the share is 8.75 ms at 35 ms left), and zero once even that floor would leave under the 20 ms minimum
    search (park at once)."""
    engine = Engine(_mock_worker(), options=EngineOptions())
    try:
        cmd = parse_command("go movetime 1000")
        assert isinstance(cmd, CmdGo)
        assert engine._go_wait_budget_s(cmd, age_s=age_s) == pytest.approx(expect_s, abs=1e-9)
    finally:
        engine.close()


def test_the_queued_check_counts_time_spent_after_the_wait(timed_out, monkeypatch) -> None:
    """The at-once check is taken after the join AND the tree sync: a `go`
    with room for a search at handler entry (930 ms of a 970 ms deadline)
    whose tree sync then takes 50 ms is answered at once, not searched."""
    worker = _mock_worker()
    worker.run.return_value = SearchResult("e2e4", None, 16, ("e2e4",), 0)
    engine = Engine(worker, options=EngineOptions())
    real_sync = engine._sync_tree_root

    def slow_sync(moves):
        time.sleep(0.05)
        return real_sync(moves)

    monkeypatch.setattr(engine, "_sync_tree_root", slow_sync)
    try:
        engine.dispatch(parse_command("position startpos"))
        n0 = len(timed_out.stamps)
        engine.dispatch(stamp_received(parse_command("go movetime 1000"), time.monotonic() - 0.93))
        assert timed_out.wait_bestmove_after(n0) is not None
        assert "behind earlier commands" in timed_out.text()
        assert worker.run.call_count == 0
    finally:
        engine.close()


@pytest.mark.parametrize("go", [
    "go nodes 100", "go depth 3", "go infinite", "go ponder", "go mate 2",
])
def test_a_go_with_no_clock_searches_however_long_it_queued(timed_out, go) -> None:
    """The queued-go floor is a clock rule: a `go` with no deadline (a work
    bound, `infinite`, `ponder`, `mate`) stamped 10 s ago still searches."""
    worker = _mock_worker()
    worker.run.return_value = SearchResult("e7e5", None, 16, ("e7e5",), 0)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        n0 = len(timed_out.stamps)
        engine.dispatch(stamp_received(parse_command(go), time.monotonic() - 10.0))
        end = time.monotonic() + 5.0
        while worker.run.call_count == 0 and time.monotonic() < end:
            time.sleep(0.005)
        if go in ("go infinite", "go ponder"):
            engine.dispatch(parse_command("stop"))
        assert timed_out.wait_bestmove_after(n0) is not None
        assert worker.run.call_count >= 1
        assert "behind earlier commands" not in timed_out.text()
    finally:
        engine.close()


def test_a_go_that_never_waited_is_not_tipped_by_its_own_setup(timed_out, monkeypatch) -> None:
    """The at-once rule is for a `go` that WAITED. `go movetime 52` with no
    overhead fits a 24 ms deadline; a fresh `go` whose tree sync takes 10 ms
    (its own setup, not a queue) still searches."""
    worker = _mock_worker()
    worker.run.return_value = SearchResult("e2e4", None, 16, ("e2e4",), 0)
    engine = Engine(worker, options=EngineOptions(move_overhead_ms=0))
    real_sync = engine._sync_tree_root

    def slow_sync(moves):
        time.sleep(0.01)
        return real_sync(moves)

    monkeypatch.setattr(engine, "_sync_tree_root", slow_sync)
    try:
        assert _mt_limits(52, 0).deadline_ms == 24
        engine.dispatch(parse_command("position startpos"))
        n0 = len(timed_out.stamps)
        engine.dispatch(stamp_received(parse_command("go movetime 52"), time.monotonic()))
        assert timed_out.wait_bestmove_after(n0) is not None
        assert "behind earlier commands" not in timed_out.text()
        assert worker.run.call_count >= 1
    finally:
        engine.close()


_MATE_IN_ONE = "7k/6pp/8/8/8/8/8/R5K1 w - - 0 1"  # first legal g1h2; Ra8# on the board
_QUEEN_ATTACKED = "r5k1/8/8/8/8/8/8/Q6K w - - 0 1"  # first legal h1h2 leaves the queen en prise
_NET_EVAL_S = 0.015  # a net slower than the clock share, inside the movetime floor


def _slow_eval(move: str, delay_s: float = _NET_EVAL_S) -> Any:
    def fresh(_board, **_kw):
        time.sleep(delay_s)
        return move
    return fresh


def _dispatch_at_once(engine: Engine, fen: str, go: str, age_s: float) -> None:
    engine.dispatch(parse_command(f"position fen {fen}"))
    engine.dispatch(stamp_received(parse_command(go), time.monotonic() - age_s))


@pytest.mark.parametrize(("fen", "net_move"), [
    (_MATE_IN_ONE, "a1a8"), (_QUEEN_ATTACKED, "a1a8"),
])
@pytest.mark.parametrize(("go", "age_s"), [
    ("go movetime 1000", 0.99), ("go movetime 30", 0.0),
])
def test_a_movetime_answered_at_once_plays_the_nets_move(timed_out, fen, net_move, go, age_s) -> None:
    """A `go movetime` answered at once (queued until its movetime was gone,
    or too short to search) gets one fresh root eval even with no time left
    before its last moment: the net's move (mate, the queen saved), not the
    first legal move. The eval here takes 15 ms: more than the old share (2 ms
    and 10 ms left), inside the 30 ms movetime floor."""
    worker = _mock_worker()
    worker.fresh_root_policy_bestmove.side_effect = _slow_eval(net_move)
    engine = Engine(worker, options=EngineOptions())
    try:
        n0 = len(timed_out.stamps)
        _dispatch_at_once(engine, fen, go, age_s)
        hit = timed_out.wait_bestmove_after(n0)
        assert hit is not None
        assert hit[1].split()[1] == net_move
        assert "source=fresh_root_eval" in timed_out.text()
        assert worker.run.call_count == 0
    finally:
        engine.close()


@pytest.mark.parametrize("go", ["go wtime 30", "go movetime 30 wtime 60"])
def test_a_clock_answered_at_once_keeps_its_eval_share(timed_out, go) -> None:
    """When a side clock binds (alone, or at or below the movetime once
    MoveOverheadMs is taken off) an overrun loses on time: the eval keeps its
    small share of the clock, and a 15 ms eval falls back to the first legal
    move."""
    worker = _mock_worker()
    worker.fresh_root_policy_bestmove.side_effect = _slow_eval("a1a8")
    engine = Engine(worker, options=EngineOptions())
    try:
        n0 = len(timed_out.stamps)
        _dispatch_at_once(engine, _QUEEN_ATTACKED, go, 0.0)
        hit = timed_out.wait_bestmove_after(n0)
        assert hit is not None
        assert hit[1].split()[1] == "h1h2"
        assert "source=first_legal" in timed_out.text()
    finally:
        engine.close()


def test_a_movetime_eval_slower_than_its_floor_falls_back_to_first_legal(timed_out) -> None:
    worker = _mock_worker()
    worker.fresh_root_policy_bestmove.side_effect = _slow_eval("a1a8", delay_s=0.3)
    engine = Engine(worker, options=EngineOptions())
    try:
        n0 = len(timed_out.stamps)
        t0 = time.monotonic()
        _dispatch_at_once(engine, _QUEEN_ATTACKED, "go movetime 30", 0.0)
        hit = timed_out.wait_bestmove_after(n0)
        assert hit is not None
        assert hit[1].split()[1] == "h1h2"
        assert "fresh root-eval fallback timed out after 30 ms" in timed_out.text()
        assert hit[0] - t0 < 0.3
    finally:
        engine.close()


def test_a_movetime_at_once_eval_honours_searchmoves(timed_out) -> None:
    """The eval is asked for the `searchmoves` set, and a move outside it is
    never played."""
    worker = _mock_worker()
    seen: list[tuple[str, ...]] = []

    def fresh(_board, *, searchmoves=()):
        seen.append(tuple(searchmoves))
        return "a1a8"  # outside the set

    worker.fresh_root_policy_bestmove.side_effect = fresh
    engine = Engine(worker, options=EngineOptions())
    try:
        n0 = len(timed_out.stamps)
        _dispatch_at_once(engine, _QUEEN_ATTACKED, "go movetime 30 searchmoves a1b2 h1g1", 0.0)
        hit = timed_out.wait_bestmove_after(n0)
        assert hit is not None
        assert seen == [("a1b2", "h1g1")]
        assert hit[1].split()[1] == "a1b2"
    finally:
        engine.close()


def test_a_movetime_at_once_answer_with_one_legal_move_plays_it(timed_out) -> None:
    worker = _mock_worker()
    engine = Engine(worker, options=EngineOptions())
    try:
        n0 = len(timed_out.stamps)
        _dispatch_at_once(engine, "7k/8/8/8/8/8/6q1/7K w - - 0 1", "go movetime 30", 0.0)
        hit = timed_out.wait_bestmove_after(n0)
        assert hit is not None
        assert hit[1].split()[1] == "h1g2"
    finally:
        engine.close()
