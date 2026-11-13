"""The deadline watchdog: a clocked `go` whose search stalls (a hung evaluator
batch, an overrunning compaction) is still answered once, before the flag,
without touching the worker, and the late result never prints a second
bestmove.

Ordering tests drive `Engine._watchdog_fire` directly with the armed entry and
gate the worker with events, so both orders of every race are forced instead
of left to timing. End-to-end tests let the real watchdog thread fire on a
worker that blocks until released.
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import chess
import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchResult


class _Out:
    """Thread-safe stand-in for stdout, with a wait for the n-th bestmove.

    The REAL `_println` writes here, so its silencing of parked / watchdogged
    threads (and the claim owner's bypass) is part of what every test sees."""

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

    def bestmoves(self) -> list[str]:
        with self._cond:
            return [ln.split()[1] for ln in self.lines if ln.startswith("bestmove ")]

    def wait_bestmoves(self, n: int, timeout: float = 5.0) -> list[str]:
        with self._cond:
            self._cond.wait_for(
                lambda: sum(ln.startswith("bestmove ") for ln in self.lines) >= n, timeout,
            )
        return self.bestmoves()

    def text(self) -> str:
        with self._cond:
            return "\n".join(self.lines)


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    # Swap the engine module's view of stdout (pytest re-installs its own
    # sys.stdout at the start of each test's call phase, after fixtures run).
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _worker() -> MagicMock:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.realized_search_path.return_value = "gumbel"
    return worker


def _ok(board: chess.Board, move: str | None = None) -> SearchResult:
    move = move or next(iter(board.legal_moves)).uci()
    return SearchResult(move, None, 8, (move,), 0)


def _publish(info_cb, move: str, *, multipv: int | None = None) -> None:
    info_cb(nodes=64, elapsed_ms=5, score_cp=10, pv=(move,), tbhits=0,
            score_mate=None, multipv=multipv, wdl=None)


class _Gate:
    """A worker.run that blocks until released (a hung evaluator batch)."""

    def __init__(self, before=None, result_move: str | None = None) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.before = before
        self.result_move = result_move
        self.calls = 0

    def __call__(self, board, **kw):
        self.calls += 1
        if self.before is not None:
            self.before(board, kw)
        self.entered.set()
        assert self.release.wait(10)
        return _ok(board, self.result_move)


def _engine(worker, **opts) -> Engine:
    return Engine(worker, options=EngineOptions(**opts))


def _join_all(engine: Engine) -> None:
    for thread in [engine._search_thread, *engine._orphan_threads]:
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive()


def _legal(board: chess.Board, moves: list[str]) -> None:
    for uci in moves:
        assert chess.Move.from_uci(uci) in board.legal_moves, uci


# -- end to end: the real watchdog thread fires on a hung worker -------------


def test_hang_at_the_first_batch_is_answered_once_before_the_flag(out) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)  # MoveOverheadMs 30
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        started = time.monotonic()
        engine.dispatch(parse_command("go movetime 100"))
        assert gate.entered.wait(5)
        moves = out.wait_bestmoves(1)
        elapsed_ms = (time.monotonic() - started) * 1000
        assert len(moves) == 1
        board = chess.Board()
        board.push_uci("e2e4")
        _legal(board, moves)
        # Search deadline 70 ms (100 - 30); fires 20 ms after it. Timing is a
        # lower bound only: the upper bound is pinned exactly by the
        # fire-time test below, not by a wall-clock assertion.
        assert elapsed_ms >= 70
        text = out.text()
        assert "deadline watchdog answers unsearched source=first_legal" in text
        assert engine._watchdog_fires == 1
        assert engine._consecutive_search_failures == 1  # counted in the streak
        # The late result is discarded by the claim: no second bestmove.
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
        assert engine._consecutive_search_failures == 1  # not reset by the late result
    finally:
        gate.release.set()
        engine.close()


def test_hang_mid_search_answers_with_the_published_pv_move(out) -> None:
    gate = _Gate(before=lambda board, kw: _publish(kw["info_cb"], "g1f3"))
    worker = _worker()
    worker.run.side_effect = gate
    worker.root_policy_bestmove.return_value = "d2d4"  # the snapshot outranks it
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("position startpos"))
        engine.dispatch(parse_command("go wtime 2000 btime 2000"))
        moves = out.wait_bestmoves(1)
        assert moves == ["g1f3"]
        assert "source=last_pv move=g1f3" in out.text()
        gate.release.set()
        _join_all(engine)
        assert out.bestmoves() == ["g1f3"]
    finally:
        gate.release.set()
        engine.close()


def test_without_a_pv_the_cached_root_policy_answers(out) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    worker.root_policy_bestmove.return_value = "d2d4"
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("go movetime 60"))
        assert out.wait_bestmoves(1) == ["d2d4"]
        assert "source=root_policy move=d2d4" in out.text()
        # Never a fresh eval on a hung evaluator.
        worker.fresh_root_policy_bestmove.assert_not_called()
    finally:
        gate.release.set()
        engine.close()


def test_hang_during_ponder_then_ponderhit_arms_the_real_clock(out) -> None:
    # The ponder phase searches the board BEFORE e7e5 and publishes a PV there.
    # A uci string that is also legal on the real board ("g1f3") shows the
    # snapshot is keyed by position, not merely re-validated for legality.
    gate = _Gate(before=lambda board, kw: _publish(kw["info_cb"], "g1f3"))
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 80"))
        assert gate.entered.wait(5)
        assert engine._watchdog.armed is None  # ponder: no clock yet
        engine.dispatch(parse_command("ponderhit"))
        armed = engine._watchdog.armed
        assert armed is not None
        assert armed.started_at == engine._ponderhit_at
        moves = out.wait_bestmoves(1)
        real = chess.Board()
        real.push_uci("e2e4")
        real.push_uci("e7e5")
        _legal(real, moves)
        text = out.text()
        assert "source=last_pv" not in text
        assert "source=first_legal" in text
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
    finally:
        gate.release.set()
        engine.close()


class _Wedged:
    """Proxy for an evaluator whose batch calls block until released."""

    def __init__(self, inner, release: threading.Event, entered: threading.Event) -> None:
        self._inner, self._release, self._entered = inner, release, entered

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr) or not name.startswith("evaluate"):
            return attr

        def wedged(*args, **kwargs):
            self._entered.set()
            assert self._release.wait(20)
            return attr(*args, **kwargs)

        return wedged


def test_multi_gpu_cpu_cpu_one_hung_evaluator_still_answers(out) -> None:
    """A real two-device pool on cpu,cpu; device 1's evaluator wedges on its
    first batch, so the search never returns. The watchdog answers from the
    root prior the primary already cached, without touching any evaluator."""
    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.uci.search import SearchWorker

    def evaluator():
        cfg = ModelConfig(input_extra_features="v1", embed_dim=16, num_layers=1,
                          num_heads=2, ffn_mult=2.0)
        model = build_model(cfg)
        model.eval()
        return ThreadSafeGPUDispatcher(DirectGPUEvaluator(
            model, device="cpu", max_batch=64, use_amp=False, n_slots=2,
        ))

    release, entered = threading.Event(), threading.Event()
    worker = SearchWorker(
        evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    worker.install_multi_gpu_pucv(
        [evaluator(), _Wedged(evaluator(), release, entered)], gather=8, as_factories=False,
    )
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 300"))
        moves = out.wait_bestmoves(1, timeout=20)
        assert entered.is_set()  # device 1 really was wedged when it fired
        assert len(moves) == 1
        board = chess.Board()
        board.push_uci("e2e4")
        _legal(board, moves)
        text = out.text()
        assert "deadline watchdog answers unsearched" in text
        assert "source=first_legal" not in text  # the cached prior or a PV answered
        release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
    finally:
        release.set()
        engine.close()
        worker.close()


# -- arming -----------------------------------------------------------------


@pytest.mark.parametrize("go", [
    "go infinite", "go ponder wtime 1000 btime 1000", "go nodes 50", "go depth 3",
])
@pytest.mark.usefixtures("out")
def test_unclocked_searches_are_not_armed(go) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command(go))
        assert gate.entered.wait(5)
        assert engine._watchdog.armed is None
        assert engine._watchdog.thread is None  # never even started
    finally:
        gate.release.set()
        engine.dispatch(parse_command("stop"))
        _join_all(engine)
        engine.close()


@pytest.mark.parametrize(("overhead", "grace"), [(0, 10.0), (30, 20.0), (100, 75.0), (1000, 750.0)])
@pytest.mark.usefixtures("out")
def test_fire_time_is_after_the_search_deadline_and_before_the_flag(overhead, grace) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker, move_overhead_ms=overhead)
    try:
        engine.dispatch(parse_command("go movetime 5000"))
        armed = engine._watchdog.armed
        assert armed is not None
        deadline_ms = 5000 - overhead
        fire_ms = (armed.fire_at - armed.started_at) * 1000
        assert fire_ms == pytest.approx(deadline_ms + grace)
        assert armed.started_at == engine._go_received_at
        if overhead >= 20:  # the floor: below 20 ms the engine already races the flag
            assert fire_ms <= 5000 - 10 + 1e-6  # >= 10 ms margin before the flag (float slack)
    finally:
        gate.release.set()
        _join_all(engine)
        engine.close()


def test_a_healthy_short_movetime_search_never_fires(out) -> None:
    """Thirty searches that each spend their whole clock and then finish an
    in-flight batch (5 ms) after the hard stop, as a healthy search on a GPU
    does: zero watchdog fires."""
    worker = _worker()

    def spend_the_clock(board, **kw):
        kw["stop_event"].wait(5)  # the phase timer sets it at the deadline
        time.sleep(0.005)  # the batch already in flight completes
        return _ok(board)

    worker.run.side_effect = spend_the_clock
    engine = _engine(worker)
    try:
        for _ in range(30):
            engine.dispatch(parse_command("go movetime 50"))
            thread = engine._search_thread
            assert thread is not None
            thread.join(5)
        assert len(out.bestmoves()) == 30
        assert engine._watchdog_fires == 0
        assert "deadline watchdog" not in out.text()
    finally:
        engine.close()


# -- ordering: exactly one bestmove under both orders of each race -----------


def _armed_go(*, gate: _Gate, go: str = "go movetime 60000"):
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)
    engine.dispatch(parse_command(go))
    assert gate.entered.wait(5)
    entry = engine._watchdog.armed
    assert entry is not None
    return engine, entry


def test_watchdog_first_then_the_search_finishes(out) -> None:
    gate = _Gate(result_move="e2e4")
    engine, entry = _armed_go(gate=gate)
    try:
        engine._watchdog_fire(entry)
        assert len(out.bestmoves()) == 1
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1  # the generation's claim is taken
    finally:
        gate.release.set()
        engine.close()


def test_search_finishes_first_then_the_watchdog(out) -> None:
    gate = _Gate(result_move="e2e4")
    engine, entry = _armed_go(gate=gate)
    try:
        gate.release.set()
        _join_all(engine)
        assert out.bestmoves() == ["e2e4"]
        assert engine._watchdog.armed is None  # disarmed by the emit
        engine._watchdog_fire(entry)  # as if it had already popped the entry
        assert out.bestmoves() == ["e2e4"]
        assert engine._watchdog_fires == 0
    finally:
        engine.close()


def test_stop_first_then_the_watchdog(out, monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_GO_WAIT_NO_CLOCK_S", 0.02)
    gate = _Gate()
    engine, entry = _armed_go(gate=gate)
    try:
        engine.dispatch(parse_command("stop"))  # parked and answered
        assert len(out.bestmoves()) == 1
        engine._watchdog_fire(entry)
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
        assert engine._watchdog_fires == 0
    finally:
        gate.release.set()
        engine.close()


def test_watchdog_first_then_stop(out, monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_GO_WAIT_NO_CLOCK_S", 0.02)
    gate = _Gate()
    engine, entry = _armed_go(gate=gate)
    try:
        engine._watchdog_fire(entry)
        assert len(out.bestmoves()) == 1
        engine.dispatch(parse_command("stop"))  # parks the stuck search, no answer
        assert len(out.bestmoves()) == 1
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
    finally:
        gate.release.set()
        engine.close()


def test_a_newer_go_after_the_watchdog_answered_is_served_normally(out, monkeypatch) -> None:
    """The stuck search becomes an orphan; the next go is answered once."""
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.02)
    gate = _Gate()
    engine, entry = _armed_go(gate=gate, go="go movetime 60000")
    try:
        engine._watchdog_fire(entry)
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 40"))
        moves = out.wait_bestmoves(2)
        assert len(moves) == 2
        board = chess.Board()
        board.push_uci("e2e4")
        _legal(board, moves[1:])
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 2
    finally:
        gate.release.set()
        engine.close()


def test_a_stale_entry_never_answers_a_newer_go(out, monkeypatch) -> None:
    """go 2 arrives while go 1 is still unanswered (its search hung, its
    watchdog not yet due): go 2 gets the busy-worker answer, and go 1's entry
    firing afterwards must not print a second bestmove into that window."""
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.02)
    gate = _Gate()
    engine, stale = _armed_go(gate=gate, go="go movetime 60000")
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 60000"))
        before = out.wait_bestmoves(1)  # go 2's busy-worker answer
        assert len(before) == 1
        engine._watchdog_fire(stale)
        assert out.bestmoves() == before
        assert engine._watchdog_fires == 0
    finally:
        gate.release.set()
        engine.dispatch(parse_command("stop"))
        _join_all(engine)
        engine.close()


def test_stop_arriving_while_the_watchdog_answers(out, monkeypatch) -> None:
    """The watchdog has claimed and is computing its answer when `stop` parks
    the search: park sees the claim and adds nothing; the watchdog's answer
    (from its own board copy, never `_live_go`) still prints. One bestmove."""
    monkeypatch.setattr(engine_module, "_GO_WAIT_NO_CLOCK_S", 0.02)
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)
    in_fallback, finish = threading.Event(), threading.Event()
    real = engine._bestmove_fallback

    def held(*args, **kwargs):
        if threading.current_thread() is engine._watchdog.thread:
            in_fallback.set()
            assert finish.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_bestmove_fallback", held)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 60"))
        assert in_fallback.wait(5)  # claimed, answer not yet printed
        engine.dispatch(parse_command("stop"))  # parks the search; must not answer
        assert out.bestmoves() == []
        finish.set()
        moves = out.wait_bestmoves(1)
        assert len(moves) == 1
        board = chess.Board()
        board.push_uci("e2e4")
        _legal(board, moves)
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
        assert engine._consecutive_search_failures == 1  # the watchdog's answer, once
    finally:
        finish.set()
        gate.release.set()
        engine.close()


def test_after_the_watchdog_answers_the_stuck_search_stays_silent(out) -> None:
    """The owner print is the watchdog's last act; the stuck search's later
    lines (an info line, its own bestmove) never reach stdout."""
    late = threading.Event()

    worker = _worker()

    def run(board, **kw):
        late.wait(10)
        _publish(kw["info_cb"], "a2a3")
        return _ok(board, "a2a4")

    worker.run.side_effect = run
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("go movetime 60"))
        assert len(out.wait_bestmoves(1)) == 1
        before = out.text()
        late.set()
        _join_all(engine)
        assert out.text() == before
    finally:
        late.set()
        engine.close()


def test_a_reply_only_go_is_never_armed(out, monkeypatch) -> None:
    """A go whose handler raised is answered by the reply-only thread (counted
    once at the go, answered by park if it stalls); the watchdog stays out."""
    worker = _worker()
    engine = _engine(worker)

    def boom() -> None:
        raise RuntimeError("handler fault")

    monkeypatch.setattr(engine, "_note_leaf_tb_inactive", boom)
    try:
        engine.dispatch(parse_command("go movetime 60"))
        assert len(out.wait_bestmoves(1)) == 1
        assert engine._watchdog.armed is None
        assert engine._watchdog.thread is None
        assert engine._consecutive_search_failures == 1
    finally:
        engine.close()


# -- interactions -----------------------------------------------------------


FORCED = "7k/8/8/8/8/8/6q1/7K w - - 0 1"  # Kxg2 is the only legal move


def test_forced_root_healthy_banks_early_and_never_fires(out) -> None:
    """#1265: a forced root banks after its first chunk; the emit disarms."""
    worker = _worker()
    worker.run.side_effect = lambda b, **kw: _ok(b, "h1g2")
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command(f"position fen {FORCED}"))
        engine.dispatch(parse_command("go wtime 60000 btime 60000"))
        assert engine._search_thread is not None
        engine._search_thread.join(5)
        assert out.bestmoves() == ["h1g2"]
        assert engine._watchdog.armed is None
        assert engine._watchdog_fires == 0
    finally:
        engine.close()


def test_forced_root_with_a_hung_evaluator_answers_the_only_move(out) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command(f"position fen {FORCED}"))
        engine.dispatch(parse_command("go movetime 60"))
        assert out.wait_bestmoves(1) == ["h1g2"]
        assert engine._watchdog_fires == 1
    finally:
        gate.release.set()
        engine.close()


def test_compaction_overrunning_the_clock_is_answered(out) -> None:
    """#1264: the deferred compaction runs inside the worker under this move's
    clock and cannot be interrupted. Within budget the search answers itself;
    an overrun past the deadline is a stall and the watchdog answers."""
    worker = _worker()
    compaction_done = threading.Event()

    def compacting(board, **kw):
        if worker.run.call_count == 1:
            kw["stop_event"].wait(5)  # compaction within budget, then search
            return _ok(board, "e2e4")
        compaction_done.wait(10)  # compaction overruns the whole clock
        return _ok(board, "d2d4")

    worker.run.side_effect = compacting
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("go movetime 60"))
        assert engine._search_thread is not None
        engine._search_thread.join(5)
        assert out.bestmoves() == ["e2e4"]
        assert engine._watchdog_fires == 0
        engine.dispatch(parse_command("go movetime 60"))
        moves = out.wait_bestmoves(2)
        assert len(moves) == 2
        assert moves[1] != "d2d4"
        assert engine._watchdog_fires == 1
        compaction_done.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 2
    finally:
        compaction_done.set()
        engine.close()


def test_watchdog_answer_cost_is_far_below_margin(out) -> None:
    """The answer the margin budgets for: a snapshot read and two lines."""
    gate = _Gate(before=lambda board, kw: _publish(kw["info_cb"], "g1f3"))
    engine, entry = _armed_go(gate=gate)
    try:
        t0 = time.perf_counter()
        engine._watchdog_fire(entry)
        cost_ms = (time.perf_counter() - t0) * 1000
        assert out.bestmoves() == ["g1f3"]
        assert cost_ms < engine_module._WATCHDOG_MIN_MARGIN_MS / 2
    finally:
        gate.release.set()
        engine.close()


@pytest.mark.usefixtures("out")
def test_the_watchdog_thread_is_one_and_stops_on_close() -> None:
    worker = _worker()
    worker.run.side_effect = lambda b, **kw: _ok(b)
    engine = _engine(worker)
    for _ in range(5):
        engine.dispatch(parse_command("go movetime 1000"))
        assert engine._search_thread is not None
        engine._search_thread.join(5)
    thread = engine._watchdog.thread
    assert thread is not None
    assert thread.is_alive()
    named = [t for t in threading.enumerate() if t.name == "uci-deadline-watchdog"]
    assert named.count(thread) == 1
    engine.close()
    assert not thread.is_alive()
    # Closed: a later arm is refused rather than restarting the thread.
    engine.dispatch(parse_command("go movetime 1000"))
    if engine._search_thread is not None:
        engine._search_thread.join(5)
    assert engine._watchdog.thread is thread
    assert not thread.is_alive()
