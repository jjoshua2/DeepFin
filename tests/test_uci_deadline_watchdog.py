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


# overhead 0: the last moment is never before the search's own deadline, which
# is then the budget itself (an engine with no overhead already races the flag).
@pytest.mark.parametrize(("overhead", "margin"), [(0, 0.0), (30, 3.0), (100, 3.0), (1000, 3.0)])
@pytest.mark.usefixtures("out")
def test_thresholds_are_the_search_deadline_and_the_budget_minus_the_margin(overhead, margin) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker, move_overhead_ms=overhead)
    try:
        engine.dispatch(parse_command("go movetime 5000"))
        armed = engine._watchdog.armed
        assert armed is not None
        assert armed.started_at == engine._go_received_at
        assert (armed.deadline_at - armed.started_at) * 1000 == pytest.approx(5000 - overhead)
        assert armed.last_moment_at is not None
        assert (armed.last_moment_at - armed.started_at) * 1000 == pytest.approx(5000 - margin)
        assert armed.check_at == armed.deadline_at
    finally:
        gate.release.set()
        _join_all(engine)
        engine.close()


@pytest.mark.usefixtures("out")
def test_a_clock_game_budget_is_the_side_to_moves_remaining_clock() -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))  # black to move
        engine.dispatch(parse_command("go wtime 7000 btime 9000"))
        armed = engine._watchdog.armed
        assert armed is not None
        assert armed.last_moment_at is not None
        assert (armed.last_moment_at - armed.started_at) * 1000 == pytest.approx(9000 - 3.0)
    finally:
        gate.release.set()
        _join_all(engine)
        engine.close()


@pytest.mark.parametrize(("go", "margin", "opts"), [
    ("go movetime 200", True, {}),
    ("go wtime 2000 btime 2000", False, {}),
    ("go nodes 64", False, {}),
    ("go infinite", False, {}),
    ("go movetime 200 nodes 64", True, {}),
    # Time management OFF (the A/B baseline arm): a clock game has no
    # optimum_ms, and must still not get the margin -- that arm reproduces
    # the pre-time-management engine exactly.
    ("go wtime 2000 btime 2000", False, {"optimum_fraction": 0.0}),
    ("go wtime 2000 btime 2000 nodes 64", False, {}),
])
@pytest.mark.usefixtures("out")
def test_only_movetime_asks_the_worker_for_the_fixed_time_stop_margin(go, margin, opts) -> None:
    worker = _worker()
    seen: list[dict] = []

    def run(board, **kw):
        seen.append(kw)
        if kw["deadline"].deadline_ms is None:
            kw["stop_event"].wait(5)
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker, **opts)
    try:
        engine.dispatch(parse_command(go))
        if go == "go infinite":
            engine.dispatch(parse_command("stop"))
        _join_all(engine)
        assert seen[0]["fixed_time_margin"] is margin
        assert callable(seen[0]["on_chunk"])  # the watchdog heartbeat
    finally:
        engine.close()


@pytest.mark.usefixtures("out")
def test_a_ponder_phase_never_asks_for_the_margin_and_ponderhit_does() -> None:
    worker = _worker()
    seen: list[dict] = []
    released = threading.Event()

    def run(board, **kw):
        seen.append(kw)
        if len(seen) == 1:
            kw["stop_event"].wait(5)
            released.set()
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 200"))
        engine.dispatch(parse_command("ponderhit"))
        assert released.wait(5)
        _join_all(engine)
        assert seen[0]["fixed_time_margin"] is False  # ponder phase
        assert seen[1]["fixed_time_margin"] is True  # real phase, movetime
    finally:
        engine.close()


@pytest.mark.usefixtures("out")
def test_a_heartbeat_is_published_per_chunk_for_the_current_go() -> None:
    worker = _worker()

    def run(board, **kw):
        for n in (8, 16, 24):
            kw["on_chunk"](n)
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("go movetime 200"))
        _join_all(engine)
        beat = engine._heartbeat
        assert beat is not None
        assert beat[0] == engine._search_gen
        assert beat[2] is not None  # a chunk time after the second chunk
        assert engine._prev_chunk_ms == beat[2]
        assert engine._prev_first_chunk_ms is not None
    finally:
        engine.close()


def _silenced_phase(engine, *, first_chunk_s: float, then_chunks: int, gap_s: float) -> None:
    """Drive one movetime phase whose search the watchdog answers (silences)
    mid-run: a first chunk after ``first_chunk_s``, then the thread is
    silenced, then ``then_chunks`` more chunks ``gap_s`` apart."""
    worker = engine._worker

    def run(board, **kw):
        time.sleep(first_chunk_s)
        kw["on_chunk"](8)
        engine_module._SILENCED_THREADS.add(threading.current_thread())
        for i in range(then_chunks):
            time.sleep(gap_s)
            kw["on_chunk"](16 + 8 * i)
        return _ok(board)

    worker.run.side_effect = run
    engine.dispatch(parse_command("go movetime 60000"))
    _join_all(engine)


@pytest.mark.usefixtures("out")
def test_a_search_the_watchdog_answered_never_teaches_the_chunk_scale() -> None:
    engine = _engine(_worker())
    try:
        engine._prev_chunk_ms = 20.0
        engine._prev_first_chunk_ms = 60.0
        # One chunk, silenced, then a 150 ms "chunk" (a hang between chunks).
        _silenced_phase(engine, first_chunk_s=0.0, then_chunks=2, gap_s=0.15)
        assert engine._prev_chunk_ms == 20.0
        assert engine._prev_first_chunk_ms == 60.0
    finally:
        engine.close()


@pytest.mark.usefixtures("out")
def test_a_slow_first_chunk_then_a_hang_does_not_teach_the_first_chunk_scale() -> None:
    engine = _engine(_worker())
    try:
        engine._prev_first_chunk_ms = 60.0
        _silenced_phase(engine, first_chunk_s=0.2, then_chunks=0, gap_s=0.0)
        assert engine._prev_first_chunk_ms == 60.0
    finally:
        engine.close()


@pytest.mark.usefixtures("out")
def test_a_healthy_phase_commits_both_samples() -> None:
    worker = _worker()

    def run(board, **kw):
        time.sleep(0.05)
        kw["on_chunk"](8)
        time.sleep(0.03)
        kw["on_chunk"](16)
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("go movetime 60000"))
        _join_all(engine)
        assert engine._prev_first_chunk_ms is not None
        assert engine._prev_first_chunk_ms >= 45.0
        assert engine._prev_chunk_ms is not None
        assert engine._prev_chunk_ms >= 25.0
    finally:
        engine.close()


def test_the_heartbeat_carries_the_threads_own_generation(out) -> None:
    """`_run_one_phase` stamps the heartbeat with the generation it is given
    (the go's), not whatever `_search_gen` is when the phase starts."""
    worker = _worker()
    seen: list = []

    def run(board, **kw):
        kw["on_chunk"](8)
        seen.append(engine._heartbeat)
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker)
    try:
        engine._search_gen = 41
        limits = engine_module.SearchLimits(deadline_ms=1000, fixed_time=True)
        engine._run_one_phase(limits, is_ponder=False, board=chess.Board(), gen=7)
        assert seen[0][0] == 7
    finally:
        engine.close()
    del out


def test_the_budget_in_the_info_string_is_exact(out) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)
    try:
        engine.dispatch(parse_command("go movetime 200"))
        assert len(out.wait_bestmoves(1)) == 1
        assert "before the 200 ms budget" in out.text()
    finally:
        gate.release.set()
        engine.close()


# -- the progress rule, decided deterministically by `_watchdog_poll` --------


def _polled(engine, *, since_start: float, deadline_in: float, last_moment_in: float | None,
            beat_ago: float | None, beat_chunk_ms: float | None, gen_offset: int = 0):
    """Poll a synthetic entry for the engine's current go; times in seconds
    relative to now. Returns ``(fired, next_check_in_s)``."""
    now = time.monotonic()
    gen = engine._search_gen
    thread = threading.Thread(target=lambda: None)
    entry = engine_module._WatchdogEntry(
        gen=gen, check_at=now, deadline_at=now + deadline_in,
        last_moment_at=None if last_moment_in is None else now + last_moment_in,
        budget_ms=None,
        board=chess.Board(), searchmoves=(), thread=thread,
        stops=(threading.Event(),), started_at=now - since_start,
    )
    engine._heartbeat = (None if beat_ago is None
                         else (gen + gen_offset, now - beat_ago, beat_chunk_ms))
    before = engine._watchdog_fires
    nxt = engine._watchdog_poll(entry)
    fired = engine._watchdog_fires > before
    return fired, (None if nxt is None else nxt - time.monotonic())


@pytest.fixture
def poll_engine(out):
    del out  # stdout capture only
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    engine = _engine(worker)
    engine.dispatch(parse_command("go movetime 600000"))  # a live, unanswered go
    assert gate.entered.wait(5)
    yield engine
    gate.release.set()
    engine.close()


def test_progress_rule_leaves_a_progressing_search_alone(poll_engine) -> None:
    # Past the deadline, last chunk 30 ms ago, chunks of 100 ms: S = 400 ms.
    fired, nxt = _polled(poll_engine, since_start=1.0, deadline_in=-0.1, last_moment_in=10.0,
                         beat_ago=0.03, beat_chunk_ms=100.0)
    assert not fired
    assert nxt == pytest.approx(0.37, abs=0.02)  # looks again when S would run out


def test_progress_rule_fires_on_no_progress_past_the_deadline(poll_engine) -> None:
    fired, _ = _polled(poll_engine, since_start=1.0, deadline_in=-0.1, last_moment_in=10.0,
                       beat_ago=0.45, beat_chunk_ms=100.0)
    assert fired


def test_progress_rule_has_a_300_ms_floor(poll_engine) -> None:
    # Tiny chunks (10 ms) still allow 300 ms without one (OS / GIL stalls).
    fired, nxt = _polled(poll_engine, since_start=1.0, deadline_in=-0.1, last_moment_in=10.0,
                         beat_ago=0.2, beat_chunk_ms=10.0)
    assert not fired
    assert nxt == pytest.approx(0.1, abs=0.02)


def test_progress_rule_waits_for_the_deadline(poll_engine) -> None:
    fired, nxt = _polled(poll_engine, since_start=1.0, deadline_in=0.3, last_moment_in=10.0,
                         beat_ago=5.0, beat_chunk_ms=100.0)
    assert not fired
    assert nxt == pytest.approx(0.3, abs=0.02)


def test_progress_rule_scales_with_the_chunk_time(poll_engine) -> None:
    # 500 ms without a chunk is a stall for 100 ms chunks, not for 200 ms ones.
    fired, _ = _polled(poll_engine, since_start=2.0, deadline_in=-0.1, last_moment_in=10.0,
                       beat_ago=0.5, beat_chunk_ms=200.0)
    assert not fired


def test_progress_rule_ignores_another_gos_heartbeat(poll_engine) -> None:
    # A fresh heartbeat from a parked search's generation is not progress.
    fired, _ = _polled(poll_engine, since_start=2.0, deadline_in=-0.1, last_moment_in=10.0,
                       beat_ago=0.0, beat_chunk_ms=100.0, gen_offset=-1)
    assert fired


def test_before_its_second_chunk_the_previous_searchs_chunk_time_is_used(poll_engine) -> None:
    poll_engine._prev_chunk_ms = 100.0  # S = 400 ms
    fired, _ = _polled(poll_engine, since_start=1.0, deadline_in=-0.1, last_moment_in=10.0,
                       beat_ago=0.45, beat_chunk_ms=None)
    assert fired


def test_with_no_chunk_time_at_all_the_stall_limit_is_1_s(poll_engine) -> None:
    poll_engine._prev_chunk_ms = None
    fired, nxt = _polled(poll_engine, since_start=2.0, deadline_in=-0.1, last_moment_in=10.0,
                         beat_ago=0.6, beat_chunk_ms=None)
    assert not fired
    assert nxt == pytest.approx(0.4, abs=0.02)


def test_before_any_chunk_the_previous_first_chunk_time_is_used(poll_engine) -> None:
    # No heartbeat yet in this search; recent searches took 100 ms to their
    # first chunk: S = 400 ms from the clock start.
    poll_engine._prev_first_chunk_ms = 100.0
    poll_engine._prev_chunk_ms = 5.0  # steady chunks are not the scale here
    fired, nxt = _polled(poll_engine, since_start=0.3, deadline_in=-0.1, last_moment_in=10.0,
                         beat_ago=None, beat_chunk_ms=None)
    assert not fired
    assert nxt == pytest.approx(0.1, abs=0.02)
    fired, _ = _polled(poll_engine, since_start=0.45, deadline_in=-0.1, last_moment_in=10.0,
                       beat_ago=None, beat_chunk_ms=None)
    assert fired


def test_one_quick_search_does_not_shrink_the_scale() -> None:
    assert engine_module._decayed_max(None, 80.0) == 80.0
    assert engine_module._decayed_max(80.0, 20.0) == pytest.approx(72.0)
    assert engine_module._decayed_max(80.0, 120.0) == 120.0


def test_a_pre_ponderhit_heartbeat_of_the_same_go_is_not_progress(poll_engine) -> None:
    # The ponder phase's last chunk (same generation) came 0.45 s ago, before
    # ponderhit 0.35 s ago: the real phase has completed no chunk yet, so the
    # first-chunk scale (400 ms) applies from ponderhit -- not a stall yet.
    poll_engine._prev_first_chunk_ms = 100.0
    fired, _ = _polled(poll_engine, since_start=0.35, deadline_in=-0.1, last_moment_in=10.0,
                       beat_ago=0.45, beat_chunk_ms=100.0)
    assert not fired


def test_the_last_moment_fires_even_on_a_progressing_search(poll_engine) -> None:
    fired, _ = _polled(poll_engine, since_start=1.0, deadline_in=-0.1, last_moment_in=-0.001,
                       beat_ago=0.0, beat_chunk_ms=40.0)
    assert fired


def test_the_next_look_is_never_after_the_last_moment(poll_engine) -> None:
    fired, nxt = _polled(poll_engine, since_start=1.0, deadline_in=-0.1, last_moment_in=0.05,
                         beat_ago=0.0, beat_chunk_ms=400.0)  # S = 1.6 s
    assert not fired
    assert nxt == pytest.approx(0.05, abs=0.02)


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
        assert cost_ms < engine_module._WATCHDOG_LAST_MOMENT_MS / 2
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


def test_watchdog_wake_latency_under_gil_contention_fits_the_last_moment_margin() -> None:
    """The last-moment margin must cover how late the watchdog thread wakes
    while another Python thread holds the GIL (measured here, in-process)."""
    lateness: list[float] = []
    done = threading.Event()
    n = 100

    def poll(entry):
        lateness.append((time.monotonic() - entry.check_at) * 1000.0)
        if len(lateness) >= n:
            done.set()
        return None

    stop = threading.Event()

    def hog() -> None:
        # Holds the GIL in pure Python half the time, 1 ms at a stretch. (The
        # wake-up waits for the holder's current stretch: measured p99 0.4 ms
        # here, 2.2 ms with 2 ms stretches, 8 ms with a thread that never
        # sleeps -- see the TCEC_SETUP caveat.)
        x = 0
        while not stop.is_set():
            end = time.perf_counter() + 0.001
            while time.perf_counter() < end:
                x += 1
            time.sleep(0.001)

    watchdog = engine_module._DeadlineWatchdog(poll)
    hog_thread = threading.Thread(target=hog, daemon=True)
    hog_thread.start()
    try:
        for i in range(n):
            now = time.monotonic()
            watchdog.arm(engine_module._WatchdogEntry(
                gen=1, check_at=now + 0.005, deadline_at=now, last_moment_at=None, budget_ms=None,
                board=chess.Board(), searchmoves=(), thread=threading.current_thread(),
                stops=(), started_at=now,
            ))
            deadline = time.monotonic() + 2.0
            while len(lateness) < i + 1 and time.monotonic() < deadline:
                time.sleep(0.001)
        assert done.wait(5)
    finally:
        stop.set()
        hog_thread.join(5)
        watchdog.close()
    lateness.sort()
    p99 = lateness[int(len(lateness) * 0.99) - 1]
    # Answer cost is < 0.5 ms (`test_watchdog_answer_cost_is_far_below_margin`).
    assert p99 + 0.5 < engine_module._WATCHDOG_LAST_MOMENT_MS, lateness
