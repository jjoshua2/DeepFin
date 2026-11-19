"""Time-forfeit fixes found by the CPU soak (735 games, 7 forfeits).

- A clocked deadline never exceeds what the clock allows, and a clock too short
  to search is answered at once, not searched for the 20 ms floor (F3).
- Sudden death (no increment) plans for 20 moves left, not 8 (F3).
- A later chunk of a timed search can shrink below ``ChunkSims`` (F2), and the
  first chunk is sized from the worker's own measured chunk rate (F5).
- The busy-worker answer reads the previous tree's choice before falling to the
  first legal move (F4); the snapshot it reads is taken between chunks.
- Main-thread torch work (health rebuild, re-warm, evaluator rebuild) runs on a
  short-lived thread, so the main thread never gains a worker-thread team (F9).

The watchdog's last moment net of ``MoveOverheadMs`` (F1) is pinned in
``test_uci_deadline_watchdog.py``.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import chess
import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci import search as search_module
from chess_anti_engine.uci.engine import Engine, EngineOptions, _on_short_lived_thread
from chess_anti_engine.uci.protocol import GoArgs, parse_command
from chess_anti_engine.uci.search import (
    _CHUNK_OVERRUN_MIN_SAMPLES,
    _DEFAULT_CHUNK_OVERRUN,
    _MIN_TIME_CAPPED_CHUNK,
    SearchResult,
    SearchWorker,
)
from chess_anti_engine.uci.time_manager import (
    _CLOCK_SEARCH_SLACK_MS,
    _MIN_DEADLINE_MS,
    Deadline,
    limits_from_go,
)

# -- F3: the clock caps the deadline ------------------------------------------


@pytest.mark.parametrize("remaining", [1, 22, 40, 60, 74, 75, 100, 300, 1000])
@pytest.mark.parametrize("inc", [0, 10])
def test_a_clocked_deadline_never_exceeds_the_clock(remaining, inc) -> None:
    limits = limits_from_go(
        GoArgs(wtime_ms=remaining, btime_ms=remaining, winc_ms=inc, binc_ms=inc),
        side_to_move_is_white=True, move_overhead_ms=30,
    )
    assert limits.deadline_ms is not None
    cap = max(0, remaining - 30 - _CLOCK_SEARCH_SLACK_MS)
    assert limits.deadline_ms <= cap
    if limits.deadline_ms < _MIN_DEADLINE_MS:
        assert limits.clock_exhausted
    if inc:  # (sudden death has its own, earlier at-once rule; tested below)
        assert limits.clock_exhausted is (limits.deadline_ms < _MIN_DEADLINE_MS)
    if limits.optimum_ms is not None:
        assert limits.optimum_ms <= limits.deadline_ms


def test_the_soak_clocks_are_exhausted_and_a_workable_clock_is_not() -> None:
    def lim(ms: int):
        return limits_from_go(GoArgs(wtime_ms=ms, btime_ms=ms, winc_ms=10, binc_ms=10),
                              side_to_move_is_white=True, move_overhead_ms=30)

    # `go wtime 1` and `go wtime 22` were searched for the 20 ms floor.
    for ms in (1, 22):
        assert lim(ms).clock_exhausted
        assert lim(ms).deadline_ms == 0
    # 75 ms - 30 overhead - 25 slack = 20 ms: exactly the floor, searched.
    assert not lim(75).clock_exhausted
    assert lim(75).deadline_ms == 20
    assert lim(74).clock_exhausted


def test_movetime_and_nodes_are_not_clock_capped() -> None:
    mt = limits_from_go(GoArgs(movetime_ms=1), side_to_move_is_white=True, move_overhead_ms=30)
    assert mt.deadline_ms == _MIN_DEADLINE_MS
    assert not mt.clock_exhausted
    nodes = limits_from_go(GoArgs(nodes=64), side_to_move_is_white=True, move_overhead_ms=30)
    assert nodes.deadline_ms is None
    assert not nodes.clock_exhausted


def test_sudden_death_plans_for_twenty_moves_in_the_endgame() -> None:
    # 8 pieces, no increment, no overhead (no reserve): 30000 / 20 = 1500.
    eg = limits_from_go(GoArgs(wtime_ms=30000), side_to_move_is_white=True, pieces=8)
    assert eg.deadline_ms == 1500
    # With an increment the 8-move floor stands (the increment carries the
    # endgame): reserve 9000, usable 21000 -> 21000/8 + 1500 = 4125.
    inc = limits_from_go(GoArgs(wtime_ms=30000, winc_ms=3000), side_to_move_is_white=True, pieces=8)
    assert inc.deadline_ms == 4125
    # The opening estimate (30 moves at 32 pieces) is above both floors.
    op = limits_from_go(GoArgs(wtime_ms=30000), side_to_move_is_white=True, pieces=32)
    assert op.deadline_ms == 1000


def test_sudden_death_answers_at_once_when_the_clock_cannot_pay_for_twenty_searches() -> None:
    # 20 more minimal searches cost 20 * (20 + 25) = 900 ms beyond MoveOverheadMs.
    def lim(ms: int, **kw: Any):
        return limits_from_go(GoArgs(wtime_ms=ms, btime_ms=ms, **kw), side_to_move_is_white=True,
                              move_overhead_ms=30)

    assert lim(929).clock_exhausted
    assert not lim(931).clock_exhausted
    # An increment or a movestogo refill funds further searches: searched.
    assert not lim(500, winc_ms=50).clock_exhausted
    assert not lim(500, movestogo=5).clock_exhausted
    # The time-management OFF baseline keeps the old allocation (only the hard
    # clock cap applies to it).
    off = limits_from_go(GoArgs(wtime_ms=500, btime_ms=500), side_to_move_is_white=True,
                         move_overhead_ms=30, optimum_fraction=0.0)
    assert not off.clock_exhausted


# -- F2 / F5: chunk sizing ----------------------------------------------------


def _bare_worker(chunk_sims: int = 2048) -> Any:
    w = object.__new__(SearchWorker)
    w._chunk_sims = chunk_sims
    w._rpg_pool = None
    return w


def test_a_later_chunk_can_shrink_below_chunk_sims() -> None:
    w = _bare_worker(2048)
    # 1000 sims in ~1000 ms (1 sim/ms), ~100 ms left: the old floor kept 2048.
    d = Deadline(deadline_ms=1100, now=time.monotonic() - 1.0)
    got = w._time_capped_chunk(4096, d, total_nodes=1000)
    assert _MIN_TIME_CAPPED_CHUNK <= got < 2048
    assert got <= int(1.0 * 100 / _DEFAULT_CHUNK_OVERRUN) + 2
    # Nearly no time left: the small floor, not a whole base chunk.
    d0 = Deadline(deadline_ms=1001, now=time.monotonic() - 1.0)
    assert w._time_capped_chunk(4096, d0, total_nodes=1000) == _MIN_TIME_CAPPED_CHUNK
    # Untimed (go nodes / infinite): never capped.
    assert w._time_capped_chunk(4096, Deadline(None), total_nodes=1000) == 4096


def test_root_parallel_gumbel_keeps_the_chunk_sims_floor() -> None:
    w = _bare_worker(2048)
    w._rpg_pool = object()
    d0 = Deadline(deadline_ms=1001, now=time.monotonic() - 1.0)
    assert w._time_capped_chunk(8192, d0, total_nodes=1000) == 2048


def _feed(w: Any, chunks: list[tuple[int, float]]) -> None:
    for sims, ms in chunks:
        w._note_chunk_overrun(sims, ms)
        w._note_chunk_time(sims, ms)


def test_the_last_moment_outlasts_one_gil_switch_interval() -> None:
    """The watchdog's last-moment answer needs the GIL; a thread that never
    yields holds it for one switch interval (5 ms). With a 3 ms margin, one
    answer under injected stalls on a loaded host landed past the hard limit."""
    switch_ms = sys.getswitchinterval() * 1000.0
    margin_ms = engine_module._WATCHDOG_LAST_MOMENT_MS
    assert margin_ms >= 3.0 + switch_ms


def test_the_first_chunk_is_sized_from_the_chunk_time_model() -> None:
    w = _bare_worker(2048)
    d = Deadline(deadline_ms=300, now=time.monotonic())
    # Before any measurement: the conservative bootstrap (6 sims/ms) -> 1800.
    assert 1700 <= w._time_capped_chunk(8192, d, 0) <= 1800
    # A slow device: 6 ms fixed + 0.5 ms per sim (2 sims/ms of throughput).
    _feed(w, [(16, 14.0), (64, 38.0), (256, 134.0), (32, 22.0), (128, 70.0)])
    fitted, fixed, per_sim, _, _ = w._chunk_model()
    assert fitted
    assert fixed == pytest.approx(6.0, abs=0.01)
    assert per_sim == pytest.approx(0.5, abs=1e-4)
    # usable 300 - 6 (2 % reserve) = 294; first chunk gets 1/2 of it: 147 ms
    # -> (147 - 6) / 0.5 = 282 sims (B is still the uncalibrated 2).
    got = w._time_capped_chunk(8192, d, 0)
    assert 270 <= got <= 283
    # A different topology (rebuilt evaluator, pool install) forgets it.
    w._evaluator = object()
    assert w._chunk_model() is None
    assert w._time_capped_chunk(8192, d, 0) >= 1600


def test_small_chunks_do_not_poison_the_throughput() -> None:
    # Many 16-sim chunks at 8 ms (6 fixed + 16 x 0.125): 2 sims/ms per chunk,
    # but the throughput is 8 sims/ms. One larger chunk lets the model see it.
    w = _bare_worker()
    _feed(w, [(16, 8.0)] * 6 + [(128, 22.0), (16, 8.0), (64, 14.0)])
    fitted, fixed, per_sim, _, _ = w._chunk_model()
    assert fitted
    assert per_sim == pytest.approx(0.125, rel=0.02)
    assert fixed == pytest.approx(6.0, rel=0.05)
    # The time cap for 70 ms is ~ (70 - 6) / 0.125 = 512, not 2 x 70 = 140.
    assert w._sims_for_ms(70.0) >= 500


def test_unseparable_sizes_predict_conservatively() -> None:
    w = _bare_worker()
    _feed(w, [(64, 20.0)] * 4)  # one size only: fixed vs per-sim unknown
    fitted, *_ = w._chunk_model()
    assert not fitted
    assert w._predict_chunk_ms(16) == pytest.approx(20.0)  # never faster than the mean chunk
    assert w._predict_chunk_ms(256) == pytest.approx(80.0)  # never faster per sim
    assert w._sims_for_ms(10.0) == 0
    assert w._sims_for_ms(40.0) == 128


def test_a_slow_spell_does_not_lock_the_rate_in() -> None:
    """The model decays toward recent chunks: after a slow spell (a loaded box)
    the minimal-search price, and with it the sudden-death rule, recovers."""
    w = _bare_worker()
    slow = [(16, 40.0), (64, 100.0), (128, 180.0)] * 3   # 20 ms + 1.25 ms/sim
    fast = [(16, 8.0), (64, 14.0), (128, 22.0)] * 6      # 6 ms + 0.125 ms/sim
    _feed(w, slow)
    slow_min = w.measured_min_chunk_ms()
    assert slow_min == pytest.approx(40.0, rel=0.05)

    def at_once(clock: int) -> bool:
        return limits_from_go(GoArgs(wtime_ms=clock, btime_ms=clock), side_to_move_is_white=True,
                              move_overhead_ms=30,
                              min_chunk_ms=w.measured_min_chunk_ms()).clock_exhausted

    assert at_once(1300)  # 20 x (20 + 40 + 8) = 1360 ms
    _feed(w, fast)
    assert w.measured_min_chunk_ms() < 12.0  # most of the way back to 8 ms
    assert not at_once(1300)  # 20 x (20 + ~9 + 8) ~ 750 ms


# -- F4: the previous tree's choice --------------------------------------------


def _tiny_worker() -> SearchWorker:
    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(7)
    model = build_model(ModelConfig(input_extra_features="v1", embed_dim=16, num_layers=1,
                                    num_heads=2, ffn_mult=2.0))
    model.eval()
    ev = ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        model, device="cpu", max_batch=256, use_amp=False, n_slots=2))
    return SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )


def test_the_reply_book_names_a_searched_answer_two_plies_on() -> None:
    worker = _tiny_worker()
    try:
        board = chess.Board()
        worker.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
        tree, rid = worker._tree, worker._root_id
        assert tree is not None
        assert rid is not None
        # The book's root entry is the tree's most-visited root move.
        root_best = worker.tree_root_bestmove(board)
        assert root_best is not None
        assert worker.reply_book_bestmove(board) == root_best
        # Find one (our move, reply) pair the tree visited, and its answer.
        ours = search_module._most_visited_children(tree, rid, 1, None)
        (action, node), = ours
        after = board.copy(stack=False)
        after.push(search_module.index_to_move(action, board))
        (reply, reply_node), = search_module._most_visited_children(tree, node, 1, None)
        pos = after.copy(stack=False)
        pos.push(search_module.index_to_move(reply, after))
        answer = search_module._most_visited_children(tree, reply_node, 1, None)
        assert answer, "the 512-sim tree should visit two plies deep"
        expected = search_module.index_to_move(answer[0][0], pos).uci()
        assert worker.reply_book_bestmove(pos) == expected
        assert chess.Move.from_uci(expected) in pos.legal_moves
        # searchmoves are honoured; an unknown position has no entry.
        assert worker.reply_book_bestmove(pos, searchmoves=("a1a1",)) is None
        assert worker.reply_book_bestmove(chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")) is None
        # The snapshot survives a tree reset: it is what a parked go reads.
        worker.reset_tree()
        assert worker.reply_book_bestmove(pos) == expected
    finally:
        worker.close()


def test_a_real_search_teaches_the_chunk_time_model() -> None:
    worker = _tiny_worker()
    try:
        assert worker.measured_min_chunk_ms() is None
        worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=128)
        assert worker._cm_n >= 1
        ms = worker.measured_min_chunk_ms()
        assert ms is not None
        assert ms > 0.0
        # The next timed search's first chunk is sized from it, not the bootstrap
        # (6 sims/ms x 200 ms = 1200).
        d = Deadline(deadline_ms=10_000, now=time.monotonic())
        b = max(worker._chunk_overrun_hi(), 2.0)
        fit = worker._sims_for_ms((10_000 - 25) / b)
        assert fit is not None
        expect = min(worker._chunk_sims, max(_MIN_TIME_CAPPED_CHUNK, fit))
        assert abs(worker._time_capped_chunk(8192, d, 0) - expect) <= 2
    finally:
        worker.close()


def test_the_startup_warmup_calibrates_the_chunk_model() -> None:
    """After the warmup the model is fitted (it has seen several chunk sizes)
    and the overrun bound has its three judged chunks, so the first move at a
    short budget is not priced as if every chunk cost the whole warmup chunk."""
    worker = _tiny_worker()
    worker._chunk_sims = 512  # a realistic ChunkSims (production: 2048), above the calibration sizes
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.warmup_search()
        model = worker._chunk_model()
        assert model is not None
        fitted, _, per_sim, _, _ = model
        assert fitted
        assert per_sim > 0.0
        assert worker._overrun_n >= _CHUNK_OVERRUN_MIN_SAMPLES
        # A 16-sim chunk is priced near its own cost, far below the warmup chunk.
        small = worker._predict_chunk_ms(16)
        big = worker._predict_chunk_ms(worker._chunk_sims)
        assert small is not None
        assert big is not None
        assert small < 0.5 * big
    finally:
        engine.close()
        worker.close()


def test_a_real_movetime_search_measures_its_overrun_and_prices_its_stop(monkeypatch) -> None:
    worker = _tiny_worker()
    margins: list[float] = []
    real = worker._timed_stop_margin_ms

    def spy(*a, **kw):
        margins.append(real(*a, **kw))
        return margins[-1]

    monkeypatch.setattr(worker, "_timed_stop_margin_ms", spy)
    try:
        worker.run(chess.Board(), stop_event=threading.Event(),
                   deadline=Deadline(deadline_ms=400), max_nodes=None, fixed_time_margin=True)
        assert worker._overrun_n >= 1  # time-capped chunks were timed against their prediction
        assert margins
        assert any(m > 0.0 for m in margins)
    finally:
        worker.close()


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

    def wait_bestmoves(self, n: int, timeout: float = 10.0) -> list[str]:
        with self._cond:
            self._cond.wait_for(
                lambda: sum(ln.startswith("bestmove ") for ln in self.lines) >= n, timeout,
            )
            return [ln.split()[1] for ln in self.lines if ln.startswith("bestmove ")]

    def text(self) -> str:
        with self._cond:
            return "\n".join(self.lines)


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _mock_worker() -> MagicMock:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.tree_root_bestmove.return_value = None
    worker.reply_book_bestmove.return_value = None
    worker.fresh_root_policy_bestmove.return_value = None
    worker.realized_search_path.return_value = "gumbel"
    worker.leaf_tb_inactive_mode.return_value = None
    return worker


def test_a_parked_worker_answers_from_the_previous_trees_choice(out) -> None:
    release = threading.Event()
    entered = threading.Event()

    def stuck_run(board, **_kw):
        entered.set()
        assert release.wait(20)
        move = next(iter(board.legal_moves)).uci()
        return SearchResult(move, None, 8, (move,), 0)

    worker = _mock_worker()
    worker.run.side_effect = stuck_run
    after = chess.Board()
    for mv in ("e2e4", "e7e5", "g1f3"):
        after.push_uci(mv)
    book = {after.fen(): "b8c6"}
    worker.reply_book_bestmove.side_effect = (
        lambda b, searchmoves=(), root_only=False: None if root_only else book.get(b.fen()))
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go movetime 100"))
        assert entered.wait(5)
        assert len(out.wait_bestmoves(1)) == 1  # the watchdog answered move 1
        # The stuck search still owns the worker when the next go arrives.
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3"))
        engine.dispatch(parse_command("go movetime 100"))
        moves = out.wait_bestmoves(2)
        assert moves[1] == "b8c6", out.text()
        second = out.text().split("\nbestmove ", 1)[1]
        assert "source=prev_tree move=b8c6" in second
        assert "source=first_legal" not in second
    finally:
        release.set()
        engine.close()


# -- F3: answered at once, without a search -------------------------------------


@pytest.mark.parametrize(("go", "source", "setup"), [
    ("go wtime 1 btime 1", "tree", {"tree_root_bestmove": "e7e5"}),
    ("go wtime 22 btime 22", "prev_tree", {"reply_book_bestmove": "c7c5"}),
    ("go wtime 40 btime 40", "root_policy", {"root_policy_bestmove": "d7d5"}),
])
def test_a_clock_too_short_to_search_is_answered_at_once(out, go, source, setup) -> None:
    worker = _mock_worker()
    for name, move in setup.items():
        getattr(worker, name).return_value = move
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command(go))
        moves = out.wait_bestmoves(1)
        assert moves == [next(iter(setup.values()))], out.text()
        assert "clock too short to search" in out.text()
        assert f"source={source}" in out.text()
        assert worker.run.call_count == 0  # no search was started
        # Answered from what was known: no evaluation either (wall-clock free).
        assert worker.fresh_root_policy_bestmove.call_count == 0
        # Not a fault: neither the fallback counter nor the unhealthy streak moves.
        assert engine.bestmove_fallback_used == 0
        assert engine._consecutive_search_failures == 0
        assert engine._watchdog.armed is None
    finally:
        engine.close()


def test_with_nothing_known_a_fresh_eval_answers_when_the_clock_allows(out, monkeypatch) -> None:
    worker = _mock_worker()
    engine = Engine(worker, options=EngineOptions())
    budgets: list[float | None] = []

    def fresh(_board, _searchmoves, budget_s):  # the helper-thread round trip, stubbed
        budgets.append(budget_s)
        return "g8f6"

    monkeypatch.setattr(engine, "_bounded_fresh_eval", fresh)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        # Sudden death with 900 ms: answered at once, and 5 % of the 870 ms left
        # (~43 ms) is far more than any loaded box needs to get here.
        engine.dispatch(parse_command("go wtime 900 btime 900"))
        assert out.wait_bestmoves(1) == ["g8f6"], out.text()
        assert "source=fresh_root_eval" in out.text()
        assert worker.run.call_count == 0
        assert len(budgets) == 1
        assert budgets[0] is not None
        assert budgets[0] > 0.0
    finally:
        engine.close()


def test_an_at_once_answer_spends_only_a_small_share_of_the_clock_on_an_eval(out, monkeypatch) -> None:
    # Sudden death, 900 ms on the clock: answered at once, every move from here
    # on. The eval may take at most 5 % of the 870 ms left after the overhead.
    worker = _mock_worker()
    worker.fresh_root_policy_bestmove.return_value = "g8f6"
    engine = Engine(worker, options=EngineOptions())
    budgets: list[float] = []
    real = engine._bounded_fresh_eval

    def spy(board, searchmoves, budget_s):
        budgets.append(budget_s)
        return real(board, searchmoves, budget_s)

    monkeypatch.setattr(engine, "_bounded_fresh_eval", spy)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go wtime 900 btime 900"))
        assert out.wait_bestmoves(1) == ["g8f6"]
        assert "clock too short to search" in out.text()
        assert len(budgets) == 1
        assert 0.0 < budgets[0] <= 0.05 * 0.870 + 1e-9
        assert worker.run.call_count == 0
    finally:
        engine.close()


def test_the_at_once_sources_are_tried_in_order(out) -> None:
    worker = _mock_worker()
    worker.tree_root_bestmove.return_value = "e7e5"
    worker.reply_book_bestmove.return_value = "c7c5"
    worker.root_policy_bestmove.return_value = "d7d5"
    worker.fresh_root_policy_bestmove.return_value = "g8f6"
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go wtime 1 btime 1"))
        assert out.wait_bestmoves(1) == ["e7e5"]
        worker.tree_root_bestmove.return_value = None
        engine.dispatch(parse_command("go wtime 1 btime 1"))
        assert out.wait_bestmoves(2)[1] == "c7c5"
        worker.reply_book_bestmove.return_value = None
        engine.dispatch(parse_command("go wtime 1 btime 1"))
        assert out.wait_bestmoves(3)[2] == "d7d5"
        worker.root_policy_bestmove.return_value = None
        engine.dispatch(parse_command("go wtime 80 btime 80"))
        assert out.wait_bestmoves(4)[3] == "g8f6"
    finally:
        engine.close()


@pytest.mark.parametrize("go", [
    "go nodes 64 wtime 500 btime 500", "go depth 3 wtime 500 btime 500",
    "go mate 2 wtime 500 btime 500",
])
def test_an_explicit_work_bound_is_searched_in_sudden_death(out, go) -> None:
    worker = _mock_worker()
    worker.run.side_effect = lambda board, **_kw: SearchResult("e7e5", None, 64, ("e7e5",), 0)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command(go))
        assert out.wait_bestmoves(1) == ["e7e5"]
        assert worker.run.call_count == 1
        assert "clock too short to search" not in out.text()
    finally:
        engine.close()


def test_explicit_bounds_skip_the_sudden_death_rule_but_not_an_exhausted_clock() -> None:
    def exhausted(args: GoArgs) -> bool:
        return limits_from_go(args, side_to_move_is_white=True, move_overhead_ms=30).clock_exhausted

    assert not exhausted(GoArgs(wtime_ms=500, btime_ms=500, nodes=64))
    assert not exhausted(GoArgs(wtime_ms=500, btime_ms=500, depth=3))
    assert not exhausted(GoArgs(wtime_ms=500, btime_ms=500, mate=2))
    assert exhausted(GoArgs(wtime_ms=1, btime_ms=1, nodes=64))


def test_the_sudden_death_threshold_scales_with_the_measured_speed() -> None:
    def exhausted(ms: int, min_chunk_ms: float | None) -> bool:
        return limits_from_go(GoArgs(wtime_ms=ms, btime_ms=ms), side_to_move_is_white=True,
                              move_overhead_ms=30, min_chunk_ms=min_chunk_ms).clock_exhausted

    # Default (nothing measured): 20 x 45 ms = 900 ms after the overhead.
    assert exhausted(929, None)
    assert not exhausted(931, None)
    # Fast device, a 16-sim chunk in 0.16 ms: 20 x 28.16 ~ 563 ms -> 700 is searched.
    assert not exhausted(700, 0.16)
    assert exhausted(590, 0.16)
    # Slow device, 32 ms: 20 x (20 + 32 + 8) = 1200 ms -> 1000 is at once.
    assert exhausted(1000, 32.0)
    assert not exhausted(1300, 32.0)


def test_the_engine_prices_the_sudden_death_rule_with_the_workers_rate(out) -> None:
    worker = _mock_worker()
    worker.measured_min_chunk_ms.return_value = 0.16
    worker.run.side_effect = lambda board, **_kw: SearchResult("e7e5", None, 64, ("e7e5",), 0)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go wtime 700 btime 700"))  # default rule: at once
        assert out.wait_bestmoves(1) == ["e7e5"]
        assert worker.run.call_count == 1
    finally:
        engine.close()


# -- F4 review: the reused tree's answer ------------------------------------------


class _FakeTree:
    def __init__(self, actions, visits, qs) -> None:
        import numpy as np
        self._a = np.asarray(actions, dtype=np.int32)
        self._n = np.asarray(visits, dtype=np.int32)
        self._q = np.asarray(qs, dtype=np.float64)

    def get_children_q(self, _node, _default):
        return self._a, self._n, self._q


def _tree_worker(board: chess.Board, tree: _FakeTree) -> Any:
    w: Any = object.__new__(SearchWorker)
    w._tree, w._root_id, w._tree_fen = tree, 0, board.fen()
    return w


def _idx(board: chess.Board, uci: str) -> int:
    return int(search_module.move_to_index(chess.Move.from_uci(uci), board))


def test_the_tree_answer_needs_its_own_root_enough_visits_and_breaks_ties_by_q() -> None:
    board = chess.Board()
    a, b = _idx(board, "e2e4"), _idx(board, "d2d4")
    lo, hi = sorted((a, b))
    # Tie on visits: the higher Q wins even with the higher action id.
    w = _tree_worker(board, _FakeTree([lo, hi], [20, 20], [0.1, 0.3]))
    hi_uci = search_module.index_to_move(hi, board).uci()
    assert w.tree_root_bestmove(board) == hi_uci
    # Fewer visits than the minimum: not trusted.
    w = _tree_worker(board, _FakeTree([lo, hi], [3, 2], [0.0, 0.0]))
    assert w.tree_root_bestmove(board) is None
    # A tree rooted at another position answers nothing for this one.
    other = chess.Board()
    other.push_uci("e2e4")
    w = _tree_worker(board, _FakeTree([lo, hi], [20, 10], [0.0, 0.0]))
    assert w.tree_root_bestmove(other) is None


# -- F9 review: setoption rebuilds run off the calling thread --------------------


def test_an_evaluator_rebuild_runs_on_a_joined_short_lived_thread(out) -> None:
    del out
    worker = _mock_worker()
    seen: list[threading.Thread] = []

    def factory(*_args: Any, **_kw: Any) -> Any:
        seen.append(threading.current_thread())
        return object()

    engine = Engine(worker, options=EngineOptions(), rebuild_evaluator=cast("Any", factory))
    try:
        assert engine._rebuild_or_keep("MaxBatch", 64, 0) is not None
        assert len(seen) == 1
        assert seen[0] is not threading.current_thread()
        assert not seen[0].is_alive()  # joined before return
    finally:
        engine.close()


def test_a_name_error_in_a_rebuild_is_not_hidden_as_a_kept_setup(out) -> None:
    del out
    worker = _mock_worker()

    def factory(*_a, **_kw):
        raise NameError("undefined_helper")

    engine = Engine(worker, options=EngineOptions(), rebuild_evaluator=cast("Any", factory))
    try:
        with pytest.raises(NameError):
            engine._rebuild_or_keep("MaxBatch", 64, 0)
    finally:
        engine.close()


# -- F9 review / item 9: ponderhit with no time left -----------------------------


def test_a_ponderhit_with_no_time_left_answers_from_the_ponder_tree(out) -> None:
    gate_entered, gate_release = threading.Event(), threading.Event()

    def ponder_run(*_args: Any, **kw: Any) -> SearchResult:
        # The ponder phase printed a PV for ITS root (the opponent's move).
        _pv_line(kw, "e7e5")
        gate_entered.set()
        assert gate_release.wait(10)
        return SearchResult("e7e5", None, 16, ("e7e5",), 0)

    worker = _mock_worker()
    worker.run.side_effect = ponder_run
    worker.child_bestmove.return_value = "g1f3"
    engine = Engine(worker, options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 500 btime 500"))
        assert gate_entered.wait(5)
        engine.dispatch(parse_command("ponderhit"))
        gate_release.set()
        assert out.wait_bestmoves(1) == ["g1f3"], out.text()
        assert "source=ponder_tree" in out.text()
        assert worker.run.call_count == 1
        _assert_answer_line(out, "g1f3")
    finally:
        gate_release.set()
        engine.close()


@pytest.mark.parametrize("deadline_open", [False, True], ids=["deadline_gone", "clock_exhausted_only"])
def test_a_real_ponder_tree_answers_a_ponderhit_with_no_time_left(out, monkeypatch, deadline_open) -> None:
    """The answer is read from the real ponder tree BEFORE the root advances:
    after `advance_root` the tree is rooted at the hit position and
    `child_bestmove(ponder_root, reply)` finds nothing (a MagicMock cannot see
    that). ``clock_exhausted_only``: the sudden-death rule says "answer at
    once" while the 20 ms deadline is still open -- the path on which the root
    DOES advance, so reading the tree after it would find nothing."""
    if deadline_open:
        real_limits_from_go = engine_module.limits_from_go

        def open_deadline(*a: Any, **kw: Any) -> Any:
            limits = real_limits_from_go(*a, **kw)
            if limits.clock_exhausted:  # same verdict, a deadline that cannot lapse here
                limits = dataclasses.replace(limits, deadline_ms=60_000)
            return limits

        monkeypatch.setattr(engine_module, "limits_from_go", open_deadline)
    worker = _tiny_worker()
    # Predict the reply the ponder search is sure to visit: this net's own
    # favourite after e2e4 (an unvisited reply has no tree answer).
    after = chess.Board()
    after.push_uci("e2e4")
    reply = worker.run(after, stop_event=threading.Event(), deadline=Deadline(None),
                       max_nodes=256).bestmove_uci
    worker.reset_tree()
    runs: list[bool] = []
    real_run = worker.run

    def counting_run(*a: Any, **kw: Any) -> SearchResult:
        runs.append(bool(kw.get("max_nodes") is None))
        return real_run(*a, **kw)

    worker.run = counting_run  # type: ignore[method-assign]
    engine = Engine(worker, options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command(f"position startpos moves e2e4 {reply}"))
        engine.dispatch(parse_command("go ponder wtime 500 btime 500"))  # sudden death: at once
        end = time.monotonic() + 20
        while time.monotonic() < end:
            tree, rid = worker._tree, worker._root_id
            if tree is not None and rid is not None and int(tree.node_count()) > 200:
                break
            time.sleep(0.02)
        engine.dispatch(parse_command("ponderhit"))
        moves = out.wait_bestmoves(1, timeout=20)
        assert "source=ponder_tree" in out.text(), out.text()
        real = chess.Board()
        for mv in ("e2e4", reply):
            real.push_uci(mv)
        assert chess.Move.from_uci(moves[0]) in real.legal_moves
        assert len(runs) == 1  # only the ponder phase searched
        _assert_answer_line(out, moves[0])
    finally:
        engine.close()
        worker.close()


def test_the_trees_own_choice_for_this_position_beats_the_prior(monkeypatch) -> None:
    """Busy-worker / watchdog fallback: when the snapshot's root IS this
    position, its most-visited move (>= 8 visits) is preferred to the raw prior;
    for any other position the prior still comes first."""
    # Snapshot after every chunk, so the last one is the final tree (at the
    # 200 ms default a loaded host can leave it a chunk behind).
    monkeypatch.setattr(search_module, "_REPLY_BOOK_INTERVAL_S", 0.0)
    worker = _tiny_worker()
    engine = Engine(worker, options=EngineOptions())
    try:
        board = chess.Board()
        worker.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
        tree_best = worker.tree_root_bestmove(board)
        assert tree_best is not None
        prior = next(m.uci() for m in board.legal_moves if m.uci() != tree_best)
        worker.root_policy_bestmove = lambda board, *, searchmoves=(): prior  # type: ignore[method-assign]
        assert engine._bestmove_fallback(board, (), allow_fresh_eval=False) == (tree_best, "prev_tree")
        other = chess.Board()
        other.push_uci("a2a3")  # not the snapshot's root
        other_prior = next(iter(other.legal_moves)).uci()
        worker.root_policy_bestmove = lambda board, *, searchmoves=(): other_prior  # type: ignore[method-assign]
        assert engine._bestmove_fallback(other, (), allow_fresh_eval=False) == (other_prior, "root_policy")
    finally:
        engine.close()
        worker.close()


# -- #1364: every answer this PR adds ends on a PV line for the played move -----


def _pv_line(kw: dict, move: str) -> None:
    kw["info_cb"](nodes=64, elapsed_ms=3, score_cp=5, pv=(move,), tbhits=0,
                  score_mate=None, multipv=None, wdl=None)


def _assert_answer_line(out: Any, move: str) -> None:
    """The line right before ``bestmove move`` is the score-less, node-less PV
    line for ``move``, and it is the last rank-1 PV line of the go."""
    lines = out.text().splitlines()
    at = lines.index(f"bestmove {move}")
    assert lines[at - 1] == f"info depth 1 pv {move}", lines[max(0, at - 6):at + 1]


def _hung(head: str, release: threading.Event):
    def run(board, **kw):
        _pv_line(kw, head)
        release.wait(10)
        mv = next(iter(board.legal_moves)).uci()
        return SearchResult(mv, None, 8, (mv,), 0)
    return run


def test_a_watchdog_answer_from_the_previous_search_is_aligned(out, monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_WATCHDOG_STALL_NO_ESTIMATE_MS", 100.0)
    monkeypatch.setattr(engine_module, "_WATCHDOG_STALL_FLOOR_MS", 50.0)
    release = threading.Event()
    worker = _mock_worker()
    worker.run.side_effect = _hung("g1f3", release)  # not in searchmoves: last_pv unusable
    start = chess.Board()
    worker.reply_book_bestmove.side_effect = (
        lambda b, searchmoves=(), root_only=False: "d2d4" if b.fen() == start.fen() else None)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos"))
        engine.dispatch(parse_command("go movetime 200 searchmoves e2e4 d2d4"))
        assert out.wait_bestmoves(1, timeout=15) == ["d2d4"], out.text()
        assert "source=prev_tree" in out.text()
        _assert_answer_line(out, "d2d4")
    finally:
        release.set()
        engine.close()


def test_a_parked_stop_answer_from_the_previous_search_is_aligned(out, monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_GO_WAIT_NO_CLOCK_S", 0.05)
    release = threading.Event()
    worker = _mock_worker()
    worker.run.side_effect = _hung("g1f3", release)
    start = chess.Board()
    worker.reply_book_bestmove.side_effect = (
        lambda b, searchmoves=(), root_only=False: "d2d4" if b.fen() == start.fen() else None)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos"))
        engine.dispatch(parse_command("go infinite searchmoves e2e4 d2d4"))
        end = time.monotonic() + 10
        while " pv g1f3" not in out.text() and time.monotonic() < end:
            time.sleep(0.01)
        engine.dispatch(parse_command("stop"))
        assert out.wait_bestmoves(1, timeout=15) == ["d2d4"], out.text()
        assert "source=prev_tree" in out.text()
        _assert_answer_line(out, "d2d4")
    finally:
        release.set()
        engine.close()


def test_the_at_once_answer_goes_through_the_bestmove_guard(out, monkeypatch) -> None:
    worker = _mock_worker()
    worker.tree_root_bestmove.return_value = "e7e5"
    engine = Engine(worker, options=EngineOptions())
    seen: list[tuple[int | None, str]] = []
    real = engine._publish_bestmove

    def spy(gen, move, ponder=None, *, pv=()):
        seen.append((gen, move))
        real(gen, move, ponder, pv=pv)

    monkeypatch.setattr(engine, "_publish_bestmove", spy)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        engine.dispatch(parse_command("go wtime 1 btime 1"))
        assert out.wait_bestmoves(1) == ["e7e5"]
        assert seen == [(engine._search_gen, "e7e5")]
        # This go printed no PV of its own, so the guard adds no line, and an
        # earlier go's line is never re-led.
        assert " pv " not in out.text()
    finally:
        engine.close()


# -- item 8: chunk sizing against the measured overrun ---------------------------


def test_later_chunks_fill_the_remainder_by_the_measured_overrun_bound() -> None:
    w = _bare_worker(4096)
    w._batch_margin_sigmas = 2.0
    # 1 ms per sim, no fixed cost; once fitted, every chunk runs as predicted.
    _feed(w, [(100, 100.0), (200, 200.0), (50, 50.0)] * 12)
    assert w._chunk_overrun_hi() == pytest.approx(1.0, abs=0.03)
    d = Deadline(deadline_ms=2000, now=time.monotonic() - 1.0)  # ~1000 ms left
    got = w._time_capped_chunk(8192, d, total_nodes=1000)
    # (1000 - 25 ms finishing reserve) / B at 1 ms per sim.
    assert int(950 / w._chunk_overrun_hi()) - 2 <= got <= 976
    # Noisy chunks: the bound widens (capped at 2), the chunk shrinks.
    for r in (1.0, 2.0) * 10:
        w._note_chunk_overrun(100, 100.0 * r)
    assert 1.5 < w._chunk_overrun_hi() <= 2.0  # never narrower than the old 0.5 share
    assert w._time_capped_chunk(8192, d, total_nodes=1000) < 700
    # The stop margin is one smallest chunk at that bound, not a whole chunk
    # (plus min(25 ms, 2 % of the budget) to finish the move).
    margin = w._timed_stop_margin_ms(1, 2000, d, 1000, fixed_time=False)
    assert margin == pytest.approx(
        _MIN_TIME_CAPPED_CHUNK * 1.0 * w._chunk_overrun_hi() + 25.0, rel=0.05,
    )
    # ClockBatchMarginSigmas 0 disables the stop margin (the knob stays live).
    w._batch_margin_sigmas = 0.0
    assert w._timed_stop_margin_ms(1, 2000, d, 1000, fixed_time=False) == 0.0


def test_later_chunks_use_the_model_not_this_searchs_average_speed() -> None:
    """This search averaged 1 sim/ms (a slow first chunk, say), but the model
    knows a chunk costs 10 ms + 0.5 ms per sim: the next chunk fills the time
    left at the model's price, about twice what the average would allow."""
    w = _bare_worker(8192)
    w._batch_margin_sigmas = 2.0
    _feed(w, [(100, 60.0), (200, 110.0), (50, 35.0)] * 12)
    assert w._chunk_overrun_hi() == pytest.approx(1.0, abs=0.03)
    d = Deadline(deadline_ms=2000, now=time.monotonic() - 1.0)  # ~1000 ms left
    got = w._time_capped_chunk(8192, d, total_nodes=1000)  # 1 sim/ms so far
    b = w._chunk_overrun_hi()
    assert got >= int(((975.0 / b) - 10.0) / 0.5) - 10  # ~1930, not ~975
    assert got <= 1932


def test_the_overrun_bound_never_drops_below_one() -> None:
    w = _bare_worker()
    w._batch_margin_sigmas = 2.0
    _feed(w, [(100, 100.0), (200, 200.0), (50, 50.0)])
    for _ in range(10):  # chunks HALF as long as predicted
        w._note_chunk_overrun(100, 50.0)
    assert w._overrun_mean < 0.7
    assert w._chunk_overrun_hi() == 1.0  # a faster-than-predicted spell never sizes past it


def test_the_overrun_bound_waits_for_three_samples() -> None:
    w = _bare_worker()
    w._batch_margin_sigmas = 2.0
    _feed(w, [(100, 100.0)])  # the model exists; its first chunk was not judged
    assert w._overrun_n == 0
    w._note_chunk_overrun(100, 100.0)
    assert w._chunk_overrun_hi() == _DEFAULT_CHUNK_OVERRUN
    w._note_chunk_overrun(100, 100.0)
    assert w._overrun_n == 2
    assert w._chunk_overrun_hi() == _DEFAULT_CHUNK_OVERRUN
    w._note_chunk_overrun(100, 100.0)
    assert w._chunk_overrun_hi() == pytest.approx(1.0)


def test_a_device_failure_restarts_the_model_and_the_overrun_bound(monkeypatch) -> None:
    w = _bare_worker(4096)
    failed: list[str] = []
    monkeypatch.setattr(w, "failed_pool_devices", lambda: tuple(failed), raising=False)
    _feed(w, [(100, 10.0)] * 6)  # 10 sims/ms
    assert w._overrun_n == 5
    failed.append("cpu:1")
    assert w._chunk_model() is None  # this topology has measured nothing yet
    _feed(w, [(100, 20.0)])
    assert w._predict_chunk_ms(100) == pytest.approx(20.0)
    assert w._overrun_n == 0
    assert w._chunk_overrun_hi() == _DEFAULT_CHUNK_OVERRUN


def test_visit_ties_go_to_the_higher_q_in_the_reply_book_order() -> None:
    import numpy as np

    class Tree:
        def get_children_q(self, _node, _default):
            return (np.array([5, 9, 7], dtype=np.int32), np.array([10, 10, 3], dtype=np.int32),
                    np.array([0.1, 0.4, 0.9]))

        def find_child(self, _node, action):
            return 100 + int(action)

    got = search_module._most_visited_children(Tree(), 0, 3, None)
    assert [a for a, _ in got] == [9, 5, 7]  # tie on 10 visits: Q 0.4 beats 0.1


# -- F9: no torch worker-thread team on the main thread -------------------------


def _native_tasks() -> set[int]:
    return {int(t) for t in os.listdir("/proc/self/task")}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/task")
def test_offloaded_torch_work_leaves_no_thread_behind() -> None:
    import torch

    if torch.get_num_threads() < 2:
        pytest.skip("needs an intra-op pool of at least 2 threads")

    def torch_work() -> float:
        x = torch.randn(8, 64, 16, 16)
        w = torch.randn(64, 64, 3, 3)
        total = 0.0
        for _ in range(3):
            total += float(torch.nn.functional.conv2d(x, w, padding=1).sum())
        return total

    counts: dict[str, int] = {}
    hold = threading.Event()
    done = threading.Event()

    def caller() -> None:
        # A long-lived thread standing in for the UCI main thread.
        before = _native_tasks()
        _on_short_lived_thread(torch_work)
        # New OS threads that are not Python threads (another test's daemon
        # thread may come and go meanwhile): a torch worker team, if any.
        python = {t.native_id for t in threading.enumerate()}
        counts["new_native"] = len(_native_tasks() - before - python)
        done.set()
        hold.wait(10)

    t = threading.Thread(target=caller)
    t.start()
    try:
        assert done.wait(60)
        assert counts["new_native"] == 0, counts
    finally:
        hold.set()
        t.join(10)


def test_on_short_lived_thread_returns_and_reraises() -> None:
    main = threading.current_thread()
    seen: list[threading.Thread] = []

    def ok() -> int:
        seen.append(threading.current_thread())
        return 7

    assert _on_short_lived_thread(ok) == 7
    assert len(seen) == 1
    assert seen[0] is not main
    with pytest.raises(ValueError, match="boom"):
        _on_short_lived_thread(lambda: (_ for _ in ()).throw(ValueError("boom")))


def test_the_game_boundary_health_rebuild_runs_off_the_calling_thread(out) -> None:
    worker = _mock_worker()
    engine = Engine(worker, options=EngineOptions())
    ran_on: list[threading.Thread] = []

    def boundary() -> bool:
        ran_on.append(threading.current_thread())
        return True

    def warm() -> None:
        ran_on.append(threading.current_thread())

    engine._game_boundary_checks = boundary  # type: ignore[method-assign]
    engine.warmup_search = warm  # type: ignore[method-assign]
    engine._warmup_dirty = True
    try:
        engine.dispatch(parse_command("isready"))
        assert "readyok" in out.text()
        assert len(ran_on) == 2
        assert all(t is not threading.current_thread() for t in ran_on)
    finally:
        engine.close()
