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
    _FIRST_CHUNK_TIME_SHARE,
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


def test_the_first_chunk_is_sized_from_the_measured_chunk_rate() -> None:
    w = _bare_worker(2048)
    d = Deadline(deadline_ms=300, now=time.monotonic())
    # Before any measurement: the conservative bootstrap (6 sims/ms) -> 1800.
    assert 1700 <= w._time_capped_chunk(8192, d, 0) <= 1800
    # A slow device: 2 sims/ms measured. A quarter of ~300 ms at 2/ms = ~150.
    w._note_chunk_rate(200, 100.0)
    assert w.measured_chunk_rate() == pytest.approx(2.0)
    got = w._time_capped_chunk(8192, d, 0)
    assert got <= int(_FIRST_CHUNK_TIME_SHARE * 2.0 * 300) + 1
    assert got >= int(_FIRST_CHUNK_TIME_SHARE * 2.0 * 280)
    # A different topology (rebuilt evaluator, pool install) forgets it.
    w._evaluator = object()
    assert w.measured_chunk_rate() is None
    assert w._time_capped_chunk(8192, d, 0) >= 1600


def test_the_chunk_rate_falls_at_once_and_rises_slowly() -> None:
    w = _bare_worker()
    w._note_chunk_rate(1000, 100.0)  # 10 /ms
    w._note_chunk_rate(100, 100.0)   # 1 /ms: falls at once
    assert w.measured_chunk_rate() == pytest.approx(1.0)
    w._note_chunk_rate(10000, 100.0)  # 100 /ms: rises at most 25 %
    assert w.measured_chunk_rate() == pytest.approx(1.25)
    w._note_chunk_rate(0, 100.0)  # a chunk that did nothing teaches nothing
    assert w.measured_chunk_rate() == pytest.approx(1.25)


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


def test_a_real_search_teaches_the_first_chunk_rate() -> None:
    worker = _tiny_worker()
    try:
        assert worker.measured_chunk_rate() is None
        worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=128)
        rate = worker.measured_chunk_rate()
        assert rate is not None
        assert rate > 0.0
        # The next timed search's first chunk is sized from it, not the bootstrap.
        d = Deadline(deadline_ms=200, now=time.monotonic())
        assert worker._time_capped_chunk(8192, d, 0) <= max(
            _MIN_TIME_CAPPED_CHUNK, int(_FIRST_CHUNK_TIME_SHARE * rate * 200) + 1,
        )
    finally:
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
    worker.reply_book_bestmove.side_effect = lambda b, searchmoves=(): book.get(b.fen())
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
        started = time.monotonic()
        engine.dispatch(parse_command(go))
        moves = out.wait_bestmoves(1)
        elapsed_ms = (time.monotonic() - started) * 1000
        assert moves == [next(iter(setup.values()))], out.text()
        assert "clock too short to search" in out.text()
        assert f"source={source}" in out.text()
        assert worker.run.call_count == 0  # no search was started
        assert elapsed_ms < 50
        # Not a fault: neither the fallback counter nor the unhealthy streak moves.
        assert engine.bestmove_fallback_used == 0
        assert engine._consecutive_search_failures == 0
        assert engine._watchdog.armed is None
    finally:
        engine.close()


def test_with_nothing_known_a_fresh_eval_answers_when_the_clock_allows(out) -> None:
    worker = _mock_worker()
    worker.fresh_root_policy_bestmove.return_value = "g8f6"
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        # 60 ms clock: no search (60 - 30 - 25 < 20), but ~27 ms to answer in.
        engine.dispatch(parse_command("go wtime 60 btime 60"))
        assert out.wait_bestmoves(1) == ["g8f6"], out.text()
        assert "source=fresh_root_eval" in out.text()
        assert worker.run.call_count == 0
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
    def exhausted(ms: int, rate: float | None) -> bool:
        return limits_from_go(GoArgs(wtime_ms=ms, btime_ms=ms), side_to_move_is_white=True,
                              move_overhead_ms=30, chunk_rate_per_ms=rate).clock_exhausted

    # Default (nothing measured): 20 x 45 ms = 900 ms after the overhead.
    assert exhausted(929, None)
    assert not exhausted(931, None)
    # Fast device, 100 sims/ms: 20 x (20 + 0.16 + 8) ~ 563 ms -> 700 is searched.
    assert not exhausted(700, 100.0)
    assert exhausted(590, 100.0)
    # Slow device, 0.5 sims/ms: 20 x (20 + 32 + 8) = 1200 ms -> 1000 is at once.
    assert exhausted(1000, 0.5)
    assert not exhausted(1300, 0.5)


def test_the_engine_prices_the_sudden_death_rule_with_the_workers_rate(out) -> None:
    worker = _mock_worker()
    worker.measured_chunk_rate.return_value = 100.0
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

    def ponder_run(*_args: Any, **_kw: Any) -> SearchResult:
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
    finally:
        gate_release.set()
        engine.close()


# -- item 8: chunk sizing against the measured overrun ---------------------------


def test_later_chunks_fill_the_remainder_by_the_measured_overrun_bound() -> None:
    w = _bare_worker(4096)
    w._batch_margin_sigmas = 2.0
    for _ in range(10):  # chunks run exactly as predicted
        w._chunk_pred_rate = 1.0
        w._note_chunk_overrun(100, 100.0)
    assert w._chunk_overrun_hi() == pytest.approx(1.0)
    d = Deadline(deadline_ms=2000, now=time.monotonic() - 1.0)  # ~1000 ms left
    got = w._time_capped_chunk(8192, d, total_nodes=1000)  # 1 sim/ms
    assert 950 <= got <= 976  # (1000 - 25 ms finishing reserve) at 1 sim/ms
    # Noisy chunks: the bound widens, the chunk shrinks.
    for r in (1.0, 2.0) * 10:
        w._chunk_pred_rate = 1.0
        w._note_chunk_overrun(100, 100.0 * r)
    assert 1.5 < w._chunk_overrun_hi() <= 2.0  # never narrower than the old 0.5 share
    # A tiny chunk is mostly fixed cost: not folded in, the bound stays put.
    n_before = w._overrun_n
    w._chunk_pred_rate = 1.0
    w._note_chunk_overrun(16, 160.0)
    assert w._overrun_n == n_before
    assert w._time_capped_chunk(8192, d, total_nodes=1000) < 700
    # The stop margin is one smallest chunk at that bound, not a whole chunk.
    margin = w._timed_stop_margin_ms(1, 2000, d, 1000, fixed_time=False)
    # (plus min(25 ms, 2 % of the budget) to finish the move).
    assert margin == pytest.approx(
        _MIN_TIME_CAPPED_CHUNK / 1.0 * w._chunk_overrun_hi() + 25.0, rel=0.05,
    )
    # ClockBatchMarginSigmas 0 disables the stop margin (the knob stays live).
    w._batch_margin_sigmas = 0.0
    assert w._timed_stop_margin_ms(1, 2000, d, 1000, fixed_time=False) == 0.0


def test_a_device_failure_restarts_the_rate_and_the_overrun_bound(monkeypatch) -> None:
    w = _bare_worker(4096)
    failed: list[str] = []
    monkeypatch.setattr(w, "failed_pool_devices", lambda: tuple(failed), raising=False)
    w._note_chunk_rate(100, 10.0)  # 10 sims/ms
    for _ in range(5):
        w._chunk_pred_rate = 10.0
        w._note_chunk_overrun(100, 10.0)
    assert w._overrun_n == 5
    failed.append("cpu:1")
    assert w.measured_chunk_rate() is None  # this topology has measured nothing yet
    w._note_chunk_rate(100, 20.0)
    assert w.measured_chunk_rate() == pytest.approx(5.0)
    assert w._overrun_n == 0
    assert w._chunk_overrun_hi() == _DEFAULT_CHUNK_OVERRUN


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
