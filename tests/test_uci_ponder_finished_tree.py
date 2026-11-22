"""An expired ponderhit is answered from the FINISHED ponder tree, never a live one.

When the ponder phase's `run()` has returned, the search thread records the
finished ponder (`Engine._finished_ponder`) before it waits for ponderhit/stop,
and clears it before it touches the tree again. While it is set the deadline
watchdog may answer the played reply from that tree (`source=ponder_tree`);
otherwise it falls through to the PV snapshot / fallback as before.

Also probes the superseded-print drop rule (`_superseded_through_gen`) on the
ponder paths: one bestmove per go, none lost.
"""
from __future__ import annotations

import threading
import time

import chess

from chess_anti_engine.uci.protocol import parse_command
from tests.test_uci_deadline_watchdog import _Gate, _engine, _join_all, _legal, _ok, _worker, out

__all__ = ["out"]  # the fixture, re-exported for pytest

_REAL = chess.Board()
_REAL.push_uci("e2e4")
_REAL.push_uci("e7e5")
_PONDER = chess.Board()
_PONDER.push_uci("e2e4")


def _wait_for(pred, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_an_expired_ponderhit_after_the_ponder_run_returned_answers_from_the_tree(out) -> None:
    worker = _worker()
    worker.run.side_effect = lambda board, **kw: _ok(board)
    worker.child_bestmove.return_value = "g1f3"  # legal on the real board
    engine = _engine(worker, ponder=True)
    original_retire = engine._retire_finished_ponder
    held = threading.Event()

    def stalled_retire() -> None:
        # The search thread woke from ponderhit but is held before it touches
        # the tree: the watchdog fires meanwhile and reads the finished tree.
        if not held.is_set():
            held.set()
            out.wait_bestmoves(1, timeout=5.0)
        original_retire()

    engine._retire_finished_ponder = stalled_retire  # type: ignore[method-assign]
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 100"))
        assert _wait_for(lambda: engine._finished_ponder is not None)
        engine.dispatch(parse_command("ponderhit"))
        assert out.wait_bestmoves(1) == ["g1f3"]
        assert held.is_set()
        text = out.text()
        assert "answers from the finished ponder tree source=ponder_tree move=g1f3" in text
        args = worker.child_bestmove.call_args_list[0].args
        assert args[0].fen() == _PONDER.fen()
        assert args[1] == chess.Move.from_uci("e7e5")
        _join_all(engine)
        assert out.bestmoves() == ["g1f3"]  # the thread's own late answer is dropped
        assert engine._finished_ponder is None
    finally:
        engine.close()


def test_a_ponder_run_still_in_flight_is_never_read(out) -> None:
    gate = _Gate()
    worker = _worker()
    worker.run.side_effect = gate
    worker.child_bestmove.return_value = "g1f3"
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 80"))
        assert gate.entered.wait(5)
        engine.dispatch(parse_command("ponderhit"))
        moves = out.wait_bestmoves(1)
        _legal(_REAL, moves)
        assert worker.child_bestmove.call_count == 0  # the tree was mid-search
        assert "source=ponder_tree" not in out.text()
        gate.release.set()
        _join_all(engine)
        assert len(out.bestmoves()) == 1
    finally:
        gate.release.set()
        engine.close()


def test_the_finished_ponder_is_clear_whenever_the_search_touches_the_tree(out) -> None:
    worker = _worker()
    engine = _engine(worker, ponder=True)
    seen: list[tuple[str, object]] = []

    def record(name):
        def call(*a, **_kw):
            seen.append((name, engine._finished_ponder))
            if name == "run":
                return _ok(a[0])
            return True if name == "advance_root" else None
        return call

    worker.run.side_effect = record("run")
    worker.advance_root.side_effect = record("advance_root")
    worker.reset_tree.side_effect = record("reset_tree")
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 400"))
        # Between the ponder run's return and ponderhit, the finished tree is readable.
        assert _wait_for(lambda: engine._finished_ponder is not None)
        finished = engine._finished_ponder
        assert finished is not None
        assert finished[0] == engine._search_gen
        engine.dispatch(parse_command("ponderhit"))
        assert len(out.wait_bestmoves(1)) == 1
        _join_all(engine)
        assert engine._finished_ponder is None
        # A ponder miss (stop) leaves nothing readable once the thread is gone.
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3 b8c6"))
        engine.dispatch(parse_command("go ponder movetime 400"))
        assert _wait_for(lambda: engine._finished_ponder is not None)
        engine.dispatch(parse_command("stop"))
        assert len(out.wait_bestmoves(2)) == 2
        _join_all(engine)
        assert engine._finished_ponder is None
        engine.dispatch(parse_command("go movetime 50"))
        assert len(out.wait_bestmoves(3)) == 3
        _join_all(engine)
    finally:
        engine.close()
    touched = [name for name, _ in seen]
    assert touched.count("run") >= 4
    assert "advance_root" in touched
    assert all(state is None for _name, state in seen), seen


# -- the superseded-print drop rule on the ponder paths ----------------------


def test_a_ponderhit_prints_exactly_one_bestmove(out) -> None:
    worker = _worker()
    worker.run.side_effect = lambda board, **kw: (time.sleep(0.02), _ok(board))[1]
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 200"))
        assert _wait_for(lambda: worker.run.call_count >= 1)
        engine.dispatch(parse_command("ponderhit"))
        moves = out.wait_bestmoves(1)
        _join_all(engine)
        time.sleep(0.05)
        assert out.bestmoves() == moves
        assert len(moves) == 1
        _legal(_REAL, moves)
    finally:
        engine.close()


def test_a_parked_go_ponder_and_the_next_ponderhit_are_each_answered_once(out) -> None:
    gate = _Gate()
    worker = _worker()
    calls = {"n": 0}

    def run(board, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return gate(board, **kw)  # the first ponder search hangs
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 200"))
        assert gate.entered.wait(5)
        engine.dispatch(parse_command("stop"))  # does not stop in time: parked, answered
        first = out.wait_bestmoves(1)
        assert len(first) == 1
        _legal(_REAL, first)
        gate.release.set()  # the parked search returns; its late answer is dropped
        assert _wait_for(lambda: not engine._orphan_threads
                         or not any(t.is_alive() for t in engine._orphan_threads))
        time.sleep(0.05)
        assert len(out.bestmoves()) == 1
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3 b8c6"))
        engine.dispatch(parse_command("go ponder movetime 200"))
        assert _wait_for(lambda: calls["n"] >= 2)
        engine.dispatch(parse_command("ponderhit"))
        both = out.wait_bestmoves(2)
        _join_all(engine)
        time.sleep(0.05)
        assert len(out.bestmoves()) == 2
        assert out.bestmoves() == both
        new_real = _REAL.copy()
        new_real.push_uci("g1f3")
        new_real.push_uci("b8c6")
        _legal(new_real, both[1:])
    finally:
        gate.release.set()
        engine.close()


def test_a_go_ponder_that_supersedes_a_hung_one_gets_the_only_bestmove(out) -> None:
    # A GUI sends a new `go ponder` while the old ponder search hangs: the old
    # one is parked WITHOUT an answer (its go is superseded: the drop rule's
    # generation bump), the new go is answered exactly once after ponderhit,
    # and the hung search's late answer never prints.
    gate = _Gate()
    worker = _worker()
    calls = {"n": 0}

    def run(board, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return gate(board, **kw)
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker, ponder=True)
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder movetime 200"))
        assert gate.entered.wait(5)
        superseded_before = engine._superseded_through_gen
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3 b8c6"))
        engine.dispatch(parse_command("go ponder movetime 200"))
        assert engine._superseded_through_gen > superseded_before  # the rule is in force
        engine.dispatch(parse_command("ponderhit"))
        moves = out.wait_bestmoves(1)
        gate.release.set()
        assert _wait_for(lambda: not any(t.is_alive() for t in engine._orphan_threads))
        time.sleep(0.1)
        assert len(out.bestmoves()) == 1
        assert out.bestmoves() == moves
        new_real = _REAL.copy()
        new_real.push_uci("g1f3")
        new_real.push_uci("b8c6")
        _legal(new_real, moves)
    finally:
        gate.release.set()
        engine.close()
