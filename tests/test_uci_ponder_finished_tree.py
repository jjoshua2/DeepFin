"""An expired ponderhit whose real phase stalls is answered from the ponder tree.

On ponderhit, BEFORE the search thread advances the root (the ponder tree is
finished and nothing runs on it), it asks the tree for its most-visited answer
to the played reply and seeds that as this go's `_pv_snapshot`
(`Engine._seed_ponder_answer`). If the real phase then stalls past its
deadline, the watchdog answers the seed -- `source=ponder_tree`, a searched
move, not counted as a fallback -- without touching the tree itself. A real
phase PV replaces the seed (then `source=last_pv`, a fallback as before), each
`go` clears it, and a ponder miss never seeds.

Also probes the superseded-print drop rule (`_superseded_through_gen`) on the
ponder paths: one bestmove per go, none lost.
"""
from __future__ import annotations

import threading
import time

import chess

from chess_anti_engine.uci.protocol import parse_command
from tests.test_uci_deadline_watchdog import _Gate, _engine, _join_all, _legal, _ok, _publish, _worker, out

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


def _hung_real_phase(worker, *, before_hang=None) -> threading.Event:
    """worker.run: the ponder phase returns at once; the real phase hangs."""
    hang = threading.Event()
    calls = {"n": 0}

    def run(board, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _ok(board)
        if before_hang is not None:
            before_hang(board, kw)
        hang.wait(10)
        return _ok(board)

    worker.run.side_effect = run
    return hang


def _ponderhit(engine, movetime: int = 100, extra: str = "") -> None:
    engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
    engine.dispatch(parse_command(f"go ponder movetime {movetime}{extra}"))
    assert _wait_for(lambda: engine._worker.run.call_count >= 1)
    time.sleep(0.05)  # the ponder run has returned; the thread waits for the GUI
    engine.dispatch(parse_command("ponderhit"))


def test_a_stalled_real_phase_after_ponderhit_answers_from_the_ponder_tree(out) -> None:
    worker = _worker()
    hang = _hung_real_phase(worker)
    worker.child_bestmove.return_value = "g1f3"  # legal on the real board
    engine = _engine(worker, ponder=True)
    try:
        _ponderhit(engine)
        assert out.wait_bestmoves(1) == ["g1f3"]
        assert "answers from the ponder tree source=ponder_tree move=g1f3" in out.text()
        # a searched move: neither counted as a fallback nor extending the streak
        assert engine.bestmove_fallback_used == 0
        assert engine._consecutive_search_failures == 0
        (call,) = worker.child_bestmove.call_args_list
        assert call.args[0].fen() == _PONDER.fen()
        assert call.args[1] == chess.Move.from_uci("e7e5")
        assert call.kwargs["min_visits"] == 8
    finally:
        hang.set()
        engine.close()


def test_a_thin_ponder_child_does_not_seed_and_the_watchdog_falls_back(out) -> None:
    worker = _worker()
    hang = _hung_real_phase(worker)
    worker.child_bestmove.return_value = None  # no child with 8+ visits
    engine = _engine(worker, ponder=True)
    try:
        _ponderhit(engine)
        moves = out.wait_bestmoves(1)
        _legal(_REAL, moves)
        assert "source=ponder_tree" not in out.text()
        assert engine.bestmove_fallback_used == 1
        assert engine._consecutive_search_failures == 1
    finally:
        hang.set()
        engine.close()


def test_a_real_phase_pv_replaces_the_seed(out) -> None:
    worker = _worker()
    hang = _hung_real_phase(worker, before_hang=lambda board, kw: _publish(kw["info_cb"], "b1c3"))
    worker.child_bestmove.return_value = "g1f3"
    engine = _engine(worker, ponder=True)
    try:
        _ponderhit(engine)
        assert out.wait_bestmoves(1) == ["b1c3"]
        assert "source=last_pv move=b1c3" in out.text()
        assert engine.bestmove_fallback_used == 1  # the usual watchdog accounting
    finally:
        hang.set()
        engine.close()


def test_a_seed_outside_the_gos_searchmoves_is_refused(out) -> None:
    worker = _worker()
    hang = _hung_real_phase(worker)
    worker.child_bestmove.return_value = "g1f3"  # not in searchmoves
    engine = _engine(worker, ponder=True)
    try:
        _ponderhit(engine, extra=" searchmoves b1c3")
        assert out.wait_bestmoves(1) == ["b1c3"]
        assert "source=ponder_tree" not in out.text()
        assert worker.child_bestmove.call_args.kwargs["searchmoves"] == ("b1c3",)
    finally:
        hang.set()
        engine.close()


def test_the_seed_never_outlives_its_go_nor_appears_on_a_ponder_miss(out) -> None:
    worker = _worker()
    worker.child_bestmove.return_value = "g1f3"
    worker.run.side_effect = lambda board, **kw: _ok(board)
    engine = _engine(worker, ponder=True)
    try:
        _ponderhit(engine, movetime=400)
        assert len(out.wait_bestmoves(1)) == 1
        _join_all(engine)
        assert engine._ponder_seed is not None  # this go seeded it
        # the next go clears it at once: no stale key can reach its watchdog
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3 b8c6"))
        engine.dispatch(parse_command("go ponder movetime 400"))
        assert engine._ponder_seed is None
        assert _wait_for(lambda: worker.run.call_count >= 3)
        time.sleep(0.05)
        engine.dispatch(parse_command("stop"))  # a ponder miss
        assert len(out.wait_bestmoves(2)) == 2
        _join_all(engine)
        assert engine._ponder_seed is None
        assert engine._pv_snapshot is None or engine._pv_snapshot is not engine._ponder_seed
    finally:
        engine.close()


def test_a_stalled_next_go_does_not_answer_the_previous_seed(out) -> None:
    worker = _worker()
    worker.child_bestmove.return_value = "g1f3"
    calls = {"n": 0}
    hang = threading.Event()

    def run(board, **_kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _ok(board)  # ponder + real phase of the first go
        hang.wait(10)  # the next go hangs
        return _ok(board)

    worker.run.side_effect = run
    engine = _engine(worker, ponder=True)
    try:
        _ponderhit(engine, movetime=400)
        assert len(out.wait_bestmoves(1)) == 1
        _join_all(engine)
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go movetime 60"))
        moves = out.wait_bestmoves(2)
        assert "source=ponder_tree" not in out.text().split("bestmove")[-2]
        assert engine.bestmove_fallback_used == 1
        _legal(_REAL, moves[1:])
    finally:
        hang.set()
        engine.close()


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
