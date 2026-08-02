"""An abandoned Stockfish search must never let a LATER search read its result.

Live defect, 2026-07-31 (shards 033944-033951, ~16k rows). A selfplay SF label
query raised part-way — the engine was still calculating, and its ``info``/
``bestmove`` lines landed in the pty buffer AFTER we stopped reading. The next
``search`` on that engine sent its own ``position``/``go``, then read, and the
first ``bestmove`` it saw was the ABANDONED query's. The engine was one search
behind from then on, permanently: every label it produced was a real ~698k-node
Stockfish search of the WRONG position.

Nothing downstream could see it. ``sf_legal_mask``, node count and depth are
computed by the caller or by the engine's genuine work, so the labels read as
sane; ``_process_sf_label_result_for_record`` silently substituted
``legal_indices[0]`` for the illegal bestmove and carried on. The raise itself
was swallowed at DEBUG by ``poll_async_sf_labels``. Measured on the live shards:
Stockfish's bestmove was illegal at the queried position in 81-92% of labelled
rows, against a 7-9% baseline, and Spearman(sf_wdl, search_wdl) fell 0.92 -> 0.02
while Spearman(material, search_wdl) held at 0.71-0.79.

These tests drive a scripted engine over the real pty transport, so they
exercise the actual read path rather than a mock of it.
"""
from __future__ import annotations

import logging
import os
import stat
import sys
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.moves.encode import legal_move_indices
from chess_anti_engine.selfplay import stockfish_turn
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import (
    StockfishDesyncError,
    StockfishTimeoutError,
    StockfishUCI,
)


# Echoes back the position it was given as its bestmove, so a stale result is
# identifiable by name. Its FIRST `go` emits the info line, then thinks for
# `stall` seconds before the bestmove — long enough for a caller with a short
# deadline to give up while the bestmove is still coming. Later searches answer
# at once, which is the live shape: the abandoned output is already waiting, so
# the engine stops timing out (coverage recovers) even though every result it
# now hands back belongs to the previous query.
_LAGGING_ENGINE = """\
import sys, time
fen = "?"
stall = float(sys.argv[1])
for line in sys.stdin:
    line = line.strip()
    if line == 'uci':
        print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('position fen '):
        fen = line[len('position fen '):].strip()
    elif line.startswith('go'):
        print('info depth 12 score cp 11 pv ' + fen, flush=True)
        time.sleep(stall)
        stall = 0.0
        print('bestmove ' + fen, flush=True)
"""

_STALL_S = 3.0
_ABANDON_TIMEOUT_S = 0.5


def _write_engine(tmp_path, name: str, stall: float = _STALL_S) -> str:
    path = tmp_path / name
    path.write_text(_LAGGING_ENGINE)
    # The pool spawns the binary with no arguments, so bake `stall` into a
    # wrapper rather than passing it through.
    wrapper = tmp_path / f"{name}.sh"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{path}" {stall}\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return str(wrapper)


def test_abandoned_search_never_serves_its_result_to_the_next_query(tmp_path):
    """The core invariant: a later search must not return an earlier one's data.

    Pre-fix this asserts on `bestmove == "A"` for a query of position "B" —
    a genuine engine result, bound to the wrong position.
    """
    engine_path = _write_engine(tmp_path, "lagging.py")
    eng = StockfishUCI(engine_path, nodes=10, read_timeout_s=_ABANDON_TIMEOUT_S)
    try:
        # Abandon the "A" search: the engine is still thinking at the deadline,
        # and A's bestmove lands in the buffer unread.
        with pytest.raises(StockfishTimeoutError):
            eng.search("A")

        eng.read_timeout_s = 30.0
        try:
            res = eng.search("B")
        except StockfishDesyncError:
            pass
        else:
            pytest.fail(
                "a search after an abandoned one returned "
                f"bestmove={res.bestmove_uci!r} for a query of 'B' — the engine "
                "served the ABANDONED query's result",
            )
        assert eng.desynced
    finally:
        eng.close()


def test_desync_is_permanent_until_the_process_is_replaced(tmp_path):
    """It never heals on its own — that is why it ran for 40 minutes live."""
    engine_path = _write_engine(tmp_path, "lagging2.py")
    eng = StockfishUCI(engine_path, nodes=10, read_timeout_s=_ABANDON_TIMEOUT_S)
    try:
        with pytest.raises(StockfishTimeoutError):
            eng.search("A")
        eng.read_timeout_s = 30.0
        for fen in ("B", "C", "D"):
            with pytest.raises(StockfishDesyncError):
                eng.search(fen)
        # ...and a cold-TT reset is not a repair either: it is one more protocol
        # exchange on a stream that is already one message behind.
        with pytest.raises(StockfishDesyncError):
            eng.new_game()
    finally:
        eng.close()


def test_healthy_engine_is_not_poisoned(tmp_path):
    """Negative control: without an abandoned search nothing is refused.

    Mutation-guard for the fix itself — a `_protocol_section` that poisoned
    unconditionally would pass every assertion above and fail this one.
    """
    engine_path = _write_engine(tmp_path, "healthy.py", stall=0.0)
    eng = StockfishUCI(engine_path, nodes=10, read_timeout_s=30.0)
    try:
        for fen in ("A", "B", "C"):
            assert eng.search(fen).bestmove_uci == fen
            assert not eng.desynced
        eng.new_game()
        assert eng.search("D").bestmove_uci == "D"
    finally:
        eng.close()


def test_pool_replaces_a_desynced_engine_and_keeps_serving(tmp_path):
    """Availability half of the fix: throughput returns, staleness does not.

    A single-engine pool makes the ownership deterministic: the one worker
    thread must swap its own process, or every later submit fails.
    """
    engine_path = _write_engine(tmp_path, "pooled.py")
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    try:
        for e in pool._engines:
            e.read_timeout_s = _ABANDON_TIMEOUT_S
        with pytest.raises(StockfishTimeoutError):
            pool.submit("A").result()
        assert pool.replacements == 1, "the desynced engine was not replaced"

        # The replacement is a fresh process, so its first search stalls too;
        # give it room to answer.
        for e in pool._engines:
            e.read_timeout_s = 30.0
        res = pool.submit("B").result()
        assert res.bestmove_uci == "B", (
            f"pool served {res.bestmove_uci!r} for a query of 'B'"
        )
    finally:
        pool.close()


def test_pool_close_reaps_the_replacement_and_refuses_later_ones(tmp_path):
    """No Stockfish process may outlive close().

    The earlier version of this test was vacuous: the replacement is installed
    when the search raises, which is before close() takes its snapshot, so
    asserting it died proved nothing about the leftover path. The race it was
    trying to cover — a worker thread spawning a replacement AFTER the snapshot
    — is now closed at the source instead: close() sets `_closing` under the
    lock before snapshotting, and `_replace_engine` returns without spawning.
    That is directly checkable, so this asserts it directly.
    """
    engine_path = _write_engine(tmp_path, "reaped.py")
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    for e in pool._engines:
        e.read_timeout_s = _ABANDON_TIMEOUT_S
    with pytest.raises(StockfishTimeoutError):
        pool.submit("A").result()
    assert pool.replacements == 1, "the desynced engine was not replaced"
    replacement = pool._engines[0]
    pool.close()

    for pid in (replacement.proc.pid,):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    # Post-close, a replacement attempt must not spawn anything to leak.
    before = pool.replacements
    pool._replace_engine(replacement)
    assert pool.replacements == before
    assert pool._engines == [replacement]


def test_replacement_failure_does_not_supersede_the_original_error(tmp_path):
    """The caller must be told what actually went wrong.

    `_replace_engine` spawns a process from inside the `except` handler. If
    building it raises, that unrelated exception would replace the original —
    and worker.py's (StockfishTimeoutError, StockfishDesyncError) handler would
    not match it, while the curriculum-move path does not catch it at all.
    """
    engine_path = _write_engine(tmp_path, "nofix.py")
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    try:
        for e in pool._engines:
            e.read_timeout_s = _ABANDON_TIMEOUT_S
        pool.path = str(tmp_path / "does-not-exist")  # replacement will fail
        with pytest.raises(StockfishTimeoutError):
            pool.submit("A").result()
        assert pool.replacements == 0
        # The poisoned engine is still installed, so the next call is refused
        # rather than answered with stale data, and it retries the replacement.
        with pytest.raises(StockfishDesyncError):
            pool.submit("B").result()
    finally:
        pool.close()


def test_base_exception_mid_section_poisons_the_engine(tmp_path):
    """`BaseException`, not `Exception`.

    A CancelledError or a KeyboardInterrupt arriving between `go` and
    `bestmove` leaves exactly the same unread output behind as a deadline
    expiry. The choice is argued in `_protocol_section`; this pins it.
    """
    engine_path = _write_engine(tmp_path, "basexc.py", stall=0.0)
    eng = StockfishUCI(engine_path, nodes=10, read_timeout_s=30.0)
    try:
        assert eng.search("A").bestmove_uci == "A"
        with pytest.raises(KeyboardInterrupt), eng._protocol_section():
            raise KeyboardInterrupt
        assert eng.desynced, "a BaseException left the engine reported as usable"
        with pytest.raises(StockfishDesyncError):
            eng.search("B")
    finally:
        eng.close()


def test_pool_refuses_then_replaces_a_poisoned_engine(tmp_path):
    """Pins the cost of the repair, which is exactly one query.

    NOT a test of `_search`'s `except BaseException` — it reaches `_search`
    through a `StockfishDesyncError`, which IS an `Exception`, so `except
    Exception` would behave identically here. That path is covered by
    `test_search_re_raises_a_base_exception_and_still_replaces`.

    The submit that MEETS the poisoned engine is refused and re-raises (that refusal is
    the point — it is never answered with stale data), the engine is replaced
    on the way out, and the next submit is served correctly. The pool does not
    retry transparently; a swallowed retry would hide the incident, and the
    lost label is counted as `failed` in the health line.
    """
    engine_path = _write_engine(tmp_path, "poolbase.py", stall=0.0)
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    try:
        assert pool.submit("A").result().bestmove_uci == "A"
        victim = pool._engines[0]
        with pytest.raises(KeyboardInterrupt), victim._protocol_section():
            raise KeyboardInterrupt
        assert victim.desynced, "a BaseException left the engine reported as usable"

        with pytest.raises(StockfishDesyncError):
            pool.submit("B").result()
        assert pool.replacements == 1
        assert pool._engines[0] is not victim

        assert pool.submit("C").result().bestmove_uci == "C"
        assert pool.replacements == 1, "service resumed without a second swap"
    finally:
        pool.close()


# ── the health line: what an operator actually reads ────────────────────────


@pytest.fixture(name="fresh_counters")
def _fresh_counters():
    stockfish_turn._sf_label_counts.update(
        labelled=0, failed=0, bestmove_illegal=0, no_legal_pv=0,
    )
    yield
    stockfish_turn._sf_label_counts.update(
        labelled=0, failed=0, bestmove_illegal=0, no_legal_pv=0,
    )


def _emit(caplog, *, n, **per_row):
    """Drive one full report window and return the single emitted record."""
    with caplog.at_level(logging.INFO, logger="chess_anti_engine.selfplay"):
        for i in range(n):
            stockfish_turn._report_sf_label_health(
                **{k: (1 if i < v else 0) for k, v in per_row.items()},
            )
    recs = [r for r in caplog.records if r.msg.startswith("sf label health")]
    assert len(recs) == 1, f"expected exactly one health line, got {len(recs)}"
    return recs[0]


@pytest.mark.usefixtures("fresh_counters")
def test_health_line_is_info_when_the_label_path_is_clean(caplog):
    n = stockfish_turn._SF_LABEL_REPORT_EVERY
    # A BELOW-THRESHOLD reading, which is not the same as a clean one. This was
    # written as "the measured clean baseline: no_legal_pv 0.0008" -- that 0.0008
    # was residual contamination in the set chosen as the baseline. The
    # structural rate is exactly 0.000000 (see _SF_NO_LEGAL_PV_WARN_RATE). What
    # this test pins is only that a sub-threshold rate does not escalate.
    rec = _emit(caplog, n=n, no_legal_pv=3, bestmove_illegal=int(0.079 * n))
    assert rec.levelno == logging.INFO


@pytest.mark.usefixtures("fresh_counters")
def test_one_desynced_engine_out_of_eight_escalates(caplog):
    """The bar the previous threshold could not clear.

    With distributed_worker_sf_workers=8, a single desynced engine puts
    no_legal_pv near 0.074 and bestmove_illegal near 0.182. The OLD rule
    (illegal > 0.25) reads that as healthy — and read episode 1, the largest,
    as healthy too at a window mean of 0.241.
    """
    n = stockfish_turn._SF_LABEL_REPORT_EVERY
    rec = _emit(
        caplog, n=n,
        no_legal_pv=int(0.074 * n), bestmove_illegal=int(0.182 * n),
    )
    assert rec.levelno == logging.WARNING
    assert stockfish_turn._SF_BESTMOVE_ILLEGAL_CONTEXT_RATE >= 0.182, (
        "this arm must be BELOW the context rate, or it proves nothing about "
        "the sensitive detector"
    )


@pytest.mark.usefixtures("fresh_counters")
def test_episode_one_window_mean_escalates(caplog):
    """Ep1: illegal 0.241 (under the old 0.25 bar), no_legal_pv 0.1241."""
    n = stockfish_turn._SF_LABEL_REPORT_EVERY
    rec = _emit(
        caplog, n=n,
        no_legal_pv=int(0.1241 * n), bestmove_illegal=int(0.241 * n),
    )
    assert rec.levelno == logging.WARNING


@pytest.mark.usefixtures("fresh_counters")
def test_a_single_abandoned_search_escalates(caplog):
    """`failed` participates: post-fix, one abandoned search IS the incident."""
    n = stockfish_turn._SF_LABEL_REPORT_EVERY
    rec = _emit(caplog, n=n, failed=1)
    assert rec.levelno == logging.WARNING
    assert "failed=1" in rec.getMessage()


def test_replacing_a_desynced_engine_is_logged_at_warning(tmp_path, caplog):
    """The repair firing must be visible, or it is indistinguishable from never
    having been needed.

    This is the observation the whole PR exists to provide. Everything else
    about a recurrence is silent BY CONSTRUCTION once the fix works: the raise
    is swallowed at DEBUG on the label path, and `bestmove_illegal` /
    `no_legal_pv` stay at baseline precisely because no stale result is ever
    produced. If this line is not emitted, a recurrence produces nothing above
    INFO anywhere in the system.
    """
    engine_path = _write_engine(tmp_path, "logged.py", stall=0.0)
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    try:
        victim = pool._engines[0]
        with pytest.raises(KeyboardInterrupt), victim._protocol_section():
            raise KeyboardInterrupt
        with caplog.at_level(logging.WARNING, logger="chess_anti_engine.stockfish"), \
                pytest.raises(StockfishDesyncError):
            pool.submit("B").result()
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "replaced a desynced engine" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            "replacing a desynced engine produced no WARNING — the repair is "
            "unobservable"
        )
        assert "replacements=1" in warnings[0].getMessage()
    finally:
        pool.close()


# ── the counter at its real call site ───────────────────────────────────────


class _FakePV:
    def __init__(self, uci):
        self.move_uci = uci
        self.wdl = None
        self.cp = 12
        self.mate = None


class _FakeRes:
    """A genuine Stockfish result for SOME OTHER position.

    Every field is well-formed — that is the entire difficulty of this bug.
    """
    def __init__(self, bestmove, pv_ucis):
        self.bestmove_uci = bestmove
        self.pvs = [_FakePV(u) for u in pv_ucis]
        self.cp = 12
        self.mate = None
        self.wdl = None
        self.nodes = 698_000
        self.depth = 12


def _fake_state() -> Any:
    """Only the five `state.game` knobs the label path reads.

    Cast at the call site rather than built as a real SelfplayState: this test
    is about the counter, and a full state would drag in a board table, an
    engine pool and a model just to reach one branch.
    """
    game = SimpleNamespace(
        sf_policy_temp=1.0, sf_policy_label_smooth=0.0,
        sf_wdl_use_cp_logistic=True, sf_wdl_cp_slope=0.01,
        sf_wdl_cp_draw_width=60.0,
    )
    return SimpleNamespace(game=game)


def _blank_record():
    rec = _NetRecord.__new__(_NetRecord)
    for slot in _NetRecord.__slots__:
        setattr(rec, slot, None)
    return rec


@pytest.mark.usefixtures("fresh_counters")
def test_no_legal_pv_is_counted_at_the_production_call_site():
    """A stale result increments `no_legal_pv`; a fresh one does not.

    Uses the real `_process_sf_label_result_for_record`, so the counter is
    exercised through the code path that actually stamps the shard, not a
    re-implementation of it.
    """
    state = _fake_state()
    # Legal set for a K+R vs K position, side to move white (canonical).
    board = chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1")
    legal = legal_move_indices(board)
    own_ucis = [m.uci() for m in board.legal_moves]

    # Healthy: bestmove and PVs are moves of THIS position.
    stockfish_turn._process_sf_label_result_for_record(
        state, rec=_blank_record(),
        res=_FakeRes(own_ucis[0], own_ucis[:4]),
        turn=True, legal_indices=legal,
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 1
    assert stockfish_turn._sf_label_counts["no_legal_pv"] == 0
    assert stockfish_turn._sf_label_counts["bestmove_illegal"] == 0

    # Stale: a real search of an unrelated position. Its moves do not exist here.
    stale = _FakeRes("g8f6", ["g8f6", "b8c6", "e7e5"])
    stockfish_turn._process_sf_label_result_for_record(
        state, rec=_blank_record(), res=stale, turn=True, legal_indices=legal,
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 2
    assert stockfish_turn._sf_label_counts["no_legal_pv"] == 1, (
        "a result belonging to another position was not counted"
    )
    assert stockfish_turn._sf_label_counts["bestmove_illegal"] == 1


@pytest.mark.usefixtures("fresh_counters")
def test_an_already_labelled_row_is_not_counted_twice():
    """Denominator hygiene: the attach is idempotent, so the counter must be too.

    A duplicate pending entry for an already-labelled record must not inflate
    `labelled`, or the rates it feeds are read against the wrong base.
    """
    state = _fake_state()
    board = chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1")
    legal = legal_move_indices(board)
    own_ucis = [m.uci() for m in board.legal_moves]
    rec = _blank_record()
    res = _FakeRes(own_ucis[0], own_ucis[:4])

    stockfish_turn._process_sf_label_result_for_record(
        state, rec=rec, res=res, turn=True, legal_indices=legal,
    )
    stockfish_turn._process_sf_label_result_for_record(
        state, rec=rec, res=res, turn=True, legal_indices=legal,
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 1


@pytest.mark.usefixtures("fresh_counters")
@pytest.mark.parametrize("run", [1, 2])
def test_counters_reset_between_windows(caplog, run):
    """Two consecutive windows must read identically on an identical signal.

    Pins the reset in BOTH places, because the shipped reset being correct is
    not the same as it being held correct:

    * Drop ``labelled=0`` from the production reset and the denominator grows
      without bound while the numerators reset — so every rate **decays toward
      zero** and the detector goes quiet while the fault continues. That is the
      exact defect class this whole change exists to fix, sitting inside the
      fix's own instrument.
    * Drop ``no_legal_pv=0`` and the numerator accumulates instead (loud, but
      wrong).
    * Misspell a key in the TEST fixture and ``dict.update`` silently adds a
      junk key and resets nothing — caught by the zero-check below, which only
      holds on run 2 if the fixture actually ran. The body deliberately leaves a
      partial window behind so run 2 has something to fail on.

    Asserting the reported counts, not just the level, is what makes this bite:
    an unreset denominator still WARNs in window 2 here, it just lies about the
    base it was computed over.
    """
    every = stockfish_turn._SF_LABEL_REPORT_EVERY
    assert stockfish_turn._sf_label_counts == {
        "labelled": 0, "failed": 0, "bestmove_illegal": 0, "no_legal_pv": 0,
    }, f"run {run}: the fixture did not reset the counters between tests"

    with caplog.at_level(logging.INFO, logger="chess_anti_engine.selfplay"):
        for _ in range(2 * every):
            stockfish_turn._report_sf_label_health(no_legal_pv=1)
        stockfish_turn._report_sf_label_health(no_legal_pv=1)  # residue for run 2

    recs = [r for r in caplog.records if r.msg.startswith("sf label health")]
    assert len(recs) == 2, f"expected two windows, got {len(recs)}"
    for i, rec in enumerate(recs, start=1):
        labelled, no_pv = rec.args[0], rec.args[1]
        assert labelled == every, (
            f"run {run} window {i} reported labelled={labelled}, want {every} — the "
            "denominator is not reset, so every rate decays toward zero while "
            "the fault continues"
        )
        assert no_pv == every, (
            f"window {i} reported no_legal_pv={no_pv}, expected {every} — the "
            "numerator is not reset"
        )
        assert rec.levelno == logging.WARNING


# ── the sync (curriculum) label path ────────────────────────────────────────


class _StubCBoard:
    """Only what `_process_sf_results` reads off a board."""

    def __init__(self, board: chess.Board) -> None:
        self._idx = legal_move_indices(board)
        self.turn = bool(board.turn)

    def legal_move_indices(self):
        return self._idx


def _sync_state(board: chess.Board, rec) -> Any:
    state = _fake_state()
    state.opponent = SimpleNamespace(wdl_regret_limit=None)
    state.cboards = [_StubCBoard(board)]
    state.samples_per_game = [[rec]]
    state.done_arr = [0]
    state.selfplay_arr = [1]
    return state


@pytest.mark.usefixtures("fresh_counters")
def test_the_sync_label_path_is_counted_too():
    """N11: curriculum labels attach through `_process_sf_results`.

    It is one of three counting sites and the only one the other two tests do
    not reach — deleting its `_report_sf_label_health` call used to fail
    nothing. A desynced engine serves the curriculum label queue exactly as it
    serves the async one, so an uncounted site is a blind spot of the same
    shape as the bug.
    """
    board = chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1")
    own_ucis = [m.uci() for m in board.legal_moves]

    rec = _blank_record()
    rec.has_policy = True
    rec.is_sf_refute_opp = False
    stockfish_turn._process_sf_results(
        _sync_state(board, rec),
        [0],
        results={0: _FakeRes(own_ucis[0], own_ucis[:4])},
        play_curriculum_moves=False,
        attach_labels=True,
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 1
    assert stockfish_turn._sf_label_counts["no_legal_pv"] == 0

    # A real search of an unrelated position, arriving on the curriculum queue.
    rec2 = _blank_record()
    rec2.has_policy = True
    rec2.is_sf_refute_opp = False
    stockfish_turn._process_sf_results(
        _sync_state(board, rec2),
        [0],
        results={0: _FakeRes("g8f6", ["g8f6", "b8c6", "e7e5"])},
        play_curriculum_moves=False,
        attach_labels=True,
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 2
    assert stockfish_turn._sf_label_counts["no_legal_pv"] == 1, (
        "the sync label path did not count a result belonging to another position"
    )
    assert stockfish_turn._sf_label_counts["bestmove_illegal"] == 1


@pytest.mark.usefixtures("fresh_counters")
def test_a_curriculum_move_query_is_not_counted_as_a_label():
    """Denominator hygiene at the same site.

    `_process_sf_results` also runs for curriculum MOVE queries, which produce
    no training row. Counting those would put something in the denominator of a
    line named "sf label health" that is not a label, diluting exactly the rate
    the restart checklist reads.
    """
    board = chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1")
    rec = _blank_record()
    rec.has_policy = True
    rec.is_sf_refute_opp = False
    stockfish_turn._process_sf_results(
        _sync_state(board, rec),
        [0],
        results={0: _FakeRes("g8f6", ["g8f6", "b8c6"])},
        play_curriculum_moves=False,
        attach_labels=False,
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 0


def test_search_re_raises_a_base_exception_and_still_replaces(tmp_path):
    """`_search`'s handler is `except BaseException`, not `except Exception`.

    Unreachable in practice — a poisoned engine refuses at entry with
    `StockfishDesyncError`, which IS an `Exception` — so this drives the case
    directly rather than claiming the pool test covers it. It does not: that
    test reaches `_search` through a `StockfishDesyncError`, and `except
    Exception` would catch it identically.
    """
    engine_path = _write_engine(tmp_path, "basesearch.py", stall=0.0)
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    try:
        assert pool.submit("A").result().bestmove_uci == "A"
        victim = pool._engines[0]

        def _abandon(*_a, **_k):
            victim._desynced = True  # what _protocol_section would have done
            raise KeyboardInterrupt

        victim.search = _abandon
        with pytest.raises(KeyboardInterrupt):
            pool.submit("B").result()
        assert pool.replacements == 1, (
            "a BaseException left the desynced engine installed — `except "
            "Exception` here would serve the NEXT query from a poisoned engine"
        )
        assert pool.submit("C").result().bestmove_uci == "C"
    finally:
        pool.close()


# ── the WIRING from each swallow site to the counter ────────────────────────
#
# The escalation RULE is pinned above (test_a_single_abandoned_search_escalates
# calls the reporter directly). What was NOT pinned is that the two places which
# swallow a label exception actually REACH the reporter. Deleting either
# `_report_sf_label_health(failed=1)` call failed no test, on the label path's
# primary post-fix evidence -- the "a value is accepted and then silently
# ignored" shape this repo keeps producing.
#
# ⚑ These assert on the COUNTER, never on the returned `failed`. The return is a
# separate local that keeps incrementing with the reporting call deleted, so a
# test written against it passes under exactly the mutation it exists to catch.


def _pending_whose_label_raises(record: Any) -> Any:
    """A pending label whose future is already failed.

    `escalated_from_res=None` matters: with an original result present,
    `_resolve_pending_label_result` swallows the failure and returns the
    airbag instead, so the except branch under test never runs.
    """
    fut: Future[Any] = Future()
    fut.set_exception(RuntimeError("stockfish went away mid-search"))
    return stockfish_turn._PendingSfLabel(
        future=fut,
        record=record,
        turn=True,
        legal_indices=np.asarray([0], dtype=np.int32),
        escalated_from_res=None,
    )


@pytest.mark.usefixtures("fresh_counters")
def test_a_swallowed_poll_failure_reaches_the_health_counter():
    record: Any = object()
    # A stand-in, not a SelfplayState: these two functions touch only
    # `pending_sf_labels` on the failure path, and building a real state would
    # drag in the whole selfplay session for no added coverage.
    state: Any = SimpleNamespace(
        pending_sf_labels=[_pending_whose_label_raises(record)],
    )

    attached, failed = stockfish_turn.poll_async_sf_labels(state)

    assert (attached, failed) == (0, 1)
    assert stockfish_turn._sf_label_counts["failed"] == 1, (
        "poll_async_sf_labels swallowed the label exception without reaching "
        "_report_sf_label_health -- the operator's only post-fix signal that an "
        "abandoned search happened is gone, while the returned `failed` still "
        "reads 1 and looks fine"
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 1, (
        "a failed label must still count toward the report window's denominator"
    )
    assert state.pending_sf_labels == [], "the failed entry must not be retried forever"


@pytest.mark.usefixtures("fresh_counters")
def test_a_swallowed_finalize_failure_reaches_the_health_counter():
    record: Any = object()
    # A stand-in, not a SelfplayState: these two functions touch only
    # `pending_sf_labels` on the failure path, and building a real state would
    # drag in the whole selfplay session for no added coverage.
    state: Any = SimpleNamespace(
        pending_sf_labels=[_pending_whose_label_raises(record)],
    )

    attached, failed = stockfish_turn.flush_async_sf_labels_for_records(state, [record])

    assert (attached, failed) == (0, 1)
    assert stockfish_turn._sf_label_counts["failed"] == 1, (
        "flush_async_sf_labels_for_records swallowed the label exception "
        "without reaching _report_sf_label_health; this is the finalize path, "
        "the last chance to label a row before it is emitted to replay"
    )
    assert stockfish_turn._sf_label_counts["labelled"] == 1


@pytest.mark.usefixtures("fresh_counters")
def test_the_health_line_binds_each_number_to_its_own_name(caplog):
    """Four DISTINCT values, so a swapped format arg cannot pass.

    The other health tests drive counters that coincide (labelled == n, and
    single-field arms leave the rest at 0), which makes several of the six
    format arguments interchangeable without any assertion noticing.
    """
    n = stockfish_turn._SF_LABEL_REPORT_EVERY
    rec = _emit(caplog, n=n, no_legal_pv=7, bestmove_illegal=11, failed=3)
    msg = rec.getMessage()

    for field, value in (
        ("labelled", n), ("no_legal_pv", 7), ("bestmove_illegal", 11), ("failed", 3),
    ):
        assert f"{field}={value}" in msg, (
            f"{field} did not report {value} in {msg!r} -- the format arguments "
            f"are positional, so a reordering silently relabels the numbers"
        )
    assert f"({7 / n:.4f})" in msg, "the no_legal_pv RATE must use labelled as denominator"
    assert rec.levelno == logging.WARNING
