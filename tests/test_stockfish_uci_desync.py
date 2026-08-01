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

import os
import stat
import sys

import pytest

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


def test_pool_close_reaps_a_replacement_engine(tmp_path):
    """A process swapped in after the close snapshot must still be reaped."""
    engine_path = _write_engine(tmp_path, "reaped.py")
    pool = StockfishPool(path=engine_path, nodes=10, num_workers=1)
    replaced_pids: list[int] = []
    try:
        for e in pool._engines:
            e.read_timeout_s = _ABANDON_TIMEOUT_S
        with pytest.raises(StockfishTimeoutError):
            pool.submit("A").result()
        # Without this the test is vacuous: with no replacement it would just
        # assert that the ORIGINAL engine died, which close() always does.
        assert pool.replacements == 1, "the desynced engine was not replaced"
        replaced_pids = [
            e.proc.pid for e in pool._engines
        ]
    finally:
        pool.close()
    for pid in replaced_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
