"""Forced roots and real node counts in ``SearchWorker.run``.

Found by the TCEC-readiness audit: with exactly one legal move on a 60 s clock
the engine searched the whole 4.7 s soft target, and the classic Gumbel path
reported ~100M "nodes" at ~20M nps for that search because every chunk counted
the REQUESTED simulations although the C path returns a forced root's sole
action without simulating at all.
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.search import SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

FORCED = "7k/8/8/8/8/8/6q1/7K w - - 0 1"  # Kxg2 is the only legal move


class _SpreadEvaluator:
    def __init__(self) -> None:
        self.rows = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        b = int(x.shape[0])
        self.rows += b
        rows = np.ascontiguousarray(x).reshape(b, -1)
        pol = np.zeros((b, POLICY_SIZE), dtype=np.float32)
        wdl = np.zeros((b, 3), dtype=np.float32)
        for i in range(b):
            digest = hashlib.blake2b(rows[i].tobytes(), digest_size=8).digest()
            r = np.random.default_rng(int.from_bytes(digest, "little"))
            pol[i, :] = r.standard_normal(POLICY_SIZE, dtype=np.float32)
            wdl[i, :] = r.standard_normal(3, dtype=np.float32)
        return pol, wdl


def _worker(chunk: int = 256) -> SearchWorker:
    return SearchWorker(
        _SpreadEvaluator(), device="cpu", chunk_sims=chunk,
        gumbel_cfg=GumbelConfig(simulations=chunk, topk=16, temperature=0.0, add_noise=False),
    )


def test_forced_root_on_a_clock_returns_immediately() -> None:
    w = _worker()
    t0 = time.monotonic()
    result = w.run(
        chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(20_000),
        max_nodes=None, optimum_ms=2_500,
    )
    elapsed = time.monotonic() - t0
    assert result.bestmove_uci == "h1g2"
    assert elapsed < 1.0, elapsed
    assert result.pv
    assert result.pv[0] == "h1g2"


def test_single_searchmove_on_a_clock_returns_immediately() -> None:
    w = _worker()
    t0 = time.monotonic()
    result = w.run(
        chess.Board(), stop_event=threading.Event(), deadline=Deadline(20_000),
        max_nodes=None, optimum_ms=2_500, root_moves=("e2e4",),
    )
    assert result.bestmove_uci == "e2e4"
    assert time.monotonic() - t0 < 1.0


def test_forced_root_without_optimum_waits_for_stop() -> None:
    """ponder / infinite run with optimum_ms=None and must not return early."""
    w = _worker()
    stop = threading.Event()
    out: list[SearchResult] = []
    th = threading.Thread(target=lambda: out.append(w.run(
        chess.Board(FORCED), stop_event=stop, deadline=Deadline(None), max_nodes=None,
        optimum_ms=None,
    )))
    th.start()
    time.sleep(0.6)
    assert th.is_alive()
    assert not out
    stop.set()
    th.join(timeout=10)
    assert not th.is_alive()
    assert out
    assert out[0].bestmove_uci == "h1g2"


def test_forced_root_reports_real_node_count_and_clean_batch_stats() -> None:
    w = _worker()
    result = w.run(
        chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(400),
        max_nodes=None, optimum_ms=None,
    )
    # The Gumbel path returns the sole action without simulating: 0 nodes,
    # not hundreds of thousands of requested-but-never-run simulations.
    assert result.nodes == 0
    assert w._batch_n == 0


def test_forced_root_under_node_limit_terminates() -> None:
    """A chunk that completes nothing can never reach a node budget; with no
    deadline the search must still end instead of idling until `stop`."""
    w = _worker()
    stop = threading.Event()
    out: list[SearchResult] = []
    th = threading.Thread(target=lambda: out.append(w.run(
        chess.Board(FORCED), stop_event=stop, deadline=Deadline(None), max_nodes=1000,
    )), daemon=True)
    th.start()
    th.join(timeout=10)
    hung = th.is_alive()
    stop.set()
    th.join(timeout=10)
    assert not hung
    assert out
    assert out[0].bestmove_uci == "h1g2"


def test_node_count_is_root_visits() -> None:
    w = _worker(chunk=256)
    result = w.run(
        chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=1024,
    )
    assert w._tree is not None
    assert w._root_id is not None
    visits = int(w._tree.get_children_visits(w._root_id)[1].sum())
    assert result.nodes == visits
    assert 0 < result.nodes <= 1024


def test_forced_root_on_walker_pool_spends_one_small_chunk() -> None:
    """The PUCT walker pool does simulate the forced child; a clocked search
    banks after one capped chunk instead of a full chunk_sims chunk."""
    w = SearchWorker(
        _SpreadEvaluator(), device="cpu", chunk_sims=2048, n_walkers=2,
        gumbel_cfg=GumbelConfig(simulations=2048, topk=16, temperature=0.0, add_noise=False),
    )
    try:
        result = w.run(
            chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(20_000),
            max_nodes=None, optimum_ms=2_500,
        )
    finally:
        w.close()
    assert result.bestmove_uci == "h1g2"
    assert 0 < result.nodes <= 64, result.nodes


def test_searchmoves_narrowing_rebuilt_root_counts_its_nodes() -> None:
    """A narrowed `searchmoves` root is rebuilt inside the chunk; its visits
    must be counted from 0, not clamped against the old root (which reported 0
    nodes and ended `go nodes` after one chunk)."""
    w = _worker(chunk=256)
    board = chess.Board()
    w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
    old_root = w._root_id
    result = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512,
                   root_moves=("e2e4", "d2d4"))
    assert w._root_id != old_root  # the C runner really allocated a fresh root
    assert w._tree is not None
    assert w._root_id is not None
    visits = int(w._tree.get_children_visits(w._root_id)[1].sum())
    assert result.nodes == 512
    assert visits == 512


def test_history_context_mismatch_rebuilt_root_counts_its_nodes() -> None:
    """Same FEN, different history: the carried root fails the history-context
    check and the runner allocates a fresh root mid-chunk."""
    a = chess.Board()
    for mv in ("e2e3", "e7e6", "d2d3", "d7d6"):
        a.push_uci(mv)
    b = chess.Board()
    for mv in ("d2d3", "d7d6", "e2e3", "e7e6"):
        b.push_uci(mv)
    assert a.fen() == b.fen()
    w = _worker(chunk=256)
    w.run(a, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
    old_root = w._root_id
    result = w.run(b, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
    assert w._root_id != old_root
    assert result.nodes == 512


def test_forced_root_idle_wait_does_not_spin_a_core() -> None:
    w = _worker()
    cpu0 = time.process_time()
    w.run(chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(800),
          max_nodes=None, optimum_ms=None)
    cpu = time.process_time() - cpu0
    assert cpu < 0.4, cpu  # spinning measured ~0.6 s CPU per 0.6 s wall


@pytest.mark.parametrize("walkers", [1, 2])
def test_forced_move_emits_one_info_line_with_score_and_pv(walkers: int, capsys) -> None:
    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.uci.protocol import parse_command

    worker = SearchWorker(
        _SpreadEvaluator(), device="cpu", chunk_sims=256, n_walkers=walkers,
        gumbel_cfg=GumbelConfig(simulations=256, topk=16, temperature=0.0, add_noise=False),
    )
    engine = Engine(worker)
    try:
        for line in (f"position fen {FORCED}", "go wtime 60000 btime 60000 winc 1000 binc 1000"):
            cmd = parse_command(line)
            assert cmd is not None
            engine.dispatch(cmd)
        engine._wait_for_search()
    finally:
        engine.close()
    out = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(("info", "bestmove"))]
    pv_lines = [ln for ln in out if ln.startswith("info") and " pv " in ln]
    assert out[-1] == "bestmove h1g2"
    assert len(pv_lines) == 1, out
    assert " score " in pv_lines[0]
    assert pv_lines[0].split(" pv ")[1].split()[0] == "h1g2"


def test_forced_root_info_line_comes_from_the_worker() -> None:
    """`run` itself reports a score + pv line for a forced root (the clocked
    bank fires after one chunk, before the first periodic info tick)."""
    w = _worker()
    lines: list[dict[str, Any]] = []
    w.run(chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(20_000),
          max_nodes=None, optimum_ms=2_500, info_cb=lambda **kw: lines.append(kw))
    assert len(lines) == 1
    assert tuple(lines[0]["pv"])[:1] == ("h1g2",)
    assert lines[0]["score_cp"] is not None


def test_solved_root_witness_branch_counts_real_simulations(monkeypatch) -> None:
    from chess_anti_engine.mcts.solved_root import SolvedRootWitnessUnavailable

    w = _worker(chunk=256)
    board = chess.Board()
    w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=256)
    original = w._run_gumbel_chunk
    raised: list[bool] = []

    def half_then_raise(chunk: int, *args: Any, **kwargs: Any) -> float:
        value = original(chunk // 2, *args, **kwargs)
        if not raised:
            raised.append(True)
            raise SolvedRootWitnessUnavailable(
                {0: 1}, (None, None, None, None, w._tree, [w._root_id]), {}, {0: 0},
            )
        return value

    monkeypatch.setattr(w, "_run_gumbel_chunk", half_then_raise)
    seen: list[int] = []
    w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=4096,
          on_chunk=seen.append)
    assert raised
    assert seen[0] == 128, seen  # the chunk really ran 128 sims, not the requested 256


def _zero_progress_witness(w: SearchWorker, stop: threading.Event, calls: list[int],
                           *, give_up_after: int) -> Any:
    """A `_run_gumbel_chunk` that simulates nothing and raises the witness
    exception every time. After `give_up_after` calls it sets `stop`, so a
    regression that retries forever fails on the call count instead of hanging."""
    from chess_anti_engine.mcts.solved_root import SolvedRootWitnessUnavailable

    def raise_without_work(chunk: int, *args: Any, **kwargs: Any) -> float:
        del chunk, args, kwargs
        calls.append(1)
        if len(calls) >= give_up_after:
            stop.set()
        raise SolvedRootWitnessUnavailable(
            {0: 1}, (None, None, None, None, w._tree, [w._root_id]), {0: 0}, {0: 0},
        )

    return raise_without_work


@pytest.mark.parametrize("bound", ["nodes", "depth"])
def test_zero_progress_witness_chunk_under_a_bound_ends_the_search(monkeypatch, bound) -> None:
    from chess_anti_engine.mcts.solved_root import SolvedRootWitnessUnavailable

    w = _worker(chunk=256)
    board = chess.Board()
    w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=256)
    stop = threading.Event()
    calls: list[int] = []
    monkeypatch.setattr(w, "_run_gumbel_chunk",
                        _zero_progress_witness(w, stop, calls, give_up_after=200))
    limits: dict[str, Any] = (
        {"max_nodes": 4096} if bound == "nodes" else {"max_nodes": None, "max_depth": 30}
    )
    with pytest.raises(SolvedRootWitnessUnavailable):
        w.run(board, stop_event=stop, deadline=Deadline(None), **limits)
    # The interrupted chunk made no progress, so the bound can never be met:
    # the first such chunk must end the go (the engine plays the witness
    # fallback move), not retry until `stop`.
    assert len(calls) == 1, len(calls)
    assert not stop.is_set()


def test_zero_progress_witness_chunk_without_a_bound_idles(monkeypatch) -> None:
    from chess_anti_engine.mcts.solved_root import SolvedRootWitnessUnavailable

    w = _worker(chunk=256)
    board = chess.Board()
    w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=256)
    stop = threading.Event()
    calls: list[int] = []
    monkeypatch.setattr(w, "_run_gumbel_chunk",
                        _zero_progress_witness(w, stop, calls, give_up_after=10_000))
    with pytest.raises(SolvedRootWitnessUnavailable):
        w.run(board, stop_event=stop, deadline=Deadline(300), max_nodes=None, optimum_ms=None)
    # 300 ms of 10 ms idle waits is ~30 retries; a busy retry loop does thousands.
    assert len(calls) <= 100, len(calls)


def test_search_that_already_reported_gets_no_extra_final_info_line(monkeypatch) -> None:
    from chess_anti_engine.uci import search as uci_search

    monkeypatch.setattr(uci_search, "_INFO_EMIT_INTERVAL_MS", 0)  # one line per chunk
    w = _worker(chunk=256)
    lines: list[dict[str, Any]] = []
    chunks: list[int] = []
    w.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None),
          max_nodes=1024, info_cb=lambda **kw: lines.append(kw), on_chunk=chunks.append)
    assert len(chunks) == 4, chunks
    assert len(lines) == len(chunks), (len(lines), chunks)
