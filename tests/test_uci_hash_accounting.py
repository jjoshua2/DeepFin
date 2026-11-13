"""UCI ``Hash`` accounting: the budget measures the tree's resident footprint.

The capacity-based ``memory_bytes()`` counted a per-node-id CBoard cache that
was ~25x larger than the boards it held, so the engine reported ~41 KB per visit
against ~5-7 KB of RSS and stopped a long search at ~1/8 of ``Hash`` (hashfull
1000) with the clock still full. These tests pin the replacement contract:
``footprint_bytes()`` matches what the tree really holds, a Hash-capped search
fills (but does not exceed) Hash and says why it stopped, and cross-move reuse
reclaims the discarded siblings instead of refusing reuse at half the cap.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any

import chess
import numpy as np

from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.uci import search as uci_search
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

_MB = 1024 * 1024


class _SpreadEvaluator:
    """Fast, deterministic, position-dependent evaluator (a degenerate constant
    evaluator would let a broken tree pass)."""

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        b = int(x.shape[0])
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


def _rss_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise AssertionError("no VmRSS")


def _expected_footprint(stats: dict[str, int]) -> int:
    return (
        stats["node_cap"] * stats["node_bytes"]
        + stats["child_count"] * stats["child_bytes"]
        + stats["cb_slot_cap"] * stats["cb_slot_bytes"]
        + stats["cb_pool_count"] * stats["cboard_bytes"]
        + stats["ht_cap"] * 4
    )


def test_node_bytes_constant_matches_c() -> None:
    c_node_bytes = MCTSTree().memory_stats()["node_bytes"]
    assert c_node_bytes == uci_search._TREE_NODE_BYTES


def test_footprint_counts_live_cache_entries_and_tracks_rss() -> None:
    w = _worker(chunk=512)
    board = chess.Board()
    stop = threading.Event()
    w.run(board, stop_event=stop, deadline=Deadline(None), max_nodes=4096)
    tree = w._tree
    assert tree is not None
    f0, rss0 = int(tree.footprint_bytes()), _rss_bytes()
    w.run(board, stop_event=stop, deadline=Deadline(None), max_nodes=24576)
    stats = tree.memory_stats()
    f1, rss1 = int(tree.footprint_bytes()), _rss_bytes()

    assert f1 == _expected_footprint(stats)
    # One cached board per evaluated leaf (+ root), not one per node id.
    visits = 4096 + 24576
    assert stats["cb_pool_count"] <= visits + 64
    assert stats["cb_pool_count"] < stats["node_count"] // 8
    # Real per-visit cost is a few KB; the old capacity figure was ~41 KB.
    assert f1 / visits < 4096
    # Growth of the footprint matches measured RSS growth within tolerance.
    df, drss = f1 - f0, rss1 - rss0
    assert df > 8 * _MB
    assert 0.5 * df <= drss <= 1.5 * df + 4 * _MB, (df, drss)
    # hashfull reports the same quantity.
    w.set_max_tree_mb(1024)
    assert w._hashfull_permille() == f1 * 1000 // (1024 * _MB)


def test_hash_capped_search_fills_hash_stays_under_it_and_says_why() -> None:
    cap_mb = 24
    w = _worker(chunk=256)
    w.set_max_tree_mb(cap_mb)
    board = chess.Board()
    strings: list[str] = []
    rss: list[int] = []

    def info_cb(**kw: Any) -> None:
        if kw.get("string"):
            strings.append(str(kw["string"]))

    def on_chunk(_nodes: int) -> None:
        rss.append(_rss_bytes())

    deadline = Deadline(120_000)
    result = w.run(
        board, stop_event=threading.Event(), deadline=deadline, max_nodes=None,
        info_cb=info_cb, on_chunk=on_chunk,
    )
    assert w.last_stop_reason == "tree_bytes"
    assert not deadline.expired()
    assert w._tree is not None
    footprint = int(w._tree.footprint_bytes())
    cap = cap_mb * _MB
    # Filled the budget (the old accounting stopped near 1/8 of it) ...
    assert footprint >= 0.85 * cap, footprint
    # ... without exceeding it (projected next-chunk growth is reserved).
    assert footprint <= cap, footprint
    hashfull = w._hashfull_permille()
    assert hashfull is not None
    assert hashfull >= 850
    # RSS grew by about the tree. At this small Hash every tree array is under
    # glibc's 32 MB dynamic mmap ceiling, so it lives in the brk heap and each
    # bounded-step realloc copy leaves a reusable hole: measured RSS growth is
    # ~1.5-2x the footprint here (exactly 1.0x with MALLOC_MMAP_THRESHOLD_
    # pinned). Arrays above 32 MB are mremap'd and add no slack, so the excess
    # is a bounded constant, not a fraction of a production-size Hash.
    # (No lower bound: in a shared test process the tree can refill heap
    # holes freed by earlier tests.)
    grew = rss[-1] - rss[0]
    assert grew <= 2 * cap + 8 * _MB, grew
    assert result.nodes > 4096
    assert any("tree memory full" in s and f"Hash {cap_mb} MB" in s for s in strings), strings


def test_ample_hash_runs_to_the_deadline() -> None:
    w = _worker(chunk=256)
    w.set_max_tree_mb(64)
    deadline = Deadline(1500)
    w.run(chess.Board(), stop_event=threading.Event(), deadline=deadline, max_nodes=None)
    assert w.last_stop_reason in ("external", "time_margin")
    assert deadline.elapsed_ms() >= 1200


def _subtree_dump(tree: MCTSTree, nid: int, depth: int) -> Any:
    actions, visits, q = tree.get_children_q(nid, 0.0)
    out = [(int(a), int(v), round(float(qq), 12)) for a, v, qq in zip(actions, visits, q)]
    if depth == 0:
        return out
    return [
        (entry, _subtree_dump(tree, int(tree.find_child(nid, entry[0])), depth - 1))
        for entry in out if entry[1] > 0
    ]


def _count_nodes(tree: MCTSTree, nid: int) -> int:
    n = 1
    actions, _ = tree.get_children_visits(nid)
    for a in actions:
        n += _count_nodes(tree, int(tree.find_child(nid, int(a))))
    return n


def test_advance_root_compacts_to_kept_subtree_when_arena_above_half_cap() -> None:
    w = _worker(chunk=256)
    board = chess.Board()
    first = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=6144)
    tree = w._tree
    assert tree is not None
    assert w._root_id is not None
    move = chess.Move.from_uci(first.bestmove_uci)
    child = int(tree.find_child(w._root_id, move_to_index(move, board)))
    before = _subtree_dump(tree, child, depth=3)
    kept_nodes = _count_nodes(tree, child)
    child_n = int(tree.get_children_visits(w._root_id)[1][
        list(tree.get_children_visits(w._root_id)[0]).index(move_to_index(move, board))
    ])
    arena_nodes = tree.node_count()
    footprint = int(tree.footprint_bytes())
    # Arena at 60% of the cap: in-place reuse used to be REFUSED here.
    w._max_tree_bytes = int(footprint / 0.6)

    assert w.advance_root(board, [move])
    assert w._tree is tree
    assert w._root_id == 0
    assert tree.node_count() == kept_nodes < arena_nodes
    assert int(tree.footprint_bytes()) < w._max_tree_bytes // 2
    assert _subtree_dump(tree, 0, depth=3) == before
    assert int(tree.get_children_visits(0)[1].sum()) + 1 >= child_n - 1

    # The compacted tree searches on from the reused root.
    after = board.copy()
    after.push(move)
    res = w.run(after, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
    assert chess.Move.from_uci(res.bestmove_uci) in after.legal_moves
    assert w._root_id == 0


def test_compaction_preserves_search_results() -> None:
    """A compacted reuse searches exactly like an in-place reuse."""
    board = chess.Board()
    results = []
    for compact in (False, True):
        w = _worker(chunk=256)
        first = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=2048)
        move = chess.Move.from_uci(first.bestmove_uci)
        assert w._tree is not None
        if compact:
            w._max_tree_bytes = 1  # half-cap 0 => compaction with no size limit
        assert w.advance_root(board, [move])
        w._max_tree_bytes = 0
        after = board.copy()
        after.push(move)
        res = w.run(after, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=2048)
        assert w._tree is not None
        assert w._root_id is not None
        results.append((res.bestmove_uci, _subtree_dump(w._tree, w._root_id, depth=2)))
    assert results[0] == results[1]


def test_growth_step_bounds_one_arena_growth() -> None:
    t = MCTSTree()
    start = t.node_capacity()
    t.set_growth_step(1000)
    t.reserve(start + 1)
    assert t.node_capacity() == start + 1000
    t.set_growth_step(0)
    t.reserve(t.node_capacity() + 1)
    assert t.node_capacity() == 2 * (start + 1000)


def test_set_max_tree_mb_applies_growth_step_to_live_tree() -> None:
    w = _worker()
    w.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=64)
    assert w._tree is not None
    w.set_max_tree_mb(1024)
    assert w._tree.memory_stats()["grow_step_max"] == (1024 * _MB // 64) // 51
    w.set_max_tree_mb(0)
    assert w._tree.memory_stats()["grow_step_max"] == 0
