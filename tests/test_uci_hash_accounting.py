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
import pytest

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


def _footprint_growth_probe() -> dict[str, Any]:
    """Grow one tree by a second search; report footprint and RSS before/after
    (run through ``_run_probe_in_fresh_process``)."""
    w = _worker(chunk=512)
    board = chess.Board()
    stop = threading.Event()
    w.run(board, stop_event=stop, deadline=Deadline(None), max_nodes=4096)
    tree = w._tree
    assert tree is not None
    f0, rss0 = int(tree.footprint_bytes()), _rss_bytes()
    w.run(board, stop_event=stop, deadline=Deadline(None), max_nodes=24576)
    f1, rss1 = int(tree.footprint_bytes()), _rss_bytes()
    w.set_max_tree_mb(1024)
    return {
        "f0": f0, "f1": f1, "rss0": rss0, "rss1": rss1,
        "stats": dict(tree.memory_stats()),
        "hashfull": w._hashfull_permille(),
    }


def test_footprint_counts_live_cache_entries_and_tracks_rss() -> None:
    r = _run_probe_in_fresh_process("_footprint_growth_probe")
    f0, f1, stats = r["f0"], r["f1"], r["stats"]

    # Growth of the footprint matches measured RSS growth.
    df, drss = f1 - f0, r["rss1"] - r["rss0"]
    assert df > 8 * _MB
    assert 0.85 * df <= drss <= 1.15 * df + 4 * _MB, (df, drss)
    assert f1 == _expected_footprint(stats)
    # One cached board per evaluated leaf (+ root), not one per node id.
    visits = 4096 + 24576
    assert stats["cb_pool_count"] <= visits + 64
    assert stats["cb_pool_count"] < stats["node_count"] // 8
    # Real per-visit cost is a few KB; the old capacity figure was ~41 KB.
    assert f1 / visits < 4096
    # hashfull reports the same quantity.
    assert r["hashfull"] == f1 * 1000 // (1024 * _MB)


_CAPPED_HASH_MB = 24


def _capped_search_probe() -> dict[str, Any]:
    """A Hash-capped search from the start position, run to its stop; reports
    the stop, the footprint, and (RSS, footprint) after every chunk."""
    w = _worker(chunk=1024)  # chunk growth (~3.4 MB) >> growth step (0.4 MB)
    w.set_max_tree_mb(_CAPPED_HASH_MB)
    strings: list[str] = []
    samples: list[tuple[int, int]] = []

    def info_cb(**kw: Any) -> None:
        if kw.get("string"):
            strings.append(str(kw["string"]))

    def on_chunk(_nodes: int) -> None:
        samples.append((_rss_bytes(), w._tree_footprint_bytes()))

    deadline = Deadline(120_000)
    result = w.run(
        chess.Board(), stop_event=threading.Event(), deadline=deadline, max_nodes=None,
        info_cb=info_cb, on_chunk=on_chunk,
    )
    return {
        "stop_reason": w.last_stop_reason,
        "deadline_expired": deadline.expired(),
        "footprint": w._tree_footprint_bytes(),
        "growth_step_nodes": w._tree_growth_step_nodes(),
        "hashfull": w._hashfull_permille(),
        "nodes": result.nodes,
        "strings": strings,
        "samples": samples,
    }


def _run_probe_in_fresh_process(name: str) -> dict[str, Any]:
    """Run ``name()`` from this module in a new interpreter; return its result.

    RSS is only a usable ruler in a fresh process: in the shared pytest process
    the tree refills heap pages earlier tests already made resident, so RSS
    growth depended on test order (CI: 25 MB of RSS for 96 MB of tree). The
    mmap threshold is pinned, which turns off glibc's dynamic threshold: every
    tree array is then its own mapping and a growth realloc moves it (mremap)
    instead of copying it inside the heap and leaving a reusable hole, so RSS
    growth measures the tree alone.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import tests.test_uci_hash_accounting as m\n"
        "print(json.dumps(getattr(m, sys.argv[2])()))\n"
    )
    env = {**os.environ, "MALLOC_MMAP_THRESHOLD_": "65536", "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code, str(root), name], capture_output=True, text=True,
        timeout=300, check=False, env=env, cwd=root,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_hash_capped_search_fills_hash_stays_under_it_and_says_why() -> None:
    r = _run_probe_in_fresh_process("_capped_search_probe")
    cap_mb = _CAPPED_HASH_MB
    assert r["stop_reason"] == "tree_bytes"
    assert not r["deadline_expired"]
    footprint = int(r["footprint"])
    cap = cap_mb * _MB
    # Filled the budget (the old accounting stopped near 1/8 of it) ...
    assert footprint >= 0.85 * cap, footprint
    # ... and ends within one growth step of it (projected next-chunk growth is
    # reserved; a node-arena step inside the last chunk is the only overshoot).
    assert footprint <= cap + int(r["growth_step_nodes"]) * 51, footprint
    assert r["hashfull"] is not None
    assert r["hashfull"] >= 850
    # RSS grew by the tree's footprint growth over the same chunks: measured
    # 1.02-1.05x with the mmap threshold pinned. (Unpinned, brk-heap realloc
    # holes made it 2.85-2.97x at this Hash in a fresh process -- allocator
    # slack, which docs/TCEC_SETUP.md budgets as "Hash + 150 MB" of RAM.)
    (rss0, foot0), (rss1, foot1) = r["samples"][0], r["samples"][-1]
    dfoot, drss = foot1 - foot0, rss1 - rss0
    assert dfoot > 8 * _MB, dfoot
    assert 0.85 * dfoot <= drss <= 1.2 * dfoot + 4 * _MB, (dfoot, drss)
    assert r["nodes"] > 4096
    assert any("tree memory full" in s and f"Hash {cap_mb} MB" in s
               for s in r["strings"]), r["strings"]


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
    assert w._compaction_pending
    # The compaction runs at the start of the next run(), on that move's clock;
    # no time budget (nodes / infinite) => always affordable.
    assert w._resolve_pending_compaction(Deadline(None), None) == "compacted"
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
            w._max_tree_bytes = 1 << 40  # no budget + cheap estimate => compact
        assert w.advance_root(board, [move])
        if compact:
            assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "compacted"
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


def test_compaction_remaps_tt_and_continued_search_matches_with_transpositions() -> None:
    """After compaction every TT entry addresses a live node, and a continued
    search that really probes transpositions matches the uncompacted tree.

    A rook endgame transposes constantly. TT hits are counted as `probe_hits`;
    with history-encoded inputs they end as `context_reject` rather than
    `reuse`, but a stale (un-remapped) slot changes which probes hit."""
    from chess_anti_engine.mcts import _mcts_tree as ext

    board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    results = []
    probe_counts = []
    for compact in (False, True):
        w = _worker(chunk=256)
        first = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=4096)
        move = chess.Move.from_uci(first.bestmove_uci)
        if compact:
            w._max_tree_bytes = 1 << 40  # no budget + cheap estimate => compact
        assert w.advance_root(board, [move])
        if compact:
            assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "compacted"
            assert w._tree is not None
            live, max_id = w._tree.tt_entry_stats()
            assert live > 0
            assert max_id < w._tree.node_count()
        w._max_tree_bytes = 0
        after = board.copy()
        after.push(move)
        before = ext.tt_stats()["probe_hits"]
        res = w.run(after, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=4096)
        probe_counts.append(ext.tt_stats()["probe_hits"] - before)
        assert w._tree is not None
        assert w._root_id is not None
        results.append((res.bestmove_uci, _subtree_dump(w._tree, w._root_id, depth=2)))
    # Transpositions were really probed on both trees (the compacted TT holds
    # fewer entries — dropped siblings' slots are cleared — so counts differ).
    assert min(probe_counts) > 0, probe_counts
    assert results[0] == results[1]


def _arena_above_half_cap(w: SearchWorker) -> tuple[chess.Board, chess.Move]:
    board = chess.Board()
    first = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=4096)
    move = chess.Move.from_uci(first.bestmove_uci)
    assert w._tree is not None
    w._max_tree_bytes = int(w._tree.footprint_bytes() / 0.6)  # arena at 60% of Hash
    assert w.advance_root(board, [move])
    assert w._compaction_pending
    return board, move


def test_no_rate_fallback_too_expensive_keeps_tree_when_it_fits(monkeypatch) -> None:
    w = _worker(chunk=256)
    _arena_above_half_cap(w)
    w._recent_visits_per_ms = 0.0  # no measured growth rate: band fallback
    tree, rid = w._tree, w._root_id
    assert tree is not None
    nodes = tree.node_count()
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e9)
    assert w._resolve_pending_compaction(Deadline(60_000), 10_000) == "kept"
    assert w._tree is tree
    assert w._root_id == rid
    assert tree.node_count() == nodes


def test_no_rate_fallback_too_expensive_and_tree_too_full_rebuilds(monkeypatch) -> None:
    w = _worker(chunk=256)
    _arena_above_half_cap(w)
    w._recent_visits_per_ms = 0.0  # no measured growth rate: band fallback
    assert w._tree is not None
    w._max_tree_bytes = int(w._tree.footprint_bytes() / 0.8)  # above the keep fraction
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e9)
    assert w._resolve_pending_compaction(Deadline(60_000), 10_000) == "rebuilt"
    assert w._tree is None


def test_cheap_compaction_runs_inside_run_under_the_move_clock() -> None:
    w = _worker(chunk=256)
    board, move = _arena_above_half_cap(w)
    after = board.copy()
    after.push(move)
    w.run(after, stop_event=threading.Event(), deadline=Deadline(60_000), max_nodes=256,
          optimum_ms=30_000)
    assert w.last_reuse_decision == "compacted"
    assert not w._compaction_pending


def test_child_pool_growth_past_int32_fails_instead_of_hanging() -> None:
    """`reserve` with a child capacity near 2^31 used to wrap the int32
    doubling to 0 and spin forever holding the tree lock (and the GIL, so the
    probe runs in a subprocess with a timeout)."""
    import subprocess
    import sys

    code = (
        "from chess_anti_engine.mcts._mcts_tree import MCTSTree\n"
        "import sys\n"
        "t = MCTSTree()\n"
        "try:\n"
        "    t.reserve(4096, int(sys.argv[1]))\n"
        "    print('ok', t.memory_stats()['child_cap'])\n"
        "except MemoryError:\n"
        "    print('MemoryError')\n"
    )
    outcomes = []
    for cap in (2**31 - 1, 2**30 + 10):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, str(cap)], capture_output=True, text=True,
                timeout=60, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(f"reserve(4096, {cap}) hung") from exc
        assert proc.returncode == 0, proc.stderr
        outcomes.append(proc.stdout.strip().splitlines()[-1])
    assert outcomes[0] == "MemoryError"  # beyond the int32 id space: refused
    assert outcomes[1] == "MemoryError" or int(outcomes[1].split()[1]) >= 2**30 + 10


def test_search_stops_before_node_ids_run_out(monkeypatch) -> None:
    monkeypatch.setattr(uci_search, "_TREE_NODE_ID_LIMIT", 20_000)
    w = _worker(chunk=256)
    strings: list[str] = []

    def info_cb(**kw: Any) -> None:
        if kw.get("string"):
            strings.append(str(kw["string"]))

    w.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(60_000), max_nodes=None,
          info_cb=info_cb)
    assert w.last_stop_reason == "tree_ids"
    assert any("node-id space full" in s for s in strings), strings


def test_growth_step_is_bounded_in_absolute_terms() -> None:
    w = _worker()
    w.set_max_tree_mb(65536)
    assert w._tree_growth_step_nodes() == uci_search._MAX_TREE_GROWTH_STEP_BYTES // 51


def test_shared_tree_headroom_never_reserves_past_the_node_id_limit(monkeypatch) -> None:
    """The shared-tree (walker / pool) pre-reserve clamps a chunk to what the
    int32 id space still holds: one sim may add `_TREE_MAX_LEAF_BRANCHING`
    nodes, so only `(limit - node_count) // 256` sims fit."""
    w = _worker()
    w.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
    assert w._tree is not None
    count = w._tree.node_count()
    branching = uci_search._TREE_MAX_LEAF_BRANCHING
    monkeypatch.setattr(uci_search, "_TREE_NODE_ID_LIMIT", count + 10 * branching + 5)
    assert w._ensure_shared_tree_headroom(4096) == 10
    monkeypatch.setattr(uci_search, "_TREE_NODE_ID_LIMIT", count)
    assert w._ensure_shared_tree_headroom(4096) == 0


def _under_a_sanitizer() -> bool:
    import os

    if any(k in os.environ for k in ("ASAN_OPTIONS", "CAE_EXT_SANITIZE")):
        return True
    if "san" in os.environ.get("LD_PRELOAD", ""):
        return True
    with open("/proc/self/maps") as f:
        return any("libasan" in line or "libtsan" in line for line in f)


@pytest.mark.skipif(_under_a_sanitizer(), reason="RLIMIT_AS cannot coexist with ASan's shadow reservations")
def test_add_root_rolls_back_when_caching_its_board_fails() -> None:
    """A real allocation failure, not a mock: in a subprocess, fill the CBoard
    pool to its capacity, cap the address space just above current use, and
    add one more root. The pool's doubling realloc fails; the root must be
    rolled back (node_count unchanged) rather than left in the arena unbound."""
    import subprocess
    import sys

    code = (
        "import resource, chess\n"
        "from chess_anti_engine.mcts._mcts_tree import MCTSTree\n"
        "from chess_anti_engine.encoding._lc0_ext import CBoard\n"
        "cb = CBoard.from_board(chess.Board())\n"
        "t = MCTSTree()\n"
        "n = 1 << 15\n"
        "t.reserve(4 * n)\n"  # node arrays + slot index: no growth on add_root
        "for _ in range(n):\n"
        "    t.add_root(1, 0.0, cb)\n"
        "assert t.memory_stats()['cb_pool_count'] == n\n"
        "vm = [int(l.split()[1]) * 1024 for l in open('/proc/self/status')\n"
        "      if l.startswith('VmSize:')][0]\n"
        "resource.setrlimit(resource.RLIMIT_AS, (vm + (16 << 20), resource.RLIM_INFINITY))\n"
        "before = t.node_count()\n"
        "try:\n"
        "    t.add_root(1, 0.0, cb)\n"
        "    print('no-error', before, t.node_count())\n"
        "except MemoryError:\n"
        "    print('MemoryError', before, t.node_count())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    outcome, before, after = proc.stdout.strip().splitlines()[-1].split()
    assert outcome == "MemoryError"  # the probe really induced the failure
    assert int(after) == int(before)


def test_chunk_sims_is_bounded_by_the_node_id_headroom() -> None:
    from chess_anti_engine.mcts.search_options import MAX_CHUNK_SIMS, SEARCH_OPTIONS
    from chess_anti_engine.uci.__main__ import _build_parser

    # One chunk's worst-case growth fits in the headroom the tree_ids stop keeps.
    assert MAX_CHUNK_SIMS * uci_search._TREE_MAX_LEAF_BRANCHING <= uci_search._TREE_ID_HEADROOM
    spin = next(o for o in SEARCH_OPTIONS if o.name == "ChunkSims")
    assert spin.hi == MAX_CHUNK_SIMS
    parser = _build_parser()
    assert parser.parse_args(["--chunk-sims", str(MAX_CHUNK_SIMS)]).chunk_sims == MAX_CHUNK_SIMS
    with pytest.raises(SystemExit):
        parser.parse_args(["--chunk-sims", str(MAX_CHUNK_SIMS + 1)])
    w = _worker()
    with pytest.raises(ValueError, match="exceeds the maximum"):
        w.set_chunk_sims(MAX_CHUNK_SIMS + 1)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        _worker(chunk=MAX_CHUNK_SIMS + 1)
