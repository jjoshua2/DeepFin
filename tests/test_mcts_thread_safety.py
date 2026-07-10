"""Thread-safety stress test for MCTSTree (phase 4).

Exercises the hot-path atomics and shard mutexes added in phase 4 by running
many Python threads against one shared tree. Not a correctness proof — we
can't assert bit-identical behavior under concurrency — but it catches:
  - segfaults from realloc during concurrent descent (gated by tree.reserve)
  - torn visit counts (tests conservation: sum of N after walkers done ==
    total number of backprops performed)
  - deadlocks (test times out if a mutex is dropped)
  - NaN Q values from torn W reads (bounded by [-1, 1] when walkers agree)

The test uses expand + backprop + apply_vloss_path / remove_vloss_path — the
primitives the phase 5 walker pool will call. It does NOT exercise the
Gumbel state machine (which still has per-tree singleton state — walker
pool will use reentrant primitives directly, not start_gumbel_sims).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chess_anti_engine.mcts._mcts_tree import MCTSTree


def _build_shallow_tree(seed: int) -> tuple[MCTSTree, list[int]]:
    """Root with 8 children, each child expanded with 4 grandchildren.
    Returns (tree, list of root+children+grandchildren node ids)."""
    rng = np.random.default_rng(seed)
    t = MCTSTree()
    # Reserve enough capacity so no realloc fires during concurrent descent.
    t.reserve(1024, 8192)
    rid = t.add_root(0, 0.0)
    actions = np.arange(8, dtype=np.int32)
    priors = rng.dirichlet(np.ones(8))
    t.expand(rid, actions, priors)

    children: list[int] = []
    for a in range(8):
        cid = t.find_child(rid, a)
        children.append(cid)
        gactions = np.arange(4, dtype=np.int32) + 100
        gpriors = rng.dirichlet(np.ones(4))
        t.expand(cid, gactions, gpriors)
    return t, [rid, *children]


def test_concurrent_apply_remove_vloss_conserves():
    """N walkers concurrently apply+remove vloss on random paths. After join,
    every node's virtual_loss must be back to 0 (conservation)."""
    t, nodes = _build_shallow_tree(seed=0)
    rid = nodes[0]
    children = nodes[1:]

    n_threads = 8
    ops_per_thread = 2000
    stop = threading.Event()
    rng_seeds = list(range(n_threads))

    def worker(seed: int) -> None:
        local_rng = np.random.default_rng(seed)
        for _ in range(ops_per_thread):
            if stop.is_set():
                return
            cid = int(local_rng.choice(children))
            path = np.array([rid, cid], dtype=np.int32)
            t.apply_vloss_path(path)
            t.remove_vloss_path(path)

    threads = [threading.Thread(target=worker, args=(s,)) for s in rng_seeds]
    start = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    stop.set()

    assert time.time() - start < 30.0, "worker threads did not finish in time"
    for th in threads:
        assert not th.is_alive(), "walker thread still running (possible deadlock)"

    # Conservation: every apply must have been matched by exactly one remove.
    for nid in range(t.node_count()):
        assert t.get_virtual_loss(nid) == 0, (
            f"node {nid} vloss leaked: {t.get_virtual_loss(nid)}")


def test_concurrent_tree_expand_idempotent():
    """Multiple threads racing to expand the same unexpanded node must all
    see expanded==True at the end, and tree.node_count must reflect ONE
    expansion's worth of children (tree_expand re-checks under the lock)."""
    t = MCTSTree()
    t.reserve(1024, 8192)
    rid = t.add_root(0, 0.0)
    actions = np.arange(8, dtype=np.int32)
    priors = np.full(8, 1.0 / 8, dtype=np.float64)
    t.expand(rid, actions, priors)

    # Pick an unexpanded leaf.
    leaf_id = t.find_child(rid, 0)
    assert not t.is_expanded(leaf_id)

    barrier = threading.Barrier(8)

    def expander() -> None:
        barrier.wait()
        t.expand(leaf_id, np.arange(4, dtype=np.int32),
                 np.full(4, 0.25, dtype=np.float64))

    threads = [threading.Thread(target=expander) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10.0)

    assert t.is_expanded(leaf_id)
    # Idempotent: only one thread's expansion should have added children.
    # Root (1) + 8 root children + 4 leaf children = 13 nodes.
    assert t.node_count() == 13, (
        f"expected 13 nodes, got {t.node_count()} — concurrent expansion "
        f"added duplicate children")


def test_concurrent_gil_released_backprop_preserves_w_and_n():
    """Bulk worker backprops must not lose W updates while N stays atomic.

    ``batch_integrate_leaves`` releases the GIL and is called concurrently by
    the multi-GPU PUCV workers. A plain ``W += q`` loses updates even though N
    advances atomically, pulling Q toward zero under contention.
    """
    t = MCTSTree()
    rid = t.add_root(0, 0.0)

    n = 256
    n_threads = 8
    loops = 100
    path_buf = np.full(n * 512, rid, dtype=np.int32)
    path_lens = np.ones(n, dtype=np.int32)
    legal_buf = np.zeros(n * 256, dtype=np.int32)
    legal_lens = np.zeros(n, dtype=np.int32)
    is_term = np.zeros(n, dtype=np.int8)
    pol = np.zeros((n, 4672), dtype=np.float32)
    wdl = np.tile(np.array([[30.0, 0.0, -30.0]], dtype=np.float32), (n, 1))
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(loops):
            t.batch_integrate_leaves(
                n, path_buf, path_lens, legal_buf, legal_lens,
                is_term, pol, wdl, 0,
            )

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    for th in threads:
        assert not th.is_alive(), "concurrent backprop worker did not finish"

    # softmax([30, 0, -30]) has Q indistinguishable from 1 at this tolerance.
    assert t.node_q(rid) == pytest.approx(1.0, abs=1e-12)


def test_reserve_grows_capacity_without_affecting_data():
    """reserve(cap) pre-grows arrays. After reserve, existing node data is
    preserved and the tree still works normally."""
    t = MCTSTree()
    rid = t.add_root(0, 0.0)
    actions = np.arange(4, dtype=np.int32)
    priors = np.full(4, 0.25, dtype=np.float64)
    t.expand(rid, actions, priors)
    n_before = t.node_count()

    t.reserve(10_000, 20_000)

    assert t.node_count() == n_before
    for a in range(4):
        assert t.find_child(rid, a) >= 0


def test_reserve_with_smaller_cap_is_noop():
    t = MCTSTree()
    t.add_root(0, 0.0)
    t.reserve(100)
    t.reserve(1)  # smaller — should be no-op, not shrink
    assert t.node_count() >= 1


@pytest.mark.slow
def test_stress_descent_and_backprop_no_crash():
    """Full stress: threads descend via select_leaves + backprop on a shared
    tree for a fixed wall-clock time. The invariant is simply 'no crash' —
    correctness under races is verified separately in the phase 5 walker
    bench vs single-threaded control."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    t = MCTSTree()
    t.reserve(50_000, 500_000)
    rid = t.add_root(0, 0.0)
    actions = np.arange(16, dtype=np.int32)
    priors = np.full(16, 1.0 / 16, dtype=np.float64)
    t.expand(rid, actions, priors)

    duration = 1.5  # seconds
    stop_at = time.time() + duration
    crashes: list[BaseException] = []

    def worker() -> None:
        try:
            root_ids = np.array([rid], dtype=np.int32)
            while time.time() < stop_at:
                leaves = t.select_leaves(root_ids, 1.5, 0.0, 0.33)
                for entry in leaves:
                    leaf_id, _, node_path, _ = entry
                    # Expand leaf if unexpanded, then backprop.
                    if not t.is_expanded(leaf_id):
                        sub_actions = np.arange(8, dtype=np.int32) + 200
                        sub_priors = np.full(8, 1.0 / 8, dtype=np.float64)
                        t.expand(leaf_id, sub_actions, sub_priors)
                    t.backprop(node_path, 0.1)
        except BaseException as e:
            crashes.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=duration + 10.0)

    assert not crashes, f"worker crashed: {crashes[0]}"
    for th in threads:
        assert not th.is_alive(), "worker thread did not finish"
    # At least one backprop must have landed at the root.
    root_q = t.node_q(rid)
    assert root_q == root_q, "root Q is NaN"  # NaN check via self-inequality


def _integrate_fixture(
    rows: int,
) -> tuple[MCTSTree, int, int, tuple[np.ndarray, ...]]:
    """Tree with one root->child edge plus batch_integrate_leaves buffers for
    ``rows`` identical non-terminal leaves whose path is [root, child].
    legal_lens are all 0 (no expand attempt) and vloss_weight is passed as 0
    by callers, so each row is a pure tree_backprop through the shared pair."""
    max_path = 512  # MCTS_MAX_PATH in _mcts_tree.c
    t = MCTSTree()
    t.reserve(64, 256)
    rid = t.add_root(0, 0.0)
    t.expand(rid, np.array([7], dtype=np.int32), np.array([1.0], dtype=np.float64))
    cid = t.find_child(rid, 7)

    path_buf = np.zeros(rows * max_path, dtype=np.int32)
    for i in range(rows):
        path_buf[i * max_path] = rid
        path_buf[i * max_path + 1] = cid
    path_lens = np.full(rows, 2, dtype=np.int32)
    legal_buf = np.zeros(rows * 256, dtype=np.int32)
    legal_lens = np.zeros(rows, dtype=np.int32)
    is_term = np.zeros(rows, dtype=np.int8)
    pol = np.zeros((rows, 4672), dtype=np.float32)
    wdl = np.zeros((rows, 3), dtype=np.float32)
    wdl[:, 0] = 3.0  # q = softmax-margin of (3,0,0): deterministic, nonzero
    bufs = (path_buf, path_lens, legal_buf, legal_lens, is_term, pol, wdl)
    return t, rid, cid, bufs


def test_concurrent_backprop_w_sum_is_exact():
    """W accumulation must be atomic, not just torn-read tolerant.

    8 threads hammer batch_integrate_leaves (GIL released inside) with paths
    through ONE shared root->child pair, all backpropping the same value q.
    Because every add is the same q, the accumulator's value depends only on
    how many adds have completed — so the final W must equal the sequential
    left fold of `total` adds EXACTLY, in any interleaving. Before the
    atomic_add_double fix in _mcts_tree.c the plain `W += v` read-modify-write
    loses colliding updates and this assertion fails reliably at these counts
    (8 threads x 40 calls x 128 rows on a 2-node path).
    """
    n_threads = 8
    iters = 40
    rows = 128

    t, rid, cid, bufs = _integrate_fixture(rows)
    path_buf, path_lens, legal_buf, legal_lens, is_term, pol, wdl = bufs

    # Reference q: one integrate on an identical fresh tree; with N=1 the
    # child's Q IS the backpropped q (0.0 + q == q).
    ref, _ref_rid, ref_cid, ref_bufs = _integrate_fixture(1)
    r_path, r_plens, r_legal, r_llens, r_term, r_pol, r_wdl = ref_bufs
    ref.batch_integrate_leaves(
        1, r_path, r_plens, r_legal, r_llens, r_term, r_pol, r_wdl, 0,
    )
    q = ref.node_q(ref_cid)
    assert q != 0.0

    start = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            start.wait()
            for _ in range(iters):
                t.batch_integrate_leaves(
                    rows, path_buf, path_lens, legal_buf, legal_lens,
                    is_term, pol, wdl, 0,
                )
        except BaseException as e:  # pragma: no cover — surfaced via assert
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60.0)
    assert not errors, f"worker crashed: {errors[0]}"

    total = n_threads * iters * rows
    expected_child = 0.0
    expected_root = 0.0
    for _ in range(total):
        expected_child += q   # leaf perspective: +q per backprop
        expected_root += -q   # sign alternates up the path
    # node_q = W/N with N == total; both sides are the same IEEE double
    # division, so equality holds iff W matches the fold EXACTLY. A lost
    # update shifts W by a whole |q| (~0.86) — far beyond quotient rounding.
    assert t.node_q(cid) == expected_child / total
    assert t.node_q(rid) == expected_root / total
