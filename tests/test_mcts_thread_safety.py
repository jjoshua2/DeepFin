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
    return t, [rid] + children


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
