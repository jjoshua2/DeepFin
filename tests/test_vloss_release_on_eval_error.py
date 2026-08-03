"""Virtual loss must not survive an evaluator that raises mid-search.

SHARED_BROKER_AUDIT B7. Every batched-VL search here splits one atomic
operation across two C calls with a network evaluation in between::

    tree.walker_descend_puct(...)      # APPLIES vloss
    pol, wdl = evaluator.evaluate(...) # <-- raises (broker TimeoutError, I1)
    tree.walker_integrate_leaf(...)    # REMOVES vloss, then backprops

No site guarded that span. ``MCTSTree.remove_vloss_path`` existed, was
exported, and was called from nowhere in ``chess_anti_engine/``. The tree is
caller-owned and outlives the failed run (``SearchWorker._tree`` is reused
across chunks and plies), so the leak biased selection away from those subtrees
for the rest of the game — invisibly, since nothing reads virtual loss back.

Measured pre-fix by ``sb_cwalk_vloss_leak_on_eval_error.py``: 5/5 failed runs
leaked 8-48 units; 1/1 clean run leaked none. The clean-run control is repeated
here, because a cleanup that fires unconditionally would also read as "no leak"
while corrupting a healthy search.
"""
from __future__ import annotations

import logging
import threading

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct_vl import PucvChunker
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig

PLANES = input_plane_count()


class _FlakyEvaluator:
    """Serves ``ok_calls`` batches, then raises the broker's own failure."""

    def __init__(self, ok_calls: int) -> None:
        self.calls = 0
        self.ok_calls = ok_calls
        self._lock = threading.Lock()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert relations is None
        with self._lock:
            self.calls += 1
            n = self.calls
        if n > self.ok_calls:
            raise TimeoutError("inference broker timed out after 30.000s")
        rng = np.random.default_rng(n)
        b = int(x.shape[0])
        return (
            rng.normal(size=(b, 4672)).astype(np.float32),
            rng.normal(size=(b, 3)).astype(np.float32),
        )


def _total_vloss(tree: MCTSTree) -> tuple[int, int]:
    total = nonzero = 0
    for nid in range(tree.node_count()):
        v = int(tree.get_virtual_loss(nid))
        if v:
            total += v
            nonzero += 1
    return total, nonzero


def _fresh_tree() -> tuple[MCTSTree, CBoard, int]:
    tree = MCTSTree()
    tree.reserve(4096, 32768)
    root_cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = root_cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size, dtype=np.float64))
    return tree, root_cb, rid


def _run_walkers(*, ok_calls: int, walkers: int, gather: int, sims: int) -> tuple[int, int, bool]:
    tree, root_cb, rid = _fresh_tree()
    pool = WalkerPool(
        WalkerPoolConfig(
            n_walkers=walkers, c_puct=1.5, fpu_at_root=0.0, fpu_reduction=0.33,
            vloss_weight=3, gather=gather, input_planes=PLANES,
        ),
        _FlakyEvaluator(ok_calls),
    )
    raised = False
    try:
        pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
                 target_sims=sims, stop_event=threading.Event())
    except Exception:
        raised = True
    finally:
        pool.close()
    total, nonzero = _total_vloss(tree)
    return total, nonzero, raised


def test_walker_pool_control_clean_run_ends_at_zero_vloss() -> None:
    """Negative control. Without it, a cleanup that ran unconditionally would
    pass every other assertion in this file while corrupting healthy searches."""
    total, nonzero, raised = _run_walkers(ok_calls=10_000, walkers=4, gather=4, sims=200)
    assert not raised
    assert (total, nonzero) == (0, 0)


def test_walker_pool_releases_vloss_when_the_evaluator_raises() -> None:
    """The repro's exact shape: 4 walkers, gather 8, evaluator dies after 2 calls.

    Repeated 5x for the same reason the repro was: WHICH leaves are in flight
    when the evaluator raises is timing-dependent, so a single run can miss.
    Pre-fix all 5 leaked (totals 48/32/16/16/16 in one sitting).
    """
    for trial in range(5):
        total, nonzero, raised = _run_walkers(ok_calls=2, walkers=4, gather=8, sims=400)
        assert raised, f"[{trial}] the flaky evaluator must actually fail the run"
        assert (total, nonzero) == (0, 0), (
            f"[{trial}] leaked {total} virtual loss over {nonzero} nodes; the "
            "tree the caller keeps is permanently biased away from those subtrees"
        )


def test_walker_pool_leaves_a_partially_integrated_batch_consistent() -> None:
    """The evaluator returns, then integration itself fails part-way.

    ``walker_integrate_leaf`` removes vloss for the leaf it consumes, so only
    the UN-integrated tail may be released.

    ⚑ The end-state total CANNOT see a double release: ``remove_vloss_path``
    floors at 0, so releasing an already-integrated path still ends at 0 on an
    idle tree. It is not harmless — the decrement lands on shared ancestors and
    steals a CONCURRENT walker's in-flight virtual loss, biasing selection back
    toward the subtree everyone is already in. So this test counts the release
    CALLS instead of summing the result; a mutation that releases the whole
    batch survives every total-based assertion and dies here.
    """
    tree, root_cb, rid = _fresh_tree()

    class _FlakyIntegrateTree:
        """Delegating proxy: MCTSTree is a C type, its methods are read-only."""

        def __init__(self, inner: MCTSTree) -> None:
            self._inner = inner
            self.integrated = 0
            self.descended_nonterminal = 0
            self.removed = 0

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def walker_descend_puct(
            self, root_id: int, root_cboard: CBoard, c_puct: float,
            fpu_root: float, fpu_reduction: float, vloss_weight: int,
            enc_out: np.ndarray, rel_out: np.ndarray | None = None,
        ) -> tuple:
            out = self._inner.walker_descend_puct(
                root_id, root_cboard, c_puct, fpu_root, fpu_reduction,
                vloss_weight, enc_out, rel_out,
            )
            if out[3] is None:  # term_q None == non-terminal == vloss applied
                self.descended_nonterminal += 1
            return out

        def remove_vloss_path(self, path: np.ndarray) -> None:
            self.removed += 1
            self._inner.remove_vloss_path(path)

        def walker_integrate_leaf(
            self, node_path: np.ndarray, legal: np.ndarray,
            pol_logits: np.ndarray, wdl_logits: np.ndarray, vloss_weight: int,
        ) -> None:
            if self.integrated >= 3:
                raise RuntimeError("integrate blew up")
            self._inner.walker_integrate_leaf(
                node_path, legal, pol_logits, wdl_logits, vloss_weight,
            )
            self.integrated += 1

    proxy = _FlakyIntegrateTree(tree)
    pool = WalkerPool(
        WalkerPoolConfig(
            n_walkers=1, c_puct=1.5, fpu_at_root=0.0, fpu_reduction=0.33,
            vloss_weight=3, gather=8, input_planes=PLANES,
        ),
        _FlakyEvaluator(10_000),
    )
    try:
        with pytest.raises(RuntimeError, match="integrate blew up"):
            pool.run(tree=proxy, root_id=rid, root_cboard=root_cb,  # pyright: ignore[reportArgumentType]
                     target_sims=64, stop_event=threading.Event())
    finally:
        pool.close()
    assert proxy.integrated == 3, "integration must have partially succeeded"
    assert proxy.descended_nonterminal > proxy.integrated, (
        "the batch must have had un-integrated leaves left over"
    )
    assert proxy.removed == proxy.descended_nonterminal - proxy.integrated, (
        f"released {proxy.removed} paths for "
        f"{proxy.descended_nonterminal - proxy.integrated} un-integrated leaves "
        f"({proxy.integrated} already had their vloss removed by "
        "walker_integrate_leaf); a double release steals a concurrent walker's "
        "in-flight vloss and the end-state total cannot see it"
    )
    total, nonzero = _total_vloss(tree)
    assert (total, nonzero) == (0, 0), f"leaked {total} over {nonzero} nodes"


class _InplaceEvaluator:
    """The ``get_input_buffer``/``evaluate_inplace_async`` shape PucvChunker uses."""

    def __init__(self, ok_calls: int, gather: int) -> None:
        self.ok_calls = ok_calls
        self.calls = 0
        self._bufs = [
            np.zeros((gather, PLANES, 8, 8), dtype=np.float32) for _ in range(2)
        ]
        self.n_slots = 2

    def get_input_buffer(self, n: int, *, slot: int = 0) -> np.ndarray:
        return self._bufs[slot][:n]

    def evaluate_inplace_async(
        self, n: int, *, slot: int = 0, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        assert 0 <= slot < self.n_slots
        assert relations is None
        self.calls += 1
        if self.calls > self.ok_calls:
            raise TimeoutError("inference broker timed out after 30.000s")
        rng = np.random.default_rng(self.calls)
        return (
            rng.normal(size=(n, 4672)).astype(np.float32),
            rng.normal(size=(n, 3)).astype(np.float32),
            None,
        )

    def evaluate_inplace(
        self, n: int, *, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        pol, wdl, _ = self.evaluate_inplace_async(n, slot=slot)
        return pol, wdl


def _run_chunker(*, ok_calls: int, gather: int, sims: int) -> tuple[int, int, bool]:
    tree, root_cb, rid = _fresh_tree()
    chunker = PucvChunker(
        _InplaceEvaluator(ok_calls, gather),
        gather=gather, c_puct=1.5, fpu_at_root=0.0, fpu_reduction=0.33,
        vloss_weight=3, input_planes=PLANES,
    )
    raised = False
    try:
        chunker.run(tree, rid, root_cb, sims)
    except Exception:
        raised = True
    total, nonzero = _total_vloss(tree)
    return total, nonzero, raised


def test_pucv_chunker_control_clean_run_ends_at_zero_vloss() -> None:
    total, nonzero, raised = _run_chunker(ok_calls=10_000, gather=8, sims=128)
    assert not raised
    assert (total, nonzero) == (0, 0)


def test_pucv_chunker_releases_vloss_when_the_evaluator_raises() -> None:
    """The identical unguarded shape the audit called PLAUSIBLE but never ran.

    ``batch_descend_puct`` applies vloss for the whole batch and
    ``batch_integrate_leaves`` removes it; the pipelined chunker can have TWO
    batches outstanding when the evaluator dies (the one just descended and the
    one awaiting its result), so both must be released.
    """
    total, nonzero, raised = _run_chunker(ok_calls=2, gather=8, sims=128)
    assert raised
    assert (total, nonzero) == (0, 0), f"leaked {total} over {nonzero} nodes"


class _FlakyInplaceEvaluator(_InplaceEvaluator):
    """Same shape as ``_InplaceEvaluator`` but at 146 planes for the pool cfg."""

    def __init__(self, ok_calls: int, gather: int, planes: int) -> None:
        super().__init__(ok_calls, gather)
        self._bufs = [
            np.zeros((gather, planes, 8, 8), dtype=np.float32) for _ in range(2)
        ]


def test_multi_gpu_pucv_pool_releases_vloss_when_the_evaluator_raises() -> None:
    """The third site with the identical shape.

    ``MultiGpuPucvPool`` has a standing test that vloss is fully unwound after a
    SUCCESSFUL run; it had none for a failed one, which is the case where the
    unwinding was never performed at all.
    """
    from chess_anti_engine.uci.multi_gpu_pucv_pool import (
        MultiGpuPucvConfig,
        MultiGpuPucvPool,
    )

    planes = input_plane_count("v1")
    tree, root_cb, rid = _fresh_tree()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(
            n_gpus=1, gather=8, vloss_weight=3, input_planes=planes,
        ),
        evaluators=[_FlakyInplaceEvaluator(2, 8, planes)],
    )
    try:
        with pytest.raises(TimeoutError):
            pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
                     target_sims=128, stop_event=threading.Event())
    finally:
        pool.close()
    total, nonzero = _total_vloss(tree)
    assert (total, nonzero) == (0, 0), f"leaked {total} over {nonzero} nodes"


def test_the_release_is_observable_when_it_fires(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fix only ever runs while an exception is unwinding, so without a log
    line there is no trace that it took effect on the production path.

    Paired with its negative control: a clean run must stay silent, or the line
    is noise rather than a signal.
    """
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.mcts.vloss"):
        _run_walkers(ok_calls=10_000, walkers=4, gather=4, sims=200)
    assert not [r for r in caplog.records if "released virtual loss" in r.message], (
        "a healthy search must not log a release"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.mcts.vloss"):
        _run_walkers(ok_calls=2, walkers=4, gather=8, sims=400)
    hits = [r for r in caplog.records if "released virtual loss" in r.message]
    assert hits, "the release fired but left no observable trace"
    assert "B7" in hits[0].message
