"""Audit W3: root-parallel Gumbel pinned virtual-mean pending accounting.

``RootParallelGumbelConfig.vloss_mode`` defaulted to 1 (virtual-mean) and was
deliberately not fed from the UCI ``PUCVPendingMode`` option, so an operator
reading ``legacy`` off that option was reading a knob RPG never consulted.

Under virtual-mean a pending, zero-visit child scores ``q_parent == parent_Q``
exactly — no pessimism at all — and its prior is counted into
``visited_policy``, which raises the FPU penalty on its untouched siblings
(``_mcts_tree.c`` ``tree_select_child``). Both effects steer the next descent
back onto the leaf already in flight, so the search re-walks the same
positions. ``nodes``/``nps`` are simulation counts and do not move when this
happens, which is why it survived: the regime is invisible from the engine's
output.

These tests assert the *descent*, not the config: the mode is read off the
live ``PucvChunker`` each group runs, and the exploration test counts distinct
tree nodes for an identical simulation budget and an identical number of
evaluator rows.
"""
from __future__ import annotations

import threading
from typing import Any

import chess
import numpy as np

from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci.root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
)
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline
from tests.test_root_parallel_gumbel import (
    _DeterministicStubEvaluator,
    _make_evaluator,
    _root_eval,
)

# RPG's own production shape (RootParallelGumbelConfig defaults), which is
# where the 2.6x exploration gap was measured.
_SIMS = 1024
_GATHER = 512


def _pool(**cfg_kwargs: Any) -> tuple[RootParallelGumbelPool, list[Any]]:
    evaluators = [
        _DeterministicStubEvaluator(max_batch=_GATHER) for _ in range(2)
    ]
    cfg = RootParallelGumbelConfig(
        n_groups=2, gather=_GATHER, split_idle_groups=False, **cfg_kwargs,
    )
    gcfg = GumbelConfig(
        input_extra_features="v1", simulations=_SIMS, topk=16,
        add_noise=True, gumbel_scale=1.0, temperature=0.0,
    )
    pool = RootParallelGumbelPool(
        cfg, gcfg, evaluators=evaluators, rng=np.random.default_rng(7),
    )
    return pool, evaluators


def _search(pool: RootParallelGumbelPool) -> tuple[int, Any]:
    """Run one full chunk; return (distinct tree nodes, pool stats)."""
    pol, wdl = _root_eval()
    tree = MCTSTree()
    tree.reserve(50_000, 500_000)
    pool.prepare_root(
        tree=tree, board=chess.Board(), pol_logits=pol, wdl_logits=wdl,
    )
    pool.run(target_sims=_SIMS, stop_event=threading.Event())
    return int(tree.node_count()), pool.last_stats()


def test_rpg_default_descends_with_legacy_pending_accounting() -> None:
    """The default reaches the descent as legacy, read off the live chunkers."""
    pool, _evs = _pool()
    try:
        assert pool.realized_vloss_mode() == (0, 0)
    finally:
        pool.close()


def test_virtual_mean_collapses_explored_positions() -> None:
    """Same sims, same evaluator rows, far fewer distinct positions.

    This is the defect itself rather than a proxy for it: if the two modes
    ever stop differing here, the fix has stopped mattering and the default
    flip below is measuring nothing.
    """
    legacy_pool, legacy_evs = _pool(vloss_mode=0)
    try:
        legacy_nodes, legacy_stats = _search(legacy_pool)
    finally:
        legacy_pool.close()
    vmean_pool, vmean_evs = _pool(vloss_mode=1)
    try:
        vmean_nodes, vmean_stats = _search(vmean_pool)
    finally:
        vmean_pool.close()

    legacy_rows = sum(int(e.rows) for e in legacy_evs)
    vmean_rows = sum(int(e.rows) for e in vmean_evs)
    assert legacy_rows == vmean_rows == _SIMS, (
        "equal-compute premise broken: the modes submitted different row counts"
    )
    assert sum(p.sims_completed for p in legacy_stats.phases) == sum(
        p.sims_completed for p in vmean_stats.phases
    )
    assert legacy_nodes > 1.5 * vmean_nodes, (
        f"legacy explored {legacy_nodes} distinct nodes, virtual-mean "
        f"{vmean_nodes} — the W3 regime is not reproducing"
    )


def test_default_config_lands_on_the_exploring_side() -> None:
    """The shipped default must explore like legacy, not like virtual-mean.

    Anchored to a measured value rather than to ``vloss_mode == 0`` so that
    re-pinning virtual-mean by another route (a config edit, a pool that
    rebuilds its chunkers from a stale field) still fails.
    """
    vmean_pool, _ = _pool(vloss_mode=1)
    try:
        vmean_nodes, _ = _search(vmean_pool)
    finally:
        vmean_pool.close()
    default_pool, _ = _pool()
    try:
        default_nodes, stats = _search(default_pool)
    finally:
        default_pool.close()
    assert default_nodes > 1.5 * vmean_nodes
    assert stats.vloss_mode == (0, 0)
    assert stats.tree_nodes == default_nodes


def test_set_vloss_mode_changes_the_descent_not_just_the_config() -> None:
    """Flipping the mode on a live pool must move the search, not a field.

    The chunkers are built once per group thread at construction, so a setter
    that only wrote ``_cfg`` would leave every descent on the old regime while
    every readout said otherwise. Asserted on explored positions so it holds
    even if the readout is later re-sourced.
    """
    pool, _evs = _pool()
    try:
        legacy_nodes, _ = _search(pool)
        pool.set_vloss_mode(1)
        vmean_nodes, stats = _search(pool)
        assert stats.vloss_mode == (1, 1)
        assert legacy_nodes > 1.5 * vmean_nodes, (
            f"set_vloss_mode(1) left the descent at {vmean_nodes} vs "
            f"{legacy_nodes} distinct nodes — the flip did not reach it"
        )
    finally:
        pool.close()


def test_pucv_pending_mode_reaches_the_rpg_descent() -> None:
    """``PUCVPendingMode`` is no longer a decoy for the Gumbel pool."""
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", simulations=64,
            add_noise=False, temperature=0.0,
        ),
        chunk_sims=64, n_walkers=1,
    )
    try:
        worker.install_root_parallel_gumbel(
            [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
            gather=8, as_factories=False,
        )
        assert worker.realized_rpg_vloss_mode() == (0, 0)
        worker.set_pucv_vloss_mode(1)
        assert worker.realized_rpg_vloss_mode() == (1, 1)
        worker.set_pucv_vloss_mode(0)
        assert worker.realized_rpg_vloss_mode() == (0, 0)
        # A reinstall must carry the option, not fall back to the dataclass.
        worker.set_pucv_vloss_mode(1)
        worker.install_root_parallel_gumbel(
            [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
            gather=8, as_factories=False,
        )
        assert worker.realized_rpg_vloss_mode() == (1, 1)
    finally:
        worker.close()


def test_rpg_search_reports_realized_mode_and_distinct_nodes() -> None:
    """The stats a UCI search reads carry the regime it actually ran."""
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", simulations=64,
            add_noise=False, temperature=0.0,
        ),
        chunk_sims=64, n_walkers=1,
    )
    try:
        worker.install_root_parallel_gumbel(
            [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
            gather=8, as_factories=False,
        )
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(10_000), max_nodes=128,
        )
        assert result.bestmove_uci
        stats = worker.last_root_parallel_gumbel_stats()
        assert stats is not None
        assert stats.vloss_mode == (0, 0)
        # `nodes` counts simulations; tree_nodes counts positions. They are
        # different numbers and the search must publish both.
        assert stats.tree_nodes > 0
        assert worker._tree is not None
        assert stats.tree_nodes == int(worker._tree.node_count())
    finally:
        worker.close()
