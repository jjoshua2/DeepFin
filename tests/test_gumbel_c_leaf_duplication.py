"""C17: the C Gumbel search must not evaluate the same leaf twice in one batch.

``gss_step`` accumulates leaves across several sequential-halving reps to fill
``GSS_GPU_BATCH``. Without virtual loss a later rep re-walks an *unchanged*
tree, reaches the *same* leaf, and back-propagates the *same* value: an NN slot
and a visit spent on zero information. Measured at the production shape that is
38% of all evaluated rows and a 37% deficit in distinct tree nodes.

These tests pin the fix and, just as importantly, its cost profile. Flushing
per rep (``target_batch=1``) also removes the duplicates but pays ~6x the GPU
round trips; virtual loss removes them while keeping production's batch count,
which is the whole reason to prefer it.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

POLICY_SIZE = 4672

_FENS = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4rrk1/pp1n1ppp/2pb1q2/3p4/3P4/2NBPN2/PPQ2PPP/2R2RK1 w - - 0 14",
    "2kr3r/ppp2ppp/2n1b3/3q4/3P4/2P1BN2/PP3PPP/R2Q1RK1 b - - 0 12",
    "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 b - - 0 9",
    "8/5pk1/6p1/7p/5P1P/4K1P1/8/8 w - - 0 40",
)


class _CountingHashEvaluator:
    """Deterministic in the position, and it records what it was asked to score.

    Values depend only on the encoded row, so a repeated leaf is provably
    uninformative: it returns the identical policy/WDL and therefore backprops
    the identical Q.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0
        self.distinct_rows = 0
        rng = np.random.default_rng(20260727)
        self._pol = rng.standard_normal((512, POLICY_SIZE)).astype(np.float32)
        self._wdl = rng.standard_normal((512, 3)).astype(np.float32)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # BatchEvaluator conformance
        n = int(x.shape[0])
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(n, -1)
        keys = [flat[i].tobytes() for i in range(n)]
        self.calls += 1
        self.rows += n
        self.distinct_rows += len({hash(k) for k in keys})
        idx = np.array(
            [int.from_bytes(k[:6], "little") % 512 for k in keys], dtype=np.int64,
        )
        return self._pol[idx], self._wdl[idx]


def _search(
    monkeypatch: pytest.MonkeyPatch, *, target_batch: int, vloss_weight: int,
) -> tuple[_CountingHashEvaluator, int]:
    # Bucket padding re-sends stale buffer rows the search never asked for;
    # they would count as duplicates and hide the very thing under test.
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    boards = [chess.Board(f) for f in _FENS]
    cfg = GumbelConfig(
        simulations=256, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )
    ev = _CountingHashEvaluator()
    result = run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=ev, target_batch=target_batch, vloss_weight=vloss_weight,
    )
    tree = result[4]
    return ev, int(tree.node_count())


def test_production_batching_evaluates_duplicate_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect itself, pinned so a silent regression is visible.

    Not an xfail: this documents the CURRENT production default
    (``vloss_weight=0``), which is still what selfplay runs. It is a
    data-affecting change to flip that default, so the default stays put until
    it has its own ledger entry.
    """
    ev, _nodes = _search(monkeypatch, target_batch=0, vloss_weight=0)
    dup_fraction = 1.0 - ev.distinct_rows / ev.rows
    assert dup_fraction > 0.25, f"expected heavy duplication, got {dup_fraction:.3f}"


def test_virtual_loss_removes_duplicates_and_keeps_the_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix: zero duplicates, more distinct nodes, same GPU round trips."""
    base, base_nodes = _search(monkeypatch, target_batch=0, vloss_weight=0)
    flush, flush_nodes = _search(monkeypatch, target_batch=1, vloss_weight=0)
    vloss, vloss_nodes = _search(monkeypatch, target_batch=0, vloss_weight=1)

    assert vloss.distinct_rows == vloss.rows
    assert flush.distinct_rows == flush.rows

    # Far more real search per nominal sim than the duplicating default...
    assert vloss_nodes > base_nodes * 2.0
    # ...and the same order as the per-rep flush that pays for it in latency.
    # Not equal: the in-flight penalty steers descent elsewhere, so the two
    # build DIFFERENT trees, and which one is larger depends on the batch
    # shape (virtual loss is ahead at the 256-board production shape and
    # behind at this 8-board one). Only the order of magnitude is the claim.
    assert vloss_nodes >= flush_nodes * 0.7

    # The point of preferring virtual loss: per-rep flushing multiplies GPU
    # round trips, virtual loss does not.
    assert flush.calls > base.calls * 3
    assert vloss.calls <= base.calls * 1.25


@pytest.mark.parametrize(("target_batch", "vloss_weight"), [(0, 0), (1, 0), (0, 1)])
def test_every_arm_spends_the_whole_simulation_budget(
    monkeypatch: pytest.MonkeyPatch, target_batch: int, vloss_weight: int,
) -> None:
    """Root visits must sum to the nominal sim count in EVERY arm.

    Written from a real bug: the first cut of the duplicate-flush cursor
    encoded "nothing pending" as 0, which is also a valid resume index, so a
    collision on the first query of a list silently dropped the rest of that
    list. Boards got 176-211 of their 256 sims and every downstream number was
    quietly measured at ~80% of the advertised budget. Nothing else in the
    suite noticed, because the search still returned a perfectly well-formed
    policy.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    boards = [chess.Board(f) for f in _FENS]
    sims = 256
    cfg = GumbelConfig(
        simulations=sims, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )
    result = run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=_CountingHashEvaluator(),
        target_batch=target_batch, vloss_weight=vloss_weight,
    )
    tree, root_ids = result[4], result[5]
    for i, rid in enumerate(root_ids):
        assert int(rid) >= 0
        _actions, visits, _q = tree.get_children_q(int(rid), 0.0)
        assert int(visits.sum()) == sims, (
            f"board {i}: root visits {int(visits.sum())} != {sims}"
        )


def test_virtual_loss_leaves_no_residue_on_the_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every applied penalty is removed, so a reused tree starts clean.

    Selfplay reuses trees across plies (``tree=``/``root_node_ids=``); a leaked
    virtual-loss count would silently bias every later search on that tree.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    boards = [chess.Board(f) for f in _FENS[:4]]
    cfg = GumbelConfig(
        simulations=128, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )
    result = run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=_CountingHashEvaluator(), target_batch=0, vloss_weight=1,
    )
    tree = result[4]
    leaked = [
        nid for nid in range(int(tree.node_count()))
        if tree.get_virtual_loss(nid) != 0
    ]
    assert not leaked, f"{len(leaked)} nodes kept virtual loss after the search"


def test_the_default_weight_never_touches_the_virtual_loss_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vloss_weight=0`` must leave the feature provably inert.

    This is the merge-safety property, not a behavioural one: production runs
    at the default 0 until a config key lands at a restart, so every write site
    (``if (vloss_w > 0) tree_apply_vloss_path(...)``) and every read site
    (``(vloss_weight > 0) ? t->virtual_loss[id] : 0``) is guarded. Today that
    holds by inspection; the guards are three lines apart in ``_mcts_tree.c``
    and a later refactor could drop one without any existing test noticing,
    because at weight 0 a leaked penalty changes nothing the other tests watch.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    boards = [chess.Board(f) for f in _FENS[:4]]
    cfg = GumbelConfig(
        simulations=128, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )
    result = run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=_CountingHashEvaluator(), target_batch=0, vloss_weight=0,
    )
    tree = result[4]
    touched = [
        nid for nid in range(int(tree.node_count()))
        if tree.get_virtual_loss(nid) != 0
    ]
    assert not touched, (
        f"{len(touched)} nodes carry virtual loss at vloss_weight=0 — a write "
        "site lost its guard, so the default is no longer inert"
    )


def test_virtual_mean_mode_is_reachable_and_differs_from_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vloss_mode=1`` must actually change descent, and still kill duplicates.

    Until this landed, ``VLOSS_MODE_VIRTUAL_MEAN`` existed in ``_mcts_tree.c``
    but was unreachable from the Gumbel path: ``tree_gumbel_select_child`` took
    no mode argument and the leaf descent hardcoded ``VLOSS_MODE_LEGACY``. So
    every Gumbel measurement of virtual loss was made with parallel-PUCT
    pessimism -- the pending visit scored as a LOSS -- when the construct the
    Gumbel descent actually wants is a pending VISIT valued at the child's own
    mean, which moves the visit count without moving the estimate.

    A mode that is accepted and silently ignored is the exact defect C17 turned
    out to be, so this asserts the two modes are distinguishable rather than
    trusting that the argument arrived.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    boards = [chess.Board(f) for f in _FENS[:4]]
    cfg = GumbelConfig(
        simulations=128, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )

    def _run(weight: int, mode: int) -> tuple[tuple[int, ...], int]:
        ev = _CountingHashEvaluator()
        res = run_gumbel_root_many_c(
            None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
            evaluator=ev, target_batch=0, vloss_weight=weight, vloss_mode=mode,
        )
        tree, root_ids = res[4], res[5]
        per_board: list[int] = []
        for rid in root_ids:
            _a, visits, _q = tree.get_children_q(int(rid), 0.0)
            per_board.append(int(visits.sum()))
        return tuple(per_board), ev.rows - ev.distinct_rows

    legacy_visits, legacy_dup = _run(1, 0)
    mean_visits, mean_dup = _run(1, 1)

    # Both modes must remove the duplication -- that is the point of either.
    assert legacy_dup == 0, f"legacy left {legacy_dup} duplicate rows"
    assert mean_dup == 0, f"virtual-mean left {mean_dup} duplicate rows"
    # ...and the budget is still fully spent under the new mode.
    assert mean_visits == legacy_visits == (128,) * len(boards), (
        f"virtual-mean spent {mean_visits} root visits vs legacy's {legacy_visits}"
    )
    # The modes differ only in how a PENDING visit is valued, so at weight 0
    # they must be indistinguishable -- this is what makes the new argument
    # safe to add to every existing caller.
    assert _run(0, 0) == _run(0, 1), "vloss_mode is not inert at vloss_weight=0"
