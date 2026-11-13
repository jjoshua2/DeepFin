"""Cross-move reuse: keep / compact / rebuild is chosen from projected growth.

A fixed footprint band (reuse in place below Hash/2, keep uncompacted below
3/4 Hash, compact only if it costs <= 5% of the move budget) let a reused tree
starve the next search: with H - F of room left it stopped on ``tree_bytes``
long before its optimum. The resolver now compares projected effective search
time for each option (see ``SearchWorker._resolve_pending_compaction``).
"""
from __future__ import annotations

import hashlib
import threading

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci import search as uci_search
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

_GIB = float(1 << 30)


class _SpreadEvaluator:
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


def _tree(w: SearchWorker) -> MCTSTree:
    assert w._tree is not None
    return w._tree


def _rerooted(w: SearchWorker, *, hash_fraction: float, visits: int = 4096) -> None:
    """Search startpos, size Hash so the arena sits at `hash_fraction` of it,
    and reroot onto the played move (the next run() would resolve it)."""
    board = chess.Board()
    first = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=visits)
    assert w._tree is not None
    w._max_tree_bytes = int(w._tree.footprint_bytes() / hash_fraction)
    assert w.advance_root(board, [chess.Move.from_uci(first.bestmove_uci)])
    assert w._compaction_pending


def test_search_records_the_growth_rate_the_rule_consumes() -> None:
    w = _worker()
    _rerooted(w, hash_fraction=0.3)
    v, b = w._recent_visits_per_ms, w._recent_bytes_per_visit
    assert v > 0.0
    # The tiny-net tree holds a few hundred bytes to a few KB per visit.
    assert 50.0 < b < 20_000.0, b
    w._resolve_pending_compaction(Deadline(60_000), 6_000)
    # The consumer used the producer's numbers (not a default or a stale copy).
    assert w.last_reuse_inputs["visits_per_ms"] == v
    assert w.last_reuse_inputs["bytes_per_visit"] == b
    assert "eff_kept_ms" in w.last_reuse_inputs


def test_a_move_that_fits_in_the_room_left_keeps_the_tree_for_free() -> None:
    w = _worker()
    _rerooted(w, hash_fraction=0.3)
    tree, rid = w._tree, w._root_id
    room = w._max_tree_bytes - _tree(w).footprint_bytes()
    rate = w._recent_visits_per_ms * w._recent_bytes_per_visit
    budget = int(0.5 * room / rate)  # projected growth is half the room
    assert budget > 0
    assert w._resolve_pending_compaction(Deadline(None), budget) == "kept"
    assert w._tree is tree
    assert w._root_id == rid


def test_reviewer_case_in_place_below_half_hash_now_compacts() -> None:
    """Review: Hash 64 MB, 6 s optimum, F=29 MB reused in place (below H/2) ->
    stopped on tree_bytes at 1.26 s. A move whose projected growth exceeds the
    room left must not simply reuse in place when compaction is cheap."""
    w = _worker()
    _rerooted(w, hash_fraction=29 / 64)
    room = w._max_tree_bytes - _tree(w).footprint_bytes()
    rate = w._recent_visits_per_ms * w._recent_bytes_per_visit
    budget = int(5.0 * room / rate)  # would stop on tree_bytes at ~1/5 of the budget
    arena_nodes = _tree(w).node_count()
    assert w._resolve_pending_compaction(Deadline(None), budget) == "compacted"
    assert w._root_id == 0
    assert w._tree is not None
    assert w._tree.node_count() < arena_nodes
    inputs = w.last_reuse_inputs
    assert inputs["eff_compacted_ms"] > inputs["eff_kept_ms"]


def test_reviewer_case_kept_band_with_expensive_compaction_rebuilds(monkeypatch) -> None:
    """Review: F=41 MB of 64 MB, compaction forced too expensive -> "kept",
    stopped at 0.85 s. Keeping only pays while the carried subtree is worth
    more than the room the discarded siblings hold; here it is not."""
    w = _worker()
    _rerooted(w, hash_fraction=41 / 64)
    room = w._max_tree_bytes - _tree(w).footprint_bytes()
    rate = w._recent_visits_per_ms * w._recent_bytes_per_visit
    budget = int(5.0 * room / rate)
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e12)
    assert w._resolve_pending_compaction(Deadline(None), budget) == "rebuilt"
    assert w._tree is None
    inputs = w.last_reuse_inputs
    assert inputs["eff_rebuilt_ms"] > inputs["eff_kept_ms"] >= inputs["eff_compacted_ms"]


@pytest.mark.parametrize(
    ("footprint_gib", "expected"),
    [(40.0, "kept"), (60.0, "compacted")],
)
def test_production_scale_hash_64gb_120s(monkeypatch, footprint_gib: float, expected: str) -> None:
    """Review: at Hash 64 GB with a 120 s optimum the old 5% rule never
    compacted (a 40 GB arena estimates 16 s > 6 s), so the arena alternated
    kept/rebuilt. With ~20 visits/ms at ~3.3 KB/visit and a carried subtree of
    10% of the arena: 40 GB keeps (the move fits in the 24 GB left); 60 GB
    compacts (4 GB of room would stop the move at ~65 s; compacting costs 24 s)."""
    w = _worker()
    _rerooted(w, hash_fraction=0.3)
    footprint = footprint_gib * _GIB
    w._max_tree_bytes = int(64 * _GIB)
    w._recent_visits_per_ms, w._recent_bytes_per_visit = 20.0, 3300.0
    monkeypatch.setattr(w, "_tree_footprint_bytes", lambda: int(footprint))
    monkeypatch.setattr(w, "_kept_subtree_visits", lambda: int(0.1 * footprint / 3300.0))
    assert w._resolve_pending_compaction(Deadline(None), 120_000) == expected
    assert w.last_reuse_inputs["est_ms"] == pytest.approx(footprint_gib * 400.0)


def test_no_rate_falls_back_to_the_budget_fraction_rule(monkeypatch) -> None:
    w = _worker()
    _rerooted(w, hash_fraction=0.6)
    w._recent_visits_per_ms = 0.0
    assert w._resolve_pending_compaction(Deadline(None), 60_000) == "compacted"
    _rerooted(w, hash_fraction=0.6)
    w._recent_visits_per_ms = 0.0
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e12)
    assert w._resolve_pending_compaction(Deadline(None), 60_000) == "kept"


def test_no_time_budget_compacts_only_under_the_cost_cap(monkeypatch) -> None:
    """Ponder (before ponderhit) and infinite have no budget, and `stop`
    cannot interrupt the C compaction: a cheap one still runs, an expensive one
    falls back to keep / rebuild instead of delaying `bestmove`."""
    w = _worker()
    _rerooted(w, hash_fraction=0.6)
    assert w._resolve_pending_compaction(Deadline(None), None) == "compacted"

    _rerooted(w, hash_fraction=0.6)
    footprint = _tree(w).footprint_bytes()
    # Price this arena just above the cap.
    per_gib = 1.01 * uci_search._NO_BUDGET_COMPACTION_MAX_MS * _GIB / footprint
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", per_gib)
    tree = w._tree
    assert w._resolve_pending_compaction(Deadline(None), None) == "kept"
    assert w._tree is tree

    _rerooted(w, hash_fraction=0.9)
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e12)
    assert w._resolve_pending_compaction(Deadline(None), None) == "rebuilt"
    assert w._tree is None
