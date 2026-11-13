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
from chess_anti_engine.uci.search import SearchResult, SearchWorker
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


def test_ponder_and_infinite_compact_only_under_the_cost_cap(monkeypatch) -> None:
    """Ponder (before ponderhit) and `go infinite` (`stop_sensitive`) have no
    budget, and `stop` cannot interrupt the C compaction: a cheap one still
    runs, an expensive one falls back to keep / rebuild."""
    w = _worker()
    _rerooted(w, hash_fraction=0.6)
    assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "compacted"

    _rerooted(w, hash_fraction=0.6)
    footprint = _tree(w).footprint_bytes()
    # Price this arena just above the cap.
    per_gib = 1.01 * uci_search._STOP_SENSITIVE_COMPACTION_MAX_MS * _GIB / footprint
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", per_gib)
    tree = w._tree
    assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "kept"
    assert w._tree is tree

    _rerooted(w, hash_fraction=0.9)
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e12)
    assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "rebuilt"
    assert w._tree is None


def test_go_nodes_keeps_while_the_move_fits_and_ignores_the_cost_cap(monkeypatch) -> None:
    """`go nodes N` has no clock: keep in place while F + N*b fits in Hash
    (what main did below H/2), compact otherwise however long it takes."""
    w = _worker()
    _rerooted(w, hash_fraction=0.3)
    room = w._max_tree_bytes - _tree(w).footprint_bytes()
    fits = int(0.5 * room / w._recent_bytes_per_visit)
    assert w._resolve_pending_compaction(Deadline(None), None, max_nodes=fits) == "kept"

    _rerooted(w, hash_fraction=0.3)
    room = w._max_tree_bytes - _tree(w).footprint_bytes()
    too_many = int(2.0 * room / w._recent_bytes_per_visit)
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e12)  # cost is irrelevant here
    assert w._resolve_pending_compaction(Deadline(None), None, max_nodes=too_many) == "compacted"


def test_go_depth_keeps_below_half_hash_and_compacts_above() -> None:
    w = _worker()
    _rerooted(w, hash_fraction=0.4)
    assert w._resolve_pending_compaction(Deadline(None), None) == "kept"
    _rerooted(w, hash_fraction=0.6)
    assert w._resolve_pending_compaction(Deadline(None), None) == "compacted"


def test_growth_rate_is_dropped_on_any_reset_and_on_a_hash_change() -> None:
    """ucinewgame, evaluator swaps and every topology change go through
    `reset_tree`; a Hash change does not, so `set_max_tree_mb` drops it too."""
    w = _worker()
    _rerooted(w, hash_fraction=0.3)
    assert w._recent_visits_per_ms > 0.0
    w.reset_tree()
    assert w._recent_visits_per_ms == 0.0
    assert w._recent_bytes_per_visit == 0.0

    _rerooted(w, hash_fraction=0.3)
    w._max_tree_bytes = 64 * 1024 * 1024
    w.set_max_tree_mb(64)  # unchanged: the rate stays
    assert w._recent_visits_per_ms > 0.0
    w.set_max_tree_mb(65)
    assert w._recent_visits_per_ms == 0.0

    _rerooted(w, hash_fraction=0.3)
    w.set_evaluator(_SpreadEvaluator())
    assert w._recent_visits_per_ms == 0.0

    _rerooted(w, hash_fraction=0.3)
    w.set_num_threads(2)
    assert w._recent_visits_per_ms == 0.0
    w.set_num_threads(1)


def test_ucinewgame_drops_the_growth_rate() -> None:
    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.uci.protocol import parse_command

    w = _worker()
    engine = Engine(w)  # construction applies the Hash option (a cap change)
    try:
        _rerooted(w, hash_fraction=0.3)
        assert w._recent_visits_per_ms > 0.0
        cmd = parse_command("ucinewgame")
        assert cmd is not None
        engine.dispatch(cmd)
        engine._wait_for_search()
    finally:
        engine.close()
    assert w._recent_visits_per_ms == 0.0


def test_carried_search_is_capped_at_the_budget(monkeypatch) -> None:
    """Review: an under-measured rate inflated carried = N_k/v so keep won
    with a full arena (eff 13043 vs rebuild 1500) and the search stopped on
    tree_bytes after 301 ms. Carried search counts at most the budget B, so a
    full arena cannot outbid a rebuild that can use the whole hard deadline."""
    w = _worker()
    _rerooted(w, hash_fraction=0.97)
    rate = w._recent_visits_per_ms * w._recent_bytes_per_visit
    optimum = int(w._max_tree_bytes / rate / 3)  # a rebuild fills Hash at ~3x the optimum
    hard = 3 * optimum
    # A kept subtree "worth" 10 optimums of search at the measured rate.
    monkeypatch.setattr(w, "_kept_subtree_visits",
                        lambda: int(10 * optimum * w._recent_visits_per_ms))
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 1e12)
    assert w._resolve_pending_compaction(Deadline(hard), optimum) == "rebuilt"
    assert w.last_reuse_inputs["carried_ms"] == optimum


def test_room_is_projected_to_the_hard_deadline_not_the_optimum() -> None:
    """Review: a clocked search ran ~2.95 s against a 1.5 s optimum, so room
    projected over the optimum chose keep at 232/256 MB and the search
    stopped on tree_bytes at 1172 ms. Room is projected to the hard deadline."""
    w = _worker()
    _rerooted(w, hash_fraction=0.5)
    room = w._max_tree_bytes - _tree(w).footprint_bytes()
    rate = w._recent_visits_per_ms * w._recent_bytes_per_visit
    optimum = int(room / rate / 1.5)  # the room outlasts the optimum ...
    hard = 3 * optimum  # ... but not the hard deadline
    assert w._resolve_pending_compaction(Deadline(hard), optimum) != "kept"
    assert w.last_reuse_inputs["horizon_ms"] >= 0.9 * hard
    assert w.last_reuse_inputs["budget_ms"] == optimum


def test_compaction_is_charged_against_the_optimum(monkeypatch) -> None:
    """A compaction costing more than the move's optimum is never chosen, even
    when the hard deadline leaves time for it."""
    w = _worker()
    _rerooted(w, hash_fraction=0.97)
    rate = w._recent_visits_per_ms * w._recent_bytes_per_visit
    optimum = max(10, int(w._max_tree_bytes / rate))
    footprint = _tree(w).footprint_bytes()
    monkeypatch.setattr(uci_search, "_COMPACTION_MS_PER_GIB", 2.0 * optimum * _GIB / footprint)
    assert w._resolve_pending_compaction(Deadline(1000 * optimum), optimum) == "rebuilt"


def test_bytes_per_visit_is_nodes_added_per_visit_times_footprint_per_node() -> None:
    w = _worker()
    board = chess.Board()
    w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=2048)
    nodes_before = _tree(w).node_count()
    result = w.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=2048)
    node_count = _tree(w).node_count()
    assert nodes_before > 0  # the tree was reused, so footprint/visits differs
    expected = (node_count - nodes_before) / result.nodes * _tree(w).footprint_bytes() / node_count
    assert w._recent_bytes_per_visit == pytest.approx(expected, rel=1e-12)
    assert w._recent_bytes_per_visit != pytest.approx(
        _tree(w).footprint_bytes() / result.nodes, rel=1e-3,
    )


def test_compaction_transient_rules_out_compaction_near_the_cap(monkeypatch) -> None:
    w = _worker()
    _rerooted(w, hash_fraction=0.6)
    tree = _tree(w)
    stats = tree.memory_stats()
    footprint = float(tree.footprint_bytes())
    kept = footprint / 2
    assert w._compaction_transient_bytes(footprint, kept) == pytest.approx(
        8.0 * stats["child_count"] * 0.5 + 4.0 * stats["cb_pool_count"],
    )
    # With no slack beyond Hash, a compaction whose transient would cross it is
    # not chosen, even a cheap one; the band fallback keeps the tree.
    monkeypatch.setattr(uci_search, "_COMPACTION_TRANSIENT_MARGIN_BYTES", 0)
    w._max_tree_bytes = int(footprint)  # arena at the cap
    assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "rebuilt"
    assert w.last_reuse_inputs["can_compact"] == 0.0


def test_compaction_keeps_a_subtree_larger_than_half_hash() -> None:
    """The compaction limit is Hash, not Hash/2: a kept subtree between H/2
    and H is compacted, not refused."""
    probe = _worker()
    _rerooted(probe, hash_fraction=0.3)
    probe._max_tree_bytes = 1 << 40
    assert probe._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "compacted"
    kept = _tree(probe).footprint_bytes()

    w = _worker()  # same deterministic search
    _rerooted(w, hash_fraction=0.3)
    w._max_tree_bytes = int(1.5 * kept)  # kept subtree = 2/3 of Hash
    assert w._resolve_pending_compaction(Deadline(None), None, stop_sensitive=True) == "compacted"
    assert _tree(w).footprint_bytes() == kept


def test_a_rolled_back_setoption_does_not_resurrect_a_dropped_rate() -> None:
    """`search_state` / `restore_search_state` (#1261's setoption rollback)
    restore the Hash cap but not the measured rate or the reuse flag: the rate
    is re-measured by the next search instead."""
    w = _worker()
    _rerooted(w, hash_fraction=0.3)
    w._max_tree_bytes = 64 * 1024 * 1024
    state = w.search_state()
    assert "_recent_visits_per_ms" not in state[0]
    assert "_compaction_pending" not in state[0]
    w.set_max_tree_mb(128)
    assert w._recent_visits_per_ms == 0.0
    assert w.restore_search_state(state)
    assert w._max_tree_bytes == 64 * 1024 * 1024
    assert w._recent_visits_per_ms == 0.0
    assert w._compaction_pending  # the scheduled reuse decision is untouched


class _RecordingWorker(SearchWorker):
    def __init__(self) -> None:
        super().__init__(_SpreadEvaluator(), device="cpu", chunk_sims=64,
                         gumbel_cfg=GumbelConfig(simulations=64, topk=8, temperature=0.0,
                                                 add_noise=False))
        self.stop_sensitive: list[bool] = []
        self.entered = threading.Event()

    def run(self, board: chess.Board, **kwargs: object) -> SearchResult:  # type: ignore[override]
        self.stop_sensitive.append(bool(kwargs.get("stop_sensitive", False)))
        self.entered.set()
        return super().run(board, **kwargs)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("go", "expected"),
    [("go infinite", True), ("go ponder", True), ("go nodes 200", False),
     ("go depth 2", False), ("go wtime 60000 btime 60000", False)],
)
def test_engine_marks_only_ponder_and_infinite_stop_sensitive(go: str, expected: bool) -> None:
    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.uci.protocol import parse_command

    worker = _RecordingWorker()
    engine = Engine(worker)
    try:
        for line in ("position startpos moves e2e4", go):
            cmd = parse_command(line)
            assert cmd is not None
            engine.dispatch(cmd)
        assert worker.entered.wait(10)
        if expected:
            stop = parse_command("stop")
            assert stop is not None
            engine.dispatch(stop)
        engine._wait_for_search()
    finally:
        engine.close()
    assert worker.stop_sensitive[:1] == [expected]
