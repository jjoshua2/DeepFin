"""The C Gumbel search must evaluate every leaf when its encode buffer fills.

Before ABI 11, ``gss_prepare_batch`` (``_mcts_tree.c``) backed a leaf that found
the buffer full with the ROOT's Q as a ``SOLVED_UNKNOWN`` pseudo-terminal.
``tree_backprop`` reads that value from the LEAF's side to move, so every
starved leaf at an odd depth got the wrong sign and every one at an even depth
a fabricated value. The buffer is capped at the evaluator's max batch, so any
caller whose cap sat below boards x topk lost real evaluations (2026-11-20: the
simple-RL smoke collect at cap 512, 128 boards, topk 16 absorbed ~31% of leaf
visits). It now flushes the batch and resumes at the leaf that did not fit,
and counts each flush.

Driven on CPU through the real C search with synthetic evaluators. The cap is
imposed through ``search_leaf_row_cap`` (the copying path's buffer cap), which
sizes the C buffer exactly as the in-place path's ``_max_batch`` does.
"""
from __future__ import annotations

import dataclasses

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import buffer_full_flushes, run_gumbel_root_many_c
from chess_anti_engine.moves import POLICY_SIZE

# Quiet middlegames: no mate or stalemate within the few plies 64 sims reach.
_FENS = (
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
    "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6",
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5",
)


class _Evaluator:
    """Constant (draw) values and flat policy, or position-hashed values.

    Counts the leaf rows it is asked to evaluate. ``cap`` (when set) is the
    copying path's leaf-buffer cap.
    """

    supports_legal_bf16 = False

    def __init__(self, *, cap: int | None, hashed: bool) -> None:
        self.search_leaf_row_cap = cap
        self.hashed = hashed
        self.rows: list[int] = []
        rng = np.random.default_rng(11)
        self._pol = (rng.standard_normal((997, POLICY_SIZE)) * 2.0).astype(np.float32)
        self._wdl = rng.standard_normal((997, 3)).astype(np.float32)

    def evaluate_encoded(self, x, relations=None):
        del relations
        n = int(x.shape[0])
        self.rows.append(n)
        if not self.hashed:
            return np.zeros((n, POLICY_SIZE), np.float32), np.zeros((n, 3), np.float32)
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(n, -1)
        idx = np.array([int.from_bytes(flat[i].tobytes()[:6], "little") % 997 for i in range(n)])
        return self._pol[idx], self._wdl[idx]


def _search(ev: _Evaluator, *, root_wdl: np.ndarray | None = None, sims: int = 64, topk: int = 16):
    gumbel_c_mod._COMPILED_BATCH_BUCKETS = ()
    boards = [chess.Board(f) for f in _FENS]
    cfg = dataclasses.replace(GumbelConfig(simulations=sims, topk=topk, temperature=0.0, add_noise=False),
                              full_tree=True)
    pre_pol = pre_wdl = None
    if root_wdl is not None:
        pre_pol = np.zeros((len(boards), POLICY_SIZE), np.float32)
        pre_wdl = np.repeat(root_wdl[None, :], len(boards), axis=0).astype(np.float32)
    tree = MCTSTree()
    out = run_gumbel_root_many_c(None, boards, device="cpu", rng=np.random.default_rng(3), cfg=cfg, evaluator=ev,
                                 target_batch=0, vloss_weight=1, tree=tree, root_node_ids=None,
                                 pre_pol_logits=pre_pol, pre_wdl_logits=pre_wdl)
    return out, tree


def test_a_starved_leaf_is_evaluated_not_backed_up_with_the_roots_q() -> None:
    """Every leaf the net evaluates is a draw (Q 0); the root is told it is lost
    (Q ~ -0.99). Any backed-up value other than 0 can only be the root's Q
    leaking into the tree through the buffer-full path. Fails on the parent:
    with 4 boards x topk 16 = 64 leaves per rep against an 8-row buffer, most
    root children carry +-0.99 there."""
    lost = np.array([-8.0, -8.0, 8.0], np.float32)  # W, D, L logits: root Q ~ -1
    before = buffer_full_flushes()
    (_probs, _acts, _vals, _masks, tree, root_ids), _ = _search(_Evaluator(cap=8, hashed=False), root_wdl=lost)
    assert buffer_full_flushes() > before  # the cap really bound
    for rid in root_ids:
        _actions, visits, qs = tree.get_children_q(int(rid), 0.0)
        visited = np.asarray(visits) > 0
        assert visited.sum() >= 2
        np.testing.assert_array_equal(np.asarray(qs)[visited], 0.0)


def test_the_counter_stays_zero_when_the_cap_does_not_bind() -> None:
    before = buffer_full_flushes()
    _search(_Evaluator(cap=None, hashed=True))
    assert buffer_full_flushes() == before
    tree = MCTSTree()
    assert tree.gumbel_buffer_full_flushes() == 0


def test_a_capped_search_evaluates_as_many_leaves_as_an_uncapped_one() -> None:
    """Flush-and-resume keeps every simulation an evaluation. Fails on the parent,
    which evaluated only ~buffer-many leaves per rep and absorbed the rest."""
    free = _Evaluator(cap=None, hashed=True)
    (_p, _a, _v, _m, free_tree, free_ids), _ = _search(free)
    capped = _Evaluator(cap=8, hashed=True)
    before = buffer_full_flushes()
    (_p, _a, _v, _m, cap_tree, cap_ids), _ = _search(capped)
    flushes = buffer_full_flushes() - before
    assert flushes > 0
    assert max(capped.rows[1:]) <= 8  # no batch above the cap reached the evaluator
    free_leaves, cap_leaves = sum(free.rows[1:]), sum(capped.rows[1:])  # rows[0] is the root batch
    assert cap_leaves >= 0.9 * free_leaves, (cap_leaves, free_leaves)
    for t, ids in ((free_tree, free_ids), (cap_tree, cap_ids)):
        for rid in ids:
            _a, visits, _q = t.get_children_q(int(rid), 0.0)
            assert int(np.asarray(visits).sum()) == 64  # sequential-halving budget, both runs


def test_the_process_counter_is_the_sum_of_the_per_search_tree_counters() -> None:
    """The consumer reads the C tree's own per-search counter (reset by
    start_gumbel_sims) and banks it; the two must agree."""
    before = buffer_full_flushes()
    out, _ = _search(_Evaluator(cap=8, hashed=True))
    tree = out[4]
    assert tree.gumbel_buffer_full_flushes() > 0
    assert buffer_full_flushes() - before == tree.gumbel_buffer_full_flushes()


def test_an_empty_encode_buffer_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        _search(_Evaluator(cap=0, hashed=True))


# --- consumers bank the observation -----------------------------------------


def test_the_arena_record_carries_the_flushes_its_searches_took(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from scripts import arena_standard as arena
    from tests.test_arena_evaluator_hoist import _run, _stub_model_loader

    _stub_model_loader(monkeypatch)

    def _loop(flushes: int):
        def _play(_c, _r, openings, **_kw):
            gumbel_c_mod._BUFFER_FULL_FLUSHES += flushes  # what a capped search banks
            return [1.0] * len(openings)
        return _play

    for flushes in (0, 3):
        for name in ("play_paired_games_matched_sims", "play_paired_games_matched_sims_rolling"):
            monkeypatch.setattr(arena, name, _loop(flushes))
        sub = tmp_path / str(flushes)
        sub.mkdir()
        record = _run(sub)
        assert record["buffer_full_flushes"] == flushes


def test_a_collect_with_buffer_full_flushes_is_recorded_and_not_sealed(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _argv, _setup

    openings, model_config = _setup(tmp_path, monkeypatch)
    argv = _argv(tmp_path / "run", openings, model_config)
    argv[argv.index("--generations") + 1] = "1"
    real = gumbel_c_mod.buffer_full_flushes
    calls = {"n": 0}

    def _bumped() -> int:  # the second read (after the generation) sees 4 more flushes
        calls["n"] += 1
        return real() + (4 if calls["n"] >= 2 else 0)

    monkeypatch.setattr(gumbel_c_mod, "buffer_full_flushes", _bumped)
    with pytest.raises(RuntimeError, match="4 buffer-full flush"):
        simple_rl_loop.main(argv)
    collect = tmp_path / "run" / "gen_0001" / "collect"
    assert json.loads((collect / "stats.json").read_text())["buffer_full_flushes"] == 4
    assert not (collect / "manifest.json").exists()  # not sealed: redone whole

    monkeypatch.setattr(gumbel_c_mod, "buffer_full_flushes", real)
    # Earlier searches in the process (other collects, arenas) must not count.
    monkeypatch.setattr(gumbel_c_mod, "_BUFFER_FULL_FLUSHES", gumbel_c_mod._BUFFER_FULL_FLUSHES + 7)
    assert simple_rl_loop.main(argv) == 0
    assert json.loads((collect / "stats.json").read_text())["buffer_full_flushes"] == 0


def test_the_arena_bound_is_read_off_the_built_evaluators(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A builder that realises a smaller cap than requested must flip the
    recorded bound: the search reads the object's _max_batch, not the flag."""
    from scripts import arena_standard as arena
    from tests.test_arena_evaluator_hoist import _run, _stub_model_loader, _stub_play_loops

    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)

    class _Small:
        _max_batch = 64
        n_slots = 2

    monkeypatch.setattr(arena, "build_arena_evaluator", lambda *_a, **_kw: _Small())
    record = _run(tmp_path, device="cuda", eval_max_batch=4096)
    assert record["eval_max_batch_realised"] == 64
    assert record["eval_leaf_cap_bound"] is True
