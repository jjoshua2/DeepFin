"""RPG temperature reaches native priors without changing plain PUCT or raw cache."""
from __future__ import annotations

import contextlib
import itertools
import threading
from collections.abc import Generator

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.puct_vl import PucvChunker
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import COMPACT_TO_FULL_POLICY
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.root_parallel_gumbel import _CandidateArena, _WorkItem
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


class _PolicyEvaluator:
    n_slots = 2

    def __init__(self, *, compact: bool = False) -> None:
        self.calls = 0
        self.full = (np.arange(POLICY_SIZE, dtype=np.float32) % 7) / 3.0
        self.raw = self.full[COMPACT_TO_FULL_POLICY].copy() if compact else self.full.copy()
        self.buffers = [np.zeros((4, 146, 8, 8), dtype=np.float32) for _ in range(2)]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self.buffers[slot][:bsz]

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        self.calls += 1
        return np.tile(self.raw, (len(x), 1)), np.zeros((len(x), 3), dtype=np.float32)

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        pol, wdl = self.evaluate_encoded(self.buffers[slot][:bsz])
        return pol, wdl, None


@contextlib.contextmanager
def _worker(
    temperature: float = 1.0, *, compact: bool = False, cache: bool = False,
) -> Generator[tuple[SearchWorker, list[_PolicyEvaluator]], None, None]:
    evaluator = _PolicyEvaluator(compact=compact)
    worker = SearchWorker(
        evaluator, device="cpu", chunk_sims=8,
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", policy_temp=temperature,
            add_noise=False, temperature=0.0,
        ),
        eval_cache_entries=8 if cache else 0,
    )
    groups = [_PolicyEvaluator(compact=compact) for _ in range(2)]
    try:
        yield worker, groups
    finally:
        if worker._walker_pool is not None:
            worker._walker_pool.close()
        worker.close()


def _prepare(worker: SearchWorker) -> None:
    worker._root_pol_logits = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    worker._root_wdl_logits = np.zeros((1, 3), dtype=np.float32)
    worker._ensure_rpg_root_prepared(chess.Board())


def _touch(
    worker: SearchWorker, groups: list[_PolicyEvaluator], budget: int,
) -> tuple[MCTSTree, _CandidateArena, PucvChunker]:
    pool = worker._rpg_pool
    assert pool is not None
    assert pool._state is not None
    st = pool._state
    arena = pool._arena_for(st, int(st.search_legal[0]))
    chunker = pool._chunkers[0]
    assert chunker is not None
    completed = pool._run_item(
        0, groups[0], chunker, _WorkItem(arena, budget, 0), threading.Event(),
    )
    assert completed == budget
    return st.tree, arena, chunker


def _assert_native_priors(
    tree: MCTSTree, node: int, board: CBoard, full_logits: np.ndarray, temperature: float,
) -> None:
    # Zero-value, zero-FPU descents allocate visits in proportion to the actual
    # native edge priors. Do not expand the leaves, so deeper policy cannot
    # confound the observation. Release each applied VL before the backprop.
    enc = np.empty((1, 146, 8, 8), dtype=np.float32)
    for _ in range(2048):
        _, path, _, terminal = tree.walker_descend_puct(
            node, board, 1.4, 0.0, 0.0, 3, enc,
        )
        assert terminal is None
        tree.remove_vloss_path(path)
        tree.backprop(path, 0.0)
    actions, visits = tree.get_children_visits(node)
    logits = full_logits[actions].astype(np.float64) / temperature
    assert float(logits.max() - logits.min()) > 0.1
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    observed = visits / visits.sum()
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.002)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


def _leaf_board(
    tree: MCTSTree, arena: _CandidateArena, chunker: PucvChunker,
) -> tuple[int, CBoard]:
    b = chunker._bufs[0]
    path = b["path_buf"][:int(b["path_lens"][0])]
    assert int(path[0]) == arena.cid
    assert isinstance(arena.cboard, CBoard)
    cb = arena.cboard.copy()
    for parent, child in itertools.pairwise(path):
        actions, _ = tree.get_children_visits(int(parent))
        action = next(int(a) for a in actions if tree.find_child(int(parent), int(a)) == int(child))
        cb.push_index(action)
    return int(path[-1]), cb


def _set_temperature(engine: Engine, temperature: float) -> None:
    command = parse_command(f"setoption name PolicyTemperature value {temperature}")
    assert command is not None
    engine.dispatch(command)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 5.0])
def test_rpg_candidate_temperature_reaches_native_priors(compact: bool, temperature: float) -> None:
    with _worker(temperature, compact=compact) as (worker, groups):
        worker.install_root_parallel_gumbel(groups, gather=2)
        _prepare(worker)
        tree, arena, _ = _touch(worker, groups, 1)
        _assert_native_priors(tree, arena.cid, arena.cboard, groups[0].full, temperature)
        np.testing.assert_array_equal(groups[0].raw, groups[0].full[COMPACT_TO_FULL_POLICY] if compact else groups[0].full)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 5.0])
def test_rpg_deeper_temperature_reaches_native_priors(
    compact: bool, cache: bool, temperature: float,
) -> None:
    with _worker(temperature, compact=compact, cache=cache) as (worker, groups):
        worker.install_root_parallel_gumbel(groups, gather=2)
        _prepare(worker)
        tree, arena, chunker = _touch(worker, groups, 2)
        leaf, cb = _leaf_board(tree, arena, chunker)
        _assert_native_priors(tree, leaf, cb, groups[0].full, temperature)
        if chunker._cache is not None:
            assert chunker.cache_stats() is not None
            for _, policy, _ in chunker._cache._cache.values():
                np.testing.assert_array_equal(policy, groups[0].raw)


@pytest.mark.parametrize("compact", [False, True])
def test_live_rpg_temperature_reuses_raw_cached_outputs(compact: bool) -> None:
    with _worker(compact=compact, cache=True) as (worker, groups):
        worker.install_root_parallel_gumbel(groups, gather=2)
        engine = Engine(worker, options=EngineOptions(policy_temp=1.0))
        _prepare(worker)
        old_tree, _, chunker = _touch(worker, groups, 2)
        raw_cache = chunker._cache
        assert raw_cache is not None
        calls = groups[0].calls
        cached_rows = [value[1].copy() for value in raw_cache._cache.values()]
        for temperature in (2.0, 0.5):
            _set_temperature(engine, temperature)
            assert worker._tree is None
            assert worker._rpg_pool is not None
            assert worker._rpg_pool._chunkers[0] is chunker
            _prepare(worker)
            tree, arena, current = _touch(worker, groups, 2)
            assert tree is not old_tree
            assert current is chunker
            leaf, cb = _leaf_board(tree, arena, chunker)
            _assert_native_priors(tree, leaf, cb, groups[0].full, temperature)
            # Only candidate expansion needs an eval; the deeper row is a hit.
            assert groups[0].calls == calls + 1
            calls = groups[0].calls
            stats = raw_cache.stats()
            assert stats.hits > 0
            assert stats.stores == 1
            for before, value in zip(cached_rows, raw_cache._cache.values(), strict=True):
                np.testing.assert_array_equal(value[1], before)
        tree_before_same_value = worker._tree
        _set_temperature(engine, 0.5)
        assert worker._tree is tree_before_same_value


def test_live_classic_temperature_invalidates_root() -> None:
    with _worker() as (worker, _):
        engine = Engine(worker, options=EngineOptions(policy_temp=1.0))
        worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=8)
        assert worker._tree is not None
        _set_temperature(engine, 2.0)
        assert worker._tree is None
        assert worker._root_pol_logits is None
        worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=8)
        root = worker._tree
        assert root is not None
        _set_temperature(engine, 2.0)
        assert worker._tree is root


@pytest.mark.parametrize("path", ["walker", "pucv", "pucv_pool"])
def test_policy_temperature_keeps_plain_puct_inert(path: str, capsys: pytest.CaptureFixture[str]) -> None:
    with _worker() as (worker, groups):
        if path == "walker":
            worker.set_num_threads(2)
        elif path == "pucv":
            worker.set_use_pucv(True, gather=2)
        else:
            worker.install_multi_gpu_pucv(groups, gather=2)
        engine = Engine(worker, options=EngineOptions(policy_temp=1.0))
        worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=8)
        root = worker._tree
        assert root is not None
        assert worker.realized_search_path() == path
        root_id = worker._root_id
        assert root_id is not None
        signature = root.get_children_visits(root_id)
        _set_temperature(engine, 2.0)
        assert "NO EFFECT" in capsys.readouterr().out
        assert worker._tree is root
        current = root.get_children_visits(root_id)
        np.testing.assert_array_equal(current[0], signature[0])
        np.testing.assert_array_equal(current[1], signature[1])
