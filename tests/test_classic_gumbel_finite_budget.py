from __future__ import annotations

import threading

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel as python_gumbel
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


class NeutralEvaluator:
    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        policy = np.broadcast_to(np.linspace(-1.0, 1.0, POLICY_SIZE, dtype=np.float32),
                                 (len(x), POLICY_SIZE)).copy()
        return policy, np.zeros((len(x), 3), np.float32)


@pytest.mark.parametrize('budget', [1, 8, 9, 18, 19, 21, 25, 33, 100])
def test_uci_classic_budget_caps_actual_root_child_visits(budget: int) -> None:
    worker = SearchWorker(NeutralEvaluator(), device='cpu', n_walkers=1, chunk_sims=512,
                          gumbel_cfg=GumbelConfig(topk=20, add_noise=False, temperature=0.0))
    try:
        result = worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None),
                            max_nodes=budget)
        assert worker._tree is not None
        assert worker._root_id is not None
        _, visits = worker._tree.get_children_visits(worker._root_id)
        assert int(visits.sum()) == result.nodes == budget
    finally:
        worker.close()


@pytest.mark.parametrize('halving_div', [2, 4])
@pytest.mark.parametrize(('target_batch', 'vloss_weight'), [(1, 0), (512, 0), (512, 1)])
def test_native_per_game_budgets_and_reused_roots(
    halving_div: int, target_batch: int, vloss_weight: int,
) -> None:
    budgets = [1, 9, 18, 33]
    boards = [chess.Board() for _ in budgets]
    cfg = GumbelConfig(topk=20, add_noise=False, temperature=0.0, halving_div=halving_div)
    result = run_gumbel_root_many_c(
        None, boards, device='cpu', rng=np.random.default_rng(31), cfg=cfg,
        evaluator=NeutralEvaluator(), per_game_simulations=budgets,
        target_batch=target_batch, vloss_weight=vloss_weight,
    )
    tree, roots = result[4:6]
    before = []
    for i, budget in enumerate(budgets):
        _, visits = tree.get_children_visits(roots[i])
        before.append(int(visits.sum()))
        assert before[-1] == budget
        assert result[3][i][result[1][i]]
        assert np.isfinite(result[0][i]).all()
        assert float(result[0][i].sum()) == pytest.approx(1.0)
    resumed = run_gumbel_root_many_c(
        None, boards, device='cpu', rng=np.random.default_rng(32), cfg=cfg,
        evaluator=NeutralEvaluator(), per_game_simulations=budgets,
        tree=tree, root_node_ids=roots, target_batch=target_batch, vloss_weight=vloss_weight,
    )
    for i, budget in enumerate(budgets):
        _, visits = resumed[4].get_children_visits(resumed[5][i])
        assert int(visits.sum()) - before[i] == budget


@pytest.mark.parametrize('halving_div', [2, 4])
@pytest.mark.parametrize('temperature', [0.0, 0.7])
def test_python_public_search_caps_each_game_budget(
    monkeypatch: pytest.MonkeyPatch, halving_div: int, temperature: float,
) -> None:
    budgets = [1, 9, 18, 33]
    visits: list[int] = []
    original = python_gumbel._build_improved_policy_for_board

    def observe(st: python_gumbel._BoardSearchState, *, root_q: float,
                cfg: GumbelConfig, rng: np.random.Generator) -> tuple[np.ndarray, int, float]:
        visits.append(sum(child.N for child in st.root.children.values()))
        return original(st, root_q=root_q, cfg=cfg, rng=rng)

    monkeypatch.setattr(python_gumbel, '_build_improved_policy_for_board', observe)
    result = python_gumbel.run_gumbel_root_many(
        None, [chess.Board() for _ in budgets], device='cpu', rng=np.random.default_rng(31),
        cfg=GumbelConfig(topk=20, add_noise=False, temperature=temperature, halving_div=halving_div),
        evaluator=NeutralEvaluator(), per_game_simulations=budgets,
    )
    assert visits == budgets
    for probs, action, mask in zip(result[0], result[1], result[3], strict=True):
        assert mask[action]
        assert np.isfinite(probs).all()
        assert float(probs.sum()) == pytest.approx(1.0)
