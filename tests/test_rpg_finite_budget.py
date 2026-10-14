"""Finite RPG chunks respect their advertised simulation budget."""
from __future__ import annotations

import threading

import chess
import numpy as np
import pytest
from numpy.typing import NDArray

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci.search import SearchWorker


class _Evaluator:
    n_slots = 2

    def __init__(self) -> None:
        self._buffers = [np.zeros((512, 146, 8, 8), dtype=np.float32) for _ in range(2)]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> NDArray[np.float32]:
        return self._buffers[slot][:bsz]

    def evaluate_inplace_async(self, bsz: int, *, slot: int = 0, relations: object = None) -> tuple[NDArray[np.float32], NDArray[np.float32], None]:
        assert slot in (0, 1)
        assert relations is None
        return np.zeros((bsz, 4672), dtype=np.float32), np.zeros((bsz, 3), dtype=np.float32), None


    def evaluate_encoded(self, x: NDArray[np.float32], relations: object = None) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert relations is None
        pol, wdl, _ = self.evaluate_inplace_async(len(x))
        return pol, wdl


@pytest.mark.parametrize(('topk', 'target'), [(32, 65), (16, 65), (16, 33), (32, 1), (32, 3), (16, 2048)])
@pytest.mark.parametrize('auto_schedule', [True, False])
def test_rpg_finite_chunk_budget(topk: int, target: int, auto_schedule: bool) -> None:
    cfg = GumbelConfig(input_extra_features='v1', topk=topk, add_noise=False, temperature=0.0, simulations=max(target, 65))
    worker = SearchWorker(_Evaluator(), device='cpu', gumbel_cfg=cfg, chunk_sims=max(target, 65))
    knob = -1 if auto_schedule else 0
    try:
        worker.install_root_parallel_gumbel([_Evaluator(), _Evaluator()], gather=512, open_vpa=knob, min_vpa=knob, min_keep=knob)
        worker._root_pol_logits = np.zeros((1, 4672), dtype=np.float32)
        worker._root_wdl_logits = np.zeros((1, 3), dtype=np.float32)
        worker._ensure_rpg_root_prepared(chess.Board())
        _, completed = worker._run_rpg_chunk(target, threading.Event())
        stats = worker.last_root_parallel_gumbel_stats()
        assert stats is not None
        assert completed == target
        assert sum(stats.group_sims) == target
        assert sum(phase.sims_completed for phase in stats.phases) == target
        assert worker._tree is not None
        assert worker._root_id is not None
        _, visits = worker._tree.get_children_visits(worker._root_id)
        assert int(visits.sum()) == target
    finally:
        worker.close()
