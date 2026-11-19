"""`--walkers N` / `SearchWalkers > 1` must not default to a starved GPU.

With a per-walker leaf gather of 1 every walker submitted one or two leaves per
evaluator call: on a GPU two walkers ran ~450 nps against classic's ~13,400, so
a documented option silently played ~30x weaker. The default is now a real
batch; the classic path (one walker) never reads it, which the identity check
below demonstrates on a fixed-node search.
"""
from __future__ import annotations

import threading

import chess
import numpy as np
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.__main__ import _build_parser
from chess_anti_engine.uci.engine import DEFAULT_LEAF_GATHER, EngineOptions
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


def test_walker_gather_defaults_to_a_real_batch() -> None:
    assert DEFAULT_LEAF_GATHER >= 16
    assert EngineOptions().leaf_gather == DEFAULT_LEAF_GATHER
    assert _build_parser().parse_args(["--checkpoint", "x"]).walker_gather == DEFAULT_LEAF_GATHER


def _fixed_node_search(walker_gather: int) -> tuple[str, int, tuple[str, ...], list[int]]:
    torch.manual_seed(1234)
    torch_model = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    torch_model.eval()
    evaluator = ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        torch_model, device="cpu", max_batch=128, use_amp=False, n_slots=2,
    ))
    worker = SearchWorker(
        evaluator, device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=128, add_noise=False),
        chunk_sims=128, n_walkers=1, walker_gather=walker_gather,
    )
    try:
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(60_000), max_nodes=384,
        )
        assert worker._tree is not None
        assert worker._root_id is not None
        _actions, visits = worker._tree.get_children_visits(worker._root_id)
        return result.bestmove_uci, result.nodes, result.pv, np.asarray(visits).tolist()
    finally:
        worker.close()


def test_leaf_gather_is_inert_on_the_single_walker_path() -> None:
    assert _fixed_node_search(1) == _fixed_node_search(DEFAULT_LEAF_GATHER)
