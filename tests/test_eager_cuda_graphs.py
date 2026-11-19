"""``DirectGPUEvaluator(cuda_graphs=True)`` must return the eager forward's bits.

The graphs replay the evaluator's own eager forward, so every output must be
bitwise identical to a graphs-off evaluator on the same rows -- on the sync
path, the async path, both slots, repeated shapes (replay, not capture) and an
interleaving that would expose shared-pool reuse. Off CUDA (or for a compiled
model) the flag is refused rather than silently dropped.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.model import ModelConfig, build_model

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")


def _model(device: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(ModelConfig(embed_dim=32, num_layers=2, num_heads=2, use_smolgen=True)).to(device).eval()


def _rows(rng: np.random.Generator, n: int, planes: int) -> np.ndarray:
    return (rng.random((n, planes, 8, 8)) < 0.15).astype(np.float32)


def test_refused_off_cuda() -> None:
    model = _model("cpu")
    with pytest.raises(ValueError, match=r"cuda_graphs=True is unsupported"):
        DirectGPUEvaluator(model, device="cpu", max_batch=8, cuda_graphs=True)
    assert DirectGPUEvaluator(model, device="cpu", max_batch=8).cuda_graphs is False


@needs_cuda
def test_refused_for_compiled_model() -> None:
    model = torch.compile(_model("cuda"))
    with pytest.raises(ValueError, match=r"torch\.compile"):
        DirectGPUEvaluator(model, device="cuda", max_batch=8, cuda_graphs=True)  # pyright: ignore[reportArgumentType]


@needs_cuda
def test_graph_outputs_are_bitwise_eager_on_every_path() -> None:
    model = _model("cuda")
    eager = DirectGPUEvaluator(model, device="cuda", max_batch=64, n_slots=2, legal_bf16=False)
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=64, n_slots=2, legal_bf16=False,
                               cuda_graphs=True)
    assert graph.cuda_graphs
    planes = eager.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(0)
    # Repeated sizes exercise replay after capture; the async/sync interleave
    # and the slot alternation would expose an output overwritten by another
    # graph's replay from the shared pool.
    for step, bsz in enumerate([16, 3, 16, 64, 3, 16, 33, 64]):
        x = _rows(rng, bsz, planes)
        slot = step % 2
        outs = []
        for ev in (eager, graph):
            ev.get_input_buffer(bsz, slot=slot)[:] = x
            if step % 3 == 0:
                pol, wdl = ev.evaluate_inplace(bsz, slot=slot)
            else:
                pol_t, wdl_t, event = ev.evaluate_inplace_async(bsz, slot=slot)
                assert event is not None
                event.synchronize()
                pol, wdl = pol_t.numpy().copy(), wdl_t.numpy().copy()
            outs.append((pol, wdl))
        (pe, we), (pg, wg) = outs
        assert pe.tobytes() == pg.tobytes(), f"policy differs at step {step} bsz {bsz}"
        assert we.tobytes() == wg.tobytes(), f"wdl differs at step {step} bsz {bsz}"
    assert graph.cuda_graph_shapes == (3, 16, 33, 64)
    assert eager.cuda_graph_shapes == ()


@needs_cuda
def test_back_to_back_async_slots_keep_their_own_outputs() -> None:
    # Two async submits in flight before either is read: the pipelined arena
    # pattern. Each slot's pinned outputs must hold its own rows' results.
    model = _model("cuda")
    eager = DirectGPUEvaluator(model, device="cuda", max_batch=32, n_slots=2, legal_bf16=False)
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=32, n_slots=2, legal_bf16=False,
                               cuda_graphs=True)
    planes = eager.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(1)
    xa, xb = _rows(rng, 32, planes), _rows(rng, 16, planes)
    results = []
    for ev in (eager, graph):
        ev.get_input_buffer(32, slot=0)[:] = xa
        ra = ev.evaluate_inplace_async(32, slot=0)
        ev.get_input_buffer(16, slot=1)[:] = xb
        rb = ev.evaluate_inplace_async(16, slot=1)
        for receipt in (ra, rb):
            assert receipt[2] is not None
            receipt[2].synchronize()
        results.append([(r[0].numpy().copy(), r[1].numpy().copy()) for r in (ra, rb)])
    for (pe, we), (pg, wg) in zip(results[0], results[1]):
        assert pe.tobytes() == pg.tobytes()
        assert we.tobytes() == wg.tobytes()
