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
        DirectGPUEvaluator(model, device="cuda", max_batch=8, legal_bf16=False, cuda_graphs=True)  # pyright: ignore[reportArgumentType]


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


@needs_cuda
def test_sync_forward_issued_while_an_async_one_is_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both paths copy their rows into the SAME static graph input and replay
    # graphs from one pool, so the sync path must wait for an async replay
    # still queued on the side stream. Made deterministic by delaying the
    # replay issued on the async stream (after its H2D copy): without the wait
    # the sync call overwrites the static input before the async replay reads
    # it. The sync call runs on a caller-owned stream, the case the explicit
    # wait exists for. Measured limitation: on the test host (torch 2.10,
    # CUDA 12.8, WSL2) the sync call is ordered behind the delayed async
    # replay even with the wait removed (it blocks for the injected ~70 ms
    # sleep on the legacy default stream and on a caller stream alike), so
    # this test checks the outputs under the race but does not by itself
    # kill a removed wait_stream there.
    model = _model("cuda")
    eager = DirectGPUEvaluator(model, device="cuda", max_batch=64, n_slots=2, legal_bf16=False)
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=64, n_slots=2, legal_bf16=False,
                               cuda_graphs=True)
    planes = eager.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(2)
    xa, xb = _rows(rng, 32, planes), _rows(rng, 16, planes)
    caller_stream = torch.cuda.Stream()

    def race(ev: DirectGPUEvaluator) -> tuple[np.ndarray, ...]:
        ev.get_input_buffer(32, slot=0)[:] = xa
        ev.get_input_buffer(16, slot=1)[:] = xb
        pol_t, wdl_t, event = ev.evaluate_inplace_async(32, slot=0)
        with torch.cuda.stream(caller_stream):
            pol_b, wdl_b = ev.evaluate_inplace(16, slot=1)
        assert event is not None
        event.synchronize()
        return pol_t.numpy().copy(), wdl_t.numpy().copy(), pol_b, wdl_b

    want = race(eager)
    race(graph)  # capture both shapes first
    orig_replay = torch.cuda.CUDAGraph.replay
    orig_async = graph.evaluate_inplace_async
    in_async = [False]
    delays = [0]

    def delayed_replay(self: torch.cuda.CUDAGraph) -> None:
        if in_async[0]:  # enqueued on the async stream, after its H2D copy
            torch.cuda._sleep(200_000_000)  # ~0.1 s of GPU spin before the replay
            delays[0] += 1
        orig_replay(self)

    def flagged_async(*a: object, **kw: object) -> object:
        in_async[0] = True
        try:
            return orig_async(*a, **kw)  # pyright: ignore[reportArgumentType]
        finally:
            in_async[0] = False

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", delayed_replay)
    monkeypatch.setattr(graph, "evaluate_inplace_async", flagged_async)
    got = race(graph)
    assert delays[0] == 1
    for e, g in zip(want, got):
        assert e.tobytes() == g.tobytes()


@needs_cuda
def test_refused_with_compact_legal_bf16_and_for_compiled_submodules() -> None:
    model = _model("cuda")
    with pytest.raises(ValueError, match="legal_bf16=True"):
        DirectGPUEvaluator(model, device="cuda", max_batch=8, cuda_graphs=True)  # legal_bf16 default
    sub = next(m for m in model.modules() if m is not model and any(True for _ in m.parameters()))
    sub.compile()
    with pytest.raises(ValueError, match=r"torch\.compile"):
        DirectGPUEvaluator(model, device="cuda", max_batch=8, legal_bf16=False, cuda_graphs=True)


@needs_cuda
def test_stats_count_replays_and_eager_fallbacks() -> None:
    model = _model("cuda")
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=64, legal_bf16=False,
                               cuda_graphs=True, cuda_graph_max_rows=16,
                               forward_chunk_rows=64)  # above the graph cap: the eager fallback
    planes = graph.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(4)
    for bsz in (8, 8, 32):
        graph.get_input_buffer(bsz)[:] = _rows(rng, bsz, planes)
        graph.evaluate_inplace(bsz)
    stats = graph.cuda_graph_stats
    assert stats is not None
    assert (stats["captures"], stats["replays"], stats["eager_over_max_rows"]) == (1, 2, 1)
    assert stats["eager_relations"] == 0
    assert DirectGPUEvaluator(model, device="cuda", max_batch=8).cuda_graph_stats is None


@needs_cuda
def test_model_swap_reallocation_and_inference_only_toggle_recapture() -> None:
    # Graphs read the parameter storages and control flow seen at capture; a
    # replaced model, reallocated parameters or a flipped forward flag must
    # not replay the old binding.
    model = _model("cuda")
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=16, legal_bf16=False, cuda_graphs=True)
    planes = graph.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(5), 8, planes)

    def eager_out(m: torch.nn.Module) -> tuple[bytes, bytes]:
        ev = DirectGPUEvaluator(m, device="cuda", max_batch=16, legal_bf16=False)
        ev.get_input_buffer(8)[:] = x
        pol, wdl = ev.evaluate_inplace(8)
        return pol.tobytes(), wdl.tobytes()

    def graph_out() -> tuple[bytes, bytes]:
        graph.get_input_buffer(8)[:] = x
        pol, wdl = graph.evaluate_inplace(8)
        return pol.tobytes(), wdl.tobytes()

    assert graph_out() == eager_out(model)
    torch.manual_seed(1)
    other = build_model(ModelConfig(embed_dim=32, num_layers=2, num_heads=2, use_smolgen=True)).cuda().eval()
    graph.model = other
    assert graph_out() == eager_out(other)
    with torch.no_grad():
        for prm in other.parameters():
            prm.data = prm.data * 1.5  # new storages
    assert graph_out() == eager_out(other)
    with torch.no_grad():
        for prm in other.parameters():
            prm.mul_(0.5)  # in place: same storages, replay sees the new values
    assert graph_out() == eager_out(other)
    setattr(other, "_inference_only", True)  # a plain attribute on a Module
    assert graph_out() == eager_out(other)
    stats = graph.cuda_graph_stats
    assert stats is not None
    assert stats["recaptures"] == 3


@needs_cuda
def test_batches_above_the_row_cap_run_eager_and_are_not_captured() -> None:
    model = _model("cuda")
    eager = DirectGPUEvaluator(model, device="cuda", max_batch=64, legal_bf16=False)
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=64, legal_bf16=False,
                               cuda_graphs=True, cuda_graph_max_rows=16,
                               forward_chunk_rows=64)  # above the graph cap: the eager fallback
    planes = eager.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(3)
    for bsz in (8, 16, 17, 64, 8):
        x = _rows(rng, bsz, planes)
        outs = []
        for ev in (eager, graph):
            ev.get_input_buffer(bsz)[:] = x
            outs.append(ev.evaluate_inplace(bsz))
        (pe, we), (pg, wg) = outs
        assert pe.tobytes() == pg.tobytes(), bsz
        assert we.tobytes() == wg.tobytes(), bsz
    assert graph.cuda_graph_shapes == (8, 16)


@needs_cuda
def test_sync_path_waits_on_the_async_stream_while_graphs_are_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Structural pin for the wait the race test above cannot observe on a host
    # that serialises cross-stream copies: both paths write the SAME static
    # graph input, so a sync forward must order itself after the async stream.
    model = _model("cuda")
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=16, n_slots=2, legal_bf16=False,
                               cuda_graphs=True)
    planes = graph.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(6), 8, planes)
    graph.get_input_buffer(8, slot=0)[:] = x
    _p, _w, event = graph.evaluate_inplace_async(8, slot=0)  # creates the async stream
    assert event is not None
    event.synchronize()
    calls: list[tuple[object, object]] = []
    orig = torch.cuda.Stream.wait_stream

    def spy(self: torch.cuda.Stream, other: torch.cuda.Stream) -> None:
        calls.append((self, other))
        orig(self, other)

    monkeypatch.setattr(torch.cuda.Stream, "wait_stream", spy)
    graph.get_input_buffer(8, slot=1)[:] = x
    graph.evaluate_inplace(8, slot=1)
    assert any(other == graph._stream and me == torch.cuda.current_stream() for me, other in calls), calls


@needs_cuda
def test_relations_and_graph_count_fallbacks_are_counted_and_equal_eager() -> None:
    model = _model("cuda")
    eager = DirectGPUEvaluator(model, device="cuda", max_batch=16, legal_bf16=False)
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=16, legal_bf16=False, cuda_graphs=True)
    assert graph._eager_graphs is not None
    graph._eager_graphs._max_graphs = 1  # one shape graphed, every other shape eager
    planes = eager.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(7)
    rel = np.zeros((8, 5, 64, 64), dtype=np.uint8)
    rel[:, 0, np.arange(64), np.arange(64)] = 1
    for bsz, relations in ((8, None), (4, None), (8, rel), (8, None)):
        x = _rows(rng, bsz, planes)
        outs = []
        for ev in (eager, graph):
            ev.get_input_buffer(bsz)[:] = x
            outs.append(ev.evaluate_inplace(bsz, relations=relations))
        (pe, we), (pg, wg) = outs
        assert pe.tobytes() == pg.tobytes()
        assert we.tobytes() == wg.tobytes()
    stats = graph.cuda_graph_stats
    assert stats is not None
    assert (stats["captures"], stats["replays"]) == (1, 2)
    assert stats["eager_over_max_graphs"] == 1
    assert stats["eager_relations"] == 1


@needs_cuda
def test_parameter_replacement_is_caught_by_the_registration_hook_and_backstop() -> None:
    # The binding caches the parameter list: a registered replacement
    # (``m.weight = Parameter``) must invalidate it at once, and a write that
    # bypasses registration (``m._parameters[...] = ...``) within
    # RETRAVERSE_EVERY calls.
    model = _model("cuda")
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=16, legal_bf16=False, cuda_graphs=True)
    assert graph._eager_graphs is not None
    planes = graph.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(8), 8, planes)

    def outs(ev: DirectGPUEvaluator) -> bytes:
        ev.get_input_buffer(8)[:] = x
        pol, wdl = ev.evaluate_inplace(8)
        return pol.tobytes() + wdl.tobytes()

    def eager_now() -> bytes:
        return outs(DirectGPUEvaluator(model, device="cuda", max_batch=16, legal_bf16=False))

    outs(graph)
    linear = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    linear.weight = torch.nn.Parameter(linear.weight.detach() * 2.0)
    assert outs(graph) == eager_now()  # caught on the very next forward
    assert graph.cuda_graph_stats is not None
    assert graph.cuda_graph_stats["recaptures"] == 1
    linear._parameters["weight"] = torch.nn.Parameter(linear.weight.detach() * 0.5)  # bypasses hooks
    every = graph._eager_graphs.RETRAVERSE_EVERY
    want = eager_now()
    got = [outs(graph) for _ in range(every)]
    assert got[-1] == want  # caught within RETRAVERSE_EVERY calls
    assert graph.cuda_graph_stats["recaptures"] == 2
