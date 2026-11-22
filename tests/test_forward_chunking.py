"""``DirectGPUEvaluator(forward_chunk_rows=N)``: no network forward above N rows.

A call above N rows runs as consecutive forwards of at most N rows, in row
order, into one output (dense and compact-legal paths). A call of at most N
rows runs the unchunked code. With CUDA graphs on, N defaults to the graph row
cap, so every chunk replays a graph; without graphs the default is no bound.
Why: peak device memory follows the rows of the largest forward, and the
full-width collect's round 0 handed ~5,000 rows to one forward per process
(``mem_wide2_20261122``).
"""
from __future__ import annotations

from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.inference import DirectGPUEvaluator, forward_chunk_spans
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model import ModelConfig, build_model

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
N = 8


def _model(device: str = "cpu") -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(ModelConfig(input_extra_features="v2_threats", embed_dim=16, num_layers=1,
                                   num_heads=2, ffn_mult=2.0)).to(device).eval()


def _rows(rng: np.random.Generator, n: int, planes: int) -> np.ndarray:
    return (rng.random((n, planes, 8, 8)) < 0.15).astype(np.float32)


def _evaluate(ev: DirectGPUEvaluator, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ev.get_input_buffer(len(x))[:] = x
    return ev.evaluate_inplace(len(x))


def test_spans_cover_the_rows_in_order() -> None:
    spans = forward_chunk_spans(5 * N + 3, N)
    assert spans == tuple((i * N, min((i + 1) * N, 5 * N + 3)) for i in range(6))
    assert spans[-1] == (5 * N, 5 * N + 3)
    assert forward_chunk_spans(N, N) == ((0, N),)
    assert forward_chunk_spans(0, N) == ()
    with pytest.raises(ValueError, match="chunk_rows"):
        forward_chunk_spans(10, 0)


def test_default_is_no_bound_without_graphs_and_the_override_is_validated() -> None:
    model = _model()
    assert DirectGPUEvaluator(model, device="cpu", max_batch=64).forward_chunk_rows is None
    assert DirectGPUEvaluator(model, device="cpu", max_batch=64, forward_chunk_rows=N).forward_chunk_rows == N
    with pytest.raises(ValueError, match="forward_chunk_rows"):
        DirectGPUEvaluator(model, device="cpu", max_batch=64, forward_chunk_rows=0)


def test_a_chunked_call_equals_the_concatenation_of_single_chunk_calls() -> None:
    model = _model()
    rows = 5 * N + 3
    chunked = DirectGPUEvaluator(model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False)
    plain = DirectGPUEvaluator(model, device="cpu", max_batch=rows, legal_bf16=False)
    planes = plain.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(1), rows, planes)
    pol, wdl = _evaluate(chunked, x)
    assert chunked.forward_chunk_stats == {"chunked_forwards": 1, "forward_chunks": 6}
    parts = [_evaluate(plain, x[lo:hi]) for lo, hi in forward_chunk_spans(rows, N)]
    assert pol.tobytes() == np.concatenate([p for p, _ in parts]).tobytes()
    assert wdl.tobytes() == np.concatenate([w for _, w in parts]).tobytes()
    # Against ONE unchunked forward of all rows: equal up to batch-size effects
    # of the CPU kernels (measured, printed for the PR record).
    pol1, wdl1 = _evaluate(plain, x)
    d_pol, d_wdl = float(np.abs(pol - pol1).max()), float(np.abs(wdl - wdl1).max())
    print(f"cpu fp32 chunked vs unchunked: max|dpol|={d_pol:.3g} max|dwdl|={d_wdl:.3g}")
    assert d_pol < 1e-4
    assert d_wdl < 1e-4


def test_a_call_of_at_most_n_rows_is_not_chunked_and_is_bit_identical() -> None:
    model = _model()
    chunked = DirectGPUEvaluator(model, device="cpu", max_batch=64, forward_chunk_rows=N, legal_bf16=False)
    plain = DirectGPUEvaluator(model, device="cpu", max_batch=64, legal_bf16=False)
    planes = plain.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(2), N, planes)
    a, b = _evaluate(chunked, x), _evaluate(plain, x)
    assert chunked.forward_chunk_stats == {"chunked_forwards": 0, "forward_chunks": 0}
    assert a[0].tobytes() == b[0].tobytes()
    assert a[1].tobytes() == b[1].tobytes()


def test_legal_chunks_cut_the_compact_policy_at_the_cumulative_counts() -> None:
    """The compact-legal loop, run on CPU through the same method CUDA uses:
    a zero-count row on a chunk boundary and a chunk with no legal moves."""
    model = _model()
    rows = 3 * N + 2
    ev = DirectGPUEvaluator(model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False)
    planes = ev.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(3)
    x = torch.from_numpy(_rows(rng, rows, planes))
    counts = rng.integers(1, 6, size=rows).astype(np.int32)
    counts[N] = 0                 # first row of chunk 2: zero legal moves on the boundary
    counts[N - 1] = 0             # last row of chunk 1
    counts[2 * N:3 * N] = 0       # chunk 3 owns no compact entries at all
    flat = rng.integers(0, 1858, size=int(counts.sum())).astype(np.int32)
    n_legal = int(counts.sum())
    whole_pol = torch.zeros(n_legal, dtype=torch.uint16)
    whole_wdl = torch.zeros((rows, 3), dtype=torch.float32)
    plain = DirectGPUEvaluator(model, device="cpu", max_batch=rows, legal_bf16=False)
    expected_pol, expected_wdl = [], []
    for lo, hi in forward_chunk_spans(rows, N):
        f_lo, f_hi = int(counts[:lo].sum()), int(counts[:hi].sum())
        p = torch.zeros(f_hi - f_lo, dtype=torch.uint16)
        w = torch.zeros((hi - lo, 3), dtype=torch.float32)
        plain._legal_forward_into(x[lo:hi], flat[f_lo:f_hi], counts[lo:hi], f_hi - f_lo, p, w)
        expected_pol.append(p)
        expected_wdl.append(w)
    ev._legal_forward_chunks_into(x, flat, counts, forward_chunk_spans(rows, N), whole_pol, whole_wdl)
    assert torch.equal(whole_pol, torch.cat(expected_pol))
    assert torch.equal(whole_wdl, torch.cat(expected_wdl))
    # and the compact entries are each row's own: compare with the dense policy
    dense_pol, dense_wdl = _evaluate(plain, x.numpy())
    rows_idx = np.repeat(np.arange(rows), counts)
    want = torch.from_numpy(dense_pol[rows_idx, flat]).to(torch.bfloat16).view(torch.uint16)
    assert torch.equal(whole_pol, want)
    np.testing.assert_allclose(whole_wdl.numpy(), dense_wdl, atol=1e-5)


_BOARDS = (
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNB1KB1R w KQkq - 1 5",
)


def _search(ev: Any) -> list[bytes]:
    out = run_gumbel_root_many_c(
        None, [chess.Board(f) for f in _BOARDS], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(input_extra_features="v2_threats", simulations=16, topk=8,
                         add_noise=False, temperature=0.0),
        evaluator=ev,
    )
    return [_bits(o) for o in out[:4]]


def _bits(o: Any) -> bytes:
    if isinstance(o, np.ndarray):
        return o.dtype.str.encode() + repr(o.shape).encode() + o.tobytes()
    if isinstance(o, (list, tuple)):
        return b"[" + b",".join(_bits(v) for v in o) + b"]"
    return repr(o).encode()


def test_a_search_that_never_exceeds_n_is_bit_identical_and_a_chunked_one_runs_chunks() -> None:
    model = _model()
    plain = _search(DirectGPUEvaluator(model, device="cpu", max_batch=512))
    wide = DirectGPUEvaluator(model, device="cpu", max_batch=512, forward_chunk_rows=512)
    assert _search(wide) == plain
    assert wide.forward_chunk_stats["chunked_forwards"] == 0
    small = DirectGPUEvaluator(model, device="cpu", max_batch=512, forward_chunk_rows=4)
    chunked = _search(small)
    assert small.forward_chunk_stats["chunked_forwards"] > 0
    print(f"cpu search chunked at 4 rows: identical={chunked == plain}")


@needs_cuda
def test_cuda_chunked_dense_and_legal_match_and_peak_follows_the_chunk() -> None:
    model = _model("cuda")
    rows, n = 5 * 512 + 3, 512
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=rows, legal_bf16=False, cuda_graphs=True)
    assert graph.forward_chunk_rows == 512  # derived from cuda_graph_max_rows
    planes = graph.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(4), rows, planes)
    pol, wdl = _evaluate(graph, x)
    stats = graph.cuda_graph_stats
    assert stats is not None
    assert (stats["eager_over_max_rows"], stats["forward_chunks"]) == (0, 6)
    eager = DirectGPUEvaluator(model, device="cuda", max_batch=rows, legal_bf16=False)
    pol1, wdl1 = _evaluate(eager, x)
    print(f"cuda chunked-graph vs one eager forward: max|dpol|={np.abs(pol - pol1).max():.3g} "
          f"max|dwdl|={np.abs(wdl - wdl1).max():.3g}")
    # peak allocated: a call of 5n + 3 rows ~ a call of n rows plus its outputs
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _evaluate(DirectGPUEvaluator(model, device="cuda", max_batch=rows, legal_bf16=False,
                                 forward_chunk_rows=n), x[:n])
    peak_n = torch.cuda.max_memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    _evaluate(DirectGPUEvaluator(model, device="cuda", max_batch=rows, legal_bf16=False,
                                 forward_chunk_rows=n), x)
    peak_big = torch.cuda.max_memory_allocated() - base
    outputs = rows * (1858 + 3) * 4
    assert peak_big <= peak_n + 2 * outputs, (peak_big, peak_n, outputs)
