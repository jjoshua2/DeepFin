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


def _bf16_ulp(magnitude: float) -> float:
    """bf16 spacing at ``magnitude`` (8 significand bits: 2**(exponent - 7))."""
    return float(2.0 ** (np.floor(np.log2(max(magnitude, 2.0 ** -126))) - 7))


def _peak_bytes(ev: DirectGPUEvaluator, x: np.ndarray) -> tuple[int, int]:
    """(peak, residual) device bytes of one call, each against ITS OWN base taken
    right before the call: a stale base would charge one call with what an earlier
    call left allocated."""
    import gc

    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _evaluate(ev, x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base, torch.cuda.memory_allocated() - base


CUDA_N = 512
CUDA_ROWS = 5 * CUDA_N + 3


@needs_cuda
def test_cuda_graph_chunks_replay_and_match_one_eager_forward_within_bf16_rounding() -> None:
    model = _model("cuda")
    graph = DirectGPUEvaluator(model, device="cuda", max_batch=CUDA_ROWS, legal_bf16=False, cuda_graphs=True)
    assert graph.forward_chunk_rows == CUDA_N  # derived from cuda_graph_max_rows
    planes = graph.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(4), CUDA_ROWS, planes)
    pol, wdl = _evaluate(graph, x)
    stats = graph.cuda_graph_stats
    assert stats is not None
    assert (stats["eager_over_max_rows"], stats["forward_chunks"]) == (0, 6)
    pol1, wdl1 = _evaluate(DirectGPUEvaluator(model, device="cuda", max_batch=CUDA_ROWS, legal_bf16=False), x)
    # Both run under bf16 autocast and return the bf16 outputs widened to fp32, so a
    # batch-size-dependent kernel can move a value by one bf16 step at its own
    # magnitude; allow two steps at the largest magnitude, never more.
    d_pol, d_wdl = float(np.abs(pol - pol1).max()), float(np.abs(wdl - wdl1).max())
    tol_pol = 2 * _bf16_ulp(float(np.abs(pol1).max()))
    tol_wdl = 2 * _bf16_ulp(float(np.abs(wdl1).max()))
    assert d_pol <= tol_pol, (d_pol, tol_pol)
    assert d_wdl <= tol_wdl, (d_wdl, tol_wdl)


@needs_cuda
def test_cuda_chunked_eager_fp32_matches_unchunked_eager() -> None:
    """Like for like: the same eager fp32 forward, chunked vs one call."""
    model = _model("cuda")
    kw: dict[str, Any] = {"device": "cuda", "max_batch": CUDA_ROWS, "legal_bf16": False, "use_amp": False}
    chunked = DirectGPUEvaluator(model, forward_chunk_rows=CUDA_N, **kw)
    plain = DirectGPUEvaluator(model, **kw)
    assert plain.forward_chunk_rows is None
    planes = plain.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(5), CUDA_ROWS, planes)
    pol, wdl = _evaluate(chunked, x)
    pol1, wdl1 = _evaluate(plain, x)
    assert chunked.forward_chunk_stats == {"chunked_forwards": 1, "forward_chunks": 6}
    assert float(np.abs(pol - pol1).max()) <= 1e-3
    assert float(np.abs(wdl - wdl1).max()) <= 1e-3


@needs_cuda
def test_cuda_chunked_compact_legal_matches_unchunked() -> None:
    model = _model("cuda")
    kw: dict[str, Any] = {"device": "cuda", "max_batch": CUDA_ROWS, "legal_bf16": True, "use_amp": False}
    chunked = DirectGPUEvaluator(model, forward_chunk_rows=CUDA_N, **kw)
    plain = DirectGPUEvaluator(model, **kw)
    planes = plain.get_input_buffer(1).shape[1]
    rng = np.random.default_rng(6)
    x = _rows(rng, CUDA_ROWS, planes)
    counts = rng.integers(0, 40, size=CUDA_ROWS).astype(np.int32)
    counts[CUDA_N] = 0
    counts[2 * CUDA_N:3 * CUDA_N] = 0
    flat = rng.integers(0, 1858, size=int(counts.sum())).astype(np.int32)
    a_pol, a_wdl = chunked.evaluate_legal_bf16(x, flat, counts)
    b_pol, b_wdl = plain.evaluate_legal_bf16(x, flat, counts)
    assert chunked.forward_chunk_stats["forward_chunks"] == 6
    a = torch.from_numpy(a_pol).view(torch.bfloat16).float().numpy()
    b = torch.from_numpy(b_pol).view(torch.bfloat16).float().numpy()
    assert a.shape == b.shape
    assert float(np.abs(a - b).max()) <= 2 * _bf16_ulp(float(np.abs(b).max()))
    assert float(np.abs(a_wdl - b_wdl).max()) <= 1e-3


@needs_cuda
def test_cuda_peak_memory_follows_the_chunk_not_the_call() -> None:
    """The point of chunking. Against the UNCHUNKED peak of the same rows (fails if
    the dense chunk loop forwards all rows at once), and against an accounting
    bound whose every term is named."""
    model = _model("cuda")
    kw: dict[str, Any] = {"device": "cuda", "max_batch": CUDA_ROWS, "legal_bf16": False}
    chunked = DirectGPUEvaluator(model, forward_chunk_rows=CUDA_N, **kw)
    plain = DirectGPUEvaluator(model, **kw)
    planes = plain.get_input_buffer(1).shape[1]
    x = _rows(np.random.default_rng(7), CUDA_ROWS, planes)
    _evaluate(chunked, x)  # warm every shape (cuBLAS / allocator state) before measuring
    _evaluate(plain, x)
    _evaluate(plain, x[:CUDA_N])
    peak_one, res_one = _peak_bytes(plain, x[:CUDA_N])  # one N-row forward
    peak_full, res_full = _peak_bytes(plain, x)          # one 5N+3-row forward
    peak_chunk, res_chunk = _peak_bytes(chunked, x)      # the same rows in 6 chunks
    detail = {"peak_one": peak_one, "peak_full": peak_full, "peak_chunk": peak_chunk,
              "residual": (res_one, res_full, res_chunk)}
    assert peak_chunk <= 0.6 * peak_full, detail
    call_outputs = CUDA_ROWS * (1858 + 3) * 4   # the call's fp32 policy + wdl, held for the whole loop
    assert peak_chunk <= peak_one + call_outputs, detail
