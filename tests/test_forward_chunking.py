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
from chess_anti_engine.moves import POLICY_SIZE  # the evaluator's dense policy width (4672 search actions)
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
    flat = rng.integers(0, POLICY_SIZE, size=int(counts.sum())).astype(np.int32)
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


def _storages(*ts: Any) -> list[Any]:
    from torch.multiprocessing.reductions import StorageWeakRef

    return [StorageWeakRef(t.untyped_storage()) for t in ts if isinstance(t, torch.Tensor)]


def test_no_chunk_tensor_outlives_its_chunk() -> None:
    """At each chunk's start, nothing the previous chunk produced is still
    referenced: neither the network's outputs nor what the chunk step returned
    (the dense step scatters the policy 1858 -> 4672 into a NEW tensor, the large
    one, so the model's outputs alone would not see it kept). The loops hold only
    the call's own output across chunks. Dense and compact-legal loops, on CPU."""
    # Storage, not Python object: ``.detach()`` / ``.float()`` make new tensor objects
    # over the same memory, so an object weakref would miss a kept view.
    model = _model()
    rows = 5 * N + 3
    ev = DirectGPUEvaluator(model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False)
    prev: list[Any] = []
    alive: list[int] = []
    seen = [0]

    def post(_m: Any, _a: Any, out: Any) -> None:
        prev.extend(_storages(*out.values()))
        seen[0] += 1

    hook = model.register_forward_hook(post)
    m_any: Any = model
    inner_fwd_legal = m_any.forward_legal_policy  # the compact-legal forward bypasses __call__ hooks

    def fwd_legal(*a: Any, **k: Any) -> Any:
        out = inner_fwd_legal(*a, **k)
        post(model, a, out)
        return out

    m_any.forward_legal_policy = fwd_legal
    inner_dense = ev._dense_forward_chunk
    inner_legal = ev._legal_forward_into

    def dense(*a: Any, **k: Any) -> Any:
        alive.append(sum(not r.expired() for r in prev))
        prev.clear()
        res = inner_dense(*a, **k)
        prev.extend(_storages(*res))   # what the step returned: the scattered 4672-wide policy and wdl
        return res

    def legal(*a: Any, **k: Any) -> Any:
        alive.append(sum(not r.expired() for r in prev))
        prev.clear()
        return inner_legal(*a, **k)

    ev._dense_forward_chunk = dense  # type: ignore[method-assign]
    ev._legal_forward_into = legal  # type: ignore[method-assign]
    try:
        x = _rows(np.random.default_rng(8), rows, ev.get_input_buffer(1).shape[1])
        ev.get_input_buffer(rows)[:] = x
        pol, wdl = ev._dense_forward_on_stream(rows, slot=0, relations=None)
        assert tuple(pol.shape) == (rows, POLICY_SIZE)
        assert tuple(wdl.shape) == (rows, 3)
        del pol, wdl
        assert alive == [0] * 6
        assert seen[0] == 6  # the hook saw every chunk's outputs (not vacuous)
        prev.clear()
        alive.clear()
        counts = np.random.default_rng(9).integers(1, 6, size=rows).astype(np.int32)
        flat = np.random.default_rng(10).integers(0, POLICY_SIZE, size=int(counts.sum())).astype(np.int32)
        ev._legal_forward_chunks_into(
            torch.from_numpy(x), flat, counts, forward_chunk_spans(rows, N),
            torch.zeros(int(counts.sum()), dtype=torch.uint16), torch.zeros((rows, 3), dtype=torch.float32),
        )
        assert alive == [0] * 6
        assert seen[0] == 12
    finally:
        hook.remove()
        del m_any.forward_legal_policy


def test_chunks_slice_the_relations_input_with_their_rows() -> None:
    """A relations-input model: each chunk must forward ITS rows' relations
    (chunk 0's relations fed to every chunk would be silently wrong output)."""
    torch.manual_seed(0)
    cfg = ModelConfig(input_extra_features="v2_threats", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
                      use_dynamic_relations=True, dynamic_relation_count=5)
    model = build_model(cfg).eval()
    weight = getattr(model, "dynamic_relation_weight", None)
    assert isinstance(weight, torch.nn.Parameter)
    with torch.no_grad():
        weight.normal_(0.0, 2.0)  # zero-initialised: the relations would not change the output
    rows = 3 * N + 2
    chunked = DirectGPUEvaluator(model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False)
    plain = DirectGPUEvaluator(model, device="cpu", max_batch=rows, legal_bf16=False)
    rng = np.random.default_rng(11)
    x = _rows(rng, rows, plain.get_input_buffer(1).shape[1])
    rel = (rng.random((rows, 5, 64, 64)) < 0.3).astype(np.uint8)
    chunked.get_input_buffer(rows)[:] = x
    pol, wdl = chunked.evaluate_inplace(rows, relations=rel)
    assert chunked.forward_chunk_stats["chunked_forwards"] == 1
    plain.get_input_buffer(rows)[:] = x
    want_pol, want_wdl = plain.evaluate_inplace(rows, relations=rel)
    np.testing.assert_allclose(pol, want_pol, atol=1e-4)
    np.testing.assert_allclose(wdl, want_wdl, atol=1e-4)
    # not vacuous: the relations change the output
    plain.get_input_buffer(rows)[:] = x
    no_rel_pol, _ = plain.evaluate_inplace(rows)
    assert float(np.abs(no_rel_pol - want_pol).max()) > 1e-3


def test_cpu_async_entries_chunk_too() -> None:
    """Off CUDA the async entries used to forward the whole call unchunked: the
    bound was accepted and ignored there. They chunk like the sync entry."""
    model = _model()
    rows = 5 * N + 3
    chunked = DirectGPUEvaluator(model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False)
    plain = DirectGPUEvaluator(model, device="cpu", max_batch=rows, legal_bf16=False)
    x = _rows(np.random.default_rng(9), rows, plain.get_input_buffer(1).shape[1])
    chunked.get_input_buffer(rows)[:] = x
    pol, wdl, _ev = chunked.evaluate_inplace_async(rows)
    assert chunked.forward_chunk_stats == {"chunked_forwards": 1, "forward_chunks": 6}
    pol2, wdl2, _ev2 = chunked.evaluate_encoded_async(x)
    assert chunked.forward_chunk_stats == {"chunked_forwards": 2, "forward_chunks": 12}
    want_pol, want_wdl = _evaluate(plain, x)
    for p_, w_ in ((pol, wdl), (pol2, wdl2)):
        np.testing.assert_allclose(np.asarray(p_), want_pol, atol=1e-4)
        np.testing.assert_allclose(np.asarray(w_), want_wdl, atol=1e-4)
    small = DirectGPUEvaluator(model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False)
    small.get_input_buffer(N)[:] = x[:N]
    small.evaluate_inplace_async(N)
    assert small.forward_chunk_stats["chunked_forwards"] == 0
    # the bucket-padding branch of both entries (rows padded up to a bucket, then sliced back)
    padded = DirectGPUEvaluator(
        model, device="cpu", max_batch=rows, forward_chunk_rows=N, legal_bf16=False, pad_to_compiled_buckets=True,
    )
    padded.get_input_buffer(rows)[:] = x
    pol3, wdl3, _ev3 = padded.evaluate_inplace_async(rows)
    pol4, wdl4, _ev4 = padded.evaluate_encoded_async(x)
    assert padded.forward_chunk_stats["chunked_forwards"] == 2
    for p_, w_ in ((pol3, wdl3), (pol4, wdl4)):
        np.testing.assert_allclose(np.asarray(p_), want_pol, atol=1e-4)
        np.testing.assert_allclose(np.asarray(w_), want_wdl, atol=1e-4)


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


def _peak_bytes(ev: DirectGPUEvaluator, x: np.ndarray) -> tuple[int, int, int]:
    """(allocated peak, requested peak, residual) device bytes of one call, each
    against ITS OWN base taken right before the call, from an EMPTY allocator cache.

    ``allocated`` counts caching-allocator blocks. A block carved from a larger free
    block is not split when the remainder is <= 1 MiB, so the same tensors count
    up to 1 MiB more each depending on what earlier calls left cached; emptying
    the cache first makes every measured call start from the same state.
    ``requested`` is the sum of the sizes the code asked for, independent of the
    allocator's block rounding."""
    import gc

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    base_req = torch.cuda.memory_stats()["requested_bytes.all.current"]
    torch.cuda.reset_peak_memory_stats()
    _evaluate(ev, x)
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    return (torch.cuda.max_memory_allocated() - base, stats["requested_bytes.all.peak"] - base_req,
            torch.cuda.memory_allocated() - base)


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


def _forwarded_rows(model: torch.nn.Module) -> tuple[list[int], Any]:
    """Record the batch rows of every network forward (a forward pre-hook)."""
    seen: list[int] = []
    handle = model.register_forward_pre_hook(lambda _m, a: seen.append(int(a[0].shape[0])))
    return seen, handle


@needs_cuda
def test_cuda_chunked_eager_fp32_matches_unchunked_eager() -> None:
    """Like for like: the same eager fp32 forward, chunked vs one call, with TF32
    off (TF32 matmuls round inputs to 10 mantissa bits, ~1e-3 relative, which
    would hide a misplaced row; fp32 kernels differ by batch size only at the
    last bits). Every wdl and policy row is compared, so a chunk written at the
    wrong offset fails."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model = _model("cuda")
        kw: dict[str, Any] = {"device": "cuda", "max_batch": CUDA_ROWS, "legal_bf16": False, "use_amp": False}
        chunked = DirectGPUEvaluator(model, forward_chunk_rows=CUDA_N, **kw)
        plain = DirectGPUEvaluator(model, **kw)
        assert plain.forward_chunk_rows is None
        planes = plain.get_input_buffer(1).shape[1]
        x = _rows(np.random.default_rng(5), CUDA_ROWS, planes)
        seen, handle = _forwarded_rows(model)
        try:
            pol, wdl = _evaluate(chunked, x)
        finally:
            handle.remove()
        pol1, wdl1 = _evaluate(plain, x)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    assert chunked.forward_chunk_stats == {"chunked_forwards": 1, "forward_chunks": 6}
    assert seen == [hi - lo for lo, hi in forward_chunk_spans(CUDA_ROWS, CUDA_N)]
    d_pol, d_wdl = float(np.abs(pol - pol1).max()), float(np.abs(wdl - wdl1).max())
    print(f"cuda fp32 (TF32 off) chunked vs unchunked: max|dpol|={d_pol:.3g} max|dwdl|={d_wdl:.3g}")
    assert d_pol <= 1e-5
    assert d_wdl <= 1e-5


@needs_cuda
def test_cuda_eager_chunks_above_the_graph_cap_use_their_own_device_copy() -> None:
    """--eval-chunk-rows above the graph row cap: each chunk is an eager forward
    above the cap (counted in eager_over_max_rows), reading its own device copy;
    the graph's static input holds only the cap's rows and must not be written."""
    model = _model("cuda")
    cap, n = CUDA_N // 4, CUDA_N
    kw: dict[str, Any] = {"device": "cuda", "max_batch": CUDA_ROWS, "legal_bf16": False}
    ev = DirectGPUEvaluator(model, cuda_graphs=True, cuda_graph_max_rows=cap, forward_chunk_rows=n, **kw)
    assert ev.forward_chunk_rows == n
    x = _rows(np.random.default_rng(12), CUDA_ROWS, ev.get_input_buffer(1).shape[1])
    pol, wdl = _evaluate(ev, x)
    stats = ev.cuda_graph_stats
    assert stats is not None
    assert stats["forward_chunks"] == 6
    assert stats["eager_over_max_rows"] == 5  # the five full chunks; the 3-row tail replays a graph
    pol1, wdl1 = _evaluate(DirectGPUEvaluator(model, **kw), x)
    assert float(np.abs(pol - pol1).max()) <= 2 * _bf16_ulp(float(np.abs(pol1).max()))
    assert float(np.abs(wdl - wdl1).max()) <= 2 * _bf16_ulp(float(np.abs(wdl1).max()))


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
    flat = rng.integers(0, POLICY_SIZE, size=int(counts.sum())).astype(np.int32)
    seen, handle = _forwarded_rows(model)
    try:
        a_pol, a_wdl = chunked.evaluate_legal_bf16(x, flat, counts)
        # the async entry is the one the threaded / dispatcher search paths call
        chunked.get_input_buffer(CUDA_ROWS)[:] = x
        c_pol_t, c_wdl_t, c_ev = chunked.evaluate_inplace_legal_bf16_async(CUDA_ROWS, flat, counts)
        if c_ev is not None:
            c_ev.synchronize()
        c_pol, c_wdl = c_pol_t.numpy().copy(), c_wdl_t.numpy().copy()
    finally:
        handle.remove()
    b_pol, b_wdl = plain.evaluate_legal_bf16(x, flat, counts)
    assert chunked.forward_chunk_stats["forward_chunks"] == 12
    # no network forward above N rows (the counter alone cannot see a loop that counts and then forwards all rows)
    assert max(seen) <= CUDA_N, seen
    assert len(seen) == 12, seen
    assert c_pol.tobytes() == a_pol.tobytes()
    assert c_wdl.tobytes() == a_wdl.tobytes()
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
    peak_one, req_one, res_one = _peak_bytes(plain, x[:CUDA_N])  # one N-row forward
    peak_full, req_full, res_full = _peak_bytes(plain, x)          # one 5N+3-row forward
    peak_chunk, req_chunk, res_chunk = _peak_bytes(chunked, x)     # the same rows in 6 chunks
    detail = {"peak_one": peak_one, "peak_full": peak_full, "peak_chunk": peak_chunk,
              "requested": (req_one, req_full, req_chunk), "residual": (res_one, res_full, res_chunk)}
    assert peak_chunk <= 0.6 * peak_full, detail
    # The only allocation the loop holds across chunks is the call's own output: the
    # fp32 dense policy (POLICY_SIZE = 4672 search actions, not the net's 1858 lc0
    # outputs) and the fp32 wdl, 2,563 x 4,675 x 4 = 47.9 MB requested.
    pol_bytes, wdl_bytes = CUDA_ROWS * POLICY_SIZE * 4, CUDA_ROWS * 3 * 4
    assert req_chunk <= req_one + pol_bytes + wdl_bytes, detail
    # As allocator blocks: a block above 1 MiB may occupy its whole segment, which is
    # rounded up to 2 MiB (the remainder of <= 1 MiB is not split off); a small block
    # is rounded up to 512 B. The first GPU run's excess over one N-row forward was
    # exactly that: 48,234,496 + 31,232 = 48,265,728 bytes.
    def ceil_to(n: int, m: int) -> int:
        return -(-n // m) * m

    held = ceil_to(pol_bytes, 2 * 2**20) + ceil_to(wdl_bytes, 512)
    assert peak_chunk <= peak_one + held, detail
