"""Compiled-path evaluator rebuilds and repeated devices (real-GPU run 2026-11-18).

On the default compiled single-GPU path a mid-session rebuild (``EvalCacheEntries``,
``MaxBatch``, ``SearchWalkers``) failed its warm-up with
``CUBLAS_STATUS_NOT_INITIALIZED`` inside a CUDA-graph capture, after which every
search raised ``beginAllocateToPool: already recording to mempool_id`` and the
game-boundary health check still passed. ``chess_anti_engine.cudagraph_threads``
holds the mechanism and the fixes; these tests drive them on CPU by injecting
the failures, plus GPU-only tests (skipped without CUDA) that reproduce the
cross-thread capture on a toy model.
"""
from __future__ import annotations

import threading
import weakref
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import chess_anti_engine.uci.__main__ as uci_main
from chess_anti_engine import cudagraph_threads, inference
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci import wedge
from chess_anti_engine.uci.device_guard import DegradingMultiGPUDispatcher
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command

from tests.test_uci_strict_startup import _flaky_engine, _make_unhealthy, _ok


def _model() -> torch.nn.Module:
    m = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    m.eval()
    return m


def _on_thread(fn) -> Any:
    box: dict[str, Any] = {}

    def body() -> None:
        try:
            box["r"] = fn()
        except BaseException as exc:  # pragma: no cover - surfaced below
            box["e"] = exc

    t = threading.Thread(target=body)
    t.start()
    t.join(30)
    if "e" in box:
        raise box["e"]
    return box.get("r")


# --- a failed capture is closed so the previous evaluator keeps working ------


class _FakeManager:
    """Stands in for ``cudagraph_trees.CUDAGraphTreeManager`` on the failing stack."""

    def __init__(self, device_index: int, pool: Any) -> None:
        self.device_index = device_index
        self.cuda_graphs_thread_pool = pool

    def record_function(self) -> None:
        raise RuntimeError("CUDA error: operation failed due to a previous error during capture")


_FakeManager.__module__ = "torch._inductor.cudagraph_trees"


def _captured_failure(manager: _FakeManager) -> BaseException:
    """The evaluator-level error, chained from the manager's failed record."""
    try:
        try:
            manager.record_function()
        except RuntimeError as inner:
            raise RuntimeError("evaluator warmup failed at batch=340") from inner
    except RuntimeError as outer:
        return outer
    raise AssertionError("unreachable")


def test_release_failed_capture_ends_the_failing_managers_pool_scope_once(monkeypatch) -> None:
    """The scope a failed capture left open is found on the failing stack (the
    manager whose record raised -- possibly another thread's) and ended once;
    with nothing open, or no manager on the stack, it is a no-op."""
    ended: list[tuple[int, Any]] = []
    open_scopes = {(0, ("pool", 1))}

    def end(idx: int, pool: Any) -> None:
        if (idx, pool) not in open_scopes:
            raise RuntimeError("endAllocatePool: not currently recording to mempool_id")
        open_scopes.discard((idx, pool))
        ended.append((idx, pool))

    monkeypatch.setattr(torch._C, "_cuda_endAllocateToPool", end, raising=False)
    monkeypatch.setattr(cudagraph_threads, "released_captures", [])
    exc = _captured_failure(_FakeManager(0, ("pool", 1)))
    assert cudagraph_threads.release_failed_capture(exc) is True
    assert ended == [(0, ("pool", 1))]
    assert cudagraph_threads.release_failed_capture(exc) is False  # nothing open now
    assert cudagraph_threads.release_failed_capture(RuntimeError("CUDA out of memory")) is False
    assert cudagraph_threads.released_captures == [0]


def test_the_torch_internals_the_release_reads_exist_in_this_torch() -> None:
    """``release_failed_capture`` reads private torch internals and falls back
    to a warning without them: a torch upgrade that renames one must fail here,
    not silently bring the poisoning back. Read off the installed torch."""
    import inspect

    import torch._inductor.cudagraph_trees as ct

    # Present even in a CPU-only build (torch 2.14.0+cpu on CI), so always pinned.
    assert callable(getattr(torch._C, "_cuda_endAllocateToPool", None))
    if torch.backends.cuda.is_built():  # absent from a CPU-only build
        assert callable(getattr(torch._C, "_cuda_clearCublasWorkspaces", None))
    assert ct.__name__ == cudagraph_threads._CUDAGRAPH_TREES_MODULE
    manager = ct.CUDAGraphTreeManager
    assert manager.__module__ == cudagraph_threads._CUDAGRAPH_TREES_MODULE
    for method in ("run", "record_function"):  # the frames a failed record has
        assert method in vars(manager), method
    init = inspect.getsource(manager.__init__)
    assert "self.device_index" in init
    assert "self.cuda_graphs_thread_pool" in init
    # The pool the manager records into is that one (a node's pool is the manager's).
    assert "cuda_graphs_thread_pool" in inspect.getsource(manager.record_function)


def test_a_capture_failure_the_release_cannot_read_is_warned_once(monkeypatch, caplog) -> None:
    """No cudagraph-trees manager on the stack (or no endAllocateToPool) for a
    failure that looks like a capture: say so loudly, once, with the torch
    version, instead of returning False in silence."""
    import logging

    monkeypatch.setattr(cudagraph_threads, "_internals_warned", False)
    caplog.set_level(logging.WARNING, logger=cudagraph_threads.__name__)
    assert cudagraph_threads.release_failed_capture(RuntimeError("CUDA out of memory")) is False
    assert not caplog.records  # not a capture: nothing to say
    capture = RuntimeError("CUDA error: operation failed due to a previous error during capture")
    assert cudagraph_threads.release_failed_capture(capture) is False
    assert cudagraph_threads.release_failed_capture(capture) is False
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1, warned
    assert torch.__version__ in warned[0]
    assert "no torch._inductor.cudagraph_trees manager" in warned[0]

    caplog.clear()
    monkeypatch.setattr(cudagraph_threads, "_internals_warned", False)
    monkeypatch.delattr(torch._C, "_cuda_endAllocateToPool", raising=False)
    wrapped = _captured_failure(_FakeManager(0, ("pool", 1)))
    assert cudagraph_threads.release_failed_capture(wrapped) is False
    assert "_cuda_endAllocateToPool is missing" in caplog.records[0].getMessage()


def test_compiled_forward_scope_prepares_first_and_releases_on_failure(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cudagraph_threads, "prepare_thread_for_capture",
                        lambda dev: calls.append(f"prepare {dev}"))
    monkeypatch.setattr(cudagraph_threads, "release_failed_capture",
                        lambda exc: calls.append(f"release {type(exc).__name__}") or True)
    with cudagraph_threads.compiled_forward_scope("cuda:0"):
        calls.append("forward")
    assert calls == ["prepare cuda:0", "forward"]  # success: nothing released
    calls.clear()
    def failing_forward() -> None:
        with cudagraph_threads.compiled_forward_scope("cuda:0"):
            calls.append("forward")
            raise RuntimeError("operation failed due to a previous error during capture invalidated")

    with pytest.raises(RuntimeError, match="capture invalidated"):
        failing_forward()
    assert calls == ["prepare cuda:0", "forward", "release RuntimeError"]


def test_only_compiled_cuda_forwards_take_the_graph_scope(monkeypatch) -> None:
    seen: list[str] = []

    class _Scope:
        def __init__(self, dev: str) -> None:
            seen.append(dev)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: object) -> None:
            return None

    monkeypatch.setattr(inference, "compiled_forward_scope", _Scope)
    padded = DirectGPUEvaluator(_model(), device="cpu", max_batch=16, use_amp=False,
                                pad_to_compiled_buckets=True)
    eager = DirectGPUEvaluator(_model(), device="cpu", max_batch=16, use_amp=False)
    padded._use_cuda = True  # the predicate only; nothing below runs a CUDA forward
    with padded._graph_scope():
        pass
    with eager._graph_scope():
        pass
    padded._use_cuda = False
    with padded._graph_scope():
        pass
    assert seen == ["cpu"]


# --- the health re-check proves health on the evaluator it reports healthy ----


def test_health_check_fails_when_a_compiled_shape_raises_though_the_search_passes(capsys) -> None:
    """The real-GPU failure: the boundary's small warm-up search only replayed
    already-recorded shapes, so the check passed while every longer go raised.
    The re-check now reruns every compiled shape on the CURRENT evaluator (the
    one kept after a failed rebuild) and fails honestly."""
    checked: list[object] = []

    def refuse(*_a, **_kw):
        raise RuntimeError("evaluator warmup failed at batch=340")

    engine, worker, _ = _flaky_engine(refuse)
    kept = object()
    worker._evaluator = kept

    def verify(target: Any) -> None:
        checked.append(target._evaluator)
        raise RuntimeError("evaluator re-check failed at batch=340")

    engine._verify_evaluator = verify
    _make_unhealthy(engine, worker)
    worker.run.side_effect = _ok  # the small warm-up search alone would pass
    engine.dispatch(parse_command("ucinewgame"))
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert "health rebuild: evaluator rebuild failed" in out
    assert checked == [kept]  # the consumer got the worker holding the evaluator kept
    assert "engine health check passed" not in out
    assert any(ln.startswith("info string engine health check failed") and "batch=340" in ln
               for ln in out.splitlines())
    assert engine.exit_code == 1
    assert engine.quit_requested


def test_health_check_passes_only_after_the_shape_check_and_the_search(capsys) -> None:
    order: list[str] = []
    engine, worker, _ = _flaky_engine()
    engine._verify_evaluator = lambda _worker: order.append("shapes")
    _make_unhealthy(engine, worker)

    def warm(board, **_kw):
        order.append("search")
        return _ok(board)

    worker.run.side_effect = warm
    engine.dispatch(parse_command("ucinewgame"))
    engine.dispatch(parse_command("isready"))
    assert order == ["shapes", "search"]
    assert "engine health check passed; unhealthy state cleared" in capsys.readouterr().out
    assert engine.exit_code == 0


def test_engine_receives_the_verifier_from_build_engine(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    real = Engine.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        real(self, *a, **kw)

    monkeypatch.setattr(Engine, "__init__", spy)
    marker = object()
    from unittest.mock import MagicMock
    monkeypatch.setattr(uci_main, "SearchWorker", MagicMock())
    monkeypatch.setattr(uci_main, "SEARCH_OPTIONS", ())
    uci_main._build_engine(
        evaluator=MagicMock(), primary_device="cpu", chunk_sims=64, topk=8,
        n_walkers=1, vloss_weight=3, walker_gather=1, pucv_vloss_mode=0, max_batch=64,
        vl_gather=64, eval_cache_entries=0, use_multi_gpu_pucv=False,
        options=EngineOptions(), verify_evaluator=marker,
    )
    assert seen["verify_evaluator"] is marker


def test_reverify_reruns_every_compiled_shape_even_already_seen_ones() -> None:
    """The warm-up skips seen shapes; the health check must not."""
    model = _model()
    counts: dict[int, int] = defaultdict(int)
    real_forward = model.forward

    def counting(x, *a, **kw):
        counts[int(x.shape[0])] += 1
        return real_forward(x, *a, **kw)

    model.forward = counting  # type: ignore[method-assign]
    ev = DirectGPUEvaluator(model, device="cpu", max_batch=64, n_slots=2, use_amp=False,
                            pad_to_compiled_buckets=True, legal_bf16=False)
    uci_main._warmup_evaluator(ev, input_extra_features="v1")
    shapes = set(ev.reachable_forward_rows())
    assert shapes <= ev.shape_ledger.seen()  # warm: the warm-up would skip all of them
    counts.clear()
    n = uci_main.reverify_compiled_evaluator(ev, input_extra_features="v1")
    assert n == len(shapes)
    assert dict(counts) == dict.fromkeys(shapes, 2)


def test_reverify_raises_on_the_broken_shape_and_skips_eager_chains() -> None:
    model = _model()
    real_forward = model.forward

    def broken(x, *a, **kw):
        if int(x.shape[0]) == 32:
            raise RuntimeError("beginAllocateToPool: already recording to mempool_id")
        return real_forward(x, *a, **kw)

    model.forward = broken  # type: ignore[method-assign]
    ev = DirectGPUEvaluator(model, device="cpu", max_batch=64, n_slots=2, use_amp=False,
                            pad_to_compiled_buckets=True, legal_bf16=False)
    with pytest.raises(RuntimeError, match="re-check failed at batch=32"):
        uci_main.reverify_compiled_evaluator(ev, input_extra_features="v1")
    eager = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    assert uci_main.reverify_compiled_evaluator(eager, input_extra_features="v1") == 0
    assert uci_main.reverify_compiled_evaluator(None) == 0


def _counting_evaluator() -> tuple[DirectGPUEvaluator, dict[int, int], torch.nn.Module]:
    model = _model()
    counts: dict[int, int] = defaultdict(int)
    real_forward = model.forward

    def counting(x, *a, **kw):
        counts[int(x.shape[0])] += 1
        return real_forward(x, *a, **kw)

    model.forward = counting  # type: ignore[method-assign]
    ev = DirectGPUEvaluator(model, device="cpu", max_batch=64, n_slots=2, use_amp=False,
                            pad_to_compiled_buckets=True, legal_bf16=False)
    return ev, counts, model


def test_reverify_runs_below_the_eval_cache_and_reports_real_forwards(capsys) -> None:
    """Review of #1404: through an EncodedEvalCache the identical start-position
    rows collapsed to one batch-1 forward, the rest were cache hits, and the
    line still claimed every shape ran twice. The check now runs below the
    cache and prints the forwards its ledger counted."""
    from chess_anti_engine.inference_cache import EncodedEvalCache

    ev, counts, model = _counting_evaluator()
    cached = EncodedEvalCache(ev, max_entries=4096, encoding_source=model)
    shapes = set(ev.reachable_forward_rows())
    n = uci_main.reverify_compiled_evaluator(cached, input_extra_features="v1")
    assert n == len(shapes)
    assert dict(counts) == dict.fromkeys(shapes, 2)
    assert cached.stats().inner_calls == 0  # nothing went through the cache
    line = [ln for ln in capsys.readouterr().out.splitlines() if "health check:" in ln][-1]
    assert line == (f"info string health check: re-ran {len(shapes)} compiled batch shapes on "
                    f"cpu: {2 * len(shapes)} real forwards, at least 2 per shape per device")


def test_reverify_fails_when_a_shape_did_not_really_run() -> None:
    """A wrapper that answers a shape without a forward (a cache, a dedupe)
    must not let the check claim that shape: the ledger count decides."""
    ev = _counting_evaluator()[0]

    class _SkipsBatch16:
        def __init__(self, inner: DirectGPUEvaluator) -> None:
            self._inner = inner

        def evaluate_encoded(self, x, **kw):
            if x.shape[0] == 16:
                return np.zeros((16, 1858), np.float32), np.zeros((16, 3), np.float32)
            return self._inner.evaluate_encoded(x, **kw)

    with pytest.raises(RuntimeError, match=r"ran batch=16 \(dense\) 0x on cpu, not 2x"):
        uci_main.reverify_compiled_evaluator(_SkipsBatch16(ev), input_extra_features="v1")


def _degraded_chain(n: int, drop: list[int], cls: type = DegradingMultiGPUDispatcher):
    evs = []
    for k in range(n):
        ev = _counting_evaluator()[0]
        ev.shape_ledger.label = f"cpu#{k}"
        evs.append(ev)
    chain = cls(evs, labels=[f"cpu#{k}" for k in range(n)])
    chain.arm()
    for idx in drop:
        with chain._state_lock:
            chain._drop_locked(idx, "failed mid-search", "injected", stalled=False)
    return chain


@pytest.mark.parametrize(("n", "drop"), [(3, [1]), (4, [3]), (4, [0, 2]), (3, [])])
def test_reverify_passes_on_a_healthy_degraded_chain(n: int, drop: list[int]) -> None:
    """Review of #1404: with a device dropped, n consecutive calls did not reach
    every live device (the router's cursor wrapped over ALL devices), and the
    re-check failed a healthy degraded chain: `ran batch=1 (dense) 1x on cpu#2`."""
    chain = _degraded_chain(n, drop)
    try:
        assert uci_main.reverify_compiled_evaluator(chain, input_extra_features="v1") == 4
    finally:
        chain.close()


def test_degraded_routing_alternates_over_the_live_devices_only() -> None:
    """Idle calls on 3 devices with #1 dropped go 0, 2, 0, 2 (not 0, 2, 2, 0);
    with every device live the cursor wraps over all of them, as before."""
    chain = _degraded_chain(3, [1])
    healthy = _degraded_chain(3, [])
    try:
        picks = []
        for _ in range(6):
            k = chain._pick_live(set())
            chain._inflight[k] -= 1
            picks.append(k)
        assert picks == [0, 2, 0, 2, 0, 2]
        picks = []
        for _ in range(6):
            k = healthy._pick_live(set())
            healthy._inflight[k] -= 1
            picks.append(k)
        assert picks == [0, 1, 2, 0, 1, 2]
    finally:
        chain.close()
        healthy.close()


class _SkewedRouter(DegradingMultiGPUDispatcher):
    """Routes by a fixed pattern over device indices (``pattern`` set per test)."""

    pattern: tuple[int, ...] = (0,)
    _calls: int = 0

    def _pick_live(self, tried: set[int]) -> int:
        with self._select_lock:
            n = self._calls
            self._calls = n + 1
            best = self.pattern[n % len(self.pattern)]
            if best in tried or not self._live[best]:
                return -1
            self._inflight[best] += 1
            return best


def test_reverify_tops_up_a_device_the_router_reaches_rarely(monkeypatch) -> None:
    """Whatever the routing, the check keeps submitting a shape until every live
    device has run it twice: device 1 gets one call in three here."""
    monkeypatch.setattr(_SkewedRouter, "pattern", (0, 0, 1))
    chain = _degraded_chain(2, [], cls=_SkewedRouter)
    try:
        assert uci_main.reverify_compiled_evaluator(chain, input_extra_features="v1") == 4
    finally:
        chain.close()


def test_reverify_fails_when_a_live_device_is_never_reached(monkeypatch) -> None:
    """Bounded: a live device the router never sends work to is a real failure."""
    monkeypatch.setattr(_SkewedRouter, "pattern", (0,))
    chain = _degraded_chain(2, [], cls=_SkewedRouter)
    try:
        with pytest.raises(RuntimeError, match=r"ran batch=1 \(dense\) 0x on cpu#1, not 2x"):
            uci_main.reverify_compiled_evaluator(chain, input_extra_features="v1")
    finally:
        chain.close()


def test_reverify_says_plainly_when_nothing_is_compiled(capsys) -> None:
    eager = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    assert uci_main.reverify_compiled_evaluator(eager, input_extra_features="v1") == 0
    out = capsys.readouterr().out
    assert "no compiled evaluator under the primary (eager), so no batch shapes were re-run" in out
    assert "DEGRADED path" in out


def test_health_check_empties_every_eval_cache_the_worker_reaches(capsys) -> None:
    """A search answered from evals cached before the evaluator broke proves
    nothing: the dense cache on the evaluator chain and a pool's PUCV cache
    are both emptied before the shapes and the real search run."""
    from chess_anti_engine.inference_cache import EncodedEvalCache, PucvEvalCache

    ev, counts, model = _counting_evaluator()
    dense = EncodedEvalCache(ev, max_entries=4096, encoding_source=model)
    dense.evaluate_encoded(np.zeros((3, *_encoded_shape()), np.float32))
    pucv = PucvEvalCache(max_entries=64)
    pucv._cache[(1, 1)] = (b"", np.zeros(1), np.zeros(1))
    pool = SimpleNamespace()  # a foreign object: not walked
    pool._cache = PucvEvalCache(max_entries=64)
    pool._cache._cache[(1, 1)] = (b"", np.zeros(1), np.zeros(1))

    class _Pool:  # this package's kind of object: walked
        def __init__(self) -> None:
            self._cache = pucv

    _Pool.__module__ = "chess_anti_engine.uci.multi_gpu_pucv_pool"
    worker = SimpleNamespace(_evaluator=dense, _pucv_pool=_Pool(), _other=pool)
    class _Worker:
        pass

    _Worker.__module__ = "chess_anti_engine.uci.search"
    real_worker = _Worker()
    vars(real_worker).update(vars(worker))
    assert dense.stats().entries > 0
    counts.clear()
    uci_main.health_check_evaluator(real_worker, input_extra_features="v1")
    assert dense.stats().entries == 0
    assert pucv.stats().entries == 0
    assert len(pool._cache._cache) == 1  # outside the package: left alone
    assert "emptied 2 eval cache(s)" in capsys.readouterr().out
    assert dict(counts) == dict.fromkeys(ev.reachable_forward_rows(), 2)


def _encoded_shape() -> tuple[int, ...]:
    import chess

    return tuple(uci_main.encode_cboard(uci_main.CBoard.from_board(chess.Board()),
                                        input_extra_features="v1").shape)


# --- a failed build is released, not leaked -----------------------------------


@pytest.fixture
def no_compile(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda module, **_kw: module)


@pytest.mark.usefixtures("no_compile")
def test_failed_rebuilds_leave_no_live_evaluator_or_owner_thread(monkeypatch) -> None:
    """Every failed build closed its owner thread, so nothing keeps its
    evaluator (pinned buffers) alive: count them across six failed rebuilds."""
    import gc

    built: list[weakref.ref] = []
    real_init = DirectGPUEvaluator.__init__

    def tracking_init(self, *a, **kw):
        real_init(self, *a, **kw)
        built.append(weakref.ref(self))

    monkeypatch.setattr(DirectGPUEvaluator, "__init__", tracking_init)

    def failing_warmup(*_a, **_kw):
        raise RuntimeError("evaluator warmup failed at batch=340")

    monkeypatch.setattr(uci_main, "_warmup_evaluator", failing_warmup)
    build = uci_main._make_evaluator_factory(
        [_model()], ["cpu"], True, 1, 1, compile_mode="max-autotune",
    )
    threads_before = {t.ident for t in threading.enumerate()}
    for _ in range(6):
        with pytest.raises(RuntimeError, match="batch=340"):
            build(64)
    gc.collect()
    assert len(built) == 6
    assert [r for r in built if r() is not None] == []
    leaked = [t for t in threading.enumerate() if t.ident not in threads_before and t.is_alive()]
    assert leaked == []


@pytest.mark.usefixtures("no_compile")
def test_successful_rebuilds_release_the_replaced_evaluators() -> None:
    import gc

    built: list[weakref.ref] = []
    build = uci_main._make_evaluator_factory(
        [_model()], ["cpu"], True, 1, 1, compile_mode="max-autotune",
    )
    current: Any = None
    for cache in (0, 4096, 0, 4096, 0, 4096):
        new = build(64, cache)
        built.extend(weakref.ref(ev) for ev in uci_main.direct_evaluators(new))
        if current is not None:
            uci_main._close_chain(current)
        current = new
    gc.collect()
    assert current is not None
    live = [r for r in built if r() is not None]
    assert len(live) == 1  # the installed one only
    uci_main._close_chain(current)


# --- repeated devices, the chain fallback, the override ------------------------


class _ModeRecorder:
    def __init__(self) -> None:
        self.modes: list[str | None] = []

    def compile(self, module, *, mode=None, **_kw):
        self.modes.append(mode)
        return module


def test_graph_modes_keep_cuda_graphs_on_the_first_listing_of_each_gpu() -> None:
    f = uci_main.graph_modes_per_device
    assert f("max-autotune", ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]) == ["max-autotune"] * 4
    assert f("max-autotune", ["cuda:0", "cuda:0"]) == ["max-autotune", "default"]
    assert f("reduce-overhead", ["cuda", "cuda:0", "cuda:1"]) == [
        "reduce-overhead", "default", "reduce-overhead"]
    assert f("default", ["cuda:0", "cuda:0"]) == ["default", "default"]
    assert f("max-autotune", ["cpu", "cpu"]) == ["max-autotune", "max-autotune"]
    assert uci_main.device_labels(["cuda:0", "cuda:0", "cuda:1"]) == ["cuda:0#0", "cuda:0#1", "cuda:1"]


@pytest.fixture
def cuda_names_on_cpu(monkeypatch):
    """Let ``cuda:N`` device names reach the factories on a CPU-only box: the
    fake evaluator never touches a device and ``set_device`` is recorded."""
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_a, **_kw: None)


class _FakeDirect:
    def __init__(self, model, *, device, max_batch, n_slots, input_bf16=False,
                 legal_bf16=True, pad_to_compiled_buckets=False) -> None:
        del model, n_slots, input_bf16, legal_bf16
        self.device = device
        self._max_batch = int(max_batch)
        self.pad_to_compiled_buckets = bool(pad_to_compiled_buckets)
        self.n_slots = 2
        self.shape_ledger = SimpleNamespace(label=str(device))

    def evaluate_encoded(self, x, relations=None):
        del relations
        rows = int(x.shape[0])
        return np.zeros((rows, 4672), np.float32), np.zeros((rows, 3), np.float32)


@pytest.mark.usefixtures("cuda_names_on_cpu")
def test_chain_repeat_builds_without_graphs_and_honours_the_override(monkeypatch, capsys) -> None:
    monkeypatch.setattr(uci_main, "_REPEAT_NOTES_PRINTED", set())
    rec = _ModeRecorder()
    monkeypatch.setattr(torch, "compile", rec.compile)
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirect)
    monkeypatch.setattr(uci_main, "_warmup_evaluator", lambda *_a, **_kw: None)
    build = uci_main._make_evaluator_factory(
        [_model(), _model()], ["cuda:0", "cuda:0"], True, 1, 1, compile_mode="max-autotune",
    )
    chain = build(64)
    uci_main._close_chain(chain)
    assert rec.modes == ["max-autotune", "default"]
    out = capsys.readouterr().out
    assert "cuda:0#1 repeats a device already listed" in out
    source: Any = uci_main.find_degrade_source(chain)
    labels = [ev.shape_ledger.label for ev in source._evals]
    assert labels == ["cuda:0#0", "cuda:0#1"]

    rec.modes.clear()
    monkeypatch.setenv("CAE_UCI_MULTI_GPU_COMPILE_MODE", "max-autotune-no-cudagraphs")
    chain = build(64)
    uci_main._close_chain(chain)
    assert rec.modes == ["max-autotune-no-cudagraphs"] * 2  # announced AND applied


@pytest.mark.usefixtures("cuda_names_on_cpu")
def test_chain_cudagraph_warmup_failure_falls_back_to_no_graphs(monkeypatch, capsys) -> None:
    rec = _ModeRecorder()
    monkeypatch.setattr(torch, "compile", rec.compile)
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirect)
    closed: list[object] = []
    real_close = DegradingMultiGPUDispatcher.close

    def tracking_close(self) -> None:
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(DegradingMultiGPUDispatcher, "close", tracking_close)

    def warm(*_a, **_kw) -> None:
        if "max-autotune" in rec.modes[-2:]:
            raise RuntimeError("evaluator warmup failed at batch=1")

    monkeypatch.setattr(uci_main, "_warmup_evaluator", warm)
    build = uci_main._make_evaluator_factory(
        [_model(), _model()], ["cuda:0", "cuda:1"], True, 1, 1, compile_mode="max-autotune",
    )
    chain = build(64)
    try:
        assert rec.modes == ["max-autotune", "max-autotune", "default", "default"]
        assert len(closed) == 1  # the failed chain's device threads were stopped
        out = capsys.readouterr().out
        assert "cudagraph warmup failed on the routing chain" in out
        assert "routing chain fell back to compile mode 'default'" in out
    finally:
        uci_main._close_chain(chain)


@pytest.mark.usefixtures("cuda_names_on_cpu")
def test_single_gpu_warmup_failure_still_raises(monkeypatch) -> None:
    """No silent fallback on the single-device path: the rebuild fails and the
    caller keeps the previous evaluator (``Engine._rebuild_or_keep``)."""
    rec = _ModeRecorder()
    monkeypatch.setattr(torch, "compile", rec.compile)
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirect)

    def warm(*_a, **_kw) -> None:
        raise RuntimeError("evaluator warmup failed at batch=340")

    monkeypatch.setattr(uci_main, "_warmup_evaluator", warm)
    build = uci_main._make_evaluator_factory(
        [_model()], ["cuda:0"], True, 1, 1, compile_mode="max-autotune",
    )
    with pytest.raises(RuntimeError, match="batch=340"):
        build(64)
    assert rec.modes == ["max-autotune"]


@pytest.mark.usefixtures("cuda_names_on_cpu")
def test_pool_repeat_builds_without_graphs_and_names_the_worker(monkeypatch, capsys) -> None:
    monkeypatch.setattr(uci_main, "_REPEAT_NOTES_PRINTED", set())
    seen: list[tuple[str | None, str | None]] = []

    def fake(model, device, *, compile_mode, label=None, **_kw):
        del model, device
        seen.append((compile_mode, label))
        return SimpleNamespace()

    monkeypatch.setattr(uci_main, "_compile_and_warm_pucv_evaluator", fake)
    build = uci_main._make_multi_gpu_pucv_factory_builder(
        [_model(), _model(), _model()], ["cuda:0", "cuda:0", "cuda:1"], compile_mode="max-autotune",
    )
    for make in build(64, 32):
        make()
    assert seen == [("max-autotune", "cuda:0#0"), ("default", "cuda:0#1"), ("max-autotune", "cuda:1")]
    assert "cuda:0#1 repeats a device already listed" in capsys.readouterr().out


def test_pool_worker_retry_lines_name_the_worker(monkeypatch, capsys) -> None:
    attempts: list[str | None] = []
    monkeypatch.setattr(torch, "compile", lambda module, *, mode=None, **_kw: (attempts.append(mode), module)[1])

    def warm(_evaluator, **_kw):
        if attempts[-1] == "max-autotune":
            raise RuntimeError("operation failed due to a previous error during capture")
        return 0.01

    monkeypatch.setattr(uci_main, "_warmup_pucv_evaluator", warm)
    ev = uci_main._compile_and_warm_pucv_evaluator(
        _model(), "cpu", max_batch=64, gather=32, compile_mode="max-autotune",
        compact_bf16=False, input_history_encoding="legacy", input_extra_features="v1",
        use_relations=False, label="cuda:0#1",
    )
    out = capsys.readouterr().out
    assert attempts == ["max-autotune", "default"]
    assert ev.shape_ledger.label == "cuda:0#1"
    assert "compile/warmup failed on cuda:0#1 with mode 'max-autotune'" in out
    assert "cudagraph warmup failed on cuda:0#1; fell back to compile mode 'default'" in out


def test_a_startup_failure_on_the_chain_is_latched_as_a_startup_failure() -> None:
    """The quit line read 'failed mid-search' for a device that failed its warm-up."""
    class _Raises:
        def evaluate_encoded(self, _x, relations=None):
            del relations
            raise RuntimeError("CUBLAS_STATUS_NOT_INITIALIZED")

    class _Fine:
        def evaluate_encoded(self, x, relations=None):
            del relations
            return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)

    chain = DegradingMultiGPUDispatcher([_Raises(), _Fine()], labels=["cuda:0#0", "cuda:0#1"])
    try:
        chain.evaluate_encoded(np.zeros((1, 146, 8, 8), np.float32))
        assert wedge.device_fault() == "device cuda:0#0 failed at startup"
    finally:
        chain.close()


# --- GPU: the real cross-thread capture (skipped without CUDA) ----------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


_GPU_SCRIPT = r"""
import queue, sys, threading, torch
from chess_anti_engine import cudagraph_threads as cgt

mode = sys.argv[1]  # "prepared" | "unprepared"
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(),
                            torch.nn.Linear(256, 64)).cuda().eval()

class Owner:
    # A persistent thread running compiled forwards, like a UCI owner thread.
    def __init__(self, prepare):
        self.q = queue.Queue(); self.prepare = prepare
        threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        from chess_anti_engine.inference_dispatcher import bootstrap_cudagraph_tls
        bootstrap_cudagraph_tls()
        compiled = torch.compile(model, mode="reduce-overhead")
        while True:
            rows, box, done = self.q.get()
            try:
                if not self.prepare:  # the old behaviour: no handles made first
                    cgt.prepare_thread_for_capture = lambda device: None
                with cgt.compiled_forward_scope("cuda:0"), torch.no_grad():
                    box["r"] = round(float(compiled(torch.ones(rows, 64, device="cuda:0")).sum()), 3)
            except BaseException as exc:
                box["r"] = "ERR " + type(exc).__name__ + ": " + (str(exc).splitlines() or [""])[0]
            done.set()
    def call(self, rows):
        box = {}; done = threading.Event(); self.q.put((rows, box, done))
        return box["r"] if done.wait(600) else "ERR timeout"

a = Owner(prepare=True)
print("A8", a.call(8))              # warms shape 8 on thread A (A stays alive)
b = Owner(prepare=(mode == "prepared"))
print("B8", b.call(8))              # B records shape 8 through A's manager
print("B8", b.call(8))
print("A8", a.call(8))
print("A64", a.call(64))
print("A64", a.call(64))
print("RELEASED", len(cgt.released_captures))
"""


def _run_gpu_script(mode: str) -> list[str]:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    proc = subprocess.run(
        [sys.executable, "-c", _GPU_SCRIPT, mode], cwd=str(root), env=env,
        capture_output=True, text=True, timeout=900, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return [ln for ln in proc.stdout.splitlines() if ln[:1] in "AB" or ln.startswith("RELEASED")]


@gpu
def test_gpu_a_new_thread_records_a_shape_another_thread_warmed() -> None:
    """The R1 rebuild in a fresh process: thread A warms shape 8 and stays
    alive, a NEW thread records it through A's manager (the shared compiled
    graph binds the shape there). With the handles prepared it works and A
    keeps working; nothing needed releasing."""
    lines = _run_gpu_script("prepared")
    values = {ln.split()[1] for ln in lines if ln.startswith(("A8", "B8"))}
    assert len(values) == 1, lines
    assert not any("ERR" in ln for ln in lines), lines
    assert lines[-1] == "RELEASED 0"


@gpu
def test_gpu_a_failed_capture_leaves_the_previous_thread_working() -> None:
    """Without the preparation B's capture fails exactly as on the real GPU
    (cuBLAS handle created inside the capture); the released scope keeps A's
    graphs replaying and recording -- before, A then failed forever with
    'already recording to mempool_id'."""
    lines = _run_gpu_script("unprepared")
    assert lines[1].startswith("B8 ERR"), lines
    a_lines = [ln for ln in lines if ln.startswith("A")]
    assert not any("ERR" in ln for ln in a_lines), lines
    assert lines[-1].startswith("RELEASED "), lines
    assert lines[-1] != "RELEASED 0", lines


@pytest.mark.usefixtures("cuda_names_on_cpu")
def test_every_cuda_rebuild_first_returns_the_allocators_cached_blocks(monkeypatch) -> None:
    """Each evaluator launches on its own stream and the caching allocator
    reuses a freed block only on its stream: without this, each pool rebuild on
    the real GPU kept 0.3-0.8 GB more VRAM reserved (live evaluators flat)."""
    emptied: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # The cuBLAS workspaces first (a replaced stream's workspace stays
    # allocated, ~9 MB per rebuild), then the allocator's cached blocks.
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces",
                        lambda: emptied.append("workspaces"), raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append("empty"))
    monkeypatch.setattr(torch, "compile", lambda module, **_kw: module)
    monkeypatch.setattr(uci_main, "DirectGPUEvaluator", _FakeDirect)
    monkeypatch.setattr(uci_main, "_warmup_evaluator", lambda *_a, **_kw: None)
    build = uci_main._make_evaluator_factory(
        [_model()], ["cuda:0"], True, 1, 1, compile_mode="max-autotune",
    )
    for _ in range(3):
        uci_main._close_chain(build(64))
    assert emptied == ["workspaces", "empty"] * 3

    emptied.clear()
    monkeypatch.setattr(uci_main, "_compile_and_warm_pucv_evaluator",
                        lambda *_a, **_kw: SimpleNamespace())
    pool_build = uci_main._make_multi_gpu_pucv_factory_builder(
        [_model(), _model()], ["cuda:0", "cuda:1"], compile_mode="max-autotune",
    )
    for make in pool_build(64, 32):
        make()
    assert emptied == ["workspaces", "empty"] * 2

    emptied.clear()
    cpu_build = uci_main._make_evaluator_factory(
        [_model()], ["cpu"], True, 1, 1, compile_mode="max-autotune",
    )
    uci_main._close_chain(cpu_build(64))
    assert emptied == []  # nothing to release on a CPU-only stack
