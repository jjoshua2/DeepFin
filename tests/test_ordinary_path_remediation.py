"""CPU fault injection through real ordinary-path consumers; no CUDA qualification."""
from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import chess_anti_engine.async_safety as safety
import chess_anti_engine.inference as inference
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci import __main__ as main
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.root_parallel_gumbel import RootParallelGumbelConfig, RootParallelGumbelPool
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig


class _LoanStream:
    def __init__(self, events, *, fail_drain, error, stage):
        self.events = events
        self.fail_drain, self.error, self.stage = fail_drain, error, stage
        self.loans = []
        self.calls = 0
        self.name = "launch"

    def hold(self, array):
        self.loans.append((array, array.copy()))

    def synchronize(self):
        self.calls += 1
        self.events.append("drain")
        assert all(np.array_equal(a, saved) for a, saved in self.loans)
        if self.stage == "sync" and self.calls == 1:
            raise self.error
        if self.fail_drain:
            raise RuntimeError("secondary drain failure")
        self.loans.clear()
        self.events.append(f"complete_{self.name}")


class _InputProxy:
    def __init__(self, array, stream, events, *, stage, error, name="h2d"):
        self.array, self.events = array, events
        self.stream: Any = stream
        self.stage, self.error, self.name = stage, error, name
        self.shape = array.shape

    def __getitem__(self, index):
        return _InputProxy(self.array[index], self.stream, self.events,
                          stage=self.stage, error=self.error, name=self.name)

    def to(self, _device, *, non_blocking):
        assert non_blocking
        self.events.append(self.name)
        stream = cast(_LoanStream, self.stream() if callable(self.stream) else self.stream)
        stream.hold(self.array)
        if self.stage == self.name:
            raise self.error
        return torch.from_numpy(self.array.astype(np.float32) if self.array.dtype == np.uint16 else self.array)


class _Watchdog:
    def __init__(self, events):
        self.events, self.pending = events, set()

    def stage(self, _name):
        return contextlib.nullcontext()

    def mark_forward_start(self, _rows):
        token = object()
        self.pending.add(token)
        self.events.append("watch_start")
        return token

    def mark_forward_done(self, *, success, token):
        del success
        self.events.append("watch_done")
        self.pending.remove(token)


class _Slot(SimpleNamespace):
    _state: int = inference._STATE_REQUEST

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if value == inference._STATE_IDLE:
            self.events.append("release")
        if value == inference._STATE_RESPONSE:
            self.events.append("response")
        self._state = value


def _broker(monkeypatch, *, mode, stage, fail_drain):
    events = []
    error = RuntimeError(f"primary {stage} failure")
    stream = _LoanStream(events, fail_drain=fail_drain, error=error, stage=stage)
    broker: Any = object.__new__(inference.SlotBroker)
    broker.device = "cuda:0"
    broker._layout = SimpleNamespace(max_batch=2)
    broker._pinned_input_np = np.zeros((2, 1, 8, 8), np.float32)
    broker._pinned_input_bf16_bits_np = np.zeros((2, 1, 8, 8), np.uint16)
    broker._pinned_input = _InputProxy(broker._pinned_input_np, stream, events, stage=stage, error=error)
    broker._pinned_input_bf16 = _InputProxy(broker._pinned_input_bf16_bits_np, stream, events, stage=stage, error=error)
    broker._pinned_wdl = torch.zeros((2, 3))
    broker._pinned_wdl_np = broker._pinned_wdl.numpy()
    broker._pinned_pol = torch.zeros((2, 4672))
    broker._pinned_pol_np = broker._pinned_pol.numpy()
    broker._pinned_legal_pol_bf16_bits = torch.zeros(4, dtype=torch.uint16)
    broker._pinned_legal_pol_bf16_bits_np = broker._pinned_legal_pol_bf16_bits.numpy()
    for name in ("flat", "rows"):
        array = np.zeros(4, np.int64)
        setattr(broker, f"_pinned_legal_{name}_np", array)
        setattr(broker, f"_pinned_legal_{name}", _InputProxy(array, stream, events, stage=stage, error=error, name=f"metadata_{name}"))
    broker._hang_watchdog = _Watchdog(events)
    broker._model = torch.nn.Module()
    broker._ensure_model = lambda: events.append("ensure_model")
    broker._aot_models = None
    broker._first_inference_pending = False
    broker.compile_inference = False
    broker._consecutive_batch_failures = 0
    broker._malformed_legal_meta_total = 0
    broker._malformed_legal_meta_warned_at = 0.
    broker._stop = False
    # Changing current_stream after launch must not redirect completion/drain.
    calls = []
    other = _LoanStream(events, fail_drain=False, error=error, stage="none")
    other.name = "output"
    current = [stream]
    for name in ("_pinned_input", "_pinned_input_bf16", "_pinned_legal_flat", "_pinned_legal_rows"):
        getattr(broker, name).stream = lambda: current[0]
    def current_stream(*_args):
        calls.append(1)
        return current[0]
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    def forward(*_args, **_kwargs):
        events.append("forward")
        current[0] = other
        if stage == "forward":
            raise error
        return {"policy": torch.zeros((2, 4672)), "wdl": torch.zeros((2, 3))}
    monkeypatch.setattr(inference, "_forward_no_grad", forward)
    copy = torch.Tensor.copy_
    def copy_(tensor, source, **kwargs):
        if stage == "output" and tensor.data_ptr() == broker._pinned_wdl.data_ptr():
            current[0].hold(tensor.numpy())
            raise error
        result = copy(tensor, source, **kwargs)
        current[0].hold(tensor.numpy())
        return result
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    slot = _Slot(events=events, _state=inference._STATE_REQUEST, batch_size=1,
                 request_id=7, request_mode=mode, input=np.full((2, 1, 8, 8), 3., np.float32),
                 input_bf16_bits=np.full((2, 1, 8, 8), 3, np.uint16),
                 policy=np.full((2, 4672), 19., np.float32), wdl=np.full((2, 3), 19., np.float32),
                 policy_i32=np.array([1, 1, 0], np.int32), policy_u16=np.zeros(4, np.uint16))
    return broker, slot, stream, other, events, error, calls


_FAILURES = [(mode, stage) for mode in (inference._MODE_DENSE_F32, inference._MODE_DENSE_BF16, inference._MODE_LEGAL_BF16)
             for stage in ("h2d", "forward", "output", "sync")]
_FAILURES += [(inference._MODE_LEGAL_BF16, stage) for stage in ("metadata_flat", "metadata_rows")]


@pytest.mark.parametrize(("mode", "stage"), _FAILURES)
@pytest.mark.parametrize("fail_drain", [False, True])
def test_slot_broker_failed_launch_drains_before_retry_or_retires(monkeypatch, mode, stage, fail_drain):
    broker, slot, stream, other, events, error, calls = _broker(monkeypatch, mode=mode, stage=stage, fail_drain=fail_drain)
    if fail_drain:
        with pytest.raises(RuntimeError) as caught:
            broker._process_batch([slot])
        assert caught.value is error
        assert broker._stop
        assert slot.state == inference._STATE_REQUEST
        assert "release" not in events
        assert "watch_done" not in events
        assert broker._hang_watchdog.pending
        assert stream.loans
        assert any(entry[0] is broker for entry in safety._quarantine)
        assert any("backend poisoned" in note for note in getattr(error, "__notes__", ()))
        before = len(events)
        with pytest.raises(RuntimeError, match="poisoned"):
            broker._process_batch([slot])
        with pytest.raises(RuntimeError, match="poisoned"):
            broker._process_batch_mode([slot], mode=mode, request_ids=[7])
        assert len(events) == before
        assert all(np.array_equal(array, saved) for array, saved in stream.loans)
    else:
        assert broker._process_batch([slot]) == 1
        assert slot.state == inference._STATE_IDLE
        assert events.index("drain") < events.index("watch_done") < events.index("release")
        assert not stream.loans
        assert not broker._hang_watchdog.pending
        safety.ensure_usable(broker)
        # A subsequent real pack/launch is safe after successful drain.
        stream.stage = "none"
        broker._pinned_input.stage = "none"
        broker._pinned_input.stream = stream
        slot.input.fill(7)
        slot.input_bf16_bits.fill(7)
        slot.request_mode = inference._MODE_DENSE_F32
        slot._state = inference._STATE_REQUEST
        # Remove fault injections while keeping the same broker storage.
        monkeypatch.undo()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a: stream)
        monkeypatch.setattr(inference, "_forward_no_grad", lambda *_a, **_k: {"policy": torch.zeros((2, 4672)), "wdl": torch.zeros((2, 3))})
        assert broker._process_batch([slot]) == 0
        assert slot.state == inference._STATE_RESPONSE
        assert not stream.loans
    if stage in ("h2d", "forward"):
        assert other.calls == 0
        assert len(calls) == 1
    else:
        assert other.calls >= 1
        assert len(calls) == 2
    assert not other.loans


@pytest.mark.parametrize("mode", [inference._MODE_DENSE_F32, inference._MODE_DENSE_BF16, inference._MODE_LEGAL_BF16])
def test_slot_broker_healthy_launch_completes_on_captured_stream(monkeypatch, mode):
    broker, slot, stream, other, events, _error, calls = _broker(monkeypatch, mode=mode, stage="none", fail_drain=False)
    assert broker._process_batch([slot]) == 0
    assert slot.state == inference._STATE_RESPONSE
    assert stream.calls == 1
    assert other.calls == 1
    assert len(calls) == 2
    assert not stream.loans
    assert not other.loans
    assert not broker._hang_watchdog.pending
    assert events.index("drain") < events.index("watch_done")
    assert events.index("complete_launch") < events.index("response")
    assert events.index("complete_output") < events.index("response")


@pytest.mark.parametrize("also_fail_launch", [False, True])
def test_slot_broker_output_stream_drain_failure_retains_custody(monkeypatch, also_fail_launch):
    broker, slot, stream, other, events, error, _calls = _broker(
        monkeypatch, mode=inference._MODE_LEGAL_BF16, stage="output", fail_drain=also_fail_launch,
    )
    other.fail_drain = True
    with pytest.raises(RuntimeError) as caught:
        broker._process_batch([slot])
    assert caught.value is error
    assert stream.calls == 1
    assert other.calls == 1  # Both drains attempted.
    assert other.loans
    assert broker._hang_watchdog.pending
    assert "response" not in events
    assert "release" not in events
    assert "watch_done" not in events
    assert broker._stop
    assert any(entry[0] is broker for entry in safety._quarantine)
    with pytest.raises(RuntimeError, match="poisoned"):
        broker._process_batch([slot])


@pytest.mark.parametrize("persistent", [False, True])
def test_slot_broker_output_completion_error_preserves_primary_and_all_drains(monkeypatch, persistent):
    broker, slot, stream, other, events, _error, _calls = _broker(
        monkeypatch, mode=inference._MODE_DENSE_F32, stage="none", fail_drain=False,
    )
    error = RuntimeError("primary output completion failure")
    other.stage, other.error, other.fail_drain = "sync", error, persistent
    if persistent:
        with pytest.raises(RuntimeError) as caught:
            broker._process_batch([slot])
        assert caught.value is error
        assert other.loans
        assert broker._hang_watchdog.pending
        assert broker._stop
        assert "release" not in events
        assert "watch_done" not in events
    else:
        assert broker._process_batch([slot]) == 1
        assert slot.state == inference._STATE_IDLE
        assert not other.loans
        assert not broker._hang_watchdog.pending
        assert events.index("complete_output") < events.index("watch_done") < events.index("release")
    assert stream.calls == 2
    assert other.calls == 2
    assert "response" not in events


@pytest.mark.parametrize("stage", ["metadata_flat", "metadata_rows", "forward", "output"])
@pytest.mark.parametrize("fail_drain", [False, True])
def test_slot_broker_legal_rows_launch_retains_metadata_and_primary(monkeypatch, stage, fail_drain):
    broker, slot, stream, other, events, error, _calls = _broker(
        monkeypatch, mode=inference._MODE_LEGAL_BF16, stage=stage, fail_drain=fail_drain,
    )
    broker._model.forward_legal_policy_rows = lambda *_a: None
    def legal_rows(*_args, **_kwargs):
        events.append("legal_rows_forward")
        if stage == "forward":
            raise error
        return {"policy": torch.zeros(1), "wdl": torch.zeros((2, 3))}
    monkeypatch.setattr(inference, "_forward_legal_rows_no_grad", legal_rows)
    with pytest.raises(RuntimeError) as caught:
        broker._process_batch_mode([slot], mode=inference._MODE_LEGAL_BF16, request_ids=[7])
    assert caught.value is error
    assert other.calls == 0
    assert "metadata_flat" in events
    if stage != "metadata_flat":
        assert "metadata_rows" in events
    if fail_drain:
        assert stream.loans
        assert broker._hang_watchdog.pending
        assert "watch_done" not in events
        with pytest.raises(RuntimeError, match="poisoned"):
            broker._process_batch([slot])
    else:
        assert not stream.loans
        assert not broker._hang_watchdog.pending
        assert events.index("drain") < events.index("watch_done")
        safety.ensure_usable(broker)


class _CpuEvaluator:
    n_slots = 2

    def __init__(self, *, cleanup_fails=False):
        self.closed = []
        self.cleanup_fails = cleanup_fails

    def evaluate_encoded(self, x, relations=None):
        del relations
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)

    def evaluate_inplace_async(self, batch, **kwargs):
        del kwargs
        return np.zeros((batch, 4672), np.float32), np.zeros((batch, 3), np.float32), None

    def close(self):
        self.closed.append(threading.current_thread())
        if self.cleanup_fails:
            raise RuntimeError("secondary evaluator cleanup failure")


@pytest.mark.parametrize("kind", ["walker", "rpg_owned", "rpg_borrowed"])
@pytest.mark.parametrize("failed_index", [0, 1, 2])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_thread_start_unwinds_started_workers(monkeypatch, kind, failed_index, cleanup_fails):
    primary = RuntimeError("primary Thread.start failure")
    evaluators = [_CpuEvaluator(cleanup_fails=cleanup_fails) for _ in range(3)]
    workers = []
    captured = []
    initialized = threading.Event()
    start = threading.Thread.start
    barrier_wait = threading.Barrier.wait
    def wait(barrier, *args, **kwargs):
        if threading.current_thread() in workers:
            initialized.set()
        return barrier_wait(barrier, *args, **kwargs)
    monkeypatch.setattr(threading.Barrier, "wait", wait)
    prefix = "walker-" if kind == "walker" else "rpg-group-"
    def fail_start(thread):
        if thread.name.startswith(prefix):
            captured.append(thread._target.__self__)
            index = int(thread.name.rsplit("-", 1)[1])
            if index == failed_index:
                if workers:
                    assert initialized.wait(3), "started worker never reached initialization rendezvous"
                raise primary
            workers.append(thread)
        return start(thread)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    def construct():
        if kind == "walker":
            return WalkerPool(WalkerPoolConfig(n_walkers=3, c_puct=1., fpu_at_root=0., fpu_reduction=0., input_planes=1), evaluators[0])
        if kind == "rpg_owned":
            return RootParallelGumbelPool(RootParallelGumbelConfig(n_groups=3, gather=1, input_planes=1), GumbelConfig(),
                                          evaluator_factories=[lambda ev=ev: ev for ev in evaluators])
        return RootParallelGumbelPool(RootParallelGumbelConfig(n_groups=3, gather=1, input_planes=1), GumbelConfig(), evaluators=evaluators)
    try:
        with pytest.raises(RuntimeError) as caught:
            construct()
        assert caught.value is primary
        pool = captured[0]
        assert pool._shutdown.is_set()
        assert pool._init_done.broken
        assert not any(thread.is_alive() for thread in workers)
        assert len(pool._started_threads) == failed_index
        if kind == "rpg_owned":
            for index, ev in enumerate(evaluators):
                assert ev.closed == ([pool._threads[index]] if index < failed_index else [])
            assert all(ch is None for ch in pool._chunkers)
            assert all(ev is None for ev in pool._evals)
            if cleanup_fails and failed_index:
                assert any("startup evaluator cleanup failed" in note for note in getattr(primary, "__notes__", ()))
        else:
            assert all(not ev.closed for ev in evaluators)
    finally:
        # Baseline controls can leak at the barrier; bounded harness-only cleanup.
        for pool in captured[:1]:
            pool._init_done.abort()
            pool._shutdown.set()
            for wake in getattr(pool, "_wakes", []):
                wake.set()
            for thread in workers:
                thread.join(timeout=3)


@pytest.mark.parametrize("kind", ["walker", "rpg"])
def test_successful_pool_start_and_close_preserves_borrowed_owner(kind):
    evaluator = _CpuEvaluator()
    if kind == "walker":
        pool = WalkerPool(WalkerPoolConfig(n_walkers=2, c_puct=1., fpu_at_root=0., fpu_reduction=0., input_planes=1), evaluator)
    else:
        pool = RootParallelGumbelPool(RootParallelGumbelConfig(n_groups=2, gather=1, input_planes=1), GumbelConfig(), evaluators=[evaluator, evaluator])
    pool.close()
    pool.close()
    assert not any(thread.is_alive() for thread in pool._threads)
    assert not evaluator.closed


@pytest.mark.parametrize("method", ["clear", "close"])
def test_successful_rpg_startup_transfers_close_ownership_before_worker_resumes(monkeypatch, method):
    helpers = [_CpuEvaluator(), _CpuEvaluator()]
    primary = _CpuEvaluator()
    caller = threading.current_thread()
    paused = [threading.Event(), threading.Event()]
    release = threading.Event()
    held_threads = []
    real_wait = threading.Barrier.wait
    real_join = threading.Thread.join
    worker = SearchWorker(primary, device="cpu", n_walkers=1, chunk_sims=8)
    pool = None

    def wait(barrier, *args, **kwargs):
        result = real_wait(barrier, *args, **kwargs)
        thread = threading.current_thread()
        owner = getattr(getattr(thread, "_target", None), "__self__", None)
        if isinstance(owner, RootParallelGumbelPool) and barrier is owner._init_done:
            held_threads.append(thread)
            paused[int(thread.name.rsplit("-", 1)[1])].set()
            assert release.wait(3), "caller did not reach pool join after ownership transfer"
        return result

    def join(thread, *args, **kwargs):
        if thread in held_threads:
            # close() sets shutdown before joining; let the actual worker
            # observe that shutdown immediately after its startup barrier.
            assert pool is not None
            assert pool._shutdown.is_set()
            release.set()
        return real_join(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Barrier, "wait", wait)
    monkeypatch.setattr(threading.Thread, "join", join)
    try:
        worker.install_root_parallel_gumbel([lambda ev=ev: ev for ev in helpers], gather=1, as_factories=True)
        pool = worker._rpg_pool
        assert pool is not None
        assert all(event.wait(3) for event in paused)
        # The actual caller has copied the successful constructor's owners,
        # while both group threads remain paused after barrier release.
        assert worker._rpg_pool_evals == helpers
        assert not any(ev.closed for ev in helpers)
        if method == "clear":
            worker.clear_root_parallel_gumbel()
        else:
            worker.close()
        assert all(ev.closed == [caller] for ev in helpers)
        assert not any(thread.is_alive() for thread in held_threads)
        worker.clear_root_parallel_gumbel()
        worker.close()
        assert all(ev.closed == [caller] for ev in helpers)
    finally:
        release.set()
        worker.close()
        if pool is not None:
            pool.close()
        for thread in held_threads:
            real_join(thread, timeout=3)


class _TinyModel(torch.nn.Module):
    input_planes = 146

    def forward(self, x):
        return {"policy": torch.zeros((len(x), 4672)), "wdl": torch.zeros((len(x), 3))}


@pytest.mark.parametrize("requested", [0, 1, 64, 128])
def test_main_accepted_max_batch_reaches_actual_factory_and_consumer(monkeypatch, requested):
    capacities = []
    real_build = main._build_engine
    def build(**kwargs):
        engine = real_build(**kwargs)
        capacities.append((engine._options.max_batch, kwargs["evaluator"]._eval._max_batch))
        # Exercise the actual evaluator consumer before the main loop's equal
        # MaxBatch setoption; a zero-capacity baseline cannot answer this row.
        policy, wdl = kwargs["evaluator"].evaluate_encoded(np.zeros((1, 146, 8, 8), np.float32))
        assert policy.shape == (1, 4672)
        assert wdl.shape == (1, 3)
        return engine
    monkeypatch.setattr(main, "_build_engine", build)
    monkeypatch.setattr(main, "_load_models", lambda *_a: [_TinyModel()])
    monkeypatch.setattr(Engine, "warmup_search", lambda _self: None)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "inert-placeholder", "--device", "cpu", "--walkers", "1", "--no-compile", "--max-batch", str(requested)])
    monkeypatch.setattr(main.sys, "stdin", iter(["isready\n", f"setoption name MaxBatch value {max(64, requested)}\n", "quit\n"]))
    assert main.main() == 0
    assert capacities == [(max(64, requested), max(64, requested))]
