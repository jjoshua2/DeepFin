"""Primary UCI startup ownership uses real dispatchers, workers and idle pools."""
from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from chess_anti_engine.inference_dispatcher import BatchCoalescingDispatcher
from chess_anti_engine.uci import __main__ as main
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.search import SearchWorker


class CpuEvaluator:
    n_slots = 2

    def __init__(self):
        self.close_calls = 0

    def evaluate_encoded(self, xs, relations=None):
        del relations
        return np.zeros((len(xs), 1858), np.float32), np.zeros((len(xs), 3), np.float32)

    def evaluate_inplace_async(self, batch, *, slot=0, relations=None):
        del slot, relations
        return np.zeros((batch, 1858), np.float32), np.zeros((batch, 3), np.float32), None

    def get_input_buffer(self, batch, *, slot=0):
        del slot
        return np.zeros((batch, 112, 8, 8), np.float32)

    def close(self):
        self.close_calls += 1


class CountingDispatcher(BatchCoalescingDispatcher):
    def __init__(self, *, fail_close=False):
        super().__init__(CpuEvaluator(), max_batch=64)
        self.close_calls = 0
        self.fail_close = fail_close

    def close(self):
        self.close_calls += 1
        super().close()
        if self.fail_close:
            raise RuntimeError("primary cleanup failure")


def record_workers(monkeypatch):
    workers = []

    def build(*args, **kwargs):
        worker = SearchWorker(*args, **kwargs)
        workers.append((worker, worker._walker_pool))
        return worker

    monkeypatch.setattr(main, "SearchWorker", build)
    return workers


def configure_main(monkeypatch, primary, factory_builder):
    monkeypatch.setattr(main, "_load_models", lambda *_a, **_kw: [SimpleNamespace(), SimpleNamespace()])
    monkeypatch.setattr(main, "_make_evaluator_factory", lambda *_a, **_kw: lambda *_b, **_k: primary)
    monkeypatch.setattr(main, "_make_multi_gpu_pucv_factory_builder", lambda *_a, **_kw: factory_builder)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "inert-placeholder", "--devices", "cpu,cpu", "--walkers", "2", "--no-compile", "--max-batch", "64", "--vl-gather", "32"])
    monkeypatch.setattr(main.sys, "stdin", iter(["isready\n", "quit\n"]))


def clean_probe(workers, primary):
    for worker, walker in workers:
        # Baseline failures retain idle threads; retire them after observations.
        if worker._pucv_pool is not None:
            worker._pucv_pool.close()
        if walker is not None:
            walker.close()
    if not primary._shutdown.is_set():
        try:
            primary.close()
        except RuntimeError:
            pass


@pytest.mark.parametrize("failed_index", [0, 1])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_main_secondary_factory_failure_releases_primary_once(monkeypatch, failed_index, cleanup_fails):
    primary = CountingDispatcher(fail_close=cleanup_fails)
    workers = record_workers(monkeypatch)
    helpers = [CpuEvaluator(), CpuEvaluator()]
    factory_threads = []
    error = RuntimeError("secondary factory failure")

    def factory(index):
        factory_threads.append(threading.current_thread())
        if index == failed_index:
            raise error
        return helpers[index]

    configure_main(monkeypatch, primary, lambda *_a, **_kw: [lambda: factory(0), lambda: factory(1)])
    try:
        with pytest.raises(RuntimeError) as caught:
            main.main()
        assert "secondary factory failure" in str(caught.value)
        assert primary.close_calls == 1
        assert primary._shutdown.is_set()
        assert not primary._submitter.is_alive()
        assert factory_threads
        assert not any(t.is_alive() for t in factory_threads)
        assert len(workers) == 1
        assert not any(t.is_alive() for t in workers[0][1]._threads)
        assert helpers[1 - failed_index].close_calls == 1
        assert helpers[failed_index].close_calls == 0
    finally:
        clean_probe(workers, primary)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_main_factory_builder_failure_retires_partial_worker_and_preserves_error(monkeypatch, cleanup_fails):
    primary = CountingDispatcher(fail_close=cleanup_fails)
    workers = record_workers(monkeypatch)
    error = RuntimeError("factory builder failure")
    real_close = SearchWorker.close

    def factory_builder(*_a, **_kw):
        raise error

    if cleanup_fails:
        def failed_cleanup(worker, **kwargs):
            real_close(worker, **kwargs)
            raise RuntimeError("worker cleanup failure")
        monkeypatch.setattr(SearchWorker, "close", failed_cleanup)
    configure_main(monkeypatch, primary, factory_builder)
    try:
        with pytest.raises(RuntimeError) as caught:
            main.main()
        assert caught.value is error
        assert primary.close_calls == 1
        assert primary._shutdown.is_set()
        assert not primary._submitter.is_alive()
        assert len(workers) == 1
        walker = workers[0][1]
        assert walker._shutdown.is_set()
        assert not any(t.is_alive() for t in walker._threads)
    finally:
        monkeypatch.setattr(SearchWorker, "close", real_close)
        clean_probe(workers, primary)


def test_main_success_transfers_primary_and_closes_owned_helpers_once(monkeypatch):
    primary = CountingDispatcher()
    helpers = [CpuEvaluator(), CpuEvaluator()]
    workers = record_workers(monkeypatch)
    factory_threads = []

    def factory(index):
        factory_threads.append(threading.current_thread())
        return helpers[index]

    configure_main(monkeypatch, primary, lambda *_a, **_kw: [lambda: factory(0), lambda: factory(1)])
    monkeypatch.setattr(Engine, "warmup_search", lambda _self: None)
    try:
        assert main.main() == 0
        assert primary.close_calls == 1
        assert primary._shutdown.is_set()
        assert not primary._submitter.is_alive()
        assert [ev.close_calls for ev in helpers] == [1, 1]
        assert not any(t.is_alive() for t in factory_threads)
        assert not any(t.is_alive() for t in workers[0][1]._threads)
    finally:
        clean_probe(workers, primary)


def test_engine_close_retires_owned_idle_walkers():
    primary = CountingDispatcher()
    worker = SearchWorker(primary, device="cpu", n_walkers=2)
    pool = worker._walker_pool
    assert pool is not None
    engine = Engine(worker)
    try:
        engine.close()
        engine.close()
        assert primary.close_calls == 1
        assert pool._shutdown.is_set()
        assert not any(t.is_alive() for t in pool._threads)
    finally:
        pool.close()
        if not primary._shutdown.is_set():
            primary.close()

def test_build_failure_retires_helpers_but_retains_callers_primary(monkeypatch):
    primary = CountingDispatcher()
    workers = record_workers(monkeypatch)
    error = RuntimeError("factory builder failure")

    def factory_builder(*_a, **_kw):
        raise error

    try:
        with pytest.raises(RuntimeError) as caught:
            main._build_engine(evaluator=primary, primary_device="cpu", chunk_sims=16,
                               topk=4, n_walkers=2, vloss_weight=1, walker_gather=1,
                               pucv_vloss_mode=0, max_batch=64, vl_gather=32,
                               eval_cache_entries=0, use_multi_gpu_pucv=True,
                               rebuild_multi_gpu_pucv_factories=factory_builder)
        assert caught.value is error
        assert primary.close_calls == 0
        assert not primary._shutdown.is_set()
        assert primary._submitter.is_alive()
        assert workers[0][1]._shutdown.is_set()
        assert not any(t.is_alive() for t in workers[0][1]._threads)
    finally:
        clean_probe(workers, primary)
