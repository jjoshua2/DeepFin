from __future__ import annotations

import threading
from typing import Any

import pytest

from chess_anti_engine.uci.multi_gpu_pucv_pool import MultiGpuPucvConfig, MultiGpuPucvPool
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig
from tests.test_async_pipeline_safety import SentinelEvaluator, tree_root


class OwnedEvaluator(SentinelEvaluator):
    def __init__(self, fail_close=False):
        super().__init__('none')
        self.fail_close = fail_close
        self.closed = 0
        self.closed_on = None

    def close(self):
        self.closed += 1
        self.closed_on = threading.current_thread().name
        if self.fail_close:
            raise RuntimeError('secondary evaluator close failure')


def record_threads(monkeypatch):
    threads = []
    original = threading.Thread.start
    def start(thread):
        if thread.name.startswith(('pucv-gpu-', 'walker-')):
            threads.append(thread)
        original(thread)
    monkeypatch.setattr(threading.Thread, 'start', start)
    errors = []
    monkeypatch.setattr(threading, 'excepthook', lambda args: errors.append(args.exc_value))
    return threads, errors


def create_pool(kind, workers, evaluators, relations=False, factories=False):
    if kind == 'multi':
        cfg = MultiGpuPucvConfig(n_gpus=workers, gather=2, compute_relations=relations)
        if factories:
            return MultiGpuPucvPool(cfg, evaluator_factories=evaluators)
        return MultiGpuPucvPool(cfg, evaluators=evaluators)
    return WalkerPool(WalkerPoolConfig(n_walkers=workers, c_puct=1.4, fpu_at_root=0., fpu_reduction=.2, gather=2, compute_relations=relations), evaluators[0])


@pytest.mark.parametrize(('kind', 'stage'), [('multi', 'first'), ('multi', 'second'), ('walker', 'encoding'), ('walker', 'relations')])
@pytest.mark.parametrize('workers', [1, 2])
def test_constructor_buffer_failure_rendezvous(monkeypatch, kind, stage, workers):
    import chess_anti_engine.uci.multi_gpu_pucv_pool as multi
    import chess_anti_engine.uci.walker_pool as walker
    threads, errors = record_threads(monkeypatch)
    borrowed = [OwnedEvaluator() for _ in range(workers)]
    primary = MemoryError('tiny controlled per-worker startup allocation failure')
    counts = {}
    failed_worker = workers - 1
    if kind == 'multi':
        original = multi._alloc_buffers
        def allocate_multi_buffers(*args, **kwargs):
            name = threading.current_thread().name
            counts[name] = counts.get(name, 0) + 1
            if name == f'pucv-gpu-{failed_worker}' and counts[name] == (1 if stage == 'first' else 2):
                raise primary
            return original(*args, **kwargs)
        monkeypatch.setattr(multi, '_alloc_buffers', allocate_multi_buffers)
    else:
        name = 'empty' if stage == 'encoding' else 'zeros'
        original = getattr(walker.np, name)
        def allocate_walker_array(*args, **kwargs):
            if threading.current_thread().name == f'walker-{failed_worker}':
                raise primary
            return original(*args, **kwargs)
        monkeypatch.setattr(walker.np, name, allocate_walker_array)
    pool = None
    try:
        with pytest.raises(MemoryError) as exc:
            pool = create_pool(kind, workers, borrowed, relations=stage == 'relations')
        assert exc.value is primary
        assert all(not thread.is_alive() for thread in threads)
        assert not errors
        assert all(ev.closed == 0 for ev in borrowed)
    finally:
        if pool is not None:
            pool.close()
    # Recovery builds a separate pool after removing the fault, then completes
    # real native work. A rejected constructor never publishes a usable pool.
    monkeypatch.undo()
    fresh = create_pool(kind, workers, borrowed)
    try:
        tree, rid, cb = tree_root()
        assert fresh.run(tree=tree, root_id=rid, root_cboard=cb, target_sims=4, stop_event=threading.Event()) == 4
    finally:
        fresh.close()
    assert all(not thread.is_alive() for thread in fresh._threads)


@pytest.mark.parametrize('stage', ['buffers', 'factory', 'validation'])
@pytest.mark.parametrize('fail_close', [False, True])
def test_factory_startup_failure_retires_all_owned_evaluators(monkeypatch, stage, fail_close):
    import chess_anti_engine.uci.multi_gpu_pucv_pool as multi
    threads, errors = record_threads(monkeypatch)
    primary = MemoryError('tiny factory-owned startup failure')
    created = []
    def factory():
        idx = int(threading.current_thread().name.rsplit('-', 1)[1])
        if stage == 'factory' and idx == 1:
            raise primary
        ev = OwnedEvaluator(fail_close)
        created.append(ev)
        if stage == 'validation' and idx == 1:
            ev.n_slots = 1
        return ev
    if stage == 'buffers':
        original = multi._alloc_buffers
        def allocate(*args, **kwargs):
            if threading.current_thread().name == 'pucv-gpu-1':
                raise primary
            return original(*args, **kwargs)
        monkeypatch.setattr(multi, '_alloc_buffers', allocate)
    pool = None
    try:
        expected = ValueError if stage == 'validation' else MemoryError
        with pytest.raises(expected) as exc:
            pool = create_pool('multi', 2, [factory, factory], factories=True)
        if stage != 'validation':
            assert exc.value is primary
        assert all(not thread.is_alive() for thread in threads)
        assert not errors
        assert created
        assert all(ev.closed == 1 for ev in created)
        assert all(ev.closed_on.startswith('pucv-gpu-') for ev in created)
        if fail_close:
            assert any('secondary evaluator close' in n for n in getattr(exc.value, '__notes__', ()))
    finally:
        if pool is not None:
            pool.close()


@pytest.mark.parametrize('kind', ['multi', 'walker'])
@pytest.mark.parametrize('workers', [1, 2])
def test_clean_startup_run_stop_and_idempotent_close(kind, workers):
    borrowed = [OwnedEvaluator() for _ in range(workers)]
    pool: Any = create_pool(kind, workers, borrowed)
    args = tree_root()
    try:
        assert all(thread.is_alive() for thread in pool._threads)
        assert pool.run(tree=args[0], root_id=args[1], root_cboard=args[2], target_sims=4, stop_event=threading.Event()) == 4
        stop = threading.Event()
        stop.set()
        assert pool.run(tree=args[0], root_id=args[1], root_cboard=args[2], target_sims=4, stop_event=stop) == 0
    finally:
        pool.close()
    pool.close()
    assert all(not thread.is_alive() for thread in pool._threads)
    assert all(ev.closed == 0 for ev in borrowed)
    with pytest.raises(RuntimeError, match='closed'):
        pool.run(tree=args[0], root_id=args[1], root_cboard=args[2], target_sims=4, stop_event=threading.Event())
