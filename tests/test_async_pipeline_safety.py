from __future__ import annotations

import contextlib
import threading
import types
from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

import chess_anti_engine.inference as inference
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct_vl import PucvChunker
from chess_anti_engine.uci.multi_gpu_pucv_pool import MultiGpuPucvConfig, MultiGpuPucvPool


def tree_root(board=None):
    cb = CBoard.from_board(chess.Board() if board is None else board)
    tree = MCTSTree()
    tree.reserve(50000, 500000)
    root = tree.add_root(1, 0.)
    legal = cb.legal_move_indices().astype(np.int32)
    tree.expand(root, legal, np.full(len(legal), 1 / len(legal), dtype=np.float64))
    return tree, root, cb


class HeldEvent:
    def __init__(self, ev, slot, bsz, failure):
        self.ev, self.slot, self.bsz, self.failure = ev, slot, bsz, failure
        self.snapshot = ev.bufs[slot][:bsz].copy()
        self.synced = False
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1
        if self.failure == 'always' or (self.failure == 'once' and self.sync_calls == 1):
            raise TimeoutError('primary receipt failure')
        assert np.array_equal(self.snapshot, self.ev.bufs[self.slot][:self.bsz])
        self.synced = True


class SentinelEvaluator:
    n_slots = 2

    def __init__(self, mode):
        self.mode = mode
        self.bufs = [np.zeros((4, 146, 8, 8), np.float32) for _ in range(2)]
        self.events = []
        self.calls = 0
        self.getters = 0

    def get_input_buffer(self, bsz, slot=0):
        self.getters += 1
        assert all(e.synced or e.slot != slot for e in self.events)
        return self.bufs[slot][:bsz]

    def evaluate_encoded(self, x, relations=None):
        del relations
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)

    def evaluate_inplace_async(self, bsz, *, slot=0, relations=None):
        del relations
        self.calls += 1
        if self.calls == 2 and self.mode.startswith('submit'):
            error = TimeoutError('primary next submission failure')
            if self.mode == 'submit_safe':
                setattr(error, '_async_quiesced_owners', (id(self),))
            raise error
        failure = ('once' if self.mode == 'finish_once' else 'always') if self.calls == 1 and self.mode.startswith('finish') else ''
        event = HeldEvent(self, slot, bsz, failure)
        self.events.append(event)
        return np.zeros((bsz, 4672), np.float32), np.zeros((bsz, 3), np.float32), event


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('cache', [0, 32])
@pytest.mark.parametrize('mode', ['submit_safe', 'submit_unknown', 'finish_once', 'finish_always'])
def test_real_pipeline_failure_receipts(multi, cache, mode):
    ev = SentinelEvaluator(mode)
    runner: Any
    if multi:
        runner = MultiGpuPucvPool(MultiGpuPucvConfig(n_gpus=1, gather=4, eval_cache_entries=cache), evaluators=[ev])
        def call_pool(args):
            return runner.run(tree=args[0], root_id=args[1], root_cboard=args[2], target_sims=8, stop_event=threading.Event())
        call = call_pool
    else:
        runner = PucvChunker(ev, gather=4, eval_cache_entries=cache)
        def call_single(args):
            return runner.run(*args, 8)
        call = call_single
    try:
        args = tree_root()
        with pytest.raises(TimeoutError, match='primary'):
            call(args)
        assert max(args[0].get_virtual_loss(i) for i in range(args[0].node_count())) == 0
        assert all(e.synced for e in ev.events if e.failure != 'always')
        if mode in ('submit_unknown', 'finish_always'):
            getters = ev.getters
            with pytest.raises(RuntimeError, match='poisoned'):
                call(tree_root())
            fresh = PucvChunker(ev, gather=4)
            with pytest.raises(RuntimeError, match='poisoned'):
                fresh.run(*tree_root(), 4)
            assert ev.getters == getters
        else:
            ev.mode = 'none'
            assert call(tree_root()) == 8
            assert all(e.synced for e in ev.events)
    finally:
        if multi:
            runner.close()


class HeldStream:
    def __init__(self, fail=False):
        self.held = []
        self.synchronizations = 0
        self.fail = fail

    def wait_event(self, event):
        del event

    def hold(self, view):
        self.held.append((view, view.copy()))

    def synchronize(self):
        self.synchronizations += 1
        if self.fail:
            raise RuntimeError('cleanup drain failure')
        self.held.clear()


class Event:
    def record(self, stream=None):
        del stream

    def synchronize(self):
        pass


METHODS = [('direct', m) for m in ('evaluate_encoded', 'evaluate_inplace', 'evaluate_encoded_async', 'evaluate_inplace_async', 'evaluate_inplace_legal_bf16_async', 'evaluate_legal_bf16')] + [('aot', m) for m in ('evaluate_encoded', 'evaluate_encoded_async', 'evaluate_inplace_async')]
METHOD_STAGES = [(kind, method, stage) for kind, method in METHODS for stage in ('h2d', 'forward', 'd2h')]
METHOD_STAGES += [(kind, method, 'receipt') for kind, method in METHODS if 'async' in method or method == 'evaluate_legal_bf16']


@pytest.mark.parametrize(('kind', 'method', 'stage'), METHOD_STAGES)
@pytest.mark.parametrize('fail_drain', [False, True])
def test_public_backend_pre_receipt_failure(monkeypatch, kind, method, stage, fail_drain):
    launch = HeldStream(fail_drain)
    default = HeldStream() if kind == 'direct' and ('async' in method or method == 'evaluate_legal_bf16') else launch
    monkeypatch.setattr(torch, 'cuda', types.SimpleNamespace(Event=Event, Stream=lambda **kw: launch, current_stream=lambda *a, **kw: default, stream=lambda s: contextlib.nullcontext()))
    original_as_tensor = torch.as_tensor
    monkeypatch.setattr(torch, 'as_tensor', lambda *a, **kw: original_as_tensor(*a, **{**kw, 'device': 'cpu'}))
    original_arange = torch.arange
    monkeypatch.setattr(torch, 'arange', lambda *a, **kw: original_arange(*a, **{**kw, 'device': 'cpu'}))
    def model(x, **_kwargs):
        if stage == 'forward':
            raise RuntimeError('primary forward failure')
        return {'policy': torch.zeros((len(x), 4672)), 'wdl': torch.zeros((len(x), 3))}
    monkeypatch.setattr(inference, '_forward_no_grad', lambda mod, x, **kw: mod(x))
    # Fake CPU models/streams intentionally replace CUDA launches, while the
    # public evaluator and helper methods below are the real implementations.
    ev: Any
    if kind == 'direct':
        ev = inference.DirectGPUEvaluator(cast(Any, model), device='cpu', max_batch=2, n_slots=2, use_amp=False)
        ev.device = 'cuda:0'
        ev._use_cuda = True
        ev._stream = launch
        view = ev.get_input_buffer(2, slot=0)
        def direct_input(bsz, *, slot):
            row = ev._pinned_inputs_np[slot][:bsz]
            launch.hold(row)
            if stage == 'h2d':
                raise RuntimeError('primary input transfer failure')
            return torch.from_numpy(row)
        ev._device_input = direct_input
    else:
        ev = object.__new__(inference.AOTEvaluator)
        ev._ownership = threading.RLock()
        ev._pending = set()
        ev.device, ev._max_batch, ev._n_slots = 'cuda:0', 2, 2
        ev._models, ev._sorted_buckets = {2: model}, [2]
        ev._slot_input_bf16 = [False, False]
        ev._pinned_inputs_np = [np.zeros((2, 146, 8, 8), np.float32) for _ in range(2)]
        view = ev.get_input_buffer(2, slot=0)
        def aot_input(x=None, *, bucket, slot=0):
            if x is not None:
                ev._pinned_inputs_np[slot][:len(x)] = x
            row = ev._pinned_inputs_np[slot][:bucket]
            launch.hold(row)
            if stage == 'h2d':
                raise RuntimeError('primary input transfer failure')
            return torch.from_numpy(row)
        ev._device_input = aot_input
        ev._device_input_inplace = aot_input
    view.fill(3)
    if stage == 'receipt':
        monkeypatch.setattr(inference, 'guarded_event', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('primary receipt construction failure')))
    if stage == 'd2h':
        if kind == 'direct':
            original_copy = torch.Tensor.copy_
            def copy_(tensor, source, **kwargs):
                if tensor.data_ptr() == ev._pinned_wdls[0].data_ptr():
                    raise RuntimeError('primary output transfer failure')
                return original_copy(tensor, source, **kwargs)
            monkeypatch.setattr(torch.Tensor, 'copy_', copy_)
        else:
            original_to = torch.Tensor.to
            original_cpu = torch.Tensor.cpu
            def to(tensor, *args, **kwargs):
                if kwargs.get('device') == 'cpu' and tensor.shape[-1] == 3:
                    raise RuntimeError('primary output transfer failure')
                return original_to(tensor, *args, **kwargs)
            def cpu(tensor, *args, **kwargs):
                if tensor.shape[-1] == 3:
                    raise RuntimeError('primary output transfer failure')
                return original_cpu(tensor, *args, **kwargs)
            monkeypatch.setattr(torch.Tensor, 'to', to)
            monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    def call():
        if method.startswith('evaluate_encoded'):
            return getattr(ev, method)(view.copy())
        if method == 'evaluate_legal_bf16':
            return ev.evaluate_legal_bf16(view.copy(), np.array([1, 2], np.int32), np.array([1, 1], np.int32))
        if method == 'evaluate_inplace_legal_bf16_async':
            return ev.evaluate_inplace_legal_bf16_async(2, np.array([1, 2], np.int32), np.array([1, 1], np.int32), slot=0)
        return getattr(ev, method)(2, slot=0)
    with pytest.raises(RuntimeError, match='primary') as exc:
        call()
    assert launch.synchronizations == 1
    assert default is launch or default.synchronizations == 0
    if fail_drain:
        assert len(launch.held) == 1
        assert any('backend poisoned' in n for n in getattr(exc.value, '__notes__', ()))
        with pytest.raises(RuntimeError, match='poisoned'):
            ev.get_input_buffer(2, slot=0)
        with pytest.raises(RuntimeError, match='poisoned'):
            ev.get_input_buffer_bf16_bits(2, slot=0)
        with pytest.raises(RuntimeError, match='poisoned'):
            call()
        fresh = PucvChunker(ev, gather=2)
        with pytest.raises(RuntimeError, match='poisoned'):
            fresh.run(*tree_root(), 2)
        assert np.array_equal(view, launch.held[0][1])
    else:
        assert not launch.held
        ev.get_input_buffer(2, slot=0).fill(7)


@pytest.mark.parametrize('fail_drain', [False, True])
def test_returned_cuda_receipt_preserves_event_error(monkeypatch, fail_drain):
    from chess_anti_engine.async_safety import ensure_usable
    launch = HeldStream(fail_drain)
    default = HeldStream()
    event_error = RuntimeError('primary completion error')
    class FailedEvent(Event):
        def synchronize(self):
            raise event_error
    monkeypatch.setattr(torch, 'cuda', types.SimpleNamespace(Event=FailedEvent, current_stream=lambda *a: default, stream=lambda s: contextlib.nullcontext()))
    ev: Any = inference.DirectGPUEvaluator(cast(Any, lambda x: None), device='cpu', max_batch=2, n_slots=2, use_amp=False)
    ev.device, ev._use_cuda, ev._stream = 'cuda:0', True, launch
    view = ev.get_input_buffer(2, slot=0)
    view.fill(3)
    def device_input(bsz, *, slot):
        row = ev._pinned_inputs_np[slot][:bsz]
        launch.hold(row)
        return torch.from_numpy(row)
    ev._device_input = device_input
    monkeypatch.setattr(inference, '_forward_no_grad', lambda *a, **kw: {'policy': torch.zeros((2, 4672)), 'wdl': torch.zeros((2, 3))})
    _, _, event = ev.evaluate_inplace_async(2)
    assert event is not None
    assert launch.synchronizations == 0  # normal launch adds no drain
    with pytest.raises(RuntimeError) as exc:
        event.synchronize()
    assert exc.value is event_error
    assert launch.synchronizations == 1
    assert default.synchronizations == 0
    if fail_drain:
        with pytest.raises(RuntimeError, match='poisoned'):
            ensure_usable(ev)
        assert len(launch.held) == 1
    else:
        ensure_usable(ev)
        assert not launch.held


@pytest.mark.parametrize('multi', [False, True])
def test_vloss_cleanup_cannot_replace_primary(monkeypatch, multi):
    import chess_anti_engine.mcts.puct_vl as puct
    ev = SentinelEvaluator('submit_safe')
    def release(*_args, **_kwargs):
        raise RuntimeError('secondary vloss cleanup error')
    monkeypatch.setattr(puct, 'release_batch_rows', release)
    runner: Any = MultiGpuPucvPool(MultiGpuPucvConfig(n_gpus=1, gather=4), evaluators=[ev]) if multi else PucvChunker(ev, gather=4)
    try:
        args = tree_root()
        def call():
            if multi:
                return runner.run(tree=args[0], root_id=args[1], root_cboard=args[2], target_sims=8, stop_event=threading.Event())
            return runner.run(*args, 8)
        with pytest.raises(TimeoutError, match='primary next submission') as exc:
            call()
        assert all(e.synced for e in ev.events)
        assert any('vloss cleanup' in n for n in getattr(exc.value, '__notes__', ()))
    finally:
        if multi:
            runner.close()


@pytest.mark.parametrize('mode', ['stopped', 'finish_once', 'finish_always', 'submit_unknown'])
def test_real_rpg_candidate_failure_and_queued_stop(mode):
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.uci.root_parallel_gumbel import RootParallelGumbelConfig, RootParallelGumbelPool, _WorkItem
    ev = SentinelEvaluator(mode)
    if mode == 'submit_unknown':
        ev.calls = 1
    pool = RootParallelGumbelPool(RootParallelGumbelConfig(n_groups=1, gather=4, open_vpa=0, min_vpa=0, min_keep=0), GumbelConfig(topk=4, add_noise=False, temperature=0., input_extra_features='v1'), evaluators=[ev])
    try:
        tree, _rid, _cb = tree_root()
        pool.prepare_root(tree=tree, board=chess.Board(), pol_logits=np.zeros(4672, np.float32), wdl_logits=np.zeros(3, np.float32))
        st = pool._state
        assert st is not None
        chunker = pool._chunkers[0]
        assert chunker is not None
        arena = pool._arena_for(st, int(st.search_legal[0]))
        stop = threading.Event()
        if mode == 'stopped':
            stop.set()
            assert pool._run_item(0, ev, chunker, _WorkItem(arena, 8, 0), stop) == 0
            assert ev.getters == 0
            assert ev.calls == 0
            assert not arena.expanded
        else:
            with pytest.raises(TimeoutError, match='primary'):
                pool._run_item(0, ev, chunker, _WorkItem(arena, 8, 0), stop)
            assert not pool._owned
            assert not pool._owned_shared
            assert all(e.synced for e in ev.events if e.failure != 'always')
            if mode in ('finish_always', 'submit_unknown'):
                getters = ev.getters
                with pytest.raises(RuntimeError, match='poisoned'):
                    pool.run(target_sims=8, stop_event=threading.Event())
                assert getters == ev.getters
            else:
                ev.mode = 'none'
                assert pool._run_item(0, ev, chunker, _WorkItem(arena, 8, 0), stop) == 8
    finally:
        pool.close()


def test_clean_receipts_do_not_drain_stream():
    from chess_anti_engine.async_safety import guarded_event
    owner = types.SimpleNamespace()
    stream = HeldStream()
    event = guarded_event(owner, Event(), stream)
    event.synchronize()
    assert stream.synchronizations == 0


@pytest.mark.parametrize('wrapped', ['locked', 'owner'])
def test_dispatch_wrapper_safety_identity(wrapped):
    from chess_anti_engine.async_safety import _mark_quiesced, ensure_usable, poison, submit_async
    from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher, ThreadSafeGPUDispatcher
    ev = SentinelEvaluator('submit_safe')
    wrapper: Any = ThreadSafeGPUDispatcher(ev) if wrapped == 'locked' else CUDAOwnerDispatcher(ev)
    try:
        error = RuntimeError('safely drained backend error')
        _mark_quiesced(error, ev)
        def submit(*_args, **_kwargs):
            raise error
        ev.evaluate_inplace_async = submit
        with pytest.raises(RuntimeError) as exc:
            submit_async(wrapper, 1)
        assert exc.value is error
        ensure_usable(wrapper)
        poison(ev, error)
        with pytest.raises(RuntimeError, match='poisoned'):
            ensure_usable(wrapper)
    finally:
        if wrapped == 'owner':
            wrapper.close()


@pytest.mark.parametrize('mode', ['submit_safe', 'submit_unknown', 'finish_once', 'finish_always'])
def test_actual_engine_fallback_then_fresh_go(mode, capsys):
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.uci.engine import Engine, EngineOptions
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import SearchLimits
    ev = SentinelEvaluator(mode)
    worker = SearchWorker(ev, device='cpu', gumbel_cfg=GumbelConfig(input_extra_features='v1', add_noise=False, temperature=0.), chunk_sims=8)
    worker.set_use_pucv(True, gather=4)
    engine = Engine(worker, options=EngineOptions(vl_gather=4, chunk_sims=8))
    try:
        engine._run_one_phase(SearchLimits(max_nodes=8), is_ponder=False, board=chess.Board())
        assert engine._bestmove_fallback_used == 1
        assert all(e.synced for e in ev.events if e.failure != 'always')
        previous = worker._pucv
        getters = ev.getters
        board = chess.Board()
        board.push_uci('g1f3')
        ev.mode = 'none'
        result = engine._run_one_phase(SearchLimits(max_nodes=8), is_ponder=False, board=board)
        assert previous is worker._pucv
        if mode in ('submit_unknown', 'finish_always'):
            assert result.nodes == 0
            assert ev.getters == getters
            assert engine._bestmove_fallback_used == 2
            assert 'poisoned' in capsys.readouterr().out
        else:
            assert result.nodes == 8
            assert all(e.synced for e in ev.events)
    finally:
        engine.close()
