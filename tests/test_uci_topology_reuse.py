"""Actual Python UCI callers with inert evaluator/native boundaries.

This file deliberately imports no project, Torch, NumPy or native extension.
The caller bodies are extracted from the selected frozen source, not copied.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
import threading
import time
import unittest

SOURCE = Path(os.environ.get('UCI_TEST_SOURCE', str(Path(__file__).resolve().parents[1])))
MODULE_HASHES = {}


def definitions(relative, wanted, namespace: dict[str, Any]) -> dict[str, Any]:
    import hashlib
    path = SOURCE / relative
    data = path.read_bytes()
    MODULE_HASHES[relative] = hashlib.sha256(data).hexdigest()
    tree = ast.parse(data)
    found = {}

    def visit(node, parents=()):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parents += (node.name,)
            if '.'.join(parents) in wanted:
                found['.'.join(parents)] = copy.deepcopy(node)
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree)
    assert set(found) == set(wanted), (relative, wanted, found.keys())
    result = {}
    for name, node in found.items():
        node.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path) + ':' + name, 'exec'), namespace)
        result[name] = namespace[node.name]
    return result


class Refusal(Exception):
    pass


class Direct:
    instances: ClassVar[list] = []

    def __init__(self, model, *, device, **_kwargs):
        self.model, self.device = model, device
        self.closed = 0
        self.instances.append(self)

    def close(self):
        self.closed += 1


class Wrapper:
    def __init__(self, child, **_kwargs):
        self.children = child if isinstance(child, list) else [child]

    def close(self):
        for child in self.children:
            child.close()


def leaves(owner):
    if isinstance(owner, Direct):
        return [owner]
    return [leaf for child in owner.children for leaf in leaves(child)]


class EvalWorker:
    def __init__(self, primary):
        self.primary = primary
        self._pucv_pool = None
        self._rpg_pool: object | None = None
        self.refuse_install: BaseException | None = None
        self.pool_factory_calls = 0

    def clear_multi_gpu_pucv(self):
        if self._pucv_pool:
            self._pucv_pool.close()
        self._pucv_pool = None

    def set_evaluator(self, primary):
        old, self.primary = self.primary, primary
        self.clear_multi_gpu_pucv()
        self._rpg_pool = None
        old.close()

    def install_multi_gpu_pucv(self, factories, **_kwargs):
        self.pool_factory_calls += 1
        if self.refuse_install:
            raise self.refuse_install
        self._pucv_pool = Wrapper([factory() for factory in factories])


def engine_fixture(enabled=False, parallel='pucv', devices=2):
    Direct.instances = []
    messages, builds = [], []
    models = [SimpleNamespace(input_history_encoding='legacy', input_extra_features='v1') for _ in range(devices)]
    device_names = ['inert:' + str(i) for i in range(devices)]
    namespace: dict[str, Any] = {'os': os, 'DirectGPUEvaluator': Direct, 'MultiGPUDispatcher': Wrapper,
                     'ThreadSafeGPUDispatcher': Wrapper, 'CUDAOwnerDispatcher': Wrapper,
                     'BatchCoalescingDispatcher': Wrapper, 'EncodedEvalCache': Wrapper,
                     '_warmup_evaluator': lambda *_args, **_kwargs: None}
    defs = definitions('chess_anti_engine/uci/__main__.py', ['_make_evaluator_factory', 'main._build.rebuild_evaluator'], namespace)
    build_eval = defs['_make_evaluator_factory'](models, device_names, False, 1, 8)
    options = SimpleNamespace(use_multi_gpu_pucv=enabled, search_parallel=parallel,
                              max_batch=64, eval_cache_entries=0, vl_gather=8)
    namespace.update(build_eval=build_eval, startup_options=options)
    rebuild = defs['main._build.rebuild_evaluator']

    def observed_rebuild(*args, **kwargs):
        builds.append((options.use_multi_gpu_pucv, options.search_parallel))
        return rebuild(*args, **kwargs)

    primary = observed_rebuild(options.max_batch, options.eval_cache_entries)
    worker = EvalWorker(primary)

    def factories(max_batch, gather):
        del max_batch, gather
        return [lambda m=m, d=d: Direct(m, device=d) for m, d in zip(models, device_names)]

    if enabled and parallel == 'pucv' and devices >= 2:
        worker._pucv_pool = Wrapper([factory() for factory in factories(64, 8)])

    ns = {'_println': messages.append}
    methods = definitions('chess_anti_engine/uci/engine.py',
                          ['Engine._handle_setoption', 'Engine._set_use_multi_gpu_pucv', 'Engine._install_multi_gpu_pucv_pool'], ns)
    cls = type('ActualOptionHandlers', (), {name.rsplit('.', 1)[-1]: fn for name, fn in methods.items()})
    engine: Any = cls()
    engine._worker, engine._options = worker, options
    engine._rebuild_evaluator, engine._rebuild_multi_gpu_pucv_factories = observed_rebuild, factories
    engine._warmup_dirty = False
    engine._wait_for_search = lambda: None
    engine._reinstall_configured_search_path = lambda **kw: None
    engine._try_install_root_parallel_gumbel = lambda: None
    engine._SETOPTION_HANDLERS = {'usemultigpupucv': methods['Engine._set_use_multi_gpu_pucv']}
    engine.command = lambda value: engine._handle_setoption(SimpleNamespace(name='UseMultiGpuPUCV', value=value))
    return engine, worker, models, messages, builds


class Move(str):
    @classmethod
    def from_uci(cls, text):
        if len(text) != 4:
            raise ValueError(text)
        return cls(text)


class Board:
    # A minimal starting-position board double: both allowed moves are legal.
    legal_moves = (Move('d2d4'), Move('e2e4'), Move('g1f3'))
    def fen(self):
        return 'startpos-history-identity'


class Deadline:
    def __init__(self, **kwargs):
        pass
    def expired(self):
        return False
    def remaining_ms(self):
        return None
    def elapsed_ms(self):
        return 0


@dataclasses.dataclass
class Config:
    simulations: int = 8
    add_noise: bool = False
    gumbel_scale: float = 0.0
    temperature: float = 0.0


def search_fixture(rpg=True):
    calls = []
    def native_boundary(**kw):
        # Not a native call: deterministic tree/root sentinels only.
        tree = kw['tree']
        if tree is None:
            tree = SimpleNamespace(visits=0)
        calls.append({'input_tree': kw['tree'], 'tree': tree, 'allowed': kw['allowed_root_indices_batch'][0],
                      'policy': kw['pre_pol_logits'], 'roots': kw['root_node_ids']})
        tree.visits += kw['cfg'].simulations
        return None, [1], [0.25], None, tree, [7]
    class Witness(Exception):
        pass
    ns = {'dataclasses': dataclasses, 'threading': threading, 'time': time,
              'chess': SimpleNamespace(Move=Move), 'move_to_index': lambda move, _board: {'d2d4':1, 'e2e4':2, 'g1f3':3}[move],
              '_DEFAULT_ABORT_FACTOR': 1.0, '_SOLVED_ROOT_TB_CHECK_LIMIT': 16,
              'SolvedRootWitnessUnavailable': Witness, 'run_gumbel_root_many_c': native_boundary,
              '_try_immediate_checkmate': lambda *_args, **_kwargs: None, 'SearchResult': SimpleNamespace}
    names = ['SearchWorker.run', 'SearchWorker._run_one_chunk', 'SearchWorker._run_gumbel_chunk',
             'SearchWorker._pre_expand_root_for_pool', 'SearchWorker._chunk_budget',
             'SearchWorker._should_stop_search', 'SearchWorker.reset_tree', '_allowed_root_indices']
    defs = definitions('chess_anti_engine/uci/search.py', names, ns)
    cls = type('ActualSearchCallers', (), {name.rsplit('.', 1)[-1]: fn for name, fn in defs.items() if name.startswith('SearchWorker.')})
    worker: Any = cls()
    worker._tree = worker._root_id = worker._tree_fen = None
    worker._root_pol_logits = worker._root_wdl_logits = None
    worker._rpg_pool = SimpleNamespace(n_groups=2, gather=8) if rpg else None
    worker._pucv_pool = worker._walker_pool = worker._pucv = None
    worker._tb_probe = None
    worker._device, worker._rng, worker._cfg = 'cpu-stub', None, Config()
    worker._root_noise_scale, worker._minibatch_size, worker._vloss_weight = 0.0, 8, 1
    worker._evaluator, worker._chunk_sims, worker._max_tree_bytes = object(), 8, 0
    def invalidate():
        worker._root_pol_logits = worker._root_wdl_logits = None
    worker._invalidate_root_caches = invalidate
    worker._reset_batch_timing = lambda: None
    worker._record_batch_ms = lambda *_args: None
    def root_eval(*_args):
        worker._root_pol_logits = [[0.0, 1.0, 2.0]]
        worker._root_wdl_logits = [[0.0, 0.0, 0.0]]
    worker._ensure_root_eval_cached = root_eval
    worker._try_tb_shortcut = lambda *_args, **_kwargs: None
    worker._time_capped_chunk = lambda chunk, *_args: chunk
    worker._is_shared_tree_path = lambda _allowed: False
    worker._maybe_emit_pv_info = lambda **kw: ([], 0, 0)
    worker._clock_time_margin_ms = lambda *_args: 0.0
    worker._abort_ready = lambda *_args: False
    worker._build_final_search_result = lambda **kw: SimpleNamespace(nodes=kw['total_nodes'],
        bestmove_uci='d2d4', ponder_uci=None, pv=('d2d4',), score_cp=0, tbhits=0, score_mate=None)
    worker._test_boundary_namespace = ns
    return worker, calls


def phase_fixture(worker):
    class Witness(Exception):
        pass
    namespace = {'Deadline': Deadline, 'time': time, 'SolvedRootWitnessUnavailable': Witness,
                     '_println': lambda text: None}
    phase = definitions('chess_anti_engine/uci/engine.py', ['Engine._run_one_phase'], namespace)['Engine._run_one_phase']
    engine = SimpleNamespace(_worker=worker, _stop_event=threading.Event(), _gil_probe=None,
                             _options=SimpleNamespace(abort_factor=1.0, ponder=False),
                             _emit_info=lambda **kw: None, _emit_gil_profile=lambda **kw: None)
    limits = SimpleNamespace(deadline_ms=None, max_nodes=20, max_depth=None, optimum_ms=None,
                             searchmoves=('d2d4', 'e2e4'), is_open_ended=lambda: False)
    return lambda: phase(engine, limits, is_ponder=False, board=Board())


class TopologyTests(unittest.TestCase):
    def test_off_on_primary_and_pool_use_intended_topology(self):
        engine, worker, models, _, builds = engine_fixture(True)
        engine.command('false')
        assert len(leaves(worker.primary)) == 2
        engine.command('true')
        assert len(leaves(worker.primary)) == 1
        assert len(leaves(worker._pucv_pool)) == 2
        assert len([x for x in Direct.instances if not x.closed]) == 3
        assert all(any(x.model is model for model in models) for x in Direct.instances)
        assert builds[-1] == (True, 'pucv')

    def test_rpg_to_pucv_uses_intended_topology(self):
        engine, worker, _, _, _ = engine_fixture(True, 'gumbel')
        worker._rpg_pool = object()
        engine.command('true')
        assert len(leaves(worker.primary)) == 1
        assert worker._rpg_pool is None
        assert (engine._options.use_multi_gpu_pucv, engine._options.search_parallel) == (True, 'pucv')

    def test_already_enabled_topology_control(self):
        engine, worker, _, _, _ = engine_fixture(True)
        engine.command('true')
        assert len(leaves(worker.primary)) == 1

    def test_primary_refusal_preserves_options_owner_and_error(self):
        engine, worker, _, messages, _ = engine_fixture(False, 'gumbel')
        old = worker.primary
        error = Refusal('primary-refusal')
        def refuse(*_args, **_kwargs):
            raise error
        engine._rebuild_evaluator = refuse
        with self.assertRaises(Refusal) as caught:  # noqa: PT027 - stdlib-only inert harness
            engine.command('true')
        assert caught.exception is error
        assert worker.primary is old
        assert (engine._options.use_multi_gpu_pucv, engine._options.search_parallel) == (False, 'gumbel')
        assert all(not x.closed for x in leaves(old))
        assert not any(' on ' in text for text in messages)

    def test_pool_refusal_does_not_advertise_success_or_lose_primary_custody(self):
        engine, worker, _, messages, _ = engine_fixture(False)
        old = worker.primary
        error = Refusal('pool-refusal')
        worker.refuse_install = error
        with self.assertRaises(Refusal) as caught:  # noqa: PT027 - stdlib-only inert harness
            engine.command('true')
        assert caught.exception is error
        assert worker.primary is not old
        assert all(x.closed == 1 for x in leaves(old))
        assert all(not x.closed for x in leaves(worker.primary))
        assert not engine._options.use_multi_gpu_pucv
        assert not any(' on ' in text for text in messages)

    def test_single_device_unavailable_control(self):
        engine, worker, _, messages, _ = engine_fixture(False, devices=1)
        engine.command('true')
        assert not engine._options.use_multi_gpu_pucv
        assert worker._pucv_pool is None
        assert len(leaves(worker.primary)) == 1
        assert any('unavailable' in text for text in messages)

    def test_missing_factories_does_not_replace_primary(self):
        engine, worker, _, _, builds = engine_fixture(False)
        engine._rebuild_multi_gpu_pucv_factories = None
        old, count = worker.primary, len(builds)
        engine.command('true')
        assert worker.primary is old
        assert len(builds) == count
        assert not engine._options.use_multi_gpu_pucv

    def test_factory_list_refusal_does_not_replace_primary(self):
        engine, worker, _, _, builds = engine_fixture(False)
        old, count = worker.primary, len(builds)
        error = Refusal('factory-list-refusal')
        def refuse(*_args):
            raise error
        engine._rebuild_multi_gpu_pucv_factories = refuse
        with self.assertRaises(Refusal) as caught:  # noqa: PT027 - stdlib-only inert harness
            engine.command('true')
        assert caught.exception is error
        assert worker.primary is old
        assert len(builds) == count

    def test_base_exception_during_primary_build_restores_options(self):
        engine, worker, _, _, _ = engine_fixture(False, 'gumbel')
        old = worker.primary
        error = KeyboardInterrupt('primary-interrupt')
        def refuse(*_args, **_kwargs):
            assert (engine._options.use_multi_gpu_pucv, engine._options.search_parallel) == (True, 'pucv')
            raise error
        engine._rebuild_evaluator = refuse
        with self.assertRaises(KeyboardInterrupt) as caught:  # noqa: PT027 - stdlib-only inert harness
            engine.command('true')
        assert caught.exception is error
        assert worker.primary is old
        assert (engine._options.use_multi_gpu_pucv, engine._options.search_parallel) == (False, 'gumbel')

    def test_primary_rebuild_is_not_repeated_for_pool_only_reinstall(self):
        engine, worker, _, _, builds = engine_fixture(True)
        old, count = worker.primary, len(builds)
        engine._install_multi_gpu_pucv_pool()
        assert worker.primary is old
        assert len(builds) == count


class SearchReuseTests(unittest.TestCase):
    def test_actual_engine_phase_forwards_two_moves_and_preserves_three_chunks(self):
        worker, calls = search_fixture(True)
        result = phase_fixture(worker)()
        assert result.nodes == 20
        assert len(calls) == 3
        assert all(item['allowed'] == {1, 2} for item in calls)
        assert calls[1]['input_tree'] is calls[0]['tree']
        assert calls[2]['input_tree'] is calls[0]['tree']
        assert all(item['policy'] is not None for item in calls)

    def test_actual_run_reuses_classic_tree_across_three_restricted_chunks(self):
        worker, calls = search_fixture(True)
        result = worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=20,
                            root_moves=('d2d4', 'e2e4'))
        assert result.nodes == 20
        assert len(calls) == 3
        assert [x['allowed'] for x in calls] == [{1, 2}] * 3
        assert calls[1]['input_tree'] is calls[0]['tree']
        assert calls[2]['input_tree'] is calls[0]['tree']
        assert calls[-1]['tree'].visits == 20
        assert calls[1]['roots'] == [7]

    def test_no_rpg_continuation_control(self):
        worker, calls = search_fixture(False)
        worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=16,
                   root_moves=('d2d4', 'e2e4'))
        assert calls[1]['input_tree'] is calls[0]['tree']

    def test_restriction_filter_excludes_invalid_and_illegal_moves(self):
        worker, calls = search_fixture(False)
        worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=8,
                   root_moves=('d2d4', 'e2e4', 'a1a8', 'invalid'))
        assert calls[0]['allowed'] == {1, 2}

    def test_stop_after_first_chunk_control(self):
        worker, calls = search_fixture(True)
        stop = threading.Event()
        result = worker.run(Board(), stop_event=stop, deadline=Deadline(), max_nodes=20,
                            root_moves=('d2d4', 'e2e4'), on_chunk=lambda nodes: stop.set())
        assert result.nodes == 8
        assert len(calls) == 1

    def test_later_unrestricted_search_keeps_rpg_dispatch(self):
        worker, calls = search_fixture(True)
        worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=16,
                   root_moves=('d2d4', 'e2e4'))
        rpg_calls = []
        worker._rpg_pool.last_stats = lambda: SimpleNamespace(root_finished=False)
        worker._drop_pool_root_if_support_differs = lambda _board: None
        def prepare(*_args, **_kwargs):
            worker._tree, worker._root_id = SimpleNamespace(visits=0, set_cpuct_scaling=lambda *_args: None), 7
        worker._ensure_rpg_root_prepared = prepare
        def run_rpg(chunk, _stop):
            rpg_calls.append(chunk)
            return 0.25, chunk
        worker._run_rpg_chunk = run_rpg
        result = worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=16)
        assert result.nodes == 16
        assert rpg_calls == [16]
        assert len(calls) == 2

    def test_rpg_shaped_tree_is_discarded_once_before_classic_progress(self):
        worker, calls = search_fixture(True)
        old = SimpleNamespace(visits=999, set_cpuct_scaling=lambda *_args: None)
        worker._tree, worker._root_id, worker._tree_fen = old, 99, Board().fen()
        result = worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=20,
                            root_moves=('d2d4', 'e2e4'))
        assert result.nodes == 20
        assert calls[0]['input_tree'] is None
        assert calls[-1]['tree'] is not old
        assert calls[-1]['tree'].visits == 20

    def test_declined_root_keeps_cached_heads_and_reports_zero_work(self):
        worker, calls = search_fixture(True)
        actual_stub = worker._test_boundary_namespace['run_gumbel_root_many_c']
        def decline(**kwargs):
            data = list(actual_stub(**kwargs))
            data[5] = [-1]
            return tuple(data)
        worker._test_boundary_namespace['run_gumbel_root_many_c'] = decline
        result = worker.run(Board(), stop_event=threading.Event(), deadline=Deadline(), max_nodes=20,
                            root_moves=('d2d4', 'e2e4'))
        assert result.nodes == 0
        assert len(calls) == 1
        assert worker._root_id is None
        assert worker._root_pol_logits is not None
        assert result.bestmove_uci == 'd2d4'


if __name__ == '__main__':
    unittest.main()
