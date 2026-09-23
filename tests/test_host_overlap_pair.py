import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_host_overlap_pair as tool
from scripts.host_overlap_probe import state_digest


def qualification() -> dict[str, Any]:
    plan = {'rows_planned': 1963948, 'batches_planned': 3836,
            'full_batches_planned': 3752, 'ragged_batches_planned': 84,
            'min_batch_rows_planned': 511, 'batch_size': 512, 'seed': 121,
            'max_working_set_bytes': 12 * tool.GIB, 'peak_working_set_bytes_planned': 1000,
            'plan_sha256': 'planned', 'corpus_sha256': 'corpus'}
    receipt = {'rows_realized': 1963948, 'batches_realized': 3836,
               'peak_working_set_bytes': 1000, 'realized_sha256': 'planned',
               'corpus_sha256': 'corpus'}
    run = {'rows': 1963948, 'complete': True, 'same_game_repeats_max': 0,
           'batch_rows': [512] * 3752 + [511] * 84,
           'raw_sequence_sha256': 'raw', 'prepared_sequence_sha256': 'prepared',
           'order_sequence_sha256': 'order', 'plan': plan, 'receipt': receipt}
    return {'runs': [dict(run, enabled=False, host_overlap_reserve_bytes=0,
                         overlap_batches_consumed=0, producer_threads=['MainThread'],
                         preparation_threads=['MainThread']),
                     dict(run, enabled=True, host_overlap_reserve_bytes=100,
                          overlap_batches_consumed=3836, producer_threads=['exact-host_0'],
                          preparation_threads=['exact-host_0'])]}


@pytest.mark.parametrize('defect', ['order', 'prepared', 'reserve', 'repeat', 'coverage'])
def test_qualification_rejects_false_admission(defect):
    q = copy.deepcopy(qualification())
    assert tool.qualified_batch_count(q, 1963948) == 3836
    on = q['runs'][1]
    if defect == 'order':
        on['batch_rows'] = on['batch_rows'][::-1]
    elif defect == 'prepared':
        on['prepared_sequence_sha256'] = 'different'
    elif defect == 'reserve':
        on['host_overlap_reserve_bytes'] = 0
    elif defect == 'repeat':
        on['same_game_repeats_max'] = 1
    else:
        on['rows'] -= 1
    with pytest.raises(ValueError, match="CPU"):
        tool.qualified_batch_count(q, 1963948)


def arm(enabled, seconds=10, wall=100) -> dict[str, Any]:
    sizes = [512] * 3752 + [511] * 84
    windows = []
    for start in range(0, 3836, 88):
        batch = sizes[start:start + 88]
        windows.append({'window_index': len(windows) + 1, 'steps_requested': len(batch),
                        'train_steps_done': len(batch), 'train_samples_seen': sum(batch),
                        'train_time_s': seconds, 'batch_prefetch_wait_s': 1,
                        'loss': 1, 'policy_loss': 1, 'wdl_loss': 1,
                        'grad_nonfinite_skip_rate': 0, 'transient_cuda_retry_batches': 0,
                        'batches_drawn': len(batch)})
    observation = {'status': 'TRAINER_RETURNED_SUCCESS', 'initial_model_sha256': 'init',
                   'initial_optimizer_sha256': 'initial_opt', 'final_model_sha256': 'model',
                   'final_optimizer_sha256': 'opt', 'host_batch_overlap': enabled,
                   'overlap_batches_consumed': 3836 if enabled else 0, 'peak_rss_bytes': 100, 'peak_cuda_allocated_bytes': 100,
                   'peak_cuda_reserved_bytes': 100, 'full_process_wall_seconds': wall,
                   'first_batch_seconds': 1, 'observer_seconds': 1,
                   'update_losses': [1.0] * 3836,
                   'batches': [{'rows': size, 'order_sha256': str(i)} for i, size in enumerate(sizes)]}
    return {'summary': {'sampling': {'complete': True, 'rows_realized': 1963948,
                        'batches_realized': 3836, 'peak_working_set_bytes': 1000,
                        **({'host_overlap_reserve_bytes': 100} if enabled else {})},
                        'train_window_metrics': windows}, 'observation': observation}


def test_fixed_windows_and_full_wall_are_both_binding():
    arms = {'OFF': arm(False), 'ON': arm(True, 9)}
    for name in arms:
        for index in [0, 42, 43]:
            arms[name]['summary']['train_window_metrics'][index]['train_time_s'] = 9999
    result = tool.compare(arms, rows=1963948, batches=3836)
    assert result['overlap_screen_pass']
    assert result['on_over_off_steady_time'] == pytest.approx(0.9)
    arms['ON']['observation']['full_process_wall_seconds'] = 106
    assert not tool.compare(arms, rows=1963948, batches=3836)['overlap_screen_pass']


@pytest.mark.parametrize('key', ['final_model_sha256', 'final_optimizer_sha256', 'initial_optimizer_sha256'])
def test_final_state_difference_is_ineligible(key):
    arms = {'OFF': arm(False), 'ON': arm(True)}
    arms['ON']['observation'][key] = 'drift'
    with pytest.raises(ValueError, match='state parity'):
        tool.compare(arms, rows=1963948, batches=3836)


@pytest.mark.parametrize('defect', ['nan_update', 'missing_update', 'retry', 'loss_drift'])
def test_individual_update_and_retry_gates(defect):
    arms = {'OFF': arm(False), 'ON': arm(True)}
    on = arms['ON']
    if defect == 'nan_update':
        on['observation']['update_losses'][108] = float('nan')
    elif defect == 'missing_update':
        on['observation']['update_losses'].pop()
    elif defect == 'retry':
        on['summary']['train_window_metrics'][2]['transient_cuda_retry_batches'] = 1
    else:
        on['observation']['update_losses'][108] = 1.1
    with pytest.raises(ValueError, match=r'update|retry|loss parity'):
        tool.compare(arms, rows=1963948, batches=3836)


def test_state_digest_covers_tensor_values_and_optimizer_scalars():
    import torch
    a: dict[str, Any] = {'state': {0: {'m': torch.tensor([1.0, 2.0])}}, 'lr': 0.01}
    assert state_digest(a) == state_digest(copy.deepcopy(a))
    b = copy.deepcopy(a)
    b['state'][0]['m'][0] = 2
    assert state_digest(a) != state_digest(b)
    b = copy.deepcopy(a)
    b['lr'] = 0.02
    assert state_digest(a) != state_digest(b)


def test_final_disk_scan_enforces_32gib(tmp_path):
    budget = tool.OutputBudget(tmp_path, 32 * tool.GIB, {}, lambda: None)
    budget.refresh()
    (tmp_path / 'growth').write_bytes(b'x')
    with pytest.raises(RuntimeError, match='32GiB'):
        budget.refresh()


def test_fast_shallow_poll_catches_checkpoint_growth_between_full_scans(tmp_path):
    budget = tool.OutputBudget(tmp_path, 0, {}, lambda: None)
    budget.refresh()
    budget.next_scan = float('inf')
    with (tmp_path / 'checkpoint.pt').open('wb') as stream:
        stream.truncate(33 * tool.GIB)  # sparse: no large storage allocation
    with pytest.raises(RuntimeError, match='32GiB sampled'):
        budget.poll()


@pytest.mark.parametrize('mode', ['failed_child', 'stop'])
def test_owned_child_failure_or_stop_reaps_process(tmp_path, mode):
    pidfile = tmp_path / 'pid'
    code = ('import os,time; from pathlib import Path; '
            f'Path({str(pidfile)!r}).write_text(str(os.getpid())); '
            + ('raise SystemExit(7)' if mode == 'failed_child' else 'time.sleep(60)'))
    def guard():
        if mode == 'stop' and pidfile.exists():
            raise RuntimeError('STOP')
    with (tmp_path / 'lease').open('w') as lease, pytest.raises(RuntimeError, match=r'STOP|exit'):
        tool.run_stage([sys.executable, '-c', code], cwd=tmp_path, env=dict(os.environ),
                           log=tmp_path / 'child.log', guard=guard, lease_fd=lease.fileno())
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cpu_receipt_binds_runtime_root_config_and_inputs(tmp_path):
    prep = tmp_path / 'prep.json'
    prep.write_text(json.dumps({'status': 'PASS_BYTES_REQUIRES_FULL_STREAM_QUALIFICATION'}))
    config = tmp_path / 'config'
    config.write_text('fixed')
    q = dict(qualification(), status='PASS_EXACT_HOST_OVERLAP_CPU_QUALIFICATION',
             runtime_commit='runtime', preparation_sha256=tool.digest(prep),
             root='fixed_root', config_sha256=tool.digest(config),
             python_executable=sys.executable, dependency_distribution_versions={'torch': 'test'})
    path = tmp_path / 'qualification.json'
    path.write_text(json.dumps(q))
    terminal = tmp_path / 'complete.json'
    terminal.write_text(json.dumps({'status': 'PASS_CPU_QUALIFICATION_NOT_GPU_ADMITTED',
                                   'qualification_sha256': tool.digest(path)}))
    plan = {'cpu_complete': str(terminal), 'preparation_receipt': str(prep), 'qualification': str(path),
            'runtime_commit': 'runtime', 'rows': 1963948, 'root': 'fixed_root', 'config': str(config),
            'python': sys.executable, 'dependency_versions': {'torch': 'test'},
            'qualification_sha256': tool.digest(path), 'cpu_complete_sha256': tool.digest(terminal)}
    tool.qualified_inputs(plan)
    config.write_text('drift')
    with pytest.raises(ValueError, match='root/config'):
        tool.qualified_inputs(plan)
    config.write_text('fixed')
    q['runs'][1]['raw_sequence_sha256'] = 'mutated'
    path.write_text(json.dumps(q))
    terminal.write_text(json.dumps({'status': 'PASS_CPU_QUALIFICATION_NOT_GPU_ADMITTED',
                                   'qualification_sha256': tool.digest(path)}))
    with pytest.raises(ValueError, match='reviewed CPU'):
        tool.qualified_inputs(plan)


def test_prerequisite_bytes_are_bound_to_reviewed_plan(tmp_path):
    completion = tmp_path / 'arena.json'
    completion.write_text(json.dumps({'status': 'PASS'}))
    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text(json.dumps({'completion': {'path': str(completion),
                                                     'status_key': 'status', 'expected': 'PASS'}}))
    out = tmp_path / 'outer'
    out.mkdir()
    terminal = out / 'parent_outer_terminal.json'
    terminal.write_text(json.dumps({'returncode': 0}))
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'items': [{'id': 'E_D', 'status': 'logged',
        'command_file': str(descriptor), 'command_sha256': tool.digest(descriptor),
        'out': str(out)}]}))
    plan = {'queue': str(queue), 'queue_sha256': tool.digest(queue),
            'prerequisite_id': 'E_D', 'prerequisite_descriptor': str(descriptor),
            'prerequisite_descriptor_sha256': tool.digest(descriptor),
            'prerequisite_completion_sha256': tool.digest(completion),
            'prerequisite_outer_terminal_sha256': tool.digest(terminal)}
    tool.dependency_gate(plan)
    completion.write_text(json.dumps({'status': 'PASS', 'drift': True}))
    with pytest.raises(ValueError, match='reviewed prerequisite completion'):
        tool.dependency_gate(plan)


def test_observer_captures_saved_state_and_restores_hooks_without_gpu(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from scripts.host_overlap_probe import observe
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 10)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 20)
    class Trainer:
        def __init__(self):
            self.opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
            self._scheduler = SimpleNamespace(state_dict=dict)
            self._peak_lr = 0.1
        def zclip_state_dict(self):
            return {}
        def _iter_exact_overlapped_batches(self):
            yield {}
        def _run_optimizer_step(self, **_kwargs):
            return 1, 0.0
        def save(self, path):
            torch.save({'model': {'weight': torch.ones(1)}, 'opt': self.opt.state_dict(),
                        'scheduler': {}, 'zclip': {}, 'peak_lr': 0.1, 'step': 3836}, path)
    class Buffer:
        host_batch_overlap = True
        def sample_batch_arrays(self, _size):
            return {'game_id': np.array([1]), 'ply_index': np.array([2]),
                    'has_game_id': np.array([True]), 'has_ply_index': np.array([True])}
    original_save = Trainer.save
    original_init = Trainer.__init__
    original_step = Trainer._run_optimizer_step
    driver = SimpleNamespace(Trainer=Trainer, GameAwareEpochBuffer=Buffer,
                             build_model=lambda: torch.nn.Linear(1, 1))
    def main(_argv):
        driver.build_model()
        trainer = driver.Trainer()
        Buffer().sample_batch_arrays(1)
        trainer._run_optimizer_step(
            step_sums=SimpleNamespace(tensor=lambda _name: torch.tensor(1.25)),
            step_opt_stats={'samples_seen': 1})
        trainer.save(tmp_path / 'checkpoint.pt')
    driver.main = main
    receipt = tmp_path / 'observation.json'
    observe(driver, receipt, [])
    result = json.loads(receipt.read_text())
    assert result['status'] == 'TRAINER_RETURNED_SUCCESS'
    assert result['host_batch_overlap'] is True
    assert result['final_step'] == 3836
    assert result['final_checkpoint_sha256'] == tool.digest(tmp_path / 'checkpoint.pt')
    for key in ['initial_optimizer_sha256', 'final_optimizer_sha256', 'final_model_sha256']:
        assert len(result[key]) == 64
    assert result['peak_cuda_reserved_bytes'] == 20
    assert result['update_losses'] == [1.25]
    assert Trainer.save is original_save
    assert Trainer.__init__ is original_init
    assert Trainer._run_optimizer_step is original_step


def test_cpu_producer_real_small_epoch_raw_and_augmented_parity(tmp_path):
    from tests.test_game_aware_epoch_replay import _write
    from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
    from chess_anti_engine.train.trainer import Trainer, _SfRebuildCoverageAccumulator
    from scripts.qualify_host_overlap_cpu import measure
    root = _write(tmp_path / 'source', [[(1, 10), (2, 20)], [(1, 11), (2, 21)]])
    trainer = object.__new__(Trainer)
    trainer._sf_rebuild_coverage = _SfRebuildCoverageAccumulator()
    trainer.rebuild_sf_targets = False
    trainer.rebuild_categorical_target = False
    trainer.sf_policy_sparse_ce = False
    trainer._input_history_encoding = 'legacy'
    trainer.mirror_prob = 0.5
    trainer.device = 'cpu'
    runs = []
    for enabled in [False, True]:
        buffer = GameAwareEpochBuffer(shard_dir=root, batch_size=2, seed=121,
            input_planes=146, input_history_encoding='legacy', history_rep_fix=False,
            mirror_augmentation=True, host_batch_overlap=enabled, plan_workers=2,
            load_workers=2, max_working_set_bytes=12 * 1024**3)
        runs.append(measure(buffer, trainer, enabled, lambda: None))
    assert all(run['complete'] and run['rows'] == 4 for run in runs)
    assert runs[0]['host_overlap_reserve_bytes'] == 0
    assert runs[1]['host_overlap_reserve_bytes'] > 0
    assert runs[0]['overlap_batches_consumed'] == 0
    assert runs[1]['overlap_batches_consumed'] == 2
    assert runs[0]['producer_threads'] == ['MainThread']
    assert all(name.startswith('exact-host') for name in runs[1]['producer_threads'])
    assert runs[0]['preparation_threads'] == ['MainThread']
    assert all(name.startswith('exact-host') for name in runs[1]['preparation_threads'])
    assert runs[0]['raw_sequence_sha256'] == runs[1]['raw_sequence_sha256']
    assert runs[0]['prepared_sequence_sha256'] == runs[1]['prepared_sequence_sha256']
    assert runs[0]['prepared_sequence_sha256'] != runs[0]['raw_sequence_sha256']


def test_cpu_producer_uses_frozen_runtime_overlap_path(tmp_path):
    from tests.test_game_aware_epoch_replay import _write
    root = _write(tmp_path / 'source', [[(1, 10), (2, 20)], [(1, 11), (2, 21)]])
    script_dir = Path(__file__).resolve().parents[1] / 'scripts'
    code = '''
import importlib.util, json, sys
from types import SimpleNamespace
sys.path.insert(0, "/tmp/deepfin-factorial58-runtime")
sys.path.insert(1, sys.argv[2])
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from qualify_host_overlap_cpu import make_host_trainer, measure
spec = importlib.util.spec_from_file_location('frozen_driver',
    '/tmp/deepfin-factorial58-runtime/scripts/lc0_control_train.py')
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
driver.build_model = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('built model'))
t = make_host_trainer(driver, {}, SimpleNamespace(input_history_encoding='legacy'))
runs = []
for enabled in (False, True):
    buf = GameAwareEpochBuffer(shard_dir=__import__('pathlib').Path(sys.argv[1]),
        batch_size=2, seed=121, input_planes=146, input_history_encoding='legacy',
        history_rep_fix=False, mirror_augmentation=True, host_batch_overlap=enabled,
        plan_workers=2, load_workers=2, max_working_set_bytes=12 * 1024**3)
    runs.append(measure(buf, t, enabled, lambda: None))
print(json.dumps({'module_file': sys.modules[GameAwareEpochBuffer.__module__].__file__, 'runs': runs}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([sys.executable, '-c', code, str(root), str(script_dir)],
                          cwd='/tmp/deepfin-factorial58-runtime', env=env,
                          capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(done.stdout)
    assert result['module_file'].startswith('/tmp/deepfin-factorial58-runtime/')
    off, on = result['runs']
    assert off['overlap_batches_consumed'] == 0
    assert on['overlap_batches_consumed'] == 2
    assert off['raw_sequence_sha256'] == on['raw_sequence_sha256']
    assert off['prepared_sequence_sha256'] == on['prepared_sequence_sha256']
    assert all(name.startswith('exact-host') for name in on['producer_threads'])
