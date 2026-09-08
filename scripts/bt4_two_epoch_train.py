#!/usr/bin/env python3
"""Train one qualified original-corpus recipe for two uninterrupted epochs.

This separate training-only contract cannot produce a one-epoch arena admission.
Preparation must bank the current planner's full seed-0/1 plans and independent
source-qualified logical row-order witnesses. Policy/content hashes are corpus
specific. This command does not construct those witnesses or select a recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_direct_screen as owned
from scripts import bt4_one_epoch_screen as recipes

TRAINING_HEAD = '0ff96f006e7cb1a72278c4ccc533abecce37ce86'
SCOPE = 'original_corpus_two_epoch_training_only'
ROWS, BATCHES, SHARDS, GAMES = 18910484, 36935, 2309, 97968
LOSS = 'sum(weighted_masked_numerator*corpus_rows/objective_mask_weight)/batch_size'
PLAN_KEYS = {
    'batch_size', 'batches_planned', 'collation_working_set_batch_copies', 'corpus_identity',
    'corpus_sha256', 'full_batches_planned', 'game_identity', 'games',
    'history_rep_fix', 'input_history_encoding', 'load_workers_planned', 'max_working_set_bytes',
    'min_batch_rows_planned', 'min_optimizer_batch_fill_ratio', 'mirror_augmentation', 'mirror_working_set_batch_copies',
    'mode', 'objective_mask_weights', 'peak_working_set_bytes_planned', 'plan_sha256',
    'policy_size', 'ragged_batches_planned', 'row_order', 'rows_planned',
    'seed', 'shards', 'sources', 'validated_load_payload_copies',
}


# Import-stable driver execution keeps normal module and exception identities.
CHILD_BOOTSTRAP = """import importlib,json,os,sys
from pathlib import Path
import torch
import torch._inductor.config as inductor
import torch._dynamo.config as dynamo
blosc=importlib.import_module('numcodecs.blosc')
receipt=Path(sys.argv[1]); sys.argv=sys.argv[2:]
before=dict(torch=torch.get_num_threads(),blosc=blosc.get_nthreads(),compile=inductor.compile_threads)
torch.set_num_threads(2); blosc.set_nthreads(2); inductor.compile_threads=2; dynamo.suppress_errors=False
after=dict(torch=torch.get_num_threads(),blosc=blosc.get_nthreads(),compile=inductor.compile_threads)
assert after == dict(torch=2,blosc=2,compile=2) and dynamo.suppress_errors is False
driver=importlib.import_module('scripts.lc0_control_train')
assert Path(driver.__file__).resolve() == (Path.cwd() / sys.argv[0]).resolve()
with receipt.open('x') as stream:
 json.dump(dict(before=before,after=after,driver=str(Path(driver.__file__).resolve()),argv=sys.argv,
               cuda_initialized=torch.cuda.is_initialized(),dynamo_suppress_errors=dynamo.suppress_errors,pid=os.getpid()),stream,indent=2)
raise SystemExit(driver.main())
"""


def same(actual: Any, expected: Any, label: str) -> None:
    owned.require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
                  json.dumps(expected, sort_keys=True, allow_nan=False), label + ' differs')


def digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def check_pin(ref: dict[str, Any]) -> None:
    owned.require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute()
                  and digest(ref['sha256']), 'invalid evidence pin')
    owned.pin(ref['path'], ref['sha256'])


def pinned(ref: dict[str, Any]) -> Any:
    check_pin(ref)
    return owned.read(ref['path'])


def environment(runtime: Path, *, gpu: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.pop('CHESS_LIVE_PRODUCTION_CONFIG', None)
    env.pop('TORCHDYNAMO_SUPPRESS_ERRORS', None)
    env.update(PYTHONPATH=str(runtime), CUDA_VISIBLE_DEVICES='0' if gpu else '', PYTHONUNBUFFERED='1',
               CHESS_ANTI_ENGINE_LIVE_CONFIG=str(runtime / 'configs/pbt2_small.yaml'))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'):
        env[key] = '2'
    return env


def validate(m: dict[str, Any]) -> None:
    same(sorted(m), sorted(('schema', 'scope', 'profile', 'state', 'run', 'training_seconds', 'plan_workers',
         'load_workers', 'max_working_set_bytes', 'runtime_qualification', 'preparation', 'preregistration',
         'launcher_sha256', 'stage_helper_sha256', 'recipe_helper_sha256')), 'manifest keys')
    owned.require(m['schema'] == 1 and m['scope'] == SCOPE and
                  m['profile'] in ('H20', 'B100', 'G50', 'SoftSF10'), 'unsupported two-epoch scope/profile')
    for key in ('plan_workers', 'load_workers', 'max_working_set_bytes'):
        owned.require(type(m[key]) is int and m[key] > 0, 'explicit positive resource allocation required')
    owned.require(type(m['training_seconds']) is int and 30 < m['training_seconds'] <= 32400,
                  'training cap must include kill grace and be at most nine hours')
    for key in ('launcher_sha256', 'stage_helper_sha256', 'recipe_helper_sha256'):
        owned.require(digest(m[key]), 'unfrozen coordinator dependency')


def verify_recipe(prep: dict[str, Any], profile: str) -> Path:
    """Reuse the genuine SoftSF producer gate; BT4 lineage retains old admission."""
    corpus = recipes.CORPORA[profile]
    owned.require(corpus.is_dir() and not corpus.is_symlink() and
                  not corpus.with_name(corpus.name + '.writing').exists(), 'corpus not finally published')
    source_ref = {'path': str(recipes.SOURCE / 'derive_targets_summary.json'),
                  'sha256': recipes.COMMON_PINS[str(recipes.SOURCE / 'derive_targets_summary.json')]}
    same(prep['source'], source_ref, 'original source')
    pinned(source_ref)
    q = pinned(prep['data_qualification'])
    derived = pinned(prep['derive_summary'])
    recipe = pinned(prep['recipe_summary'])
    name = 'sf_policy_rewrite_summary.json' if profile == 'SoftSF10' else 'bt4_policy_mix_summary.json'
    owned.require(prep['derive_summary']['path'] == str(corpus / 'derive_targets_summary.json') and
                  prep['recipe_summary']['path'] == str(corpus / name), 'recipe summary paths differ')
    expected = {'schema': 1, 'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION', 'profile': profile,
                    'corpus': str(corpus), 'rows': ROWS, 'shards': SHARDS,
                    'source': {'path': str(recipes.SOURCE), 'derive_sha256': source_ref['sha256']},
                    'derive_summary': prep['derive_summary']}
    expected['rewrite_summary' if profile == 'SoftSF10' else 'mix_summary'] = prep['recipe_summary']
    for key, value in expected.items():
        same(q.get(key), value, 'dataset qualification ' + key)
    if profile == 'SoftSF10':
        recipes.verify_softsf_recipe({'profile': profile}, recipe, derived)
        same(derived['policy_target_postprocess'], {k: v for k, v in recipe.items() if k != 'outputs'}, 'SoftSF lineage')
    else:
        same(derived['policy_target_postprocess'], recipe, 'BT4 lineage')
        kind, algorithm, alpha = ('c20-global', 'stored-c20t05-then-global-bt4-v1', .2) if profile == 'H20' else (
            'global', 'legal-normalized-global-arithmetic-v1', 1. if profile == 'B100' else .5)
        for key, value in {'kind': kind, 'algorithm': algorithm, 'alpha': alpha, 'bt4_temperature': .5, 'rows': ROWS,
                               'expected_shards': SHARDS, 'source_dir': str(recipes.SOURCE),
                               'source_derive_summary_sha256': source_ref['sha256'], 'mutated_arrays': ['policy_target']}.items():
            same(recipe.get(key), value, 'BT4 recipe ' + key)
        if profile == 'H20':
            parent = recipe['c20_parent']
            same(parent['source_dir'], str(recipes.C_CORPUS), 'H20 parent')
            for key, filename in (('derive_summary', 'derive_targets_summary.json'), ('mix_summary', 'bt4_policy_mix_summary.json')):
                ref = {'path': str(recipes.C_CORPUS / filename), 'sha256': recipes.C_CORPUS_PINS[str(recipes.C_CORPUS / filename)]}
                same(parent[key], ref, 'H20 parent pin')
                original = pinned(ref)
                if key == 'mix_summary':
                    same(parent['policy_target_postprocess'], original, 'H20 stored parent')
    return corpus


def verify_plans(m: dict[str, Any], prep: dict[str, Any]) -> None:
    same(prep['schema'], 1, 'preparation schema')
    same(prep['status'], 'PASS_TWO_EPOCH_PREPARATION', 'preparation status')
    same(prep['profile'], m['profile'], 'prepared recipe')
    same(prep['runtime_qualification'], m['runtime_qualification'], 'prepared runtime')
    same(prep['preregistration'], m['preregistration'], 'prepared registration')
    for key in ('plan_workers', 'load_workers', 'max_working_set_bytes'):
        same(prep[key], m[key], 'prepared ' + key)
    owned.require(len(prep['epochs']) == 2, 'two prospective epochs required')
    for index, entry in enumerate(prep['epochs']):
        same(entry['epoch_index'], index + 1, 'prospective epoch index')
        for key in ('source_logical_order_sha256', 'corpus_logical_order_sha256', 'source_plan_sha256'):
            owned.require(digest(entry[key]), 'unfrozen source/logical schedule witness')
        same(entry['source_logical_order_sha256'], entry['corpus_logical_order_sha256'], 'logical row order')
        p = entry['plan']
        same(sorted(p), sorted(PLAN_KEYS), 'full prospective plan keys')
        for key, value in {'mode': 'game_epoch', 'seed': index, 'rows_planned': ROWS, 'batches_planned': BATCHES,
                               'shards': SHARDS, 'games': GAMES, 'batch_size': 512, 'policy_size': 1858,
                               'input_history_encoding': 'lc0_root_legacy_meta', 'history_rep_fix': True, 'mirror_augmentation': True,
                               'mirror_working_set_batch_copies': 7, 'min_batch_rows_planned': 511, 'load_workers_planned': m['load_workers'],
                               'max_working_set_bytes': m['max_working_set_bytes']}.items():
            same(p[key], value, 'prospective ' + key)
        owned.require(digest(p['plan_sha256']) and digest(p['corpus_sha256']) and
                      0 < p['peak_working_set_bytes_planned'] <= m['max_working_set_bytes'], 'invalid corpus/resource plan')
    same(prep['epochs'][0]['plan']['corpus_sha256'], prep['epochs'][1]['plan']['corpus_sha256'], 'two-epoch corpus identity')
    owned.require(prep['epochs'][0]['plan']['plan_sha256'] != prep['epochs'][1]['plan']['plan_sha256'], 'repeated epoch plan')
    # Logical order is compared above; source/candidate physical hashes need not agree.



def verify_runtime_evidence(q: dict[str, Any]) -> None:
    """Consume the actual CPU import and compiled CUDA probe receipts."""
    rt = q['runtime']
    cpu, cuda = pinned(q['cpu_qualification']), pinned(q['cuda_qualification'])
    owned.require(cpu['status'] == 'PASS_CPU_TRAINING_IMPORTS' and cpu['head'] == TRAINING_HEAD
                  and cpu['runtime'] == q['root'] and cpu['cuda_initialized'] is False,
                  'wrong CPU import qualification')
    owned.require(cuda['status'] == 'PASS_COMPILED_CUDA_TWO_EPOCH_PROBE'
                  and cuda['runtime']['head'] == TRAINING_HEAD and cuda['runtime']['path'] == q['root']
                  and cuda['model_parameters'] == 61444448 and cuda['runtime']['dynamo_suppress_errors'] is False, 'wrong compiled CUDA qualification')
    for key in ('python', 'executable', 'torch', 'numpy', 'cuda'):
        proof_key = 'cuda_build' if key == 'cuda' else key
        same(cpu[proof_key], rt[key], 'CPU runtime ' + key)
        same(cuda['runtime'][proof_key], rt[key], 'CUDA runtime ' + key)
    compiled = cuda['compile']
    owned.require(compiled['unique_graphs'] > 0 and compiled['frames_ok'] > 0 and
                  (compiled['inductor_graph_cache_events'] > 0 or compiled['generated_kernel_count_delta'] > 0),
                  'CUDA probe did not observe compiled graph capture')
    sampling = cuda['sampling']
    owned.require(sampling['complete'] is True and sampling['mode'] == 'game_epochs' and
                  sampling['epochs_requested'] == sampling['epochs_completed'] == 2 and
                  sampling['rows_realized'] == 2048 and sampling['batches_realized'] == 4,
                  'CUDA probe did not complete both passes')
    # These small metadata records are retained; no probe checkpoint is reread.
    summary = pinned(cuda['summary'])
    same(summary['sampling'], sampling, 'CUDA probe summary sampling')
    owned.require(summary['trainable_params'] == 61444448 and summary['batch_size'] == 512
                  and summary['steps_realized'] == summary['compute_loss_calls'] == 4
                  and summary['realized_after_guard']['device'] == 'cuda'
                  and summary['realized_after_guard']['use_compile'] is True, 'CUDA probe summary differs')

def check_inputs(m: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    validate(m)
    for path, key in ((__file__, 'launcher_sha256'), (owned.__file__, 'stage_helper_sha256'), (recipes.__file__, 'recipe_helper_sha256')):
        owned.pin(path, m[key])
    check_pin(m['preregistration'])
    q = pinned(m['runtime_qualification'])
    same(q['status'], 'PASS_TWO_EPOCH_TRAINING_RUNTIME', 'runtime qualification')
    same(q['head'], TRAINING_HEAD, 'qualified training implementation')
    runtime = Path(q['root'])
    owned.require(runtime.is_absolute() and runtime == runtime.resolve(), 'canonical runtime required')
    same(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip(), TRAINING_HEAD, 'actual runtime revision')
    owned.require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip(),
                  'tracked runtime changed')
    same(q['resources'], {key: m[key] for key in ('plan_workers', 'load_workers', 'max_working_set_bytes', 'training_seconds')}, 'runtime resource allocation')
    rt = q['runtime']
    verify_runtime_evidence(q)
    same(sorted(rt), sorted(('python', 'executable', 'torch', 'cuda', 'numpy', 'native_extensions', 'native_extension_sha256')), 'runtime fields')
    required = {'chess_anti_engine.encoding._features_ext', 'chess_anti_engine.encoding._lc0_ext',
                'chess_anti_engine.nnue._nnue_ext'}
    same(sorted(rt['native_extensions']), sorted(required), 'native modules')
    same(sorted(rt['native_extensions'].values()), sorted(rt['native_extension_sha256']), 'native pin coverage')
    for path, value in {**q['environment_pins'], **rt['native_extension_sha256']}.items():
        owned.pin(path, value)
    owned.require(rt['executable'] in q['environment_pins'], 'interpreter binary pin required')
    prep = pinned(m['preparation'])
    verify_plans(m, prep)
    corpus = verify_recipe(prep, m['profile'])
    same(prep['corpus'], str(corpus), 'prepared corpus')
    check_pin(prep['producer'])
    return q, prep, corpus


def train_command(m: dict[str, Any], q: dict[str, Any], corpus: Path) -> list[str]:
    return [q['runtime']['executable'], '-c', CHILD_BOOTSTRAP, str(Path(m['state']) / 'training_threads.json'),
            'scripts/lc0_control_train.py', '--config', 'configs/lc0_positive_control.yaml',
            '--shards', str(corpus), '--out-dir', m['run'], '--steps', '0', '--batch-size', '512',
            '--sampling-mode', 'game_epoch', '--epochs', '2', '--seed', '0', '--device', 'cuda',
            '--train-window-steps', '88', '--epoch-plan-workers', str(m['plan_workers']),
            '--epoch-load-workers', str(m['load_workers']), '--epoch-max-working-set-gib',
            str(m['max_working_set_bytes'] / 1024**3), '--allow-invalid-control']


def verify_summary(summary: dict[str, Any], m: dict[str, Any], prep: dict[str, Any]) -> None:
    plans = [entry['plan'] for entry in prep['epochs']]
    rows, batches = plans[0]['rows_planned'], plans[0]['batches_planned']
    sampling = summary['sampling']
    expected = {'mode': 'game_epochs', 'complete': True, 'epochs_requested': 2, 'epochs_completed': 2,
                    'sampling_seed_rule': 'seed + zero_based_epoch_index', 'augmentation_rng': 'continuous_from_epoch_one',
                    'corpus_sha256': plans[0]['corpus_sha256'], 'rows_realized': 2*rows, 'batches_realized': 2*batches, 'loss_normalization': LOSS}
    for key, value in expected.items():
        same(sampling.get(key), value, 'aggregate ' + key)
    for key, value in {'seed': 0, 'batch_size': 512, 'configured_batch_size': 512, 'warmup_steps': 1000, 'train_window_steps': 88,
                           'steps': 2*batches, 'steps_realized': 2*batches, 'compute_loss_calls': 2*batches}.items():
        same(summary[key], value, 'training ' + key)
    same(summary['corpus']['shard_dirs'], [prep['corpus']], 'actual corpus')
    owned.require(summary['realized_after_guard']['device'] == 'cuda' and
                  summary['realized_after_guard']['use_compile'] is True, 'training backend differs')
    owned.require(type(summary['valid_control']) is bool and isinstance(summary['validity_problems'], list), 'missing historical limitations')
    windows = summary['train_window_metrics']
    count = (batches + 87) // 88
    same(summary['train_windows'], 2*count, 'total window count')
    owned.require(len(windows) == 2*count and len(sampling['epochs']) == 2, 'incomplete epochs/windows')
    for epoch, (record, plan) in enumerate(zip(sampling['epochs'], plans, strict=True)):
        for key, value in {'epoch_index': epoch+1, 'steps_start': epoch*batches, 'steps_end': (epoch+1)*batches, 'window_count': count}.items():
            same(record[key], value, 'epoch ' + key)
        actual = record['sampling']
        same({k: actual[k] for k in plan}, plan, 'actual full plan')
        for key, value in {'complete': True, 'plan_workers': m['plan_workers'], 'load_workers': m['load_workers'],
                               'rows_realized': rows, 'batches_realized': batches, 'decoded_rows_resident': 0, 'decoded_bytes_resident': 0,
                               'same_game_repeats_max': 0, 'realized_sha256': plan['plan_sha256']}.items():
            same(actual[key], value, 'realized ' + key)
        owned.require(0 < actual['peak_working_set_bytes'] <= m['max_working_set_bytes'], 'realized resource bound exceeded')
        subset = windows[epoch*count:(epoch+1)*count]
        for i, window in enumerate(subset):
            steps, end = min(88, batches-i*88), min((i+1)*88, batches)
            for key, value in {'window_index': epoch*count+i+1, 'epoch_index': epoch+1, 'epoch_window_index': i+1,
                                   'steps_requested': steps, 'train_steps_done': steps, 'steps_cumulative': epoch*batches+end,
                                   'epoch_steps_cumulative': end}.items():
                same(window[key], value, 'window ' + key)
            for key in ('transient_cuda_retry_batches', 'grad_nonfinite_skip_rate'):
                owned.require(type(window[key]) in (int, float) and math.isfinite(window[key]) and window[key] == 0,
                              'training skipped or retried updates')
            owned.require(all(not isinstance(v, float) or math.isfinite(v) for v in window.values()) and
                          math.isfinite(window['loss']) and math.isfinite(window['grad_norm_mean']), 'nonfinite window')
            owned.require(type(window['train_samples_seen']) is int and plan['min_batch_rows_planned']*steps <= window['train_samples_seen'] <= steps*512,
                          'invalid window sample count')
        same(sum(w['train_samples_seen'] for w in subset), rows, 'epoch row total')


def complete(m: dict[str, Any], prep: dict[str, Any]) -> dict[str, Any]:
    run = Path(m['run'])
    summary = owned.read(run / 'summary.json')
    verify_summary(summary, m, prep)
    owned.require(not (run / 'checkpoint_epoch1.pending.pt').exists(), 'unpublished epoch-one checkpoint')
    checkpoints = {}
    for role, filename in (('epoch1', 'checkpoint_epoch1.pt'), ('last', 'checkpoint.pt')):
        path = run / filename
        owned.require(path.is_file() and not path.is_symlink(), 'missing/nonregular checkpoint')
        ref = {'role': role, 'path': str(path), 'sha256': owned.sha(path)}
        same([c for c in summary['checkpoints'] if c['role'] == role], [ref], 'completed checkpoint ' + role)
        checkpoints[role] = ref
    return {'scope': SCOPE, 'two_epoch_training_complete': True, 'profile': m['profile'], 'checkpoints': checkpoints,
                'summary': {'path': str(run / 'summary.json'), 'sha256': owned.sha(run / 'summary.json')},
                'preparation': m['preparation'], 'runtime_qualification': m['runtime_qualification'],
                'epochs': summary['sampling']['epochs'], 'historical_valid_control': summary['valid_control'],
                'historical_validity_problems': summary['validity_problems']}


def execute(m: dict[str, Any]) -> None:
    q, prep, corpus = check_inputs(m)
    runtime, state, run = Path(q['root']), Path(m['state']), Path(m['run'])
    inputs = [runtime, corpus, recipes.SOURCE] + [Path(m[k]['path']).resolve() for k in ('preparation', 'runtime_qualification', 'preregistration')]
    for out in (state, run):
        owned.require(out.is_absolute() and out == out.resolve() and out.parent.is_dir() and not out.exists(), 'fresh canonical output required')
        for other in [*inputs, run if out == state else state]:
            owned.require(out != other and out not in other.parents and other not in out.parents, 'overlapping output')
        owned.disk_guard(out)
    state.mkdir()
    owned.write(state / 'manifest.json', m)
    try:
        rt = q['runtime']
        code = '''import contextlib,json,sys
with contextlib.redirect_stdout(sys.stderr):
 import importlib,torch,numpy
 actual=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,native_extensions={k:importlib.import_module(k).__file__ for k in json.loads(sys.argv[1])})
print(json.dumps(actual))'''
        observed = json.loads(subprocess.check_output([rt['executable'], '-c', code, json.dumps(list(rt['native_extensions']))],
                              cwd=runtime, env=environment(runtime, gpu=False), text=True, timeout=60))
        same(observed, {k: v for k, v in rt.items() if k != 'native_extension_sha256'}, 'actual runtime imports')
        owned.write(state / 'runtime_probe.json', observed)
        with (owned.ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
            owned.acquire_gpu_lease(lease)
            check_inputs(m)
            owned.require(not run.exists() and not (state / 'STOP').exists(), 'existing run or stop requested')
            owned.disk_guard(run)
            owned.require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10).strip(), 'competing GPU process')
            gpu_env = environment(runtime, gpu=True)
            gpu_env.update(TORCHINDUCTOR_CACHE_DIR=str(state / 'compile_cache/torchinductor'),
                           TRITON_CACHE_DIR=str(state / 'compile_cache/triton'), TORCHINDUCTOR_COMPILE_THREADS='2')
            charge = owned.run_owned_stage(train_command(m, q, corpus), state / 'training', m['training_seconds'], lease.fileno(),
                'training', {'runtime': observed, 'scope': SCOPE}, manifest=m, stop_paths=(state / 'STOP',),
                cwd=runtime, env=gpu_env)
        # Hashing and all completion qualification happen after releasing the GPU lease.
        owned.require(0 < charge['gpu_seconds'] <= m['training_seconds'] and not (state / 'STOP').exists(), 'invalid charge or stop requested')
        check_inputs(m)
        threads = owned.read(state / 'training_threads.json')
        owned.require(threads['dynamo_suppress_errors'] is False, 'child suppressed compiler errors')
        same(threads['after'], {'torch': 2, 'blosc': 2, 'compile': 2}, 'realized child thread counts')
        same(threads['driver'], str(runtime / 'scripts/lc0_control_train.py'), 'actual child driver')
        same(threads['argv'], train_command(m, q, corpus)[4:], 'actual child argv')
        result = complete(m, prep)
        result['training_threads'] = {'path': str(state / 'training_threads.json'), 'sha256': owned.sha(state / 'training_threads.json')}
        result['training_charge_seconds'] = charge['gpu_seconds']
        owned.require(not (state / 'STOP').exists(), 'stop before completion publication')
        owned.write(state / 'two_epoch_training.complete.json', result)
    except BaseException as error:
        owned.write(state / 'failed.json', {'scope': SCOPE, 'two_epoch_training_complete': False, 'error': str(error), 'partial_artifacts_preserved': True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    m = owned.read(args.manifest)
    if args.execute:
        def interrupted(signum: int, _frame: Any) -> None:
            raise InterruptedError(f'signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        execute(m)
    else:
        q, _prep, corpus = check_inputs(m)
        print(json.dumps({'execute': False, 'scope': SCOPE, 'command': train_command(m, q, corpus)}, indent=2))


if __name__ == '__main__':
    main()
