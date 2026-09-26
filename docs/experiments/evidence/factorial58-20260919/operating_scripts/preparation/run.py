#!/usr/bin/env python3
"""Bounded CPU preparation for the registered 58M factorial; never owns GPU jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def pin(item):
    require(sha(item['path']) == item['sha256'], 'input pin differs: ' + item['path'])
    return read(item['path'])


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == data, 'existing receipt differs: ' + str(path))
        return
    temp = path.with_name(path.name + f'.{os.getpid()}.writing')
    with temp.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def verify_cohort(manifest, out, expected):
    from chess_anti_engine.replay import target_overlay as storage
    from scripts import bootstrap_factorial_targets as producer
    result = read(out / 'complete.json')
    require(result == {'status': 'COMPLETE_FACTORIAL_TARGET_COHORT', 'rows': expected['rows'],
        'shards': expected['shards'], 'base': manifest['base'],
        'roots': {arm: str(out / arm) for arm in producer.ARMS}, 'recipe': producer.RECIPE},
        'cohort completion differs')
    context = storage.BaseSeal(manifest['base_seal'])
    storage.require_base_corpus(manifest['base_seal'], Path(manifest['base']), context=context)
    for arm in producer.ARMS:
        paths = storage.shard_paths(out / arm)
        require([p.name for p in paths] == [e['shard'] for e in manifest['entries']], 'output membership differs')
        for path, entry in zip(paths, manifest['entries'], strict=True):
            proof, _ = storage._open_manifest(path, seal=context)
            require(proof['base'] == str(Path(manifest['base']) / entry['shard'])
                and proof['base_seal'] == manifest['base_seal']
                and proof['recipe'] == {**producer.RECIPE, 'arm': arm,
                    'base_summary': manifest['base_summary'], 'ceres_binding': entry['ceres_binding']},
                'completed cohort recipe differs')
    return result


def worker(plan, stage, index):
    sys.path.insert(0, plan['runtime'])
    import numcodecs.blosc
    numcodecs.blosc.set_nthreads(2)
    from chess_anti_engine.replay import target_overlay as storage
    from chess_anti_engine.replay.target_overlay_v2 import qualify_target_roots
    from scripts import target_overlay_storage as seals
    from scripts import bootstrap_factorial_targets as producer
    cohorts = plan['cohorts']
    if stage == 'qualify':
        roots = {arm: [Path(c['output']) / arm for c in cohorts] for arm in producer.ARMS}
        qualifications = {}
        for arm, paths in roots.items():
            output = HERE / 'qualifications' / f'{arm}.json'
            if not output.exists():
                qualify_target_roots(paths, output)
            r = ref(output)
            storage.qualified_paths(r, [p for root in paths for p in storage.shard_paths(root)])
            receipt = read(output)
            require(receipt['rows'] == plan['rows'] and len(receipt['shards']) == plan['shards'], 'arm totals differ')
            qualifications[arm] = r
        publish(HERE / 'complete.json', {'status': 'COMPLETE_FACTORIAL58_TARGETS',
            'plan': ref(HERE / 'plan.json'), 'base_plan': plan['base_plan'],
            'rows': plan['rows'], 'shards': plan['shards'],
            'arms': {'A': {'roots':[c['base'] for c in cohorts], 'qualification':None},
                **{arm: {'roots':list(map(str,paths)), 'qualification':qualifications[arm]} for arm,paths in roots.items()}},
            'cohorts': [ref(Path(c['output']) / 'complete.json') for c in cohorts]})
        return
    cohort = cohorts[index]
    base = Path(cohort['base'])
    directory = HERE / 'cohorts' / f'cohort{index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    seal = directory / 'base-seal.json'
    if stage == 'seal':
        if not seal.exists():
            seals.seal_base(base, seal)
        proof = storage.require_base_corpus(ref(seal), base)
        require(proof['rows'] == cohort['rows'] and len(proof['shards']) == cohort['shards'], 'base counts differ')
        return
    require(stage == 'cohort', 'unknown preparation stage')
    teacher_path = Path(cohort['ceres_manifest']['path'])
    teachers = pin(cohort['ceres_manifest']) if cohort['ceres_manifest'].get('sha256') else read(teacher_path)
    # New manifests are admitted by the pinned assembler's complete global roster.
    if not cohort['ceres_manifest'].get('sha256'):
        collected = read(plan['collected_complete'])
        require(collected['status'] == 'COMPLETE_COLLECTED_MANIFESTS_NOT_TRAINING_ADMISSION', 'collection incomplete')
        matches = [e for e in collected['manifests'] if e['path'] == str(teacher_path)]
        require(len(matches) == 1 and matches[0]['sha256'] == sha(teacher_path), 'new teacher manifest not admitted')
    entries = [{k:e[k] for k in ('shard', 'ceres', 'ceres_binding')} for e in teachers['entries']]
    require(len(entries) == cohort['shards'] and sum(e['ceres_binding']['rows'] for e in entries) == cohort['rows'],
        'teacher counts differ')
    require(all(e['ceres_binding']['source'] == cohort['sf_source'] for e in entries), 'teacher source differs')
    manifest = {'base': str(base), 'base_summary': cohort['base_summary'], 'base_seal': ref(seal), 'entries': entries}
    manifest_path = directory / 'manifest.json'
    publish(manifest_path, manifest)
    out = Path(cohort['output'])
    if not out.exists():
        producer.build_cohort(manifest, out)
    else:
        require((out / 'complete.json').is_file(), 'partial output retained; explicit recovery required: ' + str(out))
    result = verify_cohort(manifest, out, cohort)
    publish(directory / 'verified.json', {'status':'PASS_VERIFIED_FACTORIAL_COHORT',
        'manifest': ref(manifest_path), 'complete': ref(out / 'complete.json'), 'rows':result['rows']})


def resources(plan):
    require(not (HERE/'STOP').exists() and not (HERE.parent/'STOP').exists(), 'preparation STOP requested')
    require(shutil.disk_usage(HERE).free >= plan['disk_floor_gib'] * GIB, 'disk floor reached')
    mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    require(mem['MemAvailable'] >= plan['available_memory_floor_gib']*GIB, 'available memory floor reached')


def run_child(plan, argv, label, deadline):
    import psutil
    resources(plan)
    environment = {**os.environ, 'PYTHONPATH':plan['runtime'], 'CUDA_VISIBLE_DEVICES':'',
        'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
        'NUMEXPR_NUM_THREADS':'2', 'BLOSC_NTHREADS':'2', 'PYTHONUNBUFFERED':'1'}
    with (HERE / (label + '.log')).open('a') as log:
        child = subprocess.Popen(argv, cwd=plan['runtime'], env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                require(time.monotonic() < deadline, 'preparation wall limit reached')
                resources(plan)
                try:
                    process = psutil.Process(child.pid)
                    processes = [process, *process.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    require(rss <= plan['process_memory_cap_gib']*GIB, 'preparation RSS cap reached')
                except psutil.NoSuchProcess:
                    pass
                time.sleep(10)
            require(child.returncode == 0, f'{label} failed exit={child.returncode}; inspect log')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', choices=('seal','cohort','qualify'))
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    plan = pin({'path':str(args.plan), 'sha256':args.sha256})
    for item in plan['pins']:
        require(sha(item['path']) == item['sha256'], 'runtime/input pin differs: ' + item['path'])
    if args.worker:
        worker(plan, args.worker, args.index)
        return
    require(args.execute, 'explicit --execute required')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'preparation interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    deadline = time.monotonic() + plan['max_seconds']
    prefix = [plan['python'], str(Path(__file__).resolve()), '--plan',str(args.plan),'--sha256',args.sha256]
    # Queue after all14 collector jobs: no GPU-lease holding readiness waits.
    require(all(Path(p).is_file() and read(p).get('status') == 'COMPLETE'
        for p in plan['driver_completions']), 'collectors incomplete; queue preparation after collection')
    for i in range(len(plan['cohorts'])):
        run_child(plan, [*prefix,'--worker','seal','--index',str(i)],f'seal{i:02d}',deadline)
    run_child(plan, [plan['python'], plan['assembler']], 'assemble',deadline)
    for i in range(len(plan['cohorts'])):
        run_child(plan,[*prefix,'--worker','cohort','--index',str(i)],f'cohort{i:02d}',deadline)
    run_child(plan,[*prefix,'--worker','qualify'],'qualify',deadline)


if __name__ == '__main__':
    main()
