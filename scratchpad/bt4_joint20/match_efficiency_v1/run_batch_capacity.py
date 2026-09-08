"""Queue one registered component probe through the existing GPU stage owner."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

STATE = Path(__file__).resolve().parent
HELPER = Path('/tmp/deepfin-bt4-hybrid-tools/scripts/bt4_direct_screen.py')


def main():
    spec = importlib.util.spec_from_file_location('qualified_gpu_stage', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan_path = STATE / 'batch_capacity_v1' / 'plan.json'
    module.pin(plan_path, sys.argv[1])
    plan = module.read(plan_path)
    for path, expected in plan['pins'].items():
        module.pin(path, expected)
    os.sched_setaffinity(0, {4, 5})
    os.nice(19)
    rt = module.runtime_identity(plan['runtime_manifest'])
    actual = module.runtime_probe(rt)
    module.write(plan_path.parent / 'runtime_probe.json', actual)
    out = plan_path.parent / 'execution'
    stop_paths = [STATE / 'STOP', plan_path.parent / 'STOP',
                  module.ROOT / 'scratchpad/bt4_joint20/hybrid_endpoint_run01/STOP']
    module.require(not out.exists(), 'no resume/repeated probe')
    module.require(not any(p.exists() for p in stop_paths), 'stop requested')
    with (module.ROOT / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
        module.acquire_gpu_lease(lease)
        module.require(not any(p.exists() for p in stop_paths), 'stop requested while queued')
        prior = module.read(module.ROOT / 'scratchpad/bt4_joint20/hybrid_endpoint_run01/H20/C20T05.s400/complete.json')
        module.require(prior['complete'] is True, 'current registered match must finish first')
        module.require(not subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True, timeout=10).strip(), 'competing GPU process')
        for path, expected in plan['pins'].items():
            module.pin(path, expected)
        module.runtime_identity(plan['runtime_manifest'])
        cmd = ['/usr/bin/env',
               f'TORCHINDUCTOR_CACHE_DIR={plan_path.parent}/compile_cache/inductor',
               f'TRITON_CACHE_DIR={plan_path.parent}/compile_cache/triton',
               'TORCHINDUCTOR_COMPILE_THREADS=1', 'MAX_JOBS=1',
               '/usr/bin/ionice', '-c', '3', rt['executable'],
               str(STATE / 'probe_batch_capacity.py'), '--plan', str(plan_path), '--out', str(out)]
        receipt = module.run_owned_stage(cmd, out, plan['hard_seconds'], lease.fileno(),
                                         'capacity', {'runtime': actual, 'component_only': True},
                                         manifest=plan, stop_paths=stop_paths)
    complete = module.read(out / 'complete.json')
    module.require(complete['complete'] is True and complete['cells'] == 6, 'incomplete component panel')
    module.pin(out / 'cells.jsonl', complete['cells_sha256'])
    module.write(plan_path.parent / 'completed.json', {
        'complete': True, 'process': receipt, 'measurement': complete,
        'plan_sha256': module.sha(plan_path), 'no_strength_claim': True})


if __name__ == '__main__':
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
