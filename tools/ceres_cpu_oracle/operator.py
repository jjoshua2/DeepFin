"""One-hour task-local CPU oracle; reuse the qualified owned-stage lifecycle."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    if sys.flags.optimize or not math.isfinite(a.deadline_unix):
        raise ValueError('unoptimized finite-deadline execution required')
    # Existing frozen lifecycle helpers, not a new process-management implementation.
    sys.path.insert(0, '/tmp/deepfin-audited-row-exclusions')
    from scripts import bt4_direct_screen as owned
    from scripts import training_host_memory as memory
    owned.pin(a.plan, a.expected_plan_sha256)
    plan = owned.read(a.plan)
    for path, digest in plan['pins'].items():
        owned.pin(path, digest)
    work = Path(plan['work_root'])
    state = Path(plan['state'])
    owned.require(not work.exists() and not work.is_symlink() and not state.exists() and not state.is_symlink(), 'fresh task roots required')
    owned.require(os.sched_getaffinity(0) == {2, 3}, 'CPU2,3 allocation required')
    owned.require(all(os.environ.get(k) == v for k, v in plan['environment'].items()), 'isolated environment')
    memory.require_available(48)
    owned.require(shutil.disk_usage(work.parent).free >= 162 * 1024**3, '162GiB startup reserve')
    owned.require(0 < a.deadline_unix-time.time() <= 3600, 'one-hour shared deadline')
    work.mkdir()
    state.mkdir()
    deadline = a.deadline_unix-40
    last_scan = 0.
    completed = []
    def interrupted(signum, _frame):
        raise RuntimeError(f"operator interrupted by {signum}; owned cleanup required")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    def guard():
        nonlocal last_scan
        owned.require(time.time() < deadline, 'whole oracle deadline')
        owned.require(not (a.plan.parent/'STOP.request').exists(), 'STOP')
        memory.require_available(32)
        owned.require(shutil.disk_usage(work).free >= 150*1024**3, 'disk floor')
        if time.monotonic()-last_scan < 30:
            return
        last_scan = time.monotonic()
        total = 0
        def walk_error(error):
            raise error
        for attempt in range(3):
            total = 0
            try:
                for root in (work, state):
                    for base, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
                        owned.require(time.time() < deadline, 'metadata guard deadline')
                        for name in dirs+files:
                            # SDK/package symlinks are counted, never traversed.
                            total += (Path(base)/name).lstat().st_blocks*512
                break
            except FileNotFoundError:
                owned.require(attempt < 2, 'unstable metadata accounting')
        owned.require(total <= 8*1024**3, '8GiB aggregate task allowance')
    owned.write(state/'started.json', {'started_unix': time.time(), 'deadline': a.deadline_unix, 'plan_sha256': a.expected_plan_sha256})
    try:
        for stage in plan['stages']:
            guard()
            seconds = min(stage['seconds'], int(deadline-time.time()-60))
            owned.require(seconds > 60, 'stage budget exhausted')
            result = owned.run_owned_stage(stage['argv'], state/stage['name'], seconds, None,
                'oracle', {'name': stage['name']}, manifest=stage,
                stop_paths=(a.plan.parent/'STOP.request',), cwd=work, env=dict(os.environ), guard=guard)
            completed.append({'name': stage['name'], 'process': result})
            owned.write(state/'progress.json', {'completed': completed})
        last_scan = 0.
        guard()
        report = work/'results/readout.json'
        owned.require(report.exists(), 'missing CPU readout')
        owned.write(state/'complete.json', {'status': 'COMPLETE_CPU_SEMANTICS_NOT_NEURAL_PARITY',
            'ended_unix': time.time(), 'completed': completed, 'readout': {'path': str(report), 'sha256': owned.sha(report)}})
    except BaseException as error:
        owned.write(state/'failed.json', {'error': str(error), 'ended_unix': time.time(), 'completed': completed})
        raise


if __name__ == '__main__':
    main()
