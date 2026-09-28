from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bt4_pipeline_benchmark as tool


def banks(tmp_path):
    roots = []
    runs = []
    for index, name in enumerate(tool.ORDER):
        root = tmp_path / str(index)
        group: Any = zarr.open_group(str(root), mode='w')
        group.create_dataset('policy', data=np.array([[.25, .75], [.5, .5]], dtype='float32'))
        group.create_dataset('identity', data=np.array([123, 456], dtype='int64'))
        roots.append(root)
        seconds = {'A': 12., 'B': 10., 'C': 8.}[name]
        runs.append({'variant': name, 'input_sequence_sha256': 'same', 'raw_output_sha256': ['policy', 'wdl'],
                     'producer_seconds': seconds - 2, 'verification_seconds': 2.,
                     'producer_and_verification_seconds': seconds, 'session_setup_seconds': 1.,
                     'session_run_seconds': 3., 'worker_seconds': seconds + 1})
    return runs, roots


def test_comparison_uses_complete_pipeline_and_exact_outputs(tmp_path):
    runs, roots = banks(tmp_path)
    result = tool.compare(runs, roots)
    assert result['prefetch_over_optimized_serial'] == 1.25
    assert result['prefetch_decision'] == 'PASS_5_PERCENT_SCREEN'
    for r in runs:
        if r['variant'] == 'C':
            r['producer_and_verification_seconds'] = 10.
    assert tool.compare(runs, roots)['prefetch_decision'] == 'NO_5_PERCENT_GAIN'


@pytest.mark.parametrize('defect', ['inputs', 'outputs', 'order', 'stored_values'])
def test_comparison_refuses_parity_or_order_changes(tmp_path, defect):
    runs, roots = banks(tmp_path)
    if defect == 'inputs':
        runs[2]['input_sequence_sha256'] = 'changed'
    elif defect == 'outputs':
        runs[2]['raw_output_sha256'][1] = 'changed'
    elif defect == 'order':
        runs[2]['variant'] = 'A'
    else:
        group: Any = zarr.open_group(str(roots[2]), mode='a')
        group['policy'][0] = [.3, .7]
    with pytest.raises(RuntimeError):
        tool.compare(runs, roots)


def test_successful_factorial_dependencies_are_required(tmp_path):
    complete = tmp_path / 'complete.json'
    complete.write_text(json.dumps({'status': 'PASS'}))
    command = tmp_path / 'command.json'
    command.write_text(json.dumps({'completion': {'path': str(complete), 'status_key': 'status', 'expected': 'PASS'}}))
    (tmp_path / 'parent_outer_terminal.json').write_text(json.dumps({'returncode': 0}))
    queue = tmp_path / 'queue.json'
    item = {'id': 'D_B', 'status': 'logged', 'command_file': str(command),
            'command_sha256': tool.sha(command), 'out': str(tmp_path)}
    queue.write_text(json.dumps({'items': [item]}))
    plan = {'queue': str(queue), 'dependencies': [{'id': 'D_B', 'path': str(command), 'sha256': tool.sha(command)}]}
    tool.dependencies(plan)
    item['status'] = 'queued'
    queue.write_text(json.dumps({'items': [item]}))
    with pytest.raises(RuntimeError, match='not successful'):
        tool.dependencies(plan)
    item['status'] = 'logged'
    queue.write_text(json.dumps({'items': [item]}))
    command.write_text(command.read_text() + '\n')
    with pytest.raises(RuntimeError, match='pin changed'):
        tool.dependencies(plan)


def test_guard_failure_reaps_owned_worker(tmp_path):
    marker = tmp_path / 'pid'
    command = [sys.executable, '-c',
               'import os,time,pathlib; pathlib.Path(' + repr(str(marker)) + ').write_text(str(os.getpid())); time.sleep(60)']

    def guard():
        if marker.exists():
            raise RuntimeError('STOP requested')

    with pytest.raises(RuntimeError, match='STOP'):
        tool.checked_child(command, tmp_path, dict(os.environ), tmp_path / 'worker.log', guard)
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_failure_does_not_replace_existing_receipt(tmp_path):
    path = tmp_path / 'receipt.json'
    tool.publish(path, {'status': 'first'})
    with pytest.raises(FileExistsError):
        tool.publish(path, {'status': 'second'})
    assert json.loads(path.read_text()) == {'status': 'first'}


def test_comparison_observes_stop_during_final_qualification(tmp_path):
    runs, roots = banks(tmp_path)

    def guard():
        raise RuntimeError('STOP requested during comparison')

    with pytest.raises(RuntimeError, match='during comparison'):
        tool.compare(runs, roots, guard)
