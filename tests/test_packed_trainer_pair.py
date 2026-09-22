import copy
import pytest
from scripts.run_packed_trainer_pair import compare


def arm(seconds=10):
    windows = [{'steps_requested': n, 'train_steps_done': n, 'train_samples_seen': rows,
                'train_time_s': seconds, 'loss': 1.0, 'batch_prefetch_wait_s': 1.0}
               for n, rows in [(88, 45056), (88, 45056), (1, 1)]]
    return {'summary': {'sampling': {'complete': True, 'rows_realized': 90113,
        'batches_realized': 177}, 'train_window_metrics': windows},
        'observation': {'status': 'TRAINER_RETURNED_SUCCESS', 'initial_model_sha256': 'abc',
            'batches': [{'rows': 512, 'order_sha256': str(i)} for i in range(176)]
                       + [{'rows': 1, 'order_sha256': '176'}],
            'wall_seconds': 100, 'first_batch_seconds': 20, 'observer_seconds': 0.1}}


def test_actual_final_partial_samples_and_fixed_warmup():
    arms = {'external_zip': arm(20), 'nvme_directory': arm(10)}
    result = compare(arms, rows=90113, batches=177)
    assert result['arms']['external_zip']['steady_samples'] == 1
    assert result['steady_external_over_nvme'] == 0.5
    assert not result['storage_screen_pass']


@pytest.mark.parametrize('defect', ['weights', 'order', 'rows', 'loss', 'steps'])
def test_pair_refuses_invalid_evidence(defect):
    arms = {'external_zip': arm(), 'nvme_directory': arm()}
    bad = arms['external_zip']
    if defect == 'weights':
        bad['observation']['initial_model_sha256'] = 'other'
    elif defect == 'order':
        bad['observation']['batches'][0]['order_sha256'] = 'other'
    elif defect == 'rows':
        bad['summary']['sampling']['rows_realized'] -= 1
    elif defect == 'loss':
        bad['summary']['train_window_metrics'][2]['loss'] = float('nan')
    else:
        bad['summary']['train_window_metrics'][2]['train_steps_done'] = 0
    with pytest.raises(ValueError, match={
        'weights': 'initial weights', 'order': 'batch order', 'rows': 'coverage',
        'loss': 'nonfinite', 'steps': 'optimizer step'}[defect]):
        compare(copy.deepcopy(arms), rows=90113, batches=177)


def test_sources_reauthenticated_after_mutation(tmp_path):
    from scripts.run_packed_trainer_pair import authenticate_sources, digest
    source = tmp_path / 'source.zarr'
    source.mkdir()
    member = source / 'chunk'
    member.write_bytes(b'original compressed bytes')
    archive = tmp_path / 'source.zarr.zip'
    archive.write_bytes(b'banked archive')
    roots = {name: tmp_path / name for name in ('nvme_directory', 'external_zip')}
    for path in roots.values():
        path.mkdir()
    (roots['nvme_directory'] / 'shard_000000.zarr').symlink_to(source)
    (roots['external_zip'] / 'shard_000000.zarr.zip').symlink_to(archive)
    prepared = {'records': [{'source': str(source), 'zip': str(archive),
        'members': {'chunk': digest(member)}, 'zip_sha256': digest(archive)}]}
    authenticate_sources(prepared, roots, lambda: None)
    member.write_bytes(b'changed tensors with same game ids')
    with pytest.raises(ValueError, match='source bytes drift'):
        authenticate_sources(prepared, roots, lambda: None)


def test_gpu_lease_fd_inherited_by_owned_child(tmp_path):
    import fcntl
    import os
    import sys
    from scripts.run_packed_trainer_pair import run_stage
    lock = tmp_path / 'gpu.lock'
    with lock.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = ('import os,fcntl,sys; fd=int(sys.argv[1]); os.fstat(fd); '
                'fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); print("inherited")')
        run_stage([sys.executable, '-c', code, str(lease.fileno())], cwd=tmp_path,
                  env=dict(os.environ), log=tmp_path / 'child.log', guard=lambda: None,
                  lease_fd=lease.fileno())
        with lock.open('a') as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert 'inherited' in (tmp_path / 'child.log').read_text()


@pytest.mark.parametrize(('status', 'returncode', 'ok'), [('running', 0, False), ('logged', 1, False), ('logged', 0, True)])
def test_predecessor_requires_logged_outer_success(tmp_path, status, returncode, ok):
    import json
    from scripts.run_packed_trainer_pair import dependency_gate, digest
    completion = tmp_path / 'done.json'
    completion.write_text(json.dumps({'status': 'PASS'}))
    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text(json.dumps({'completion': {'path': str(completion), 'status_key': 'status', 'expected': 'PASS'}}))
    (tmp_path / 'parent_outer_terminal.json').write_text(json.dumps({'returncode': returncode}))
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'items': [{'id': 'bt4_pipeline_prefetch_20260921',
        'status': status, 'command_file': str(descriptor), 'command_sha256': digest(descriptor),
        'out': str(tmp_path)}]}))
    plan = {'queue': str(queue), 'bt4_descriptor': str(descriptor)}
    if ok:
        dependency_gate(plan)
    else:
        with pytest.raises(ValueError, match=r'predecessor|supervisor'):
            dependency_gate(plan)
