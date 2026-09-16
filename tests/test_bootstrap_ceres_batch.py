import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location('ceres_batch', SCRIPT_DIR/'bootstrap_ceres_batch.py')
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)
sys.path.pop(0)


def plan_for(tmp_path, first_rc=0):
    prerequisite = tmp_path/'prereq.json'
    prerequisite.write_text('{"status":"PASS"}')
    blocks = []
    for i in range(2):
        qual = tmp_path/f'qual{i}.json'
        spec = {'argv': [sys.executable, '-c',
            'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"status":"PASS"}\'); '
            'sys.exit(int(sys.argv[2]))', str(qual), str(first_rc if i == 0 else 0)],
            'cwd': str(tmp_path), 'env': {}, 'pins': [],
            'completion': {'path': str(qual), 'status_key': 'status', 'expected': 'PASS'}}
        path = tmp_path/f'spec{i}.json'
        path.write_text(json.dumps(spec))
        blocks.append({'id': str(i), 'command_file': str(path),
            'command_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'wall_seconds': 60, 'log_path': str(tmp_path/f'log{i}')})
    return {'prerequisite': {'path': str(prerequisite), 'status_key': 'status', 'expected': 'PASS'},
        'completion_path': str(tmp_path/'complete.json'), 'blocks': blocks,
        'internal_seconds': 300, 'success_status': 'PASS_ALL',
        'output_roots': [], 'output_cap_bytes': 10, 'disk_path': str(tmp_path),
        'startup_disk_gib': 0, 'disk_floor_gib': 0, 'startup_ram_gib': 0, 'running_ram_gib': 0}


def test_failure_stops_next_block(tmp_path, monkeypatch):
    plan = plan_for(tmp_path, first_rc=1)
    monkeypatch.setattr(batch.time, 'sleep', lambda _seconds: None)
    assert batch.run(plan) == 1
    assert not (tmp_path/'qual1.json').exists()
    assert json.loads((tmp_path/'complete.json').read_text())['status'] == 'INCOMPLETE'


def test_success_qualifies_every_block(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    monkeypatch.setattr(batch.time, 'sleep', lambda _seconds: None)
    assert batch.run(plan) == 0
    result = json.loads((tmp_path/'complete.json').read_text())
    assert result['status'] == 'PASS_ALL'
    assert result['completed_blocks'] == ['0', '1']


def test_budget_refuses_before_next_launch(tmp_path):
    plan = plan_for(tmp_path)
    plan['internal_seconds'] = 1
    assert batch.run(plan) == 1
    assert not (tmp_path/'qual0.json').exists()


def test_failed_prerequisite_prevents_all_work(tmp_path):
    plan = plan_for(tmp_path)
    (tmp_path/'prereq.json').write_text('{"status":"FAIL"}')
    assert batch.run(plan) == 1
    assert not (tmp_path/'qual0.json').exists()


def test_aggregate_output_guard_stops_before_launch(tmp_path):
    plan = plan_for(tmp_path)
    plan['output_roots'] = [str(tmp_path)]
    assert batch.run(plan) == 1
    result = json.loads((tmp_path/'complete.json').read_text())
    assert 'aggregate cap' in result['reason']
    assert not (tmp_path/'qual0.json').exists()


def test_output_scan_tolerates_atomic_chunk_rename(tmp_path, monkeypatch):
    temporary = tmp_path/'chunk.partial'
    temporary.write_bytes(b'a' * 8192)
    stable = tmp_path/'stable'
    stable.write_bytes(b'b' * 8192)
    real_lstat = Path.lstat
    def rename_then_stat(path, *args, **kwargs):
        if path == temporary and temporary.exists():
            temporary.rename(tmp_path/'chunk')
        return real_lstat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', rename_then_stat)
    first = batch.allocated_bytes([str(tmp_path)])
    assert first >= real_lstat(stable).st_blocks * 512
    second = batch.allocated_bytes([str(tmp_path)])
    assert second == sum(real_lstat(p).st_blocks * 512 for p in [tmp_path, stable, tmp_path/'chunk'])
    assert second > first


def test_output_scan_missing_directory_is_transient(tmp_path, monkeypatch):
    def disappearing_walk(root, *, followlinks, onerror):
        assert not followlinks
        onerror(FileNotFoundError(root))
        return iter([])
    monkeypatch.setattr(batch.os, 'walk', disappearing_walk)
    assert batch.allocated_bytes([str(tmp_path)]) == 0


def test_output_scan_propagates_permission_and_io_failures(tmp_path, monkeypatch):
    import errno
    import pytest
    for error in [PermissionError(errno.EACCES, 'denied'), OSError(errno.EIO, 'io failure')]:
        def failing_walk(_root, *, followlinks, onerror, error=error):
            assert not followlinks
            onerror(error)
            return iter([])
        monkeypatch.setattr(batch.os, 'walk', failing_walk)
        with pytest.raises(type(error), match=r"denied|io failure"):
            batch.allocated_bytes([str(tmp_path)])


def test_output_scan_lstat_permission_failure_propagates(tmp_path, monkeypatch):
    import pytest
    def denied(_path):
        raise PermissionError('denied')
    monkeypatch.setattr(Path, 'lstat', denied)
    with pytest.raises(PermissionError, match='denied'):
        batch.allocated_bytes([str(tmp_path)])
