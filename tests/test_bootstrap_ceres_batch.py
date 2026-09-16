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
