"""Pure metadata tests: never import or invoke the coordinator or its workloads."""
import ast
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from typing import Any

import pytest


def load_contract() -> dict[str, Any]:
    path = Path(__file__).parents[1] / 'scripts/factorial_prepare_and_train.py'
    tree = ast.parse(path.read_text())
    names = {'require', 'sha', 'preparation_admission', 'preparation_receipt'}
    selected: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == names
    scope: dict[str, Any] = {'Path': Path, 'hashlib': hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class PreparationContractTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # JSON fixtures intentionally contain heterogeneous values; invalid
        # values are assigned by the negative controls below.
        self.scope: dict[str, Any] = {}
        self.control = Path()
        self.config: dict[str, Any] = {}
        self.plan: dict[str, Any] = {}
        self.descriptor: dict[str, Any] = {}
        self.receipt: dict[str, Any] = {}

    def setUp(self):
        self.scope = load_contract()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.control = Path(temp.name)
        entry = self.control / 'run.py'
        entry.write_text('# inert metadata fixture\n')
        self.config = {'preparation_control': str(self.control),
                       'preparation_plan': {'path': str(self.control / 'plan.json'), 'sha256': 'a' * 64},
                       'preparation_complete': str(self.control / 'complete.json')}
        entry_ref = {'path': str(entry), 'sha256': hashlib.sha256(entry.read_bytes()).hexdigest()}
        self.plan = {'python': '/usr/bin/python3', 'runtime': '/fixture-runtime',
                     'base_plan': {'path': '/base-plan', 'sha256': 'b' * 64},
                     'rows': 3, 'shards': 2, 'pins': [entry_ref],
                     'cohorts': [{'base': '/base/0', 'output': '/output/0'},
                                 {'base': '/base/1', 'output': '/output/1'}]}
        self.descriptor = {'argv': [self.plan['python'], str(entry), '--plan', self.config['preparation_plan']['path'],
                                    '--sha256', 'a' * 64, '--execute'], 'cwd': self.plan['runtime'],
                           'pins': [copy.deepcopy(self.config['preparation_plan']), entry_ref],
                           'completion': {'path': self.config['preparation_complete'], 'status_key': 'status',
                                          'expected': 'COMPLETE_FACTORIAL58_TARGETS'}}
        self.receipt = {'status': 'COMPLETE_FACTORIAL58_TARGETS', 'plan': copy.deepcopy(self.config['preparation_plan']),
                        'base_plan': copy.deepcopy(self.plan['base_plan']), 'rows': 3, 'shards': 2,
                        'arms': {'A': {'roots': ['/base/0', '/base/1'], 'qualification': None}},
                        'cohorts': [{'path': '/output/0/complete.json', 'sha256': 'd' * 64},
                                    {'path': '/output/1/complete.json', 'sha256': 'e' * 64}]}
        for arm in ('B', 'C', 'D'):
            self.receipt['arms'][arm] = {'roots': [f'/output/{i}/{arm}' for i in range(2)],
                'qualification': {'path': str(self.control / 'qualifications' / f'{arm}.json'), 'sha256': 'c' * 64}}

    def test_supported_handoff_and_existing_artifact_preserved(self):
        output = Path(self.config['preparation_complete'])
        output.write_text('prior valid artifact')
        self.scope['preparation_admission'](self.config, self.plan, self.descriptor)
        self.scope['preparation_receipt'](self.config, self.plan, self.receipt)
        self.receipt['plan']['sha256'] = 'old-sha'
        with pytest.raises(RuntimeError, match='preparation receipt failed'):
            self.scope['preparation_receipt'](self.config, self.plan, self.receipt)
        assert output.read_text() == 'prior valid artifact'

    def test_rejects_wrong_command_bindings(self):
        for index in range(7):
            with self.subTest(index=index):
                value = copy.deepcopy(self.descriptor)
                value['argv'][index] = 'wrong'
                with pytest.raises(RuntimeError, match='preparation command differs'):
                    self.scope['preparation_admission'](self.config, self.plan, value)
        for key in ('cwd', 'completion', 'pins'):
            with self.subTest(key=key):
                value = copy.deepcopy(self.descriptor)
                value[key] = [] if key == 'pins' else 'wrong'
                with pytest.raises(RuntimeError, match='preparation'):
                    self.scope['preparation_admission'](self.config, self.plan, value)

    def test_rejects_changed_entrypoint(self):
        (self.control / 'run.py').write_text('# changed\n')
        with pytest.raises(RuntimeError, match='preparation entrypoint changed'):
            self.scope['preparation_admission'](self.config, self.plan, self.descriptor)

    def test_rejects_receipt_mismatch(self):
        mutations = [lambda r: r.update(status='INCOMPLETE'),
                     lambda r: r['plan'].update(path='old-plan'),
                     lambda r: r['plan'].update(sha256='old-sha'),
                     lambda r: r.update(base_plan={}), lambda r: r.update(rows=4),
                     lambda r: r.update(shards=3), lambda r: r['arms'].pop('D'),
                     lambda r: r['arms']['A']['roots'].reverse(),
                     lambda r: r['arms']['B']['roots'].reverse(),
                     lambda r: r['arms']['C'].update(qualification=None),
                     lambda r: r['arms']['D']['qualification'].update(path='/wrong'),
                     lambda r: r['cohorts'].reverse()]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                value = copy.deepcopy(self.receipt)
                mutate(value)
                with pytest.raises(RuntimeError, match='preparation'):
                    self.scope['preparation_receipt'](self.config, self.plan, value)

    def test_current_run_wires_checks_before_launch_and_before_publication(self):
        path = Path(__file__).parents[1] / 'scripts/factorial_prepare_and_train.py'
        tree = ast.parse(path.read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        calls = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        admission = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'preparation_admission')
        receipt = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'preparation_receipt')
        assert admission.lineno < min(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'launch')
        assert receipt.lineno < min(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'publish')


if __name__ == '__main__':
    unittest.main(verbosity=2)
