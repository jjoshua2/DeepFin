"""Dependency-free synthetic capture identity tests; never import project modules."""
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
import unittest
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def extract(path: str, names: set[str]) -> dict[str, Any]:
    tree = ast.parse((ROOT / 'scripts' / path).read_text())
    nodes: list[ast.stmt] = [n for n in tree.body if getattr(n, 'name', None) in names
             or (isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    # exec adds heterogeneous functions/classes, not only these seed imports.
    ns: dict[str, Any] = {'dataclass': dataclass, 're': re, 'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class CaptureIdentityTests(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest'):
        super().__init__(methodName)
        self.tmp: tempfile.TemporaryDirectory[str] | None = None
        self.parser: dict[str, Any] = {}
        self.truth: dict[str, Any] = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parser = extract('blindspot_continuation.py', {'Seed', '_COMMENT_RE', 'parse_seeds'})
        self.truth = extract('blindspot_prefilter_calib.py', {'load_ground_truth', 'build_ground_truth_index'})

    def file(self, name, text):
        assert self.tmp is not None
        p = Path(self.tmp.name) / name
        p.write_text(text)
        return str(p)

    def seeds(self, lines):
        return self.parser['parse_seeds']([self.file('seeds.txt', '\n'.join(lines))])

    def test_same_score_distinct_plies_and_bodies_and_duplicate_copy(self):
        lines = [f'{body} # nq=0.30 sq=-0.70 sev=1 game=7 ply={ply}'
                 for body, ply in [('fen A moves a', 10), ('fen A moves a', 20),
                                   ('fen B moves b', 10), ('fen A moves a', 10)]]
        seeds = self.seeds(lines)
        assert len(seeds) == 3
        assert [s.ply for s in seeds] == [10, 20, 10]

    def test_exact_ground_truth_preserves_conflicting_verdicts_at_distinct_plies(self):
        seeds = self.seeds([f'A # nq=0.30 sq=-0.70 sev=1 game=7 ply={p}' for p in (10, 20)])
        path = self.file('truth.jsonl', '\n'.join(json.dumps({'game': 7, 'line': 'A', 'ply': p,
                      'sq': -0.7, 'deep': d}) for p, d in [(10, 'LOST'), (20, 'FINE')]))
        gt = self.truth['load_ground_truth'](path, None)
        assert [self.truth['build_ground_truth_index'](gt, seeds).get((s.game_id, s.ply, s.line)) for s in seeds] == ['LOST', 'FINE']

    def test_legacy_requires_unique_full_body_not_score(self):
        seeds = self.seeds([f'A # nq=0.30 sq=-0.70 sev=1 game=7 ply={p}' for p in (10, 20)])
        gt = {(7, None, 'A'): 'LOST'}
        def join(gt, s, seeds):
            return self.truth['build_ground_truth_index'](gt, seeds).get((s.game_id, s.ply, s.line))
        assert [join(gt, s, seeds) for s in seeds] == [None, None]
        assert join(gt, seeds[0], seeds[:1]) == 'LOST'
        assert join({(7, 20, 'A'): 'LOST'}, seeds[0], seeds[:1]) is None

    def test_legacy_parser_keeps_distinct_bodies(self):
        seeds = self.seeds([f'{b} # nq=0.30 sq=-0.70 sev=1 game=7' for b in ('A', 'B', 'A')])
        assert [s.line for s in seeds] == ['A', 'B']
        assert all(s.ply is None for s in seeds)

    def test_missing_body_and_duplicate_truth(self):
        load = self.truth['load_ground_truth']
        path = self.file('truth.jsonl', json.dumps({'game': 7, 'sq': -0.7, 'deep': 'FINE'}))
        assert load(path, None) == {}
        rows = [{'game': 7, 'line': 'A', 'ply': 10, 'deep': d} for d in ('LOST', 'FINE')]
        path = self.file('truth.jsonl', '\n'.join(map(json.dumps, rows)))
        conflict = ''
        try:
            load(path, None)
        except ValueError as exc:
            conflict = str(exc)
        assert 'conflicting ground truth for capture' in conflict
        path = self.file('truth.jsonl', '\n'.join([json.dumps(rows[0])] * 2))
        assert len(load(path, None)) == 1

    def test_parse_all_preserves_explicit_ply_and_legacy_none(self):
        ns = extract('blindspot_deepsf_calibrate.py', {'_TAG', 'parse_all'})
        path = self.file('all.txt', '\n'.join([
            'A # bucket=REFUTED nq=0.30 sq=-0.70 outcome=2 game=7 ply=20',
            'B # bucket=REFUTED nq=0.30 sq=-0.70 outcome=2 game=7']))
        rows = ns['parse_all'](path)
        assert [r['ply'] for r in rows] == [20, None]
        assert [r['line'] for r in rows] == ['A', 'B']

    def test_dump_preserves_capture_ply_not_inferred_location(self):
        from types import SimpleNamespace
        tree = ast.parse((ROOT / 'scripts/blindspot_continuation.py').read_text())
        writes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'write']
        assert len(writes) == 2
        for capture_ply in (20, None):
            seed = SimpleNamespace(line='A', nq=.3, sq=-.7, game_id=7, ply=capture_ply)
            verdict = SimpleNamespace(bucket='REFUTED', outcome=2, seed_ply=99)
            for write in writes:
                value = eval(compile(ast.Expression(write.args[0]), '<dump>', 'eval'),
                             {'s': seed, 'v': verdict})
                assert ('ply=20' in value) == (capture_ply == 20)
                assert 'ply=99' not in value

    def test_scaling_budget_selection_and_body_qualification(self):
        load = self.truth['load_ground_truth']
        path = self.file('truth.jsonl', json.dumps({
            'game': 7, 'line': 'A', 'ply': 10, 'sq': -.7, 'verdict': {'100': 'LOST', '200': 'FINE'}}))
        assert load(path, 100)[7, 10, 'A'] == 'LOST'
        assert load(path, None)[7, 10, 'A'] == 'FINE'
        seeds = self.seeds([f'{b} # nq=0.30 sq=-0.70 sev=1 game=7' for b in ('A', 'B')])
        gt = {(7, None, 'A'): 'LOST', (7, None, 'B'): 'FINE'}
        assert [self.truth['build_ground_truth_index'](gt, seeds).get((s.game_id, s.ply, s.line)) for s in seeds] == ['LOST', 'FINE']

    def test_index_built_once_for_repeated_horizon_queries(self):
        class CountedList(list):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        class CountedDict(dict):
            item_passes = 0

            def items(self):
                self.item_passes += 1
                return super().items()

        Seed = self.parser['Seed']
        seeds = CountedList(Seed('A', .3, -.7, 7, 'copy', ply=p) for p in range(100))
        gt = CountedDict({(7, p, 'A'): ('LOST' if p % 2 else 'FINE') for p in range(100)})
        index = self.truth['build_ground_truth_index'](gt, seeds)
        for _horizon in range(32):
            for ply in range(100):
                assert index[7, ply, 'A'] == ('LOST' if ply % 2 else 'FINE')
        assert seeds.iterations == 1
        assert gt.item_passes == 1
        assert len(index) == 100


if __name__ == '__main__':
    unittest.main()
