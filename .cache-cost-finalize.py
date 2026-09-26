"""Audit correctness without timing, preserve banked data and publish a clean commit."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from native.bend_engine.u64_map_probe import move_cache_benchmark as b

root = Path.cwd()
out = root / 'artifacts/cache-cost-contracts'
bank = root / 'docs/experiments/evidence/move-cache-cost-screen'
manifest = json.loads((root / 'artifacts/candidate-source.json').read_text())
for path, digest in manifest.items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
subprocess.run(['git', 'diff', '--exit-code'], check=True)
assert sorted(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()) == sorted(manifest)

original = json.loads((bank / 'summary.json').read_text())
assert original['status'] == 'passed' and original['measured'] is True
assert original['raw_report_sha256'] == 'b459721b9c01a245ae0edf9d336c3bc0d6b66b2519dffffe09765dbdd7c47a00'
with (bank / 'samples.csv').open() as stream:
    samples = list(csv.DictReader(stream))
integer_fields = ('rounds', 'pair', 'order', 'milliseconds', 'requests', 'moves', 'checksum', 'hits', 'fills', 'bypasses', 'peak_rss_kib')
for row in samples:
    for field in integer_fields:
        row[field] = int(row[field])
assert len(samples) == 362
assert Counter(row['phase'] for row in samples) == {'diagnostic': 32, 'contract': 64, 'calibration': 170, 'measurement': 96}
cases = {case.name: case for case in b.cases()}
assert b.summarize(samples, list(cases)) == original['summary']

report_bytes = (out / 'report.json').read_bytes()
report = json.loads(report_bytes)
assert report['status'] == 'passed' and report['measured'] is False
assert report['contract_executions'] == len(report['samples']) == 96
assert Counter(row['phase'] for row in report['samples']) == {'diagnostic': 32, 'contract': 64}
assert report['invalid_native_inputs'] == 7 and report['shortened_work_rejected']
assert report['oracle_boards'] == 17
assert report['ordered_moves'] == original['ordered_moves']
assert report['generated_c_sha256'] == original['generated_c_sha256']
# Exact native/board dependencies must still identify the measured implementation.
for path, digest in original['sources'].items():
    if path.endswith(('.bend', '.c', '.h')):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
for path, digest in report['sources'].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
known = {}
for case in cases.values():
    for fen in (*case.prime, *case.positions):
        if fen not in known:
            raw = (out / ('oracle-' + hashlib.sha256(fen.encode()).hexdigest() + '.stdout')).read_text()
            moves, _ = b.legal.parse_moves(raw)
            assert isinstance(moves, dict)
            known[fen] = {b.pack(move) for move in moves}
for number, row in enumerate(report['samples']):
    stem = f"{number:04}-{row['case']}-{row['arm']}-{row['phase']}-{row['mode']}"
    raw = (out / (stem + '.stdout')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['stdout_sha256']
    assert (out / (stem + '.stderr')).read_text() == ''
    assert b.rss((out / (stem + '.rss')).read_text()) == row['peak_rss_kib']
    case = cases[row['case']]
    diagnostic = row['phase'] == 'diagnostic'
    assert hashlib.sha256(b.encode(case, row['arm'], row['rounds'], diagnostic).encode()).hexdigest() == row['input_sha256']
    parsed = b.parse(raw.decode(), case, row['arm'], row['rounds'], diagnostic,
                     report['ordered_moves'][case.name], [known[fen] for fen in case.positions])
    parsed.pop('order')
    assert all(row[k] == value for k, value in parsed.items())
commands = json.loads((out / 'commands.json').read_text())
for command in commands:
    assert command['timed_out'] is False
    stage = command['stage']
    if stage.startswith('invalid-'):
        assert command['exit'] == 2
        assert (out / (stage + '.stdout')).read_text() == ''
        assert (out / (stage + '.stderr')).read_text() == 'invalid benchmark bounds\n'
    else:
        assert command['exit'] == 0
        assert (out / (stage + '.stderr')).read_text() == ''
xml = ET.parse(root / 'artifacts/cache-cost-python.xml').getroot()
tests = list(xml.iter('testcase'))
assert len(tests) == 457
assert len([case for case in tests if case.attrib['classname'].endswith('test_bend_map_cache_benchmark')]) == 51
assert not any(list(xml.iter(tag)) for tag in ('failure', 'error', 'skipped'))

validation = {
    'status': 'passed', 'base_commit': os.environ['BASE'],
    'workflow_run': int(os.environ['GITHUB_RUN_ID']), 'tested_workflow_commit': os.environ['GITHUB_SHA'],
    'scope': 'Hosted correctness/static completion; banked Clang 17 timing is unchanged, no new timing panel.',
    'tested_source_sha256': manifest, 'focused_python_tests': len(tests), 'new_python_module_tests': 51,
    'historical_module_tests': 43, 'additional_admission_tests': 8,
    'diagnostic_executions': 32, 'contract_executions': 64, 'verified_native_outputs': 96,
    'invalid_native_inputs': 7, 'shortened_work_rejected': report['shortened_work_rejected'],
    'independent_oracle_boards': len(known), 'native_generated_c_sha256': report['generated_c_sha256'],
    'hosted_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
    'hosted_commands_sha256': hashlib.sha256((out / 'commands.json').read_bytes()).hexdigest(),
    'verified_commands': len(commands), 'hosted_compiler': report['cc'], 'target': report['target'],
    'banked_sample_count': len(samples), 'banked_summary_recomputed': True,
    'banked_raw_report_sha256': original['raw_report_sha256'], 'performance_panels_this_continuation': 0,
}
evidence = bank / 'hosted-validation.json'
assert not evidence.exists()
evidence.write_text(json.dumps(validation, indent=2) + '\n')
doc = root / 'docs/experiments/2026-09-26-move-cache-cost-screen.md'
doc.write_text(doc.read_text() + '''\n\n## Hosted correctness and publication follow-up\n\nThe preceding local readout and all 362 historical observations are preserved,\nincluding the interrupted pre-timing attempt, cold-cache regression and below-floor\nempty-list result. They remain Clang 17 measurements; no timing panel was rerun.\n\n''' + f'Run https://github.com/jjoshua2/DeepFin/actions/runs/{os.environ["GITHUB_RUN_ID"]} '
    + '''completed all validation stages before publication. The locked CPU environment\nand Clang 18 built the unchanged measured Bend driver from the same pinned compiler.\nAll 457 map Python cases passed without skips, including 51 in the new module.\nFocused Ruff/Basedpyright and whole-repository Ruff/Basedpyright/Vulture passed.\nThe native entry point ran without --measure: 32 diagnostic and 64 zero/three-cycle\ncontracts matched the independently built CBoard oracle and exact native ordering;\nall 96 outputs, RSS records, work/route counts and hashes were reconciled. The\nshortened-work mutation ran normally and was rejected, as were seven invalid\ninputs with the exact expected diagnostic. Generated C matches the banked driver's\nhash; original native/board dependencies remain source-identical.\n\nThe reusable Python harness now includes the same seven invalid-input cases that\npreviously ran only in the local recovery script. Eight additional admission tests\ncover their validation; a tuple-construction style edit is semantically unchanged.\nNo cache algorithm, native driver, historical sample, or expected move changed.\nThe corrected reference recomputes the banked summary from all 362 CSV rows.\n\nThe compact hosted-validation.json records tested source hashes, fresh correctness\ncounts and raw-report identities separately from the historical experiment. Its\nsource manifest identifies the pre-readout documentation; this appended section\nand that evidence file are documentation-only. This follow-up validates the\nreusable entry point, not the historical timing on another compiler or full-engine\nperformance. Source-built ordinary PR CI remains separate. Nothing was merged,\ndeployed or enabled in a production frontend. Self-review only.\n''')
paths = [*manifest, str(evidence.relative_to(root))]
subprocess.run(['git', 'add', '--', *paths], check=True)
subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
assert sorted(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()) == sorted(paths)
for path, digest in manifest.items():
    if not path.endswith('.md'):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'], check=True)
subprocess.run(['git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], check=True)
tree = subprocess.check_output(['git', 'write-tree'], text=True).strip()
commit = subprocess.check_output(['git', 'commit-tree', tree, '-p', os.environ['BASE'], '-m',
    'perf: preserve legal-move cache costs with hosted correctness qualification'], text=True).strip()
(out / 'feature.txt').write_text(commit + '\n')
subprocess.run(['git', 'push', '--force-with-lease=refs/heads/' + os.environ['FEATURE'] + ':',
                'origin', commit + ':refs/heads/' + os.environ['FEATURE']], check=True)
print('PUBLISHED_QUALIFIED_CACHE_COST', commit)
