"""One-shot evidence publication from exact completed, hash-checked hosted runs."""
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

REPO = 'jjoshua2/DeepFin'
QUALIFIED = 'c7b00c19a69bdc2242e9f1931695b2bbb73f8a7a'
PARENT = '185106dfbb271abe41a81c5256d87ecb7493d257'
BAD = '5e2c2a5eca3ce2b5e183bd98b93cccd1646eaf71'
RUN = 35936460996
FEATURE = 'feat/bend-collections-owning-qualified-20260923'


def api(path):
    return subprocess.check_output(['gh', 'api', path], timeout=60)


def artifact(run_id, artifact_id, digest):
    metadata = json.loads(api(f'repos/{REPO}/actions/artifacts/{artifact_id}'))
    assert metadata['workflow_run']['id'] == run_id and not metadata['expired']
    blob = api(f'repos/{REPO}/actions/artifacts/{artifact_id}/zip')
    assert hashlib.sha256(blob).hexdigest() == digest
    return zipfile.ZipFile(io.BytesIO(blob))


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def main():
    assert git('rev-parse', 'HEAD') == QUALIFIED
    assert not git('status', '--porcelain')
    run = json.loads(api(f'repos/{REPO}/actions/runs/{RUN}'))
    assert run['conclusion'] == 'success' and run['head_sha'] == QUALIFIED
    jobs = json.loads(api(f'repos/{REPO}/actions/runs/{RUN}/jobs'))['jobs']
    assert len(jobs) == 1 and jobs[0]['conclusion'] == 'success'
    assert all(step['conclusion'] == 'success' for step in jobs[0]['steps'])
    qualified_zip = artifact(RUN, int(os.environ['EVIDENCE_ARTIFACT']), os.environ['EVIDENCE_SHA256'])
    data = qualified_zip.read('collections-owning.json')
    report = json.loads(data)
    assert report['status'] == 'passed' and report['wrong_head_mutation_rejected'] is True
    original = json.loads(qualified_zip.read('collections-original.json'))
    assert original['status'] == 'passed' and original['lifo_mutation_rejected'] is True
    tests = ET.fromstring(qualified_zip.read('collections-python.xml'))
    assert len(list(tests.iter('testcase'))) == 61
    assert not any(list(tests.iter(name)) for name in ('failure', 'error', 'skipped'))
    src = Path('native/bend_engine/collections_probe')
    for name, sha in report['sources'].items():
        assert hashlib.sha256((src/name).read_bytes()).hexdigest() == sha
    for key, name in [('search_sha256', 'session_probe/Search.bend'), ('chess_sha256', 'legal_probe/Chess.bend')]:
        assert hashlib.sha256((src.parent/name).read_bytes()).hexdigest() == report[key]
    from native.bend_engine.collections_probe.owning_benchmark import timing_summary
    samples = report.pop('samples')
    assert Counter(row['phase'] for row in samples) == {
        'contract-generic': 27, 'contract-ubsan': 27, 'contract-native': 27,
        'calibration': 69,
        'measurement': 54,
    }
    assert timing_summary(samples) == report['timing_summary']
    assert report['ring_rows_per_mode'] == 37416
    assert len(report['ring_trace_sha256']) == 3 and len(set(report['ring_trace_sha256'].values())) == 1
    dest = Path('docs/experiments/evidence/bend-owning-collections')
    assert not dest.exists()
    dest.mkdir(parents=True)
    with (dest/'samples.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(samples[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(samples)
    report.update({'samples_file': 'samples.csv', 'source_commit': QUALIFIED,
                   'workflow_run': RUN, 'artifact_id': int(os.environ['EVIDENCE_ARTIFACT']),
                   'artifact_sha256': os.environ['EVIDENCE_SHA256'],
                   'raw_report_sha256': hashlib.sha256(data).hexdigest(),
                   'python_tests_passed': 61})
    (dest/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (dest/'original-contracts.json').write_text(json.dumps(original, indent=2)+'\n')
    diagnosis_zip = artifact(35936064206, 10783161833,
        'b8a46427727721d2604bcde4565063799be7fefabb2f954bfd016f3abdbf2ce5')
    diagnosis = json.loads(diagnosis_zip.read('results.json'))
    assert [(row['config'], row['exit']) for row in diagnosis] == [
        ('0 1 0', 0), ('1 1 0', 0), ('2 1 0', 1),
        ('0 1 0', 0), ('1 1 0', 0), ('2 1 0', 1)]
    (dest/'rejected-layout-diagnosis.json').write_text(json.dumps(diagnosis, indent=2)+'\n')
    c_source = diagnosis_zip.read('owning.c')
    assert hashlib.sha256(c_source).hexdigest() == '87dd6cb2d1184984576acf2a01cd11a4a622cf612f6afdb2fbc8fda23c6ced39'
    c_lines = c_source.decode().splitlines()
    excerpts = '\n\n'.join('Generated C lines '+str(start)+'-'+str(end)+':\n```c\n'+
        '\n'.join(c_lines[start-1:end])+'\n```' for start,end in [(1377,1379), (3720,3735), (5804,5820)])
    (dest/'rejected-layout-c.md').write_text('# Rejected unboxed layout\n\nSource commit `'+BAD+'`.\n\n'+excerpts+'\n')
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', BAD, PARENT], check=True)
    rejected = subprocess.check_output(['git','show', BAD+':native/bend_engine/collections_probe/Ring.bend'])
    assert git('rev-parse', BAD+':native/bend_engine/collections_probe/Ring.bend') == '1a9834bb089efb7de1e13c7a28eebbba782c66e5'
    (dest/'rejected-ring.bend.txt').write_bytes(rejected)
    doc = Path('docs/experiments/2026-09-23-bend-owning-collections.md')
    pending = 'Pending native execution. The 36 independent Python parser/oracle/timing tests\npassed locally; that result does not qualify the Bend implementation. Compiled\nresults and any failed attempts will be added after execution.'
    assert doc.read_text().count(pending) == 1
    doc.write_text(doc.read_text().replace(pending, 'The completed qualification below supersedes the intermediate failures.\nAll raw timing samples and compact diagnostic evidence are retained.') + '\n\n'+Path(os.environ['AMENDMENT_FILE']).read_text())
    lines = ['\n\n### Completed owning qualification\n',
             f'Run [{RUN}](https://github.com/{REPO}/actions/runs/{RUN}), source `{QUALIFIED}`.',
             'Whole-repository Ruff/Basedpyright/Vulture and explicit collection static checks passed.',
             'All 61 collection Python cases passed without skips (36 new, 25 existing).',
             'Ring: 37,416 identical exact output rows per generic/native/UBSan mode; the compiled wrong-head mutation was rejected.',
             'Owning payloads: 81 native contract executions (27 per mode), checking order, counters, identity, root state and observed history.',
             'Existing FIFO/traversal/LIFO controls passed unchanged. Modes repeat fixtures; they are not disjoint datasets.',
             '', '| Roots | Turns per measured sample | List median ms | FIFO median ms | Ring median ms | Reliable samples |',
             '|---|---|---|---|---|---|']
    for size, entry in report['timing_summary'].items():
        counts = {row['steps'] for row in samples if row['phase']=='measurement' and row['size']==int(size)}
        assert len(counts) == 1
        med = entry['median_ms']
        lines.append(f"| {size} | {counts.pop()} | {med['list']} | {med['fifo']} | {med['ring']} | {entry['reliable']} |")
    lines += ['', 'Six measured permutations per size; calibration excluded. Raw samples, output hashes and exact source hashes are retained in [evidence](evidence/bend-owning-collections/summary.json).',
              'These are collection-plus-root-update timings, not useful neural EPS, search throughput, Elo, allocation counts or tail-latency measurements. Descriptive medians from one host are not an independent review or a broad statistical performance claim.',
              'On this host, FIFO/list median ratios favor FIFO by 3.65x at 16 roots and 7.90x at 64 roots. With one root, list-append was fastest. The boxed ring did not beat FIFO at 16 or 64 roots by median; overlapping/noisy individual samples make the smaller FIFO/ring differences less conclusive. Prioritize a FIFO scheduler experiment rather than adopting the ring as a performance upgrade.',
              'The boxed ring avoids the demonstrated unboxed layout failure for the tested workloads; no general compiler correction is claimed. Neither ring nor FIFO is newly wired into production scheduling. No merge, deployment, GPU/model execution, compiler update or live configuration change occurred. Self-review only.\n']
    doc.write_text(doc.read_text()+'\n'.join(lines))
    index = Path('docs/experiments/README.md')
    index.write_text(index.read_text()+'\n- [Bounded owning ring and calibrated collection comparison](2026-09-23-bend-owning-collections.md): Search.Tree payloads, layout failure/workaround, and matched-work timings; no scheduler adoption.\n')
    readme = src/'README.md'
    readme.write_text(readme.read_text()+'''\n\n## Owning state and bounded ring follow-up\n\nSee [the owning collection experiment](../../../docs/experiments/2026-09-23-bend-owning-collections.md) for the calibrated list/FIFO/ring comparison.\n`owning_benchmark.py --benchmark` uses actual Search.Tree values and validates every final root.\nRun it as a module with the same compiler pin and an explicit --report path.\nThe bounded single-owner Ring accepts capacities 0..4096. Full pushes return\nthe incoming owning value, and empty pops leave the ring unchanged. Its slots\nare boxed zero/one-element lists to avoid a demonstrated generic Maybe-slot\nlayout mismatch in this pinned compiler. The singleton allocation cost is\nincluded in measured rotations; this is not an allocation-free or lock-free\nqueue. Use the constructor and preserve its invariants. The exposed\nrepresentation and low-level helpers are not a validated deserialization API.\nThe persistent Bend owning collections CI checks both old and new contracts.\nNo scheduler adoption follows from this microbenchmark alone.\n''')
    paths = [str(doc), str(index), str(readme), *[str(p) for p in sorted(dest.iterdir())]]
    subprocess.run(['git', 'add', '--', *paths], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    actual = git('diff','--cached','--name-only').splitlines()
    assert sorted(actual) == sorted(paths)
    assert not git('diff','--name-only')
    tree = git('write-tree')
    subprocess.run(['git','config','user.name','github-actions[bot]'],check=True)
    subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
    commit = git('commit-tree',tree,'-p',PARENT,'-m','native: qualify owning ring and matched-work collection timings')
    subprocess.run(['git','push','--force-with-lease=refs/heads/'+FEATURE+':','origin',commit+':refs/heads/'+FEATURE],check=True)
    print('PUBLISHED_QUALIFIED',commit)

if __name__ == '__main__':
    main()
