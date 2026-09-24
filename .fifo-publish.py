"""Publish only the exact completed FIFO qualification; no re-execution or deployment."""
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

REPO = 'jjoshua2/DeepFin'
BASE = 'bb1847d2d7f3c4e8b94e5dfb9f64bfc87e0c3390'
FEATURE = 'feat/bend-fifo-cohort-20260923'


def api(path):
    return subprocess.check_output(['gh', 'api', path], timeout=90)


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def main():
    run_id = int(os.environ['QUALIFIED_RUN'])
    artifact_id = int(os.environ['QUALIFIED_ARTIFACT'])
    expected_head = os.environ['QUALIFIED_HEAD']
    expected_zip = os.environ['QUALIFIED_SHA256']
    assert git('rev-parse', 'HEAD') == BASE
    assert not git('status', '--porcelain')
    run = json.loads(api(f'repos/{REPO}/actions/runs/{run_id}'))
    assert run['status'] == 'completed' and run['conclusion'] == 'success' and run['head_sha'] == expected_head
    jobs = json.loads(api(f'repos/{REPO}/actions/runs/{run_id}/jobs'))['jobs']
    assert len(jobs) == 1 and jobs[0]['conclusion'] == 'success'
    assert all(s['conclusion'] == 'success' for s in jobs[0]['steps'])
    meta = json.loads(api(f'repos/{REPO}/actions/artifacts/{artifact_id}'))
    assert meta['workflow_run']['id'] == run_id and not meta['expired']
    blob = api(f'repos/{REPO}/actions/artifacts/{artifact_id}/zip')
    assert hashlib.sha256(blob).hexdigest() == expected_zip
    z = zipfile.ZipFile(io.BytesIO(blob))
    manifest = json.loads(z.read('fifo-sources.json'))
    assert len(manifest) == 12
    patch = Path(os.environ['RUNNER_TEMP'])/'qualified-fifo.patch'
    patch.write_bytes(z.read('fifo.patch'))
    subprocess.run(['git','apply','--check',str(patch)],check=True)
    subprocess.run(['git','apply','--index',str(patch)],check=True)
    paths = git('diff','--cached','--name-only').splitlines()
    assert sorted(paths) == sorted(manifest)
    for name, expected in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    summary = json.loads(z.read('qualification/summary.json'))
    assert summary['status'] == 'passed' and summary['scheduler_equivalence_pairs'] == 120
    candidate_c = gzip.decompress(z.read('qualification/runner.c.gz'))
    assert hashlib.sha256(candidate_c).hexdigest() == summary['candidate_c_sha256']
    tests = ET.fromstring(z.read('fifo-python.xml'))
    cases = list(tests.iter('testcase'))
    assert not any(list(tests.iter(tag)) for tag in ('failure','error','skipped'))
    assert len([c for c in cases if c.attrib['classname'].endswith('test_bend_fifo_scheduler')]) == 15
    matrix = json.loads(z.read('qualification/matrix/matrix.json'))
    assert matrix['status'] == 'passed' and matrix['all_final_tree_bits_match_serial'] is True
    reports = {name.removeprefix('qualification/'):json.loads(z.read(name)) for name in z.namelist()
               if name.startswith('qualification/') and name.endswith('.json')}
    pairs = []
    for name, report in reports.items():
        if '/fifo-equivalence' in name:
            assert report['status'] == 'passed' and len(report['observations']) == 10
            for row in report['observations']:
                pairs.append({'configuration':name, **row})
    assert len(pairs) == 120
    controls = {name:r for name,r in reports.items() if '/control' in name or name.startswith('deadlines/')}
    assert len(controls) == 8 and all(r['status']=='passed' for r in controls.values())
    workers = {name:r for name,r in reports.items() if name.startswith('matrix/worker-')}
    assert len(workers)==2 and all(r['status']=='passed' and r['assertions']==1456 and r['reused_calls']==160 for r in workers.values())
    dest = Path('docs/experiments/evidence/fifo-cohort-integration')
    assert not dest.exists(); dest.mkdir(parents=True)
    summary.update({'workflow_run':run_id, 'tested_workflow_commit':expected_head,
                    'artifact_id':artifact_id, 'artifact_zip_sha256':expected_zip,
                    'base_commit':BASE,'focused_python_tests':len(cases),
                    'new_python_tests':15,'source_patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),
                    'async_matrix':matrix,'workers':workers})
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (dest/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    with (dest/'equivalence.csv').open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=list(pairs[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(pairs)
    (dest/'report-hashes.json').write_text(json.dumps({name:hashlib.sha256(z.read('qualification/'+name)).hexdigest() for name in reports},indent=2)+'\n')
    doc=Path('docs/experiments/2026-09-23-fifo-cohort-integration.md')
    text=doc.read_text()
    pending='Pending execution. No full coordinator equivalence result yet.'
    assert text.count(pending)==1
    text=text.replace(pending,'Completed qualification is recorded below; pending execution is superseded.')
    text+='\n\n### Preserved transport setup failure\n\nRun 35942974064 stopped before applying source or running any test because two characters were transcribed incorrectly in the compressed transport. The corrected transport was independently compared with the local authored patch and verified against its unchanged SHA-256. No source, oracle or tolerance was changed to recover it.\n'
    text+='\n\n### Completed qualification\n\n'
    text+=f'Run [{run_id}](https://github.com/{REPO}/actions/runs/{run_id}), job `{jobs[0]["id"]}`, completed every stage. '
    text+=f'All {len(cases)} focused Python cases passed without skips (15 new); explicit static checks and whole-repository Ruff, Basedpyright and Vulture passed.\n\n'
    text+='The freshly generated FIFO coordinator passed the unchanged CBoard/Python-chess matrix: ten batch/width configurations in synchronous and asynchronous modes, two UBSan configurations, and four held-forward cancellation/stop/quit configurations. Existing deadline controls passed in four configurations; compiled timer/compaction checks passed in normal and UBSan modes. The unchanged native worker passed 1,456 assertions and 160 reuse calls per normal/ASan+UBSan mode.\n\n'
    text+='The additional 120 parent/FIFO pairs matched chronological path/reply/batch/rule events, final report root order, every populated final tree field and every non-time work field. Only the six named time/rate values were excluded after strict parsing. These comprise five fixture workloads in sync/async execution across ten normal configurations and two batch-four UBSan configurations. Adapter checks passed 120 scenarios / 840 exact rows per normal and UBSan build; the compiled reversed-requeue mutation was rejected. Repeated modes use the same fixtures, not independent games.\n\n'
    text+='Representation rationale (self-review, not a formal proof): the logical queue is front followed by reversed rear. Removing from its front and appending retired tasks in their original order preserves the old tail-plus-returned-roots schedule. Cancellation is a pointwise transform; the live-root mask is a commutative OR across both lists. Neither operation needs to flatten the ready queue. Pending tasks remain outside it until retirement.\n\n'
    text+='No real model, GPU, full-runner performance, deadline-latency bound, EPS, Elo, formal proof or independent review is claimed. The existing opt-in cohort runner is the integration target, not production UCI or live training. Initial capacity remains 16 roots. Nothing is merged or deployed.\n\n'
    text+=f'Raw build/test reports are retained in artifact `{artifact_id}`; ZIP SHA-256 `{expected_zip}`. Candidate generated C SHA-256 `{summary["candidate_c_sha256"]}`. The exact source manifest, compact controls and 120 equivalence observations are committed under `evidence/fifo-cohort-integration/`. The source manifest describes the tested preregistration before this documentation-only readout; executable files are unchanged.\n'
    doc.write_text(text)
    paths.extend(str(p) for p in sorted(dest.iterdir()))
    subprocess.run(['git','add','--',*paths],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    assert sorted(git('diff','--cached','--name-only').splitlines())==sorted(paths)
    for name, expected in manifest.items():
        if name.endswith(('.bend','.py','.sh')):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    subprocess.run(['git','config','user.name','github-actions[bot]'],check=True)
    subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
    commit=git('commit-tree',git('write-tree'),'-p',BASE,'-m','native: use owning FIFO for bounded cohort scheduling')
    subprocess.run(['git','push','--force-with-lease=refs/heads/'+FEATURE+':','origin',commit+':refs/heads/'+FEATURE],check=True)
    print('PUBLISHED_FIFO_FEATURE',commit)


if __name__=='__main__':
    main()
