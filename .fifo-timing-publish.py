"""One-shot publication; never re-run a timing panel or merge/deploy its candidate."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

REPO = 'jjoshua2/DeepFin'
BASE = 'f45ae893a8eb0ac23285cab0d4c953adfc4f20e0'
FEATURE = 'feat/bend-fifo-timing-qualified-20260923'
SOURCE = {
    'native/bend_engine/multi_root/benchmark_fifo.py': 'b5631ad25fe5e5813ae97dcb0db1b05ac9f0d78b469bb35d868a5398af05a159',
    'native/bend_engine/multi_root/build_fifo_benchmark.sh': 'dfb18e83351d239d816840a881c51d6383a39b4f4f2206c18e2f169d648071d8',
    'tests/test_bend_fifo_benchmark.py': 'c0adff6ef851115b4cab8aaa85796ace6038222fd9908fbd484715fbcedc48e0',
    'docs/experiments/2026-09-23-fifo-runner-timing.md': '2dd0e4409ed12ab0c471048cafc459bbd28c8f9930e7f048231ddcf2653a3ff5',
}


def api(path):
    return subprocess.check_output(['gh','api',path],timeout=90)


def git(*args):
    return subprocess.check_output(['git',*args],text=True).strip()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert git('rev-parse','HEAD') == BASE
    assert not git('status','--porcelain')
    run_id = int(os.environ['QUALIFIED_RUN'])
    artifact = int(os.environ['QUALIFIED_ARTIFACT'])
    source_commit = os.environ['QUALIFIED_HEAD']
    zip_sha = os.environ['QUALIFIED_SHA256']
    run = json.loads(api(f'repos/{REPO}/actions/runs/{run_id}'))
    assert run['status']=='completed' and run['conclusion']=='success' and run['head_sha']==source_commit
    jobs = json.loads(api(f'repos/{REPO}/actions/runs/{run_id}/jobs'))['jobs']
    assert len(jobs)==1 and jobs[0]['conclusion']=='success'
    assert all(s['conclusion']=='success' for s in jobs[0]['steps'])
    metadata = json.loads(api(f'repos/{REPO}/actions/artifacts/{artifact}'))
    assert metadata['workflow_run']['id']==run_id and not metadata['expired']
    blob = api(f'repos/{REPO}/actions/artifacts/{artifact}/zip')
    assert sha(blob)==zip_sha
    z = zipfile.ZipFile(io.BytesIO(blob))
    raw = z.read('fifo-timing.json'); report = json.loads(raw)
    build = json.loads(z.read('fifo-timing-build/build.json'))
    tests = ET.fromstring(z.read('fifo-timing-tests.xml'))
    assert len(list(tests.iter('testcase')))==41
    assert not any(list(tests.iter(k)) for k in ('failure','error','skipped'))
    assert report['status']=='passed' and len(report['samples'])==72
    assert len(report['diagnostic_checks'])==6 and len(report['calibration'])==12
    assert report['binaries']=={'list':build['sha256']['list_binary'],'fifo':build['sha256']['fifo_binary']}
    assert build['sha256']['reference_c']=='fcb787241c29f10c0fa33fbd8983f9630d3ab52914c109de2449b52a93251a24'
    assert build['sha256']['candidate_c']=='7f348a6ad58c7e3abebbf876f4ada8862c9f6dbc5856f445e1e2310346d68b74'
    subprocess.run(['git','fetch','--depth=1','origin',source_commit],check=True)
    subprocess.run(['git','checkout',source_commit,'--',*SOURCE],check=True)
    for name,expected in SOURCE.items(): assert sha(Path(name).read_bytes())==expected,name
    for key,name in [('benchmark','benchmark_fifo.py'),('builder','build_fifo_benchmark.sh'),
                     ('parser','verify.py'),('equivalence','verify_fifo.py'),('callback','test_backend.c')]:
        assert sha((Path('native/bend_engine/multi_root')/name).read_bytes())==build['sha256'][key],name
    assert sha(Path('native/bend_engine/batch_backend/async_batch.cpp').read_bytes())==build['sha256']['worker']
    from native.bend_engine.multi_root.benchmark_fifo import summarize
    assert summarize(report['samples'])==report['summary']
    observations=[]
    for row in report['samples']:
        assert len(row['observations'])==row['repeats']
        for key in ('process_seconds','coordinator_seconds'):
            assert math.isclose(sum(s[key] for s in row['observations']),row[key],rel_tol=0,abs_tol=1e-12)
        for i,sample in enumerate(row['observations']):
            observations.append({k:v for k,v in row.items() if k not in ('observations','process_seconds','coordinator_seconds')}
                                | {'repeat':i} | sample)
    destination=Path('docs/experiments/evidence/fifo-runner-timing')
    assert not destination.exists();destination.mkdir(parents=True)
    with (destination/'samples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(observations[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(observations)
    summary={k:v for k,v in report.items() if k!='samples'}
    summary.update({'workflow_run':run_id,'tested_commit':source_commit,'base_commit':BASE,'artifact_id':artifact,
                    'artifact_zip_sha256':zip_sha,'raw_report_sha256':sha(raw),'source_sha256':SOURCE,
                    'focused_tests_passed':41,'measurement_groups':72,'measured_processes':len(observations),
                    'samples_file':'samples.csv'})
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (destination/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    (destination/'cpu.txt').write_bytes(z.read('fifo-timing-build/cpu.txt'))
    doc=Path('docs/experiments/2026-09-23-fifo-runner-timing.md')
    pending='Pending execution. No full-runner speedup has been established by this record.'
    assert doc.read_text().count(pending)==1
    text=doc.read_text().replace(pending,'The complete, source-qualified panel is recorded below. A passed run means valid measurements, not a speedup verdict.')
    text+='\n\n### Setup failure and local checks\n\nRun 35945422686 stopped before native builds or measurement on three new-test Ruff findings: an intentional regex needed a raw string and two combined assertions needed splitting. All expectations and measurement rules are unchanged. The 26 new admission tests passed locally. A local extended O3 semantic smoke reached five completed configurations before the command time limit; it is not a completed timing panel and is not used as performance evidence.\n'
    text+=f'\n\n### Completed hosted panel\n\nRun [{run_id}](https://github.com/{REPO}/actions/runs/{run_id}) on `{source_commit}` passed all stages. All 41 focused Python cases passed without skips (26 new and 15 existing), explicit static checks and whole-repository Ruff/Basedpyright/Vulture passed, and source reconciliation found no changes.\n\n'
    text+=f'The six diagnostic pairs matched full trees/events and work. All {len(observations)} measured child executions across 72 arm-groups matched diagnostics-off root summaries and non-time accounting. Twelve warmup observations were retained but excluded. All raw per-child timings and output hashes are in samples.csv; source/build identities, configuration, warmups, diagnostic work counts and recomputed summaries are retained beside it.\n\n'
    text+='Times below are coordinator medians per invocation (group medians divided by the common repetition count). Ratios are medians of paired group ratios, not ratios of unpaired medians. Above 1 favors FIFO.\n\n'
    text+='| Case/mode | Repeats | List ms | FIFO ms | Paired list/FIFO | Observed paired range | Screen decision |\n|---|---:|---:|---:|---:|---|---|\n'
    for name,row in report['summary'].items():
        m=row['coordinator_seconds'];r=row['repeats']
        ratio='unresolved' if m['median_list_over_fifo'] is None else f"{m['median_list_over_fifo']:.4f}"
        span='below floor' if m['paired_ratios'] is None else f"{min(m['paired_ratios']):.4f}..{max(m['paired_ratios']):.4f}"
        text+=f"| {name} | {r} | {1000*m['median_list']/r:.2f} | {1000*m['median_fifo']/r:.2f} | {ratio} | {span} | {m['decision']} |\n"
    text+='\nWhole-process timing (initialization/reporting/exit included):\n\n| Case/mode | Median paired list/FIFO | Screen decision |\n|---|---:|---|\n'
    for name,row in report['summary'].items():
        m=row['process_seconds'];ratio='unresolved' if m['median_list_over_fifo'] is None else f"{m['median_list_over_fifo']:.4f}"
        text+=f"| {name} | {ratio} | {m['decision']} |\n"
    text+='\nInterpret each result against the preregistered all-six-pairs/5% rule. Inconclusive means this screen did not establish that practical gain or regression; it is not equivalence proof. The older 3.65x/7.90x owning-loop measurements cannot be substituted for these full-coordinator measurements. This fixed batch-4, 146-channel callback screen does not establish actual model/GPU speed, playing strength, other batch sizes or performance beyond the 16-root cap. No scheduler, compiler, model, live setting or arena size was changed during this continuation. Self-review only. Nothing merged or deployed.\n'
    text+=f'\nArtifact `{artifact}`, ZIP SHA-256 `{zip_sha}`. Source hashes describe the tested preregistration before this documentation-only readout. Historical generated C inputs remain in the two previously qualified artifacts; their ordinary retention may expire. Recreating a historical input requires separately qualifying code generation, not silently accepting a different hash.\n'
    text+='\n### Reproduction\n\n```sh\nCC=clang-18 CXX=clang++-18 bash native/bend_engine/multi_root/build_fifo_benchmark.sh PARENT.c FIFO.c NEW_BUILD\npython -m native.bend_engine.multi_root.benchmark_fifo --reference NEW_BUILD/list --candidate NEW_BUILD/fifo --report NEW_REPORT.json\n```\n'
    doc.write_text(text)
    index=Path('docs/experiments/README.md')
    index.write_text(index.read_text()+'\n- [Matched full-coordinator FIFO timing](2026-09-23-fifo-runner-timing.md): exact list/FIFO sources, optimized callback runner, strict realized-work checks and paired timings; no model/GPU claim.\n')
    readme=Path('native/bend_engine/multi_root/README.md')
    readme.write_text(readme.read_text()+'\n\n## Matched FIFO timing\n\nThe opt-in `build_fifo_benchmark.sh` and `benchmark_fifo.py` compare exact qualified list/FIFO snapshots using the unchanged deterministic callback. See [the completed experiment](../../../docs/experiments/2026-09-23-fifo-runner-timing.md) for source hashes, commands, raw observations and measurement limits. Ordinary pytest exercises admission logic only; this is not a new production mode or real-model performance claim.\n')
    paths=[*SOURCE,str(index),str(readme),*[str(p) for p in sorted(destination.iterdir())]]
    subprocess.run(['git','add','--',*paths],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    assert sorted(git('diff','--cached','--name-only').splitlines())==sorted(paths)
    for name,expected in SOURCE.items():
        if name.endswith(('.py','.sh')):assert sha(Path(name).read_bytes())==expected,name
    subprocess.run(['git','config','user.name','github-actions[bot]'],check=True)
    subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
    commit=git('commit-tree',git('write-tree'),'-p',BASE,'-m','perf: retain matched full-coordinator FIFO measurements and admission tests')
    subprocess.run(['git','push','--force-with-lease=refs/heads/'+FEATURE+':','origin',commit+':refs/heads/'+FEATURE],check=True)
    print('PUBLISHED_SOURCE_IDENTICAL_TIMING',commit)


if __name__=='__main__':
    main()
