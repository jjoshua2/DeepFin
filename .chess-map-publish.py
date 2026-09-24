"""Publish a completed timing panel; never rerun measurements or merge."""
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
BASE = '89bd90c461ebca9654556ecdf5ac652285cd28a1'
MEASURED = '924bd4f6cc1c6dc66cf57839e1d3a9ac6aa5fa30'
RUN = 36064709080
FEATURE = 'feat/bend-chess-map-timing-qualified-20260924'
DOC = 'docs/experiments/2026-09-24-chess-map-timing.md'


def api(path):
    return subprocess.check_output(['gh', 'api', path], timeout=90)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    from native.bend_engine.u64_map_probe import benchmark as b
    from native.bend_engine.u64_map_probe.workload import Workload
    artifact = int(os.environ['TIMING_ARTIFACT'])
    artifact_sha = os.environ['TIMING_SHA256']
    root = Path.cwd()
    run = json.loads(api(f'repos/{REPO}/actions/runs/{RUN}'))
    assert run['status'] == 'completed' and run['conclusion'] == 'success' and run['head_sha'] == MEASURED
    jobs = json.loads(api(f'repos/{REPO}/actions/runs/{RUN}/jobs'))['jobs']
    assert len(jobs) == 1 and all(s['conclusion']=='success' for s in jobs[0]['steps'])
    metadata = json.loads(api(f'repos/{REPO}/actions/artifacts/{artifact}'))
    assert metadata['workflow_run']['id'] == RUN and not metadata['expired']
    raw_zip = api(f'repos/{REPO}/actions/artifacts/{artifact}/zip')
    assert sha(raw_zip) == artifact_sha
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    raw = z.read('chess-map-timing/report.json')
    r = json.loads(raw)
    assert r['status']=='passed' and r['measured'] is True and r['workload_source']=='chess'
    preflight = json.loads(z.read('chess-map-preflight.json'))
    assert preflight['status']=='passed' and preflight['map_source_sha256']==r['source_sha256']
    module = root/'native/bend_engine/u64_map_probe'
    for name,digest in r['source_sha256'].items():
        assert sha((module/name).read_bytes())==digest,name
    for name,digest in r['chess_corpus']['producer_source_sha256'].items():
        assert sha((root/name).read_bytes())==digest,name
    assert {c['name']:c['records_sha256'] for c in r['chess_corpus']['corpora']}==preflight['corpus_record_sha256']
    assert r['chess_corpus']['cboard_extension_sha256']==preflight['extension_sha256']
    cases = {c['name']:Workload(c['name'],c['bits'],tuple(map(tuple,c['initial'])),tuple(map(tuple,c['ops']))) for c in r['cases']}
    assert len(cases)==12 and r['contract_executions']==144
    assert len(r['reference_checks'])==4 and all((x['cases'],x['operations'])==(36,12061) for x in r['reference_checks'])
    assert sum(row['phase']=='measurement' for row in r['samples'])==144
    assert b.summarize(r['samples'],list(cases))==r['summary']
    for i,row in enumerate(r['samples']):
        stem=f"chess-map-timing/{i:04}-{row['case']}-{row['arm']}-{row['phase']}"
        text=z.read(stem+'.stdout')
        assert sha(text)==row['stdout_sha256'] and not z.read(stem+'.stderr')
        assert b.parse(text.decode(),cases[row['case']],row['rounds'])==row['milliseconds']
        assert sha(b.encode(cases[row['case']],row['rounds']).encode())==row['input_sha256']
    tests=ET.fromstring(z.read('chess-map-timing-tests.xml'))
    assert len(list(tests.iter('testcase')))==102
    assert not any(list(tests.iter(tag)) for tag in ('failure','error','skipped'))
    build=json.loads(z.read('chess-map-timing/build.json'))
    assert build['checkout_sha']==MEASURED and int(build['workflow_run'])==RUN
    subprocess.run(['git','fetch','--depth=1','origin',BASE],check=True)
    work=Path(os.environ['RUNNER_TEMP'])/'chess-map-measured'
    subprocess.run(['git','worktree','add','--detach',str(work),BASE],check=True)
    dest=work/'docs/experiments/evidence/chess-map-timing'
    assert not dest.exists();dest.mkdir(parents=True)
    samples=r.pop('samples')
    with (dest/'samples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(samples[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(samples)
    r['cases']=[{'name':c['name'],'bits':c['bits'],'entries':len(c['initial']),'operations_per_cycle':len(c['ops']),
                 'workload_sha256':sha(json.dumps(c,sort_keys=True).encode())} for c in r['cases']]
    for c in r['chess_corpus']['corpora']:
        c.pop('records')
    r.update({'measurement_run':RUN,'tested_commit':MEASURED,'base_commit':BASE,'artifact_id':artifact,
              'artifact_zip_sha256':artifact_sha,'raw_report_sha256':sha(raw),'python_tests_passed':102,
              'measured_observations':144,'calibration_observations':sum(x['phase']=='calibration' for x in samples),
              'samples_file':'samples.csv','preregistration_sha256':sha((root/DOC).read_bytes())})
    (dest/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    (dest/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    (dest/'cpu.txt').write_bytes(z.read('chess-map-timing/cpu.txt'))
    document=(root/DOC).read_text()
    assert document.count('Pending the single preregistered hosted panel.')==1
    document=document.replace('Pending the single preregistered hosted panel.','The completed single-host panel is recorded below. No source or measurement rule changed after preregistration.')
    document+=f'\n\n### Completed qualification\n\n[Run {RUN}](https://github.com/{REPO}/actions/runs/{RUN}), job `{jobs[0]["id"]}`, completed every stage on the first attempt. '
    document+='All 102 existing map Python tests passed without skips/failures/errors; focused Ruff/Basedpyright passed. The corpus source, four record hashes and six identity examples matched the already-qualified record before timing. The two map implementations each passed all 36 replay cases / 12,061 operations in explicit BMI2/POPCNT and UBSan builds. All 144 zero/one/three-cycle driver checks passed. Build modes repeat fixtures, not independent games. No new tests or full broad CPU-suite run is claimed by this documentation-only continuation.\n\n'
    document+=f'The single timed panel contains 144 measured observations and {r["calibration_observations"]} separately retained calibration observations. Every observation passed the returned-result checksum and complete final-dictionary comparison. All input/output hashes and paired summaries were recomputed from downloaded records before publication.\n\n'
    document+='| Corpus/workload | Cycles/sample | Hash median ms | Scan median ms | Median paired scan/hash | Decision |\n|---|---:|---:|---:|---:|---|\n'
    for name,item in r['summary'].items():
        ratio='below floor' if item['median_scan_over_hash'] is None else f"{item['median_scan_over_hash']:.3f}"
        document+=f"| {name} | {item['rounds']} | {item['median_ms']['hash']} | {item['median_ms']['scan']} | {ratio} | {item['decision']} |\n"
    document+='\nRatios are medians of paired scan/hash times, not ratios of the unpaired median columns. Above one favors hashing. Read the decisions against the preregistered all-six-pairs and duration rules; a below-floor or inconclusive result is retained rather than promoted. This compares the native operation loops at 64 stored keys, not whole-engine throughput, cache reuse safety, or a tuned alternative map. The structural-key/history/collision distinctions from the preceding replay remain prerequisites for any consumer.\n\n'
    document+='The practical use of this result is to choose between these two candidates for comparable numeric-key workloads. It does not justify transplanting the map into the production structural graph or neural cache without its separate identity, capacity and lifecycle contract. Neither historical synthetic results nor this panel establishes a universal fastest map.\n\n'
    document+=f'All {len(samples)} contract/calibration/measurement observations are committed in [samples.csv](evidence/chess-map-timing/samples.csv), with compact [summary and source identities](evidence/chess-map-timing/summary.json), generated-code/binary fingerprints and host details. Full workloads, legal paths, raw stdout/stderr and JUnit are retained in artifact `{artifact}`; ZIP SHA-256 `{artifact_sha}`. The source and native extension fingerprints describe this run; the extension is not presumed binary-identical to an earlier build. Artifact retention is finite, but generators, compact results and raw timing observations are committed.\n\n'
    document+='The isolated measurement/publication workflows are absent from the feature diff. No map, benchmark driver, engine key function, compiler, production workflow, default or live setting changed. All three workflows on parent `89bd90c4` had passed before this experiment; new-head checks are separate. Self-review only; no independent review, formal proof, real-model/GPU result, merge or deployment.\n'
    doc=work/DOC;assert not doc.exists();doc.write_text(document)
    index=work/'docs/experiments/README.md'
    index.write_text(index.read_text()+'\n- [Matched CBoard-key map timing](2026-09-24-chess-map-timing.md): unchanged chess-derived key fixtures and explicit CPU target; paired hash-versus-scan operation timings, not production search/cache throughput.\n')
    readme=work/'native/bend_engine/u64_map_probe/README.md'
    readme.write_text(readme.read_text()+'\n\n## Chess-derived timing result\n\nThe separately preregistered [CBoard-key timing panel](../../../docs/experiments/2026-09-24-chess-map-timing.md) measures the existing `--chess --measure` path on the explicit CPU target. The source, keys and operation tapes are unchanged from their correctness qualification. It preserves paired results and all timing observations; it does not qualify a production cache consumer or add a CI speed gate.\n')
    paths=[DOC,str(index.relative_to(work)),str(readme.relative_to(work)),*[str(p.relative_to(work)) for p in sorted(dest.iterdir())]]
    assert len(paths)==7
    subprocess.run(['git','-C',str(work),'add','--',*paths],check=True)
    subprocess.run(['git','-C',str(work),'diff','--cached','--check'],check=True)
    actual=subprocess.check_output(['git','-C',str(work),'diff','--cached','--name-only'],text=True).splitlines()
    assert sorted(actual)==sorted(paths)
    subprocess.run(['git','-C',str(work),'config','user.name','github-actions[bot]'],check=True)
    subprocess.run(['git','-C',str(work),'config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
    tree=subprocess.check_output(['git','-C',str(work),'write-tree'],text=True).strip()
    commit=subprocess.check_output(['git','-C',str(work),'commit-tree',tree,'-p',BASE,'-m','perf: record matched numeric-map timings on unchanged CBoard keys'],text=True).strip()
    subprocess.run(['git','-C',str(work),'push','--force-with-lease=refs/heads/'+FEATURE+':','origin',commit+':refs/heads/'+FEATURE],check=True)
    print('PUBLISHED_MEASURED_CHESS_MAP',commit)


if __name__=='__main__':
    main()
