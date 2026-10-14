"""Serial fail-closed qualification of the actual white-kingside consumer.

All generated files, compiler streams, mutant sources and receipts are external.
The watchdog must be the independently reviewed unlimited-VA sampled-RSS v2
adapter identified below. A resource admission is required by repository policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
PROJECT = ENGINE.parents[1]
PREFIX = 'standalone/proofs/castle_valid/'
BASE = '0ec611bad5b79682936e7dc5e36dd2498a227b67'
BASE_TREE = 'cf8fdc1681f49a3510153ffa5eb0acccd58e573c'
PIN = 'aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae'
FINGERPRINT = 'd9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4'
WATCHDOG = 'bdafed929dd1d6ed5a4cf72148340423da1ca7bd2513ce78e45ee8beab57b279'
CHECK_SCRIPT = 'set -eu\nulimit -c 0\nulimit -f 16384\nexec /usr/bin/time -v -o "$1" taskset -c "$2" "$3" --smol "$4" "$5" --check-only >"$6" 2>"$7"\n'
ENTRIES = ('consumer.bend', 'Fixtures.bend', 'GeneratedNegative.bend')
CONTROLS = (
    ('remove-coherent-EP', 'Geometry.bend', 'generated_ep_victim',
     'coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool}', 'coherent: Unit'),
    ('remove-parent-right', 'Rights.bend', 'parent_when',
     'parent: {Pos.valid_right(b,bit,rook,king,side) == True{} : Bool}', 'parent: Unit'),
    ('remove-parent-rows', 'Home.bend', 'source_king',
     '+rows: {Board.valid(b) == True{} : Bool}', '+rows: Unit'),
    ('wrong-own-castle-clearance', 'Castles.bend', 'white_empty',
     ')),1) == 0 : U32}', ')),1) == 1 : U32}'),
    ('wrong-generated-EP-flag', 'GeneratedNegative.bend', 'full_legal_list',
     'Chess.Ply{23,15,0,1}', 'Chess.Ply{23,15,0,0}'),
)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(PROJECT), *args], text=True, timeout=30).strip()

def blob(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()

def closure(entry: Path) -> list[str]:
    path = SUITE.parent / 'table_preservation/focused.py'
    spec = importlib.util.spec_from_file_location('castle_valid_closure', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted(module.closure(entry, ENGINE))

def compiler_identity(root: Path) -> dict:
    files = []
    for part in ('bend2/base.bend', 'bend2/bend.ts', 'bend2/comp.ts', 'bend2/main.ts', 'bend2/effs'):
        p = root / part
        files.extend(p.rglob('*') if p.is_dir() else [p])
    rows = {}
    for p in files:
        if p.is_symlink():
            raise RuntimeError('compiler symlink')
        if p.is_file():
            rows[str(p.relative_to(root))] = digest(p)
    value = hashlib.sha256(b''.join(p.encode() + b'\0' + bytes.fromhex(h) for p,h in sorted(rows.items()))).hexdigest()
    if len(rows) != 84 or value != FINGERPRINT:
        raise RuntimeError('compiler source identity mismatch')
    return {'pin': PIN, 'fingerprint': value, 'files': rows}

def mutate(path: Path, declaration: str, old: str, new: str) -> None:
    source = path.read_text()
    found = re.search(r'^def ' + re.escape(declaration) + r'\(', source, re.M)
    if not found:
        raise RuntimeError('missing mutation declaration')
    next_def = re.search(r'^def ', source[found.end():], re.M)
    end = found.end() + next_def.start() if next_def else len(source)
    section = source[found.start():end]
    if section.count(old) != 1:
        raise RuntimeError('nonunique mutation anchor')
    path.write_text(source[:found.start()] + section.replace(old,new) + source[end:])

def strict(receipt: dict, text: str, declaration: str) -> bool:
    forbidden = ('more than once', 'cannot infer', 'not inferable', 'non-inferrable',
                 'annotated term', 'a parameter or field', 'a match on', 'a defined name',
                 'decreasing self-call', 'rangeerror', 'maximum call stack', 'out of memory',
                 'outofmemory', 'no such file', 'unexpected', 'unsafe', 'warning',
                 '- expected : data\n', '- expected : type\n', '- observed : data\n', '- observed : type\n')
    return (receipt['reason'] == 'EXIT' and receipt['returncode'] == 1
            and receipt['remaining_live_owned'] == 0
            and '- expected' in text and '- observed' in text
            and bool(re.search(r'Location:\s*(?:[\w./]+\.)*' + re.escape(declaration) + r'\b', text))
            and not any(x in text.lower() for x in forbidden))

def validate_admission(admission: dict, cpus: set[int], wall_seconds: int) -> dict:
    limits = admission.get('admitted', {})
    if (admission.get('status') != 'ACCEPTED_SERIAL_CASTLING_CHECKER_CPU29_30_NICE19_6GIB_SAMPLED_RSS_600S'
            or admission.get('isolated_worktree') != str(PROJECT.resolve())
            or admission.get('observed_git_HEAD') != BASE or admission.get('checker_revision') != PIN
            or admission.get('reported_verified_84_file_fingerprint') != FINGERPRINT
            or set(limits.get('CPU_affinity', [])) != cpus
            or limits.get('nice') != 19 or limits.get('serial_only') is not True
            or limits.get('max_simultaneous_checker_jobs') != 1 or limits.get('GPU_use') is not False
            or limits.get('sampled_aggregate_RSS_threshold_bytes') != 6*1024**3
            or limits.get('sample_interval_seconds') != 0.05
            or limits.get('hard_RSS_ceiling_claim') is not False
            or limits.get('RSS_sampling_overshoot_acknowledged') is not True
            or limits.get('initial_execution_seconds_per_check', 0) < wall_seconds
            or limits.get('stdout_max_bytes') != 16*1024**2 or limits.get('stderr_max_bytes') != 16*1024**2):
        raise RuntimeError('resource admission does not authorize this exact serial checker invocation')
    return limits

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--bun', type=Path, required=True)
    parser.add_argument('--watchdog', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--cpus', default='29,30')
    parser.add_argument('--wall-seconds', type=int, default=600)
    parser.add_argument('--reuse-consumer-report', type=Path)
    args = parser.parse_args()
    out = args.evidence_dir.resolve()
    if out.is_relative_to(PROJECT.resolve()) or out.exists():
        raise RuntimeError('evidence requires a fresh external directory')
    cpus = {int(c) for c in args.cpus.split(',')}
    if len(cpus) > 2 or not cpus or not cpus.issubset(os.sched_getaffinity(0)) or not 1 <= args.wall_seconds <= 600:
        raise RuntimeError('CPU or wall bound exceeds this qualification admission')
    if digest(args.watchdog) != WATCHDOG:
        raise RuntimeError('unqualified watchdog')
    admission_hash = digest(args.admission)
    admission = json.loads(args.admission.read_text())
    limits = validate_admission(admission, cpus, args.wall_seconds)
    version = subprocess.check_output([str(args.bun), '--version'], text=True, timeout=15).strip()
    if version != '1.4.2' or git('rev-parse', BASE+'^{tree}') != BASE_TREE:
        raise RuntimeError('Bun or base identity mismatch')
    if git('status', '--porcelain', '--ignored'):
        raise RuntimeError('frozen qualification requires a clean worktree')
    changed = git('diff', '--name-only', BASE, 'HEAD').splitlines()
    if not changed or any(not p.startswith('native/bend_engine/'+PREFIX) for p in changed):
        raise RuntimeError('candidate delta outside new suite')
    os.sched_setaffinity(0, cpus)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='2',
        BUN_JSC_jitMemoryReservationSize='67108864', BUN_JSC_forceRAMSize='6442450944',
        BUN_JSC_validateOptions='true', DO_NOT_TRACK='1', BEND_NO_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1')
    runtime = compiler_identity(args.compiler)
    out.mkdir(parents=True)
    report = {'gate': 'NOT_COMPLETED', 'base': BASE, 'base_tree': BASE_TREE,
        'head': git('rev-parse','HEAD'), 'tree': git('rev-parse','HEAD^{tree}'), 'compiler': runtime,
        'bun_version': version, 'watchdog_sha256': WATCHDOG, 'admission_sha256': admission_hash,
        'validated_admission': {'status': admission['status'], 'admitted': limits},
        'resource_limits': {'cpus': sorted(cpus), 'nice': 19, 'rss_bytes': 6*1024**3,
            'wall_seconds': args.wall_seconds, 'stream_bytes': 16*1024**2, 'hard_rss_ceiling': False},
        'scope': 'White kingside child valid_right after actual generated complete Ply; no zero-right or child freshness assumption.',
        'positive_checks': [], 'negative_controls': [], 'guards': []}
    save = lambda: (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    paths = set().union(*(set(closure(SUITE/e)) for e in ENTRIES))
    paths.update(str(p.relative_to(ENGINE)) for p in SUITE.iterdir() if p.is_file())
    paths.update({'standalone/toolchain.json', 'standalone/verify_compiler.js', 'standalone/proofs/table_preservation/focused.py'})
    before = {p: digest(ENGINE/p) for p in sorted(paths)}
    report['sources'] = before
    head_blobs = {}
    for p in paths:
        if (ENGINE/p).is_symlink():
            raise RuntimeError('source symlink: '+p)
        expected = git('rev-parse', 'HEAD:native/bend_engine/'+p)
        if blob(ENGINE/p) != expected:
            raise RuntimeError('source differs from frozen Git head: '+p)
        head_blobs[p] = expected
    report['qualified_git_blobs'] = head_blobs
    reused = {}
    for p in paths:
        if p.startswith(PREFIX):
            continue
        baseline = git('rev-parse', BASE+':native/bend_engine/'+p)
        if blob(ENGINE/p) != baseline:
            raise RuntimeError('reused dependency differs from exact base: '+p)
        reused[p] = baseline
    report['reused_git_blobs'] = reused
    snapshot = out/'source-snapshot'
    for p in paths:
        destination=snapshot/p
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ENGINE/p,destination)
    watch = runpy.run_path(str(args.watchdog))
    def guard(stage: str) -> None:
        if (before != {p:digest(ENGINE/p) for p in paths}
                or before != {p:digest(snapshot/p) for p in paths}
                or compiler_identity(args.compiler) != runtime
                or git('rev-parse','HEAD') != report['head'] or git('status','--porcelain','--ignored')
                or digest(args.admission) != admission_hash or digest(args.watchdog) != WATCHDOG):
            raise RuntimeError('identity drift at '+stage)
        report['guards'].append({'stage':stage,'utc':datetime.now(timezone.utc).isoformat()})
        save()
    def check(name: str, entry: Path) -> tuple[dict,str]:
        guard('before-'+name)
        logs=out/'checks'/name
        logs.mkdir(parents=True)
        script=logs/'run.sh'
        script.write_text(CHECK_SCRIPT)
        command=['/bin/bash',str(script),str(logs/'time.txt'),args.cpus,str(args.bun),str(args.compiler/'bend2/main.ts'),str(entry),str(logs/'stdout.txt'),str(logs/'stderr.txt')]
        receipt=watch['run'](command,6*1024**3,args.wall_seconds)
        raw=(logs/'stdout.txt').read_bytes()+(logs/'stderr.txt').read_bytes()
        receipt.update(name=name,entry=str(entry),command=command,
            stdout_sha256=digest(logs/'stdout.txt'),stderr_sha256=digest(logs/'stderr.txt'),
            passed=receipt['reason']=='EXIT' and receipt['returncode']==0
                and (logs/'stdout.txt').read_bytes()==b'All terms check.\n' and not (logs/'stderr.txt').read_bytes()
                and receipt['remaining_live_owned']==0)
        (logs/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        guard('after-'+name)
        print(json.dumps({k:receipt[k] for k in ('name','reason','returncode','wall_seconds','peak_aggregate_rss_bytes','passed')}),flush=True)
        return receipt,raw.decode(errors='replace')
    try:
        save()
        for entry in ENTRIES:
            if entry == 'consumer.bend' and args.reuse_consumer_report:
                guard('before-reuse-consumer')
                old_path=args.reuse_consumer_report.resolve()
                old=json.loads(old_path.read_text())
                logs=old_path.parent/'checks/consumer'
                old_receipt=json.loads((logs/'receipt.json').read_text())
                old_entry=old_path.parent/'source-snapshot'/PREFIX/entry
                bound_command=['/bin/bash',str(logs/'run.sh'),str(logs/'time.txt'),args.cpus,
                    str(args.bun),str(args.compiler/'bend2/main.ts'),str(old_entry),str(logs/'stdout.txt'),str(logs/'stderr.txt')]
                previous=[r for r in old['positive_checks'] if r['name']=='consumer']
                if (previous != [old_receipt] or not old_receipt['passed']
                        or old_receipt['reason']!='EXIT' or old_receipt['returncode']!=0
                        or old_receipt['remaining_live_owned']!=0 or old['compiler']!=runtime or old['bun_version']!=version
                        or old_receipt['entry']!=str(old_entry) or old_receipt['command']!=bound_command
                        or (logs/'run.sh').read_bytes()!=CHECK_SCRIPT.encode()
                        or old['admission_sha256']!=admission_hash or old['watchdog_sha256']!=WATCHDOG
                        or old['resource_limits']!=report['resource_limits']
                        or git('rev-parse',old['head']+'^{tree}')!=old['tree']):
                    raise RuntimeError('previous consumer receipt or execution identity differs')
                checked=closure(SUITE/entry)
                for p in checked:
                    old_source=old_path.parent/'source-snapshot'/p
                    if (old['sources'].get(p)!=before[p] or digest(old_source)!=before[p]
                            or old['qualified_git_blobs'].get(p)!=head_blobs[p]
                            or git('rev-parse',old['head']+':native/bend_engine/'+p)!=head_blobs[p]):
                        raise RuntimeError('previous consumer source differs: '+p)
                if ((logs/'stdout.txt').read_bytes()!=b'All terms check.\n' or (logs/'stderr.txt').read_bytes()
                        or digest(logs/'stdout.txt')!=old_receipt['stdout_sha256']
                        or digest(logs/'stderr.txt')!=old_receipt['stderr_sha256']
                        or not all(any(g['stage']==stage for g in old['guards']) for stage in ('before-consumer','after-consumer'))):
                    raise RuntimeError('previous raw consumer evidence differs')
                receipt=dict(old_receipt)
                receipt['reused_from']={'report':str(old_path),'report_sha256':digest(old_path),
                    'head':old['head'],'tree':old['tree'],'identical_bend_inputs':len(checked),
                    'bound_check_script_sha256':digest(logs/'run.sh'),'rusage_sha256':digest(logs/'time.txt')}
                guard('after-reuse-consumer')
            else:
                receipt,_=check(Path(entry).stem,snapshot/PREFIX/entry)
            report['positive_checks'].append(receipt)
            save()
            if not receipt['passed']:
                raise RuntimeError('positive failed: '+entry)
        for name,file,declaration,old,new in CONTROLS:
            mutant=out/'mutants'/name
            shutil.copytree(snapshot,mutant)
            target=mutant/PREFIX/file
            mutate(target,declaration,old,new)
            expected=dict(before)
            expected[PREFIX+file]=digest(target)
            receipt,raw=check(name,target)
            observed={str(p.relative_to(mutant)):digest(p) for p in mutant.rglob('*') if p.is_file()}
            receipt.update(declaration=declaration,anchor=old,replacement=new,mutant_sha256=digest(target),
                exact_mutation=expected==observed,semantic_rejection=strict(receipt,raw,declaration))
            report['negative_controls'].append(receipt)
            save()
            if not receipt['exact_mutation'] or not receipt['semantic_rejection']:
                raise RuntimeError('control lacks strict intended-obligation rejection: '+name)
        guard('final')
        report['gate']='PASS'
    except Exception as failure:
        report['gate']='FAIL'
        report['failure']=repr(failure)
        raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        save()

if __name__ == '__main__':
    main()
