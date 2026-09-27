"""Opt-in complete consumer and classified semantic/import/output controls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

SUITE=Path(__file__).resolve().parent
ENGINE=SUITE.parents[2]
LAWS=['attack_query_preserves_table','ordinary_scan_preserves_table','castling_side_preserves_table',
      'prepared_filter_preserves_table','legal_moves_preserves_table']


def digest(raw: bytes) -> str: return hashlib.sha256(raw).hexdigest()


def closure(entry: Path,root: Path) -> dict[str,str]:
    out={};root=root.resolve()
    def visit(path: Path) -> None:
        # Resolve dot components without resolving symlinks, then inspect every component.
        path=Path(os.path.abspath(path));rel=path.relative_to(root)
        cursor=root
        for part in rel.parts:
            cursor=cursor/part
            if cursor.is_symlink(): raise ValueError('symlinked dependency')
        if not path.is_file(): raise ValueError('missing/nonregular source')
        name=rel.as_posix()
        if not (name.startswith('standalone/') or name in {'legal_probe/Chess.bend','bitboard_probe/Sliders.bend'}):
            raise ValueError('foreign source path')
        if name in out: return
        raw=path.read_bytes();out[name]=digest(raw)
        text='\n'.join(line.split('#',1)[0] for line in raw.decode().splitlines())
        if '@unsafe' in text or '?' in text: raise ValueError('unsafe dependency/proof hole')
        for imported in re.findall(r'^\s*import\s+(\S+)',text,re.MULTILINE):
            if imported=='Base': continue
            if not re.fullmatch(r'\.{1,2}/[A-Za-z0-9_/.]+\.bend',imported): raise ValueError('foreign import')
            visit(path.parent/imported)
    visit(entry);return dict(sorted(out.items()))


def manifest(suite: Path) -> None:
    if re.findall(r'^law (\w+):',(suite/'LAWS.bend').read_text(),re.MULTILINE)!=LAWS: raise ValueError('law inventory')
    proof=(suite/'PROOF.bend').read_text()
    if re.findall(r'^def Laws\.(\w+)\(',proof,re.MULTILINE)!=LAWS: raise ValueError('proof inventory')
    if 'import ./LAWS.bend as Laws' not in proof: raise ValueError('law import')
    if 'import ./PROOF.bend as Proof' not in (suite/'consumer.bend').read_text(): raise ValueError('proof import')


def dispatch_audit() -> dict:
    text=(SUITE/'Queries.bend').read_text().split('def attack_bits(',1)[1].split('def attack_model(',1)[0]
    cases=re.findall(r'^    case (.*):$',text,re.MULTILINE)
    cubes=[]
    for case in cases:
        bits=case.split();assert len(bits)==32
        fixed={i:(v=='True{}') for i,v in enumerate(bits) if v!='_'}
        assert all(v in {'False{}','True{}','_'} for v in bits)
        cubes.append(fixed)
    for i,a in enumerate(cubes):
        for b in cubes[i+1:]: assert any(a[k]!=b[k] for k in a.keys() & b.keys()),'overlapping dispatch clauses'
    count=sum(1 << (32-len(c)) for c in cubes);assert count==1<<32
    return {'disjoint_boolean_clauses':len(cubes),'raw_u32_values_covered':count,'not_expected_attack_masks':True}


def invoke(bun: str,compiler: Path,entry: Path) -> dict:
    start=time.monotonic()
    p=subprocess.run([bun,str(compiler/'bend2/main.ts'),str(entry)],capture_output=True,timeout=180,check=False,
                     env={**os.environ,'BEND_NO_TELEMETRY':'1','TERM':'dumb'})
    raw=p.stdout+p.stderr
    return {'exit_code':p.returncode,'raw_text':raw.decode(),'sha256':digest(raw),'seconds':time.monotonic()-start}


def safe(r: dict) -> None:
    if r['exit_code']!=0 or r['raw_text'].strip()!='All terms check.': raise AssertionError(str(r)[-5000:])


def replace(path: Path,old: str,new: str) -> None:
    s=path.read_text()
    if s.count(old)!=1: raise ValueError(f'nonunique mutation: {old!r}')
    path.write_text(s.replace(old,new))


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('compiler',type=Path);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();compiler=args.compiler.resolve();bun=os.environ.get('BUN','bun')
    check=[bun,str(ENGINE/'standalone/verify_compiler.js'),str(compiler)]
    initial=subprocess.run(check,capture_output=True,text=True,check=True)
    if initial.stderr: raise AssertionError(initial.stderr)
    manifest(SUITE);graph=closure(SUITE/'consumer.bend',ENGINE)
    paths=[ENGINE/p for p in graph]+[p for p in SUITE.iterdir() if p.is_file()]+[
        ENGINE/'standalone/toolchain.json',ENGINE/'standalone/verify_compiler.js']
    before={str(p.relative_to(ENGINE)):digest(p.read_bytes()) for p in paths}
    consumer=invoke(bun,compiler,SUITE/'consumer.bend');safe(consumer)
    partition=dispatch_audit();controls=[]
    scribble='Array.set(U64,table,131071,U64.from_parts(305419896,2596069104))'
    edits=[
      ('raw-default-drops-table','Queries.bend','attack_bits',
       'Array.get(U64, table, U32.add(320, sq))','Array.get(U64, Array.new(U64,0n,U64.zero()), U32.add(320, sq))'),
      ('queen-corrupts-table','Queries.bend','or_result','(table, U64.or(a, b))',f'({scribble}, U64.or(a, b))'),
      ('attack-reducer-corrupts-table','Checks.bend','bishop',
       '(table, Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))',f'({scribble}, Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))'),
      ('scan-corrupts-table','Targets.bend','after',
       '(table, destinations(64n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc))',f'({scribble}, destinations(64n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc))'),
      ('castling-corrupts-table','Castles.bend','finish',
       '(table, retain_move(Bool.or(start_check, transit_check), m, acc))',f'({scribble}, retain_move(Bool.or(start_check, transit_check), m, acc))'),
      ('fast-shortcut-corrupts-table','Filters.bend','fast_step','(table, Con{m, acc})',f'({scribble}, Con{{m, acc}})'),
      ('blocker-stage-corrupts-table','Filters.bend','blockers',
       'filter_fast(moves, b, sensitive, (table, Nil{}))',f'filter_fast(moves, b, sensitive, ({scribble}, Nil{{}}))'),
      ('generator-resets-table','Generation.bend','legal',
       'scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (table, Nil{}))',
       'scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (Array.new(U64,0n,U64.zero()), Nil{}))')]
    for name,entry,location,old,new in edits:
        with tempfile.TemporaryDirectory(prefix='table-state-source-') as temp:
            copied=Path(temp)/'engine';shutil.copytree(ENGINE,copied,symlinks=True)
            replace(copied/'legal_probe/Chess.bend',old,new)
            r=invoke(bun,compiler,copied/'standalone/proofs/table_preservation'/entry)
            text=r['raw_text']
            forbidden=['no such file','RangeError','Maximum call stack','more than once','a decreasing self-call','a defined name','a parameter or field scrutinee']
            if not (r['exit_code']==1 and 'expected' in text and 'observed' in text and
                    re.search(r'Location:\s*(?:\w+\.)*'+location+r'\b',text) and not any(x in text for x in forbidden)):
                raise AssertionError(f'Not intended source rejection {name}: {r}')
            controls.append({'name':name,'kind':'source semantic/refinement','rejected':True,'entry':entry,
                             'diagnostic_sha256':r['sha256'],'excerpt':text[-1800:]})
    policy=[
      ('missing-law',lambda s:replace(s/'LAWS.bend','law legal_moves_preserves_table:','def absent:')),
      ('missing-proof',lambda s:replace(s/'PROOF.bend','def Laws.legal_moves_preserves_table(','def absent(')),
      ('missing-laws-import',lambda s:replace(s/'PROOF.bend','import ./LAWS.bend as Laws\n','')),
      ('missing-proof-import',lambda s:replace(s/'consumer.bend','import ./PROOF.bend as Proof\n','')),
      ('proof-hole',lambda s:(s/'Generation.bend').write_text((s/'Generation.bend').read_text()+'\n?missing\n')),
      ('unsafe',lambda s:(s/'Generation.bend').write_text('@unsafe\n'+(s/'Generation.bend').read_text())),
      ('foreign',lambda s:(s/'Generation.bend').write_text((s/'Generation.bend').read_text()+'\nimport "oracle.c"\n')),
      ('symlink',lambda s:((s/'Core.bend').rename(s/'Core.original'),(s/'Core.bend').symlink_to('Core.original')))]
    for name,edit in policy:
        with tempfile.TemporaryDirectory(prefix='table-state-policy-') as temp:
            copied=Path(temp)/'engine';shutil.copytree(ENGINE,copied,symlinks=True);suite=copied/'standalone/proofs/table_preservation'
            edit(suite)
            try: manifest(suite);closure(suite/'consumer.bend',copied)
            except (ValueError,OSError): controls.append({'name':name,'kind':'manifest/import policy','rejected':True})
            else: raise AssertionError(f'policy accepted {name}')
    try: safe({'exit_code':0,'raw_text':'All terms check.\nWARNING: unsafe dependency'})
    except AssertionError: controls.append({'name':'warning-on-zero-exit','kind':'synthetic output-wrapper unit','rejected':True})
    else: raise AssertionError('warning accepted')
    assert before=={str(p.relative_to(ENGINE)):digest(p.read_bytes()) for p in paths}
    final=subprocess.run(check,capture_output=True,text=True,check=True);assert final.stdout==initial.stdout and not final.stderr
    report={'focused_gate':'PASS','consumer':consumer,'new_law_count':len(LAWS),'new_laws':LAWS,'new_control_count':len(controls),
            'negative_controls':controls,'control_categories':dict(Counter(c['kind'] for c in controls)),
            'raw_dispatch_partition':partition,'source_sha256s':before,'compiler_identity':initial.stdout,
            'scope':'Whole source-array identity under actual query/scan/castling/prepared-filter/full-generator operations; arbitrary array shape/contents, Boards and raw scalar metadata. Not move or attack correctness.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'source_sha256s','negative_controls','consumer'}},indent=2))


if __name__=='__main__': main()
