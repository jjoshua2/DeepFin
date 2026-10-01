"""Check sequential generation/query transport and classify rejection controls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
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
LAWS=['generation_then_check_matches_direct_check','initialized_generation_then_check_matches_geometry']
spec=importlib.util.spec_from_file_location('storage_gate',SUITE.parent/'table_preservation/focused.py')
assert spec is not None and spec.loader is not None
storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)

def invoke(bun: str,compiler: Path,entry: Path,timeout: int) -> dict:
    start=time.monotonic()
    p=subprocess.run([bun,'--smol',str(compiler/'bend2/main.ts'),str(entry)],capture_output=True,timeout=timeout,check=False,
                     env={**os.environ,'BEND_NO_TELEMETRY':'1','TERM':'dumb'})
    raw=p.stdout+p.stderr
    return {'exit_code':p.returncode,'raw_text':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest(),'seconds':time.monotonic()-start}

def manifest(suite: Path) -> None:
    if re.findall(r'^law (\w+):',(suite/'LAWS.bend').read_text(),re.MULTILINE)!=LAWS:raise ValueError('law inventory')
    text=(suite/'PROOF.bend').read_text()
    if re.findall(r'^def Laws\.(\w+)\(',text,re.MULTILINE)!=LAWS:raise ValueError('proof inventory')
    if 'import ./LAWS.bend as Laws' not in text:raise ValueError('law import')
    if 'import ./PROOF.bend as Proof' not in (suite/'consumer.bend').read_text():raise ValueError('consumer import')

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('compiler',type=Path);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--controls-only',action='store_true',help='No consumer pass is claimed in this mode.')
    args=ap.parse_args();compiler=args.compiler.resolve();bun=os.environ.get('BUN','bun')
    cmd=[bun,str(ENGINE/'standalone/verify_compiler.js'),str(compiler)]
    identity=subprocess.run(cmd,capture_output=True,text=True,check=True);assert not identity.stderr
    manifest(SUITE);graph=storage.closure(SUITE/'consumer.bend',ENGINE)
    paths={ENGINE/p for p in graph}|{p for p in SUITE.iterdir() if p.is_file()}|{SUITE.parent/'table_preservation/focused.py',SUITE.parent/'table_preservation/_validation.py',ENGINE/'standalone/verify_compiler.js',ENGINE/'standalone/toolchain.json'}
    before={str(p.relative_to(ENGINE)):storage.digest(p.read_bytes()) for p in sorted(paths)}
    consumer={'status':'NOT_RUN'}
    if not args.controls_only:
        consumer=invoke(bun,compiler,SUITE/'consumer.bend',1500);storage.safe(consumer)
    old='attach(ms,Chess.in_check(a,q,side))'
    edits=[
      ('discard-returned-table',old,'attach(ms,Chess.in_check(Array.new(U64,0n,U64.zero()),q,side))','pair'),
      ('check-opposite-side',old,'attach(ms,Chess.in_check(a,q,U32.xor(side,1)))','pair'),
      ('skip-followup-check',old,'attach(ms,(a,False{}))','pair'),
      ('discard-generated-moves',old,'attach(Nil{},Chess.in_check(a,q,side))','pair'),
      ('query-generation-board','follow(q,side,Chess.legal_moves(a,b))','follow(b,side,Chess.legal_moves(a,b))','known'),
      ('generate-altered-board','follow(q,side,Chess.legal_moves(a,b))','follow(q,side,Chess.legal_moves(a,Chess.make_move(b,Chess.Ply{4,5,0,0})))','known')]
    controls=[]
    for name,old,new,location in edits:
        with tempfile.TemporaryDirectory(prefix='generator-followup-source-') as tmp:
            root=Path(tmp)/'engine';shutil.copytree(ENGINE,root,symlinks=True)
            suite=root/'standalone/proofs/generator_followup'
            storage.replace(suite/'Runtime.bend',old,new)
            record=invoke(bun,compiler,suite/'Transport.bend',180);out=record['raw_text']
            forbidden=['no such file','RangeError','Maximum call stack','more than once','a decreasing self-call','a defined name']
            if not(record['exit_code']==1 and 'expected' in out and 'observed' in out and re.search(r'Location:\s*(?:\w+\.)*'+location+r'\b',out) and not any(x in out for x in forbidden)):
                raise AssertionError((name,record))
            controls.append({'name':name,'kind':'source semantic/refinement','rejected':True,'entry':'Transport.bend',**record})
    policy=[
      ('missing-law',lambda p:storage.replace(p/'LAWS.bend','law generation_then_check_matches_direct_check:','def absent:')),
      ('missing-proof',lambda p:storage.replace(p/'PROOF.bend','def Laws.generation_then_check_matches_direct_check(','def absent(')),
      ('missing-law-import',lambda p:storage.replace(p/'PROOF.bend','import ./LAWS.bend as Laws\n','')),
      ('missing-proof-import',lambda p:storage.replace(p/'consumer.bend','import ./PROOF.bend as Proof\n','')),
      ('proof-hole',lambda p:(p/'Transport.bend').write_text((p/'Transport.bend').read_text()+'\n?missing\n')),
      ('unsafe',lambda p:(p/'Transport.bend').write_text('@unsafe\n'+(p/'Transport.bend').read_text())),
      ('foreign',lambda p:(p/'Transport.bend').write_text((p/'Transport.bend').read_text()+'\nimport "oracle.c"\n')),
      ('symlink',lambda p:((p/'Transport.bend').rename(p/'Transport.original'),(p/'Transport.bend').symlink_to('Transport.original')))]
    for name,edit in policy:
        with tempfile.TemporaryDirectory(prefix='generator-followup-policy-') as tmp:
            root=Path(tmp)/'engine';shutil.copytree(ENGINE,root,symlinks=True);suite=root/'standalone/proofs/generator_followup';edit(suite)
            try:manifest(suite);storage.closure(suite/'consumer.bend',root)
            except (ValueError,OSError):controls.append({'name':name,'kind':'manifest/import policy','rejected':True})
            else:raise AssertionError('policy accepted '+name)
    try:storage.safe({'exit_code':0,'raw_text':'All terms check.\nWARNING: unsafe dependency'})
    except AssertionError:controls.append({'name':'warning-on-zero-exit','kind':'synthetic output-wrapper unit','rejected':True})
    else:raise AssertionError('warning accepted')
    assert before=={str(p.relative_to(ENGINE)):storage.digest(p.read_bytes()) for p in sorted(paths)}
    final=subprocess.run(cmd,capture_output=True,text=True,check=True);assert final.stdout==identity.stdout and not final.stderr
    result={'focused_gate':'NOT_RUN' if args.controls_only else 'PASS','controls_gate':'PASS','consumer':consumer,'new_law_count':2,'new_control_count':len(controls),'negative_controls':controls,'control_categories':dict(Counter(c['kind'] for c in controls)),'source_sha256s':before,'compiler_identity':identity.stdout,
            'scope':'Subsequent actual check after complete generation; exact generated list retained. Query-Board singleton/init premises do not certify generated move legality.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'consumer','source_sha256s','negative_controls'}},indent=2))

if __name__=='__main__':main()
