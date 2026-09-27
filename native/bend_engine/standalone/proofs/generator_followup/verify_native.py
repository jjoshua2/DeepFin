"""Observe generated lists and subsequent checks on the actually returned table.

Check answers use the retained independent forward-coordinate oracle. Move-list
parity is differential against a preceding direct call, not an independent proof
of all generated moves. Each request executes two generators (baseline + adapter).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SUITE=Path(__file__).resolve().parent
ENGINE=SUITE.parents[2]
MODES={'generic':[],'portable':['-DBEND_U64_PORTABLE'],'native':['-march=native'],
       'ubsan':['-fsanitize=undefined','-fno-sanitize-recover=all']}

def load(name: str,path: Path):
    spec=importlib.util.spec_from_file_location(name,path);assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

ref=load('attack_reference',SUITE.parent/'attack_witness/verify_native.py')
state=load('state_reference',SUITE.parent/'table_preservation/verify_native.py')

def digest(raw: bytes)->str:return hashlib.sha256(raw).hexdigest()

def run(cmd: list[str],timeout: int=180):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False,
                     env={**os.environ,'BEND_NO_TELEMETRY':'1','TERM':'dumb'})
    if p.returncode or p.stderr:raise AssertionError((cmd[:3],p.returncode,p.stderr[-2000:],p.stdout[-2000:]))
    return p

def fixtures()->list[dict]:
    gens=[x['input'][2:] for x in state.fixtures() if x['category']=='initialized-full-generator'][:4]
    queries=[]
    for sq in (9,27,54):
        for side in (0,1):
            opposite=63 if sq<32 else 0
            for kind in range(-1,6):
                board=ref.empty();ref.put(board,sq,5,side);ref.put(board,opposite,5,1-side)
                if kind>=0:
                    origin=next(x for x in range(64) if x not in {sq,opposite} and sq in ref.geometry(board,x,kind,1-side))
                    ref.put(board,origin,kind,1-side)
                expected=int(ref.attacked(board,sq,1-side))
                queries.append((ref.encode(board,(1-side,0,64)),side,expected,f'king-{sq}-side-{side}-attacker-{kind}'))
    for sq in (0,63):
        for side in (0,1):
            board=ref.empty();ref.put(board,sq,5,side);ref.put(board,63-sq,5,1-side)
            queries.append((ref.encode(board,(1-side,0,64)),side,0,f'edge-{sq}-side-{side}'))
    cases=[{'input':[ *g,*q,side],'expected_check':expected,'label':f'generator-{i}-{name}'}
           for i,g in enumerate(gens) for q,side,expected,name in queries]
    assert len({tuple(c['input']) for c in cases})==len(cases)
    assert {c['expected_check'] for c in cases}=={0,1}
    assert all(c['input'][:19]!=c['input'][19:38] for c in cases)
    return cases

class WrongObservation(AssertionError):pass

def observe(binary: Path,cases: list[dict],context: int)->dict:
    arguments=[str(context),*[str(x) for c in cases for x in c['input']]]
    lines=run([str(binary),*arguments]).stdout.splitlines()
    if len(lines)!=2*len(cases)+1:raise WrongObservation(('line-count',len(lines)))
    counts=0
    for i,c in enumerate(cases):
        original=lines[2*i].split();following=lines[2*i+1].split()
        if not original or original[0]!='baseline' or len(following)<2 or following[0]!='following':
            raise WrongObservation(('record-shape',i))
        if original[1:]!=following[2:]:raise WrongObservation(('ordered-moves',i,original[1:4],following[2:5]))
        if following[1]!=str(c['expected_check']):raise WrongObservation(('check',i,c['expected_check'],following[1]))
        counts+=len(original)-1
    marker=[0,0] if context==0 else [305419896,2596069104]
    if lines[-1]!='marker '+' '.join(map(str,marker)):raise WrongObservation(('marker',lines[-1],marker))
    return {'requests':len(cases),'moves_compared':counts,'checks_compared':len(cases),'marker_reads':1,
            'output_sha256':digest(('\n'.join(lines)+'\n').encode())}

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('compiler',type=Path);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();compiler=args.compiler.resolve();bun=os.environ.get('BUN','bun');cc=os.environ.get('CC','clang')
    identity=run([bun,str(ENGINE/'standalone/verify_compiler.js'),str(compiler)]).stdout
    cases=fixtures();modes=[];mutations=[]
    files=[SUITE/'Runtime.bend',SUITE/'probe.bend',Path(__file__),SUITE.parent/'attack_witness/verify_native.py',SUITE.parent/'table_preservation/verify_native.py',ENGINE/'legal_probe/Chess.bend',ENGINE/'standalone/Tables.bend',ENGINE/'standalone/Text.bend',ENGINE/'bitboard_probe/Sliders.bend']
    before={str(p.relative_to(ENGINE)):digest(p.read_bytes()) for p in files}
    with tempfile.TemporaryDirectory(prefix='generator-followup-native-') as td:
        tmp=Path(td)
        def compile(root: Path,name: str,flags: list[str])->Path:
            source=tmp/(name+'.c');binary=tmp/(name+'.bin')
            run([bun,str(compiler/'bend2/main.ts'),str(root/'standalone/proofs/generator_followup/probe.bend'),'-o',str(source)])
            run([cc,'-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',*flags,str(source),'-pthread','-lm','-o',str(binary)])
            return binary
        for mode,flags in MODES.items():
            binary=compile(ENGINE,mode,flags);observations=[]
            for context in (0,1):
                for start in range(0,len(cases),64):observations.append(observe(binary,cases[start:start+64],context))
            valid=[str(x) for x in cases[0]['input']]
            bad_side=valid.copy();bad_side[-1]='2'
            missing_king=valid.copy();missing_king[29:31]=['0','0']
            invalids=[[],['bad'],['2'],['0','1'],['0',*valid[:-1]],['0',*bad_side],['0',*missing_king],['0',*valid[:-1],'4294967296']]
            for argv in invalids:
                p=subprocess.run([str(binary),*argv],capture_output=True,text=True,timeout=120,check=False)
                assert p.returncode==2 and 'invalid followup' in p.stderr,(argv[:4],p.returncode,p.stderr)
            modes.append({'mode':mode,'requests':sum(x['requests'] for x in observations),'checks_compared':sum(x['checks_compared'] for x in observations),
                          'moves_compared':sum(x['moves_compared'] for x in observations),'marker_reads':sum(x['marker_reads'] for x in observations),
                          'invalid_rejections':len(invalids),'batches':observations})
            print('PASS '+mode,flush=True)
        mutations_to_test=[
          ('discard-returned-table','attach(ms,Chess.in_check(a,q,side))','attach(ms,Chess.in_check(Array.new(U64,17n,U64.zero()),q,side))'),
          ('check-opposite-side','attach(ms,Chess.in_check(a,q,side))','attach(ms,Chess.in_check(a,q,U32.xor(side,1)))'),
          ('discard-generated-moves','attach(ms,Chess.in_check(a,q,side))','attach(Nil{},Chess.in_check(a,q,side))'),
          ('query-generation-board','follow(q,side,Chess.legal_moves(a,b))','follow(b,side,Chess.legal_moves(a,b))')]
        for name,old,new in mutations_to_test:
            root=tmp/name;shutil.copytree(ENGINE,root,symlinks=True);p=root/'standalone/proofs/generator_followup/Runtime.bend'
            text=p.read_text();assert text.count(old)==1;p.write_text(text.replace(old,new))
            binary=compile(root,name,[])
            try:observe(binary,cases[:32],1)
            except WrongObservation as err:mutations.append({'name':name,'compiled_and_executed':True,'rejected':True,'build_mode':'generic','diagnostic':str(err)})
            else:raise AssertionError('mutation accepted: '+name)
            print('PASS mutation '+name,flush=True)
    assert before=={str(p.relative_to(ENGINE)):digest(p.read_bytes()) for p in files}
    assert identity==run([bun,str(ENGINE/'standalone/verify_compiler.js'),str(compiler)]).stdout
    report={'native_gate':'PASS','base_generation_boards':4,'base_requests':len(cases),'initialization_contexts':2,
            'requests_per_mode':len(cases)*2,'generator_calls_per_request':2,'modes':modes,'mutations':mutations,
            'fixture_sha256':digest(json.dumps(cases,sort_keys=True).encode()),'source_sha256s':before,
            'cc':run([cc,'--version']).stdout.splitlines()[0],
            'scope':'Independent subsequent check oracle, differential full ordered move-list parity, sampled retained marker. Not independent move-set correctness or full-buffer/lifetime correctness.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
