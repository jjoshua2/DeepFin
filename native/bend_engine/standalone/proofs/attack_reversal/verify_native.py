"""Actual initialized masks: independent geometry AND exhaustive endpoint reversal."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from _common import atomic_report, command, require, sha, success
from focused import ENGINE, snapshot

MODES={"generic":[],"portable":["-DBEND_U64_PORTABLE"],"native":["-march=native"],"ubsan":["-fsanitize=undefined","-fno-sanitize-recover=all"]}
REVERSE=(0,1,3,2)

def geometry(kind:int,square:int)->int:
    x,y=square%8,square//8
    if kind==0: offsets=[(dx,dy) for dx in range(-2,3) for dy in range(-2,3) if abs(dx*dy)==2]
    elif kind==1: offsets=[(dx,dy) for dx in range(-1,2) for dy in range(-1,2) if dx or dy]
    else:offsets=[(dx,1 if kind==2 else -1) for dx in (-1,1)]
    return sum(1<<((y+dy)*8+x+dx) for dx,dy in offsets if 0<=x+dx<8 and 0<=y+dy<8)

def cases()->list[list[int]]:
    return [[init,kind,sq,occ>>32,occ&0xffffffff] for init in range(4) for kind in range(4) for sq in range(64) for occ in (0,2**64-1,0x9249249249249249^(1<<sq))]

def parse(text:str,count:int)->list[int]:
    lines=text.splitlines();require(len(lines)==count,"Incomplete/extra mask output")
    values=[]
    for line in lines:
        v=line.split();require(len(v)==3 and v[0]=="mask","Malformed mask row")
        require(all(re.fullmatch(r"[0-9]+",x) is not None for x in v[1:]),"Invalid numeric mask field")
        hi,lo=map(int,v[1:]);require(0<=hi<2**32 and 0<=lo<2**32,"Mask field exceeds U32")
        values.append((hi<<32)|lo)
    return values

def reciprocity(values:list[int],rows:list[list[int]])->dict:
    require(len(values)==len(rows),"Mismatched mask count")
    table={(r[0],r[1],r[2],i%3):v for i,(r,v) in enumerate(zip(rows,values))}
    groups=sorted({r[0] for r in rows});first=None;comparisons=positives=0
    for init in groups:
        for occ in range(3):
            for kind in range(4):
                for src in range(64):
                    for dst in range(64):
                        a=(table[(init,kind,src,occ)]>>dst)&1
                        b=(table[(init,REVERSE[kind],dst,occ)]>>src)&1
                        comparisons+=1;positives+=a
                        if a!=b and first is None:first={"initialization":init,"occupancy_index":occ,"kind":kind,"source":src,"target":dst,"forward":a,"reverse":b}
    return {"pair_comparisons":comparisons,"positive_edges":positives,"first_mismatch":first}

def geometry_mismatch(values:list[int],rows:list[list[int]])->dict|None:
    require(len(values)==len(rows),"Mismatched mask count")
    for i,(v,r) in enumerate(zip(values,rows)):
        expected=geometry(r[1],r[2])
        if v!=expected:return {"row":i,"input":r,"observed":v,"expected":expected}
    return None

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("compiler",type=Path);ap.add_argument("--report",type=Path,required=True);args=ap.parse_args()
    atomic_report(args.report,{"native_gate":"NOT_COMPLETED"})
    compiler=args.compiler.resolve();bun=os.environ.get("BUN","bun");cc=os.environ.get("CC","clang")
    identity=success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]));before=snapshot(ENGINE);rows=cases()
    require(len(rows)==3072 and len(set(map(tuple,rows)))==3072,"Fixture overlap")
    modes=[];mutations=[]
    with tempfile.TemporaryDirectory(prefix="leaper-native-") as td:
        tmp=Path(td)
        def generate(root:Path,name:str)->Path:
            p=tmp/(name+".c");success(command([bun,str(compiler/"bend2/main.ts"),str(root/"standalone/proofs/attack_geometry/probe.bend"),"-o",str(p)],240));require(p.is_file(),"No emitted C");return p
        def build(source:Path,name:str,flags:list[str])->Path:
            p=tmp/name;success(command([cc,"-std=c11","-O2",*flags,str(source),"-pthread","-lm","-o",str(p)],240));return p
        def observe(binary:Path,rs:list[list[int]])->list[int]:
            values=[]
            for init in sorted({r[0] for r in rs}):
                group=[r for r in rs if r[0]==init]
                raw=success(command([str(binary),str(init),*(str(v) for r in group for v in r[1:])],180))
                values.extend(parse(raw,len(group)))
            return values
        source=generate(ENGINE,"probe")
        invalid=[[],["x"],["-1"],["4294967296"],["4"],["0","0"],["0","4","0","0","0"],["0","0","64","0","0"],["0",*(str(v) for _ in range(1025) for v in (0,0,0,0))]]
        for mode,flags in MODES.items():
            binary=build(source,mode,flags);values=observe(binary,rows)
            mismatch=geometry_mismatch(values,rows);r=reciprocity(values,rows)
            require(mismatch is None,"Native coordinate mismatch: "+str(mismatch));require(r["first_mismatch"] is None,"Native reciprocity mismatch: "+str(r))
            for batch in invalid:
                o=command([str(binary),*batch],30);require(o["exit_code"]==2 and "invalid" in o["stdout"]+o["stderr"] and "runtime error" not in o["stderr"],"Bad malformed-input rejection")
            modes.append({"mode":mode,"mask_requests":len(values),"mask_field_comparisons":len(values)*2,"invalid_rejections":len(invalid),"output_sha256":sha(json.dumps(values,separators=(",",":")).encode()),**r})
            print("PASS native "+mode,flush=True)
        for name in ["asymmetric-knight","swap-both-pawn-storages"]:
            root=tmp/name;shutil.copytree(ENGINE,root,symlinks=True);p=root/"standalone/Tables.bend";text=p.read_text()
            if name=="asymmetric-knight":
                require(text.count("case 0: (1, 2)")==1,"Nonunique knight mutation");text=text.replace("case 0: (1, 2)","case 0: (1, 1)")
            else:
                a="Array.set(U64, a, U32.add(384, sq), pawns(sq, 1))";b="Array.set(U64, a, U32.add(448, sq), pawns(sq, 4294967295))"
                require(text.count(a)==1 and text.count(b)==1,"Nonunique pawn mutation")
                text=text.replace(a,a.replace("pawns(sq, 1)","pawns(sq, 4294967295)"));text=text.replace(b,b.replace("pawns(sq, 4294967295)","pawns(sq, 1)"))
            p.write_text(text);binary=build(generate(root,name),name+"-exe",[]);rs=rows[:768];values=observe(binary,rs)
            mismatch=geometry_mismatch(values,rs);r=reciprocity(values,rs);require(mismatch is not None,"Mutation evaded geometry oracle")
            require((r["first_mismatch"] is None)==(name=="swap-both-pawn-storages"),"Unexpected reciprocity diagnostic")
            mutations.append({"name":name,"compiled_and_executed":True,"rejected":True,"coordinate_mismatch":mismatch,"reciprocity":r})
            print("PASS mutation "+name,flush=True)
    require(snapshot(ENGINE)==before,"Source drift");require(success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]))==identity,"Compiler drift")
    atomic_report(args.report,{"native_gate":"PASS","compiler_identity":identity,"cc":success(command([cc,"--version"])),"fixture_sha256":sha(json.dumps(rows,separators=(",",":")).encode()),"distinct_mask_requests":len(rows),"modes":modes,"mutations":mutations,"source_sha256s":before,"scope":"Actual initialized leaper masks; independent forward coordinates plus exhaustive bounded endpoint reversal. Not slider reciprocity, a full attacked theorem, legal move completeness, or native array lifetime."})
if __name__=="__main__":main()
