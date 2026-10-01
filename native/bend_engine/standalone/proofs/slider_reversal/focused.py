"""Check both complete slider-reversal laws and explicitly classified controls."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile
from _common import InvalidEvidence, atomic_report, closure, command, require, safe, sha, success

NAMES = ["slider_geometric_membership_reverses", "computed_slider_mask_reverses"]
SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]

def manifest(s: Path) -> None:
    laws = re.findall(r"^law (\w+):", (s/"LAWS.bend").read_text(), re.MULTILINE)
    proofs = re.findall(r"^def Laws\.(\w+)\(", (s/"PROOF.bend").read_text(), re.MULTILINE)
    require(laws == NAMES and proofs == NAMES, "Missing, extra or duplicate public law/proof")
    require("import ./LAWS.bend as Laws" in (s/"PROOF.bend").read_text(), "Missing law import")
    require("import ./PROOF.bend as Proof" in (s/"consumer.bend").read_text(), "Missing proof import")

def snapshot(root: Path) -> dict[str,str]:
    here = root/"standalone/proofs/slider_reversal"
    seen = closure(here/"consumer.bend",root)
    closure(root/"standalone/proofs/ray/probe.bend",root,seen)
    for p in [*here.glob("*.py"), here/"README.md", root/"standalone/toolchain.json", root/"standalone/verify_compiler.js"]:
        seen[p.relative_to(root).as_posix()] = sha(p.read_bytes())
    return dict(sorted(seen.items()))

def replace(p: Path, old: str, new: str, count: int = 1) -> None:
    text=p.read_text();require(text.count(old)==count,"Nonunique mutation site: "+str(p))
    p.write_text(text.replace(old,new,1))

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("compiler",type=Path);ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--controls-only",action="store_true",help="Development diagnostic, not a complete qualification")
    args=ap.parse_args();atomic_report(args.report,{"focused_gate":"NOT_COMPLETED"})
    compiler=args.compiler.resolve();bun=os.environ.get("BUN","bun")
    identity=success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]))
    manifest(SUITE);before=snapshot(ENGINE)
    def invoke(p:Path,seconds:int=240)->dict:return command([bun,"--smol",str(compiler/"bend2/main.ts"),str(p)],seconds)
    consumer=None
    if not args.controls_only:
        consumer=invoke(SUITE/"consumer.bend",900);safe(consumer)
        print("PASS complete two-law consumer",flush=True)
    prefix="standalone/proofs/";here=prefix+"slider_reversal/"
    mutations=[
      ("opposite-direction-required",here+"Spec.bend","case 0n: 1n","case 0n: 0n",here+"Rank0.bend",1,r"square[0-7]"),
      ("exclude-hit-from-strict-interior",here+"Spec.bend","Between{Nil{}},prepend(h,cut(dst,tail))","Between{h <> Nil{}},prepend(h,cut(dst,tail))",here+"Bits.bend",1,r"path"),
      ("prefix-must-test-blocker",here+"Spec.bend","Bool.and(Bool.not(U64.test_bit(occ,h)),clear(t,occ))","Bool.and(True{},clear(t,occ))",here+"Bits.bend",1,r"Booleans.append_clear"),
      ("prefix-order-must-reverse",here+"Spec.bend","case Between{xs}: Between{reverse(xs)}","case Between{xs}: Between{xs}",here+"Rank0.bend",1,r"square[0-7]"),
      ("source-bound-required",here+"LAWS.bend","for source_bound: {Nat.is_lt(src,64n) == True{} : Bool}","for source_bound: {True{} == True{} : Bool}",here+"PROOF.bend",2,r"slider_geometric_membership_reverses"),
      ("target-bound-required",here+"LAWS.bend","for target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}","for target_bound: {True{} == True{} : Bool}",here+"PROOF.bend",2,r"slider_geometric_membership_reverses"),
      ("actual-ray-must-stop-at-blocker","standalone/Tables.bend","stop = Bool.or(U32.is_eq(next, 64), U64.test_bit(occ, U32.to_nat(next)))","stop = U32.is_eq(next, 64)",prefix+"ray/Traversal.bend",1,r"actual"),
      ("shared-occupancy-required",here+"LAWS.bend","U64.test_bit(S.geometry(kind,dst,occ),src)","U64.test_bit(S.geometry(kind,dst,U64.zero()),src)",here+"PROOF.bend",1,r"slider_geometric_membership_reverses"),
    ]
    controls=[]
    forbidden=r"no such file|a defined name|consumed more than once|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault"
    for name,filename,old,new,entry,count,location in mutations:
        with tempfile.TemporaryDirectory(prefix="slider-control-") as td:
            root=Path(td)/"engine";shutil.copytree(ENGINE,root,symlinks=True)
            replace(root/filename,old,new,count);r=invoke(root/entry,600);out=r["stdout"]+r["stderr"]
            require(r["exit_code"]==1 and "expected" in out and "observed" in out and re.search(r"Location:.*(?:"+location+r")",out) and re.search(forbidden,out,re.IGNORECASE) is None,"Not intended semantic rejection: "+name+" "+str(r)[-5000:])
            controls.append({"name":name,"kind":"source semantic/refinement","rejected":True,"entry":entry,"result":r})
            print("PASS control "+name,flush=True)
    policies=[
      ("missing-law","LAWS.bend","law slider_geometric_membership_reverses:","def missing:"),
      ("missing-proof","PROOF.bend","def Laws.slider_geometric_membership_reverses(","def missing("),
      ("missing-law-import","PROOF.bend","import ./LAWS.bend as Laws","# removed law import"),
      ("missing-consumer-import","consumer.bend","import ./PROOF.bend as Proof","# removed proof import"),
      ("hole","Spec.bend",None,"\n?hole\n"),("foreign","Spec.bend",None,'\nimport "oracle.c"\n'),
      ("unsafe","Spec.bend",None,"\n@unsafe\n"),("symlink","Spec.bend",None,None)]
    for name,filename,old,new in policies:
        with tempfile.TemporaryDirectory(prefix="slider-policy-") as td:
            root=Path(td)/"engine";shutil.copytree(ENGINE,root,symlinks=True);s=root/here;p=s/filename
            if name=="symlink":p.rename(p.with_suffix(".original"));p.symlink_to(p.with_suffix(".original").name)
            elif old is None:p.write_text(p.read_text()+new)
            else:replace(p,old,new)
            try:manifest(s);closure(s/"consumer.bend",root)
            except (InvalidEvidence,FileNotFoundError):pass
            else:raise InvalidEvidence("Accepted policy violation: "+name)
            controls.append({"name":name,"kind":"manifest/import policy","rejected":True})
    try:safe({"exit_code":0,"stdout":"All terms check.\nWARNING: unsafe dependency","stderr":""})
    except InvalidEvidence:pass
    else:raise InvalidEvidence("Accepted warning on zero exit")
    controls.append({"name":"warning-on-zero-exit","kind":"synthetic output-wrapper unit","rejected":True})
    require(len(controls)==17,"Wrong control accounting")
    require(snapshot(ENGINE)==before,"Source drift")
    require(success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]))==identity,"Compiler drift")
    atomic_report(args.report,{"focused_gate":"CONTROLS_ONLY" if args.controls_only else "PASS","new_laws":2,"new_controls":17,"consumer":consumer,"negative_controls":controls,"source_sha256s":before,"compiler_identity":identity,"full_aggregate_executed":False})

if __name__=="__main__":main()
