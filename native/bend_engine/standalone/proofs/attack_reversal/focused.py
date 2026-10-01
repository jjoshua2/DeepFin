"""Check both complete public reversal laws and classified negative controls."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile
from _common import InvalidEvidence, atomic_report, closure, command, require, safe, sha, success

NAMES = ["leaper_coordinate_membership_reverses", "computed_leaper_mask_reverses"]
SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]

def manifest(s: Path) -> None:
    laws = re.findall(r"^law (\w+):", (s/"LAWS.bend").read_text(), re.MULTILINE)
    proofs = re.findall(r"^def Laws\.(\w+)\(", (s/"PROOF.bend").read_text(), re.MULTILINE)
    require(laws == NAMES and proofs == NAMES, "Missing, extra or duplicate law/proof")
    require("import ./LAWS.bend as Laws" in (s/"PROOF.bend").read_text(), "Missing law import")
    require("import ./PROOF.bend as Proof" in (s/"consumer.bend").read_text(), "Missing proof import")

def snapshot(root: Path) -> dict[str,str]:
    here=root/"standalone/proofs/attack_reversal"
    seen=closure(here/"consumer.bend",root)
    closure(root/"standalone/proofs/attack_geometry/probe.bend",root,seen)
    for p in [*here.glob("*.py"), here/"README.md", root/"standalone/toolchain.json", root/"standalone/verify_compiler.js"]:
        seen[p.relative_to(root).as_posix()]=sha(p.read_bytes())
    return dict(sorted(seen.items()))

def replace(p: Path, old: str, new: str, count: int = 1) -> None:
    text=p.read_text()
    require(text.count(old)==count, "Nonunique mutation site: "+str(p))
    p.write_text(text.replace(old,new,1))

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("compiler",type=Path);ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args();atomic_report(args.report,{"focused_gate":"NOT_COMPLETED"})
    compiler=args.compiler.resolve();bun=os.environ.get("BUN","bun")
    identity=success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]))
    manifest(SUITE);before=snapshot(ENGINE)
    def invoke(p:Path)->dict:return command([bun,"--smol",str(compiler/"bend2/main.ts"),str(p)],180)
    consumer=invoke(SUITE/"consumer.bend");safe(consumer)
    print("PASS complete two-law consumer",flush=True)
    controls=[]; prefix="standalone/proofs/"
    mutations=[
      ("pawn-reversal-must-change-color",prefix+"attack_geometry/Geometry.bend", "case WhitePawn{}: BlackPawn{}", "case WhitePawn{}: WhitePawn{}",prefix+"attack_reversal/WhitePawn.bend",1),
      ("king-edge-correspondence",prefix+"attack_geometry/Geometry.bend", "Grid.next(sq,0n) <>","Grid.next(sq,1n) <>",prefix+"attack_reversal/King.bend",1),
      ("actual-knight-offset","standalone/Tables.bend","case 0: (1, 2)","case 0: (1, 1)",prefix+"attack_geometry/Computed.bend",1),
      ("source-bound-required",prefix+"attack_reversal/LAWS.bend","for source_bound: {Nat.is_lt(src,64n) == True{} : Bool}","for source_bound: {True{} == True{} : Bool}",prefix+"attack_reversal/PROOF.bend",2),
      ("target-bound-required",prefix+"attack_reversal/LAWS.bend","for target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}","for target_bound: {True{} == True{} : Bool}",prefix+"attack_reversal/PROOF.bend",2),
      ("public-mask-must-use-reversed-kind",prefix+"attack_reversal/LAWS.bend","U64.test_bit(Slots.value(G.reverse(kind),dst),src)","U64.test_bit(Slots.value(kind,dst),src)",prefix+"attack_reversal/PROOF.bend",1),
      ("reflection-high-limb-not-low",prefix+"attack_reversal/Mask.bend","U32{word(32n,32n,Many{xs})}","U32{word(32n,0n,Many{xs})}",prefix+"attack_reversal/Mask.bend",1),
      ("computed-zero-is-not-geometry",prefix+"attack_geometry/Slots.bend","case G.Knight{}: Tables.leaps(8n,U32.from_nat(sq),0,False{},U64.zero())","case G.Knight{}: U64.zero()",prefix+"attack_geometry/Computed.bend",1),
    ]
    forbidden=r"no such file|a defined name|consumed more than once|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault"
    for name,filename,old,new,entry,count in mutations:
        with tempfile.TemporaryDirectory(prefix="leaper-control-") as td:
            root=Path(td)/"engine";shutil.copytree(ENGINE,root,symlinks=True)
            replace(root/filename,old,new,count);r=invoke(root/entry);out=r["stdout"]+r["stderr"]
            require(r["exit_code"]==1 and "expected" in out and "observed" in out and "Location:" in out and re.search(forbidden,out,re.IGNORECASE) is None,"Not a semantic rejection: "+name+" "+str(r)[-5000:])
            controls.append({"name":name,"kind":"source semantic/refinement","rejected":True,"entry":entry,"result":r})
            print("PASS control "+name,flush=True)
    policies=[
      ("missing-law","LAWS.bend","law leaper_coordinate_membership_reverses:","def missing:"),
      ("missing-proof","PROOF.bend","def Laws.leaper_coordinate_membership_reverses(","def missing("),
      ("missing-law-import","PROOF.bend","import ./LAWS.bend as Laws","# removed law import"),
      ("missing-consumer-import","consumer.bend","import ./PROOF.bend as Proof","# removed proof import"),
      ("hole","Spec.bend",None,"\n?hole\n"),
      ("foreign","Spec.bend",None,'\nimport "oracle.c"\n'),
      ("unsafe","Spec.bend",None,"\n@unsafe\n"),
      ("symlink","Spec.bend",None,None)]
    for name,filename,old,new in policies:
        with tempfile.TemporaryDirectory(prefix="leaper-policy-") as td:
            root=Path(td)/"engine";shutil.copytree(ENGINE,root,symlinks=True);s=root/prefix/"attack_reversal";p=s/filename
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
    atomic_report(args.report,{"focused_gate":"PASS","new_laws":2,"new_controls":17,"consumer":consumer,"negative_controls":controls,"source_sha256s":before,"compiler_identity":identity,"full_aggregate_executed":False})

if __name__=="__main__":main()
