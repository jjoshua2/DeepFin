"""Actual slider computations: whole-mask geometry plus blocker-aware reciprocity."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from _common import atomic_report, command, require, sha, success
from focused import ENGINE, snapshot

FULL = (1 << 64) - 1
DIRECTIONS = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
MODES = {"generic":[], "portable":["-DBEND_U64_PORTABLE"], "native":["-march=native"],
         "ubsan":["-fsanitize=undefined","-fno-sanitize-recover=all"]}


def geometry(kind: int, square: int, occupancy: int) -> int:
    require(kind in (0,1,2) and 0 <= square < 64 and 0 <= occupancy <= FULL, "Oracle domain")
    directions = DIRECTIONS[:4] if kind == 0 else DIRECTIONS[4:] if kind == 1 else DIRECTIONS
    mask = 0
    for dx, dy in directions:
        x, y = square % 8 + dx, square // 8 + dy
        while 0 <= x < 8 and 0 <= y < 8:
            bit = 1 << (8*y+x)
            mask |= bit
            if occupancy & bit:
                break
            x, y = x+dx, y+dy
    return mask


def interior(kind: int, source: int, target: int) -> tuple[int, ...] | None:
    require(kind in (0,1) and 0 <= source < 64 and 0 <= target < 64, "Interior domain")
    dx, dy = target % 8 - source % 8, target // 8 - source // 8
    if source == target or not ((dx == 0 or dy == 0) if kind == 0 else abs(dx) == abs(dy)):
        return None
    sx, sy = (dx > 0)-(dx < 0), (dy > 0)-(dy < 0)
    steps = max(abs(dx), abs(dy))
    return tuple((source//8 + j*sy)*8 + source%8 + j*sx for j in range(1,steps))


def fixtures() -> tuple[list[tuple[int,int,int,int]], list[tuple[int,int,int]], dict]:
    # A relation case is (family, source, target, one shared occupancy).
    cases: dict[tuple[int,int,int,int], None] = {}
    aligned = subsets = aligned_requests = 0
    for kind in (0,1):
        for source in range(64):
            for target in range(source+1,64):
                inside = interior(kind,source,target)
                if inside is None:
                    continue
                aligned += 1
                segment = (1<<source) | (1<<target) | sum(1<<q for q in inside)
                for subset in range(1<<len(inside)):
                    subsets += 1
                    blockers = sum(1<<q for i,q in enumerate(inside) if (subset>>i)&1)
                    for endpoints in range(4):
                        occupancy = blockers | ((endpoints&1)<<source) | (((endpoints>>1)&1)<<target)
                        for noise in (0,FULL ^ segment):
                            for family in (kind,2):
                                cases[(family,source,target,occupancy|noise)] = None
                                aligned_requests += 1
    global_masks = (0,FULL,0xA55AA55AA55AA55A,1,1<<63,(1<<27)|(1<<35))
    global_requests = 0
    for occupancy in global_masks:
        for kind in (0,1,2):
            for source in range(64):
                for target in range(64):
                    cases[(kind,source,target,occupancy)] = None
                    global_requests += 1
    pairs = list(cases)
    queries: dict[tuple[int,int,int],None] = {}
    for kind,source,target,occupancy in pairs:
        for family in ((0,1) if kind == 2 else (kind,)):
            queries[(family,source,occupancy)] = None
            queries[(family,target,occupancy)] = None
    report = {"aligned_unordered_pairs":aligned,"interior_subsets_across_pairs":subsets,
              "aligned_relation_requests_before_union":aligned_requests,
              "global_occupancy_contexts":len(global_masks),"global_relation_requests_before_union":global_requests,
              "distinct_relation_cases":len(pairs),"deduplicated_relation_cases":aligned_requests+global_requests-len(pairs),
              "distinct_native_queries":len(queries)}
    return pairs,list(queries),report


def parse(text: str, count: int) -> list[int]:
    lines = text.splitlines()
    require(len(lines) == count, "Incomplete or extra native mask output")
    values = []
    for line in lines:
        parts = line.split()
        require(len(parts) == 2 and all(re.fullmatch(r"[0-9]+",x) is not None for x in parts), "Malformed native mask row")
        hi,lo = map(int,parts)
        require(hi < 2**32 and lo < 2**32,"Native mask field exceeds U32")
        values.append((hi<<32)|lo)
    return values


def mask_for(table: dict[tuple[int,int,int],int], kind: int, square: int, occupancy: int) -> int:
    if kind == 2:
        return table[(0,square,occupancy)] | table[(1,square,occupancy)]
    return table[(kind,square,occupancy)]


def analyze(values: list[int], queries: list[tuple[int,int,int]], pairs: list[tuple[int,int,int,int]]) -> dict:
    require(len(values) == len(queries) and len(set(queries)) == len(queries), "Query/value count or uniqueness mismatch")
    table = dict(zip(queries,values))
    first_geometry = None
    for row,((kind,square,occupancy),observed) in enumerate(zip(queries,values)):
        expected = geometry(kind,square,occupancy)
        if observed != expected and first_geometry is None:
            first_geometry = {"row":row,"kind":kind,"source":square,"occupancy":occupancy,"expected":expected,"observed":observed}
    first_relation = None
    positives = 0
    for row,(kind,source,target,occupancy) in enumerate(pairs):
        forward = (mask_for(table,kind,source,occupancy)>>target)&1
        reverse = (mask_for(table,kind,target,occupancy)>>source)&1
        positives += forward
        if forward != reverse and first_relation is None:
            first_relation = {"row":row,"kind":kind,"source":source,"target":target,"occupancy":occupancy,"forward":forward,"reverse":reverse}
    return {"whole_mask_comparisons":len(queries),"relation_comparisons":len(pairs),"positive_relations":positives,
            "geometry_mismatch":first_geometry,"reciprocity_mismatch":first_relation}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("compiler",type=Path);ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args();atomic_report(args.report,{"native_gate":"NOT_COMPLETED"})
    compiler=args.compiler.resolve();bun=os.environ.get("BUN","bun");cc=os.environ.get("CC","clang")
    identity=success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]));before=snapshot(ENGINE)
    pairs,queries,coverage=fixtures()
    require(coverage["aligned_unordered_pairs"]==728 and coverage["interior_subsets_across_pairs"]==5322,"Incorrect fixture coverage")
    modes=[];mutations=[]
    with tempfile.TemporaryDirectory(prefix="slider-native-") as td:
        tmp=Path(td)
        def generate(root:Path,name:str)->Path:
            p=tmp/(name+".c")
            success(command([bun,"--smol",str(compiler/"bend2/main.ts"),str(root/"standalone/proofs/ray/probe.bend"),"-o",str(p)],240))
            require(p.is_file(),"No emitted C");return p
        def build(source:Path,name:str,flags:list[str])->Path:
            p=tmp/name;success(command([cc,"-std=c11","-O2",*flags,str(source),"-pthread","-lm","-o",str(p)],240));return p
        def observe(binary:Path)->list[int]:
            values=[]
            for start in range(0,len(queries),512):
                batch=queries[start:start+512]
                rows=[(1,source+64*kind,0,occ>>32,occ&0xffffffff,0,0) for kind,source,occ in batch]
                raw=success(command([str(binary),*(str(v) for r in rows for v in r)],30))
                values.extend(parse(raw,len(rows)))
            return values
        source=generate(ENGINE,"probe")
        invalid=[["x"],["-1","0","0","0","0","0","0"],["4294967296","0","0","0","0","0","0"],
                 ["3","0","0","0","0","0","0"],["1","128","0","0","0","0","0"],
                 ["0","64","0","0","0","0","0"],["1","0","1","0","0","0","0"],
                 ["0","0","8","0","0","0","0"],[str(v) for _ in range(1025) for v in (1,0,0,0,0,0,0)]]
        for mode,flags in MODES.items():
            binary=build(source,mode,flags);values=observe(binary);a=analyze(values,queries,pairs)
            require(a["geometry_mismatch"] is None and a["reciprocity_mismatch"] is None,"Incorrect native mask: "+str(a))
            for batch in invalid:
                r=command([str(binary),*batch],30)
                require(r["exit_code"]==2 and any(msg in r["stdout"]+r["stderr"] for msg in ("invalid", "budget or tuple error")),"Wrong malformed batch rejection")
                require("runtime error" not in r["stderr"],"Sanitizer failure is not malformed rejection")
            modes.append({"mode":mode,"invalid_rejections":len(invalid),"mask_u32_fields":2*len(values),
                          "output_sha256":sha(json.dumps(values,separators=(",",":")).encode()),**a})
            print("PASS native "+mode,flush=True)
        edits=[
          ("ignore-blockers","stop = Bool.or(U32.is_eq(next, 64), U64.test_bit(occ, U32.to_nat(next)))","stop = U32.is_eq(next, 64)",True),
          ("omit-first-blocker","acc = U64.or(acc, Bool.pick(U64, add, U64.bit(U32.to_nat(next)), U64.zero()))",
           "acc = U64.or(acc, Bool.pick(U64, Bool.and(add,Bool.not(U64.test_bit(occ,U32.to_nat(next)))), U64.bit(U32.to_nat(next)), U64.zero()))",False)]
        for name,old,new,still_reciprocal in edits:
            root=tmp/name;shutil.copytree(ENGINE,root,symlinks=True);p=root/"standalone/Tables.bend";text=p.read_text()
            require(text.count(old)==1,"Nonunique mutation site: "+name);p.write_text(text.replace(old,new))
            binary=build(generate(root,name),name+"-exe",[]);values=observe(binary);a=analyze(values,queries,pairs)
            require(a["geometry_mismatch"] is not None,"Mutation evaded independent geometry")
            require((a["reciprocity_mismatch"] is None)==still_reciprocal,"Unexpected reciprocity mutation result")
            mutations.append({"name":name,"compiled_and_executed":True,"rejected":True,**a})
            print("PASS mutation "+name,flush=True)
    require(snapshot(ENGINE)==before,"Source drift")
    require(success(command([bun,str(ENGINE/"standalone/verify_compiler.js"),str(compiler)]))==identity,"Compiler drift")
    atomic_report(args.report,{"native_gate":"PASS","compiler_identity":identity,"cc":success(command([cc,"--version"])),
                              "coverage":coverage,"fixture_sha256":sha(json.dumps({"pairs":pairs,"queries":queries},separators=(",",":")).encode()),
                              "modes":modes,"mutations":mutations,"source_sha256s":before,
                              "scope":"Actual unmasked Tables.slider and its rook/bishop union, not initialized lookup, full attacked, or legal-move safety."})

if __name__=="__main__":main()
