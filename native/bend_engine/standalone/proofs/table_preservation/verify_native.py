"""Whole-buffer before/after observations of actual threaded Chess operations.

This tests storage preservation, not correctness/completeness of the move set.
Raw arrays have independently computed index-distinct contents; initialized arrays
are compared cell-for-cell with their actual pre-operation snapshot. No proof
module or external expected answer is passed into the native candidate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

if __package__:
    from ._validation import begin_report, require, write_report
else:
    from _validation import begin_report, require, write_report

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
MODES = {'generic': [], 'portable': ['-DBEND_U64_PORTABLE'], 'native': ['-march=native'],
         'ubsan': ['-fsanitize=undefined', '-fno-sanitize-recover=all']}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                       env={**os.environ, 'BEND_NO_TELEMETRY': '1', 'TERM': 'dumb'})
    if p.returncode != 0 or p.stderr:
        raise AssertionError((cmd[:3], p.returncode, p.stderr[-2000:], p.stdout[-1000:]))
    return p


def board(pieces: list[tuple[int, int, int]], turn: int = 1, rights: int = 15, ep: int = 64) -> list[int]:
    planes = [0] * 8
    for sq, kind, side in pieces:
        planes[kind] |= 1 << sq
        planes[6 if side else 7] |= 1 << sq
    return [v for plane in planes for v in (plane >> 32, plane & 0xffffffff)] + [turn, rights, ep]


def fixtures() -> list[dict]:
    back = [3, 1, 2, 4, 5, 2, 1, 3]
    start = [(i, k, 1) for i, k in enumerate(back)] + [(56+i, k, 0) for i, k in enumerate(back)]
    start += [(8+i, 0, 1) for i in range(8)] + [(48+i, 0, 0) for i in range(8)]
    kings = [(4,5,1),(60,5,0)]
    rooks = kings + [(0,3,1),(7,3,1),(56,3,0),(63,3,0)]
    boards = [board(start),board(rooks),board(rooks,0),
              board([(4,5,1),(56,5,0),(60,3,0)],rights=0),
              board(kings+[(36,0,1),(35,0,0)],ep=43),
              board(kings+[(48,0,1),(57,3,0)],rights=0),
              board([],turn=3,rights=0xffffffff,ep=0xffffffff),
              board(kings+[(4,1,0),(28,4,1),(35,2,0)],turn=0)]
    out=[]
    for context in (0,1):
        for op in range(9):
            for i,b in enumerate(boards):
                out.append({'category':f'raw-{context}-operation-{op}', 'input':[context,op,*b]})
        for k in range(6):
            b=boards[0].copy();b[-2]=k;b[-1]=63
            out.append({'category':'raw-dispatch', 'input':[context,0,*b]})
    # Real full builder and full index-distinct 131072-cell arrays.
    for i,b in enumerate(boards):
        out.append({'category':'initialized-full-generator', 'input':[2,8,*b]})
    for op in range(9):
        out.append({'category':'large-nonuniform-operation', 'input':[3,op,*boards[1]]})
    require(len({tuple(c['input']) for c in out})==len(out), 'verify_native.py:69: validation failed')
    return out


class WrongValue(AssertionError):
    pass


def _u32_fields(line: str, tag: str, count: int) -> list[int]:
    """Accept exactly the canonical unsigned-decimal output of probe.bend."""
    fields = line.split(" ")
    require(len(fields) == count + 1 and fields[0] == tag,
            f"Invalid {tag} record: {line!r}")
    require(all(re.fullmatch(r"0|[1-9][0-9]{0,9}", x) is not None for x in fields[1:]),
            f"Noncanonical unsigned field: {line!r}")
    values = [int(x) for x in fields[1:]]
    require(all(x <= 0xffffffff for x in values), f"U32 overflow: {line!r}")
    return values


def parse_observation(stdout: str, case: dict, permit_corruption: bool = False) -> dict:
    """Validate the entire transcript before counting any compared cells.

    This checks framing, types, original synthetic contents and retained storage.
    It deliberately does not infer move legality from syntactically valid U32s.
    permit_corruption permits changed *values* for mutation diagnostics only;
    malformed, incomplete or out-of-range output is always rejected.
    """
    context, op = case['input'][:2]
    require(type(context) is int and 0 <= context <= 3, "Invalid context")
    require(type(op) is int and 0 <= op <= 8, "Invalid operation")
    size = [64, 1024, 131072, 131072][context]
    lines = stdout.splitlines()
    require(len(lines) >= 2 * size + 4, "Incomplete before/answer/after transcript")
    require(lines[0] == f'before {size}' and lines[-1] == f'done {size}',
            "Missing or incorrect buffer framing")
    try:
        at = lines.index(f'after {size}', size + 1)
    except ValueError as exc:
        raise AssertionError("Missing after-buffer header") from exc
    require(len(lines) == at + size + 2, "After-buffer size does not match header")
    before = lines[1:size + 1]
    after = lines[at + 1:at + 1 + size]
    require(len(before) == len(after) == size, "Incomplete logical buffer")
    for cell in before + after:
        _u32_fields(cell, 'cell', 2)
    if context != 2:
        expected = [f'cell {i ^ 2779096485} {(i*2654435761+12345)&0xffffffff}' for i in range(size)]
        require(before == expected, "Independent raw seed initialization mismatch")
    outputs = lines[size + 1:at]
    if op == 0:
        require(len(outputs) == 1, "Expected one word result")
        _u32_fields(outputs[0], 'word', 2)
    elif op == 1:
        require(len(outputs) == 1, "Expected one Boolean result")
        require(_u32_fields(outputs[0], 'bool', 1)[0] <= 1, "Non-Boolean result")
    else:
        require(bool(outputs) and outputs[-1] == 'end', "Missing move-list terminator")
        for move in outputs[:-1]:
            _u32_fields(move, 'move', 4)
    wrong = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    if wrong and not permit_corruption:
        raise WrongValue(f'op={op} context={context} changed cell {wrong[0]}: {before[wrong[0]]} -> {after[wrong[0]]}')
    return {'cells': len(before), 'changed_cells': wrong,
            'first_changed_before': before[wrong[0]] if wrong else None,
            'first_changed_after': after[wrong[0]] if wrong else None,
            'moves': len(outputs)-1 if op >= 2 else 0,
            'answers_sha256': digest('\n'.join(outputs).encode()),
            'before_sha256': digest('\n'.join(before).encode()),
            'after_sha256': digest('\n'.join(after).encode())}


def observe(binary: Path, case: dict, permit_corruption: bool = False) -> dict:
    stdout = run([str(binary), *map(str, case['input'])]).stdout
    return parse_observation(stdout, case, permit_corruption)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('compiler',type=Path);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();begin_report(args.report, 'native_gate');compiler=args.compiler.resolve();bun=os.environ.get('BUN','bun');cc=os.environ.get('CC','clang')
    identity=run([bun,str(ENGINE/'standalone/verify_compiler.js'),str(compiler)]).stdout
    inputs=[p for p in SUITE.iterdir() if p.is_file()]+[ENGINE/p for p in ['legal_probe/Chess.bend','bitboard_probe/Sliders.bend','standalone/Tables.bend','standalone/Text.bend','standalone/toolchain.json','standalone/verify_compiler.js']]
    hashes={str(p.relative_to(ENGINE)):digest(p.read_bytes()) for p in inputs}
    cases=fixtures(); modes=[]; mutations=[]
    with tempfile.TemporaryDirectory(prefix='table-preservation-native-') as tmp:
        tmp=Path(tmp);generated=tmp/'probe.c'
        run([bun,str(compiler/'bend2/main.ts'),str(SUITE/'probe.bend'),'-o',str(generated)])
        baseline=[]
        for mode,flags in MODES.items():
            binary=tmp/mode
            run([cc,'-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',*flags,str(generated),'-pthread','-lm','-o',str(binary)])
            results=[observe(binary,c) for c in cases]
            if baseline: require(results==baseline, 'verify_native.py:122: validation failed')
            else: baseline=results
            row=list(map(str,cases[0]['input']));invalids=[]
            for index,value in [(0,'4'),(1,'9'),(2,'-1'),(2,'4294967296'),(2,'x')]:
                bad=row.copy();bad[index]=value;invalids.append(bad)
            invalids += [row[:-1],row+['0']]
            for bad in invalids:
                proc=subprocess.run([str(binary),*bad],capture_output=True,text=True,timeout=30,check=False)
                require(proc.returncode==2 and 'invalid table preservation' in proc.stderr, 'verify_native.py:130: validation failed')
            modes.append({'mode':mode,'requests':len(cases),'complete_buffers':len(cases),'compared_cells':sum(r['cells'] for r in results),
                          'compared_u32_fields':2*sum(r['cells'] for r in results),'moves_observed':sum(r['moves'] for r in results),
                          'invalid_rejections':len(invalids),'observations_sha256':digest(json.dumps(results,sort_keys=True).encode())})
            print(f'PASS {mode}: {len(cases)} complete buffer comparisons',flush=True)
        scribble='Array.set(U64,table,131071,U64.from_parts(305419896,2596069104))'
        edits=[('scan-corrupts-unused-cell','(table, destinations(64n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc))',
                f'({scribble}, destinations(64n, U64.is_zero(targets), targets, src, pawn, ep_sq, acc))'),
               ('unchecked-filter-corrupts-unused-cell','(table, Con{m, acc})',f'({scribble}, Con{{m, acc}})'),
               ('queen-combination-corrupts-unused-cell','(table, U64.or(a, b))',f'({scribble}, U64.or(a, b))')]
        for name,old,new in edits:
            copied=tmp/name;shutil.copytree(ENGINE,copied,symlinks=True)
            chess=copied/'legal_probe/Chess.bend';source=chess.read_text();require(source.count(old)==1, 'verify_native.py:142: validation failed');chess.write_text(source.replace(old,new))
            code=tmp/(name+'.c');binary=tmp/(name+'.bin')
            run([bun,str(compiler/'bend2/main.ts'),str(copied/'standalone/proofs/table_preservation/probe.bend'),'-o',str(code)])
            run([cc,'-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',str(code),'-pthread','-lm','-o',str(binary)])
            checked=[]
            # Correct moves can hide corruption: compare all eight real-builder generator outputs.
            for i,c in enumerate(cases):
                if c['category']=='initialized-full-generator':
                    r=observe(binary,c,True)
                    require(r['answers_sha256']==baseline[i]['answers_sha256'], 'Mutation unexpectedly changed move list')
                    checked.append(r)
            require(any(r['changed_cells'] for r in checked), 'Mutant did not exhibit storage corruption')
            first=next(r for r in checked if r['changed_cells'])
            mutations.append({'name':name,'compiled_and_executed':True,'rejected':True,'build_mode':'generic',
                              'full_generator_cases':len(checked),'all_move_lists_match_clean':True,
                              'corrupt_buffers':sum(bool(r['changed_cells']) for r in checked),
                              'first_changed_cell':first['changed_cells'][0],'before':first['first_changed_before'],'after':first['first_changed_after']})
            print('PASS mutation '+name,flush=True)
    require(hashes=={str(p.relative_to(ENGINE)):digest(p.read_bytes()) for p in inputs}, 'verify_native.py:160: validation failed')
    require(run([bun,str(ENGINE/'standalone/verify_compiler.js'),str(compiler)]).stdout==identity, 'verify_native.py:161: validation failed')
    report={'native_gate':'PASS','requests_per_mode':len(cases),'complete_cells_per_mode':modes[0]['compared_cells'],
            'complete_fields_per_mode':modes[0]['compared_u32_fields'],'full_131072_cell_buffers_per_mode':17,
            'input_sha256':digest(json.dumps(cases,sort_keys=True).encode()),'modes':modes,'mutations':mutations,
            'source_sha256s':hashes,'cc':run([cc,'--version']).stdout.splitlines()[0],
            'scope':'Whole logical array values before/after actual operations. No native allocation/lifetime theorem or move-set correctness/completeness claim.'}
    write_report(args.report, report)


if __name__=='__main__': main()
