from pathlib import Path
import hashlib,json,os,subprocess,sys
work=Path(sys.argv[1]); source=Path(__file__).parent
files={
 'native/bend_engine/u64_map_probe/chess_workloads.py':'01d61baac99bd9ba7919a5ffdeaf3f545bb0aac70b65c77a0c815d360416717f',
 'tests/test_bend_map_chess.py':'7f1db59f3d26a631049f305c5c7ecf548cbf5e0b756fe6bbb8768b55ef8aae2e',
 'docs/experiments/2026-09-24-chess-key-map-replay.md':'5716a81f88a885c3d96381cc23dbd6bb903ddb3d9f2f8f5da13c956a02a89cde'}
for name,sha in files.items():
 data=(source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==sha,name
 assert not (work/name).exists();(work/name).write_bytes(data)
p=work/'native/bend_engine/u64_map_probe/benchmark.py';s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='bef54cd3ef45a97b33edad448e29414cc89b8e06d381070d652507535e141586'
changes=[
 ("    parser.add_argument('--measure', action='store_true')", "    parser.add_argument('--measure', action='store_true')\n    parser.add_argument('--chess', action='store_true', help='use actual CBoard keys from deterministic legal positions')"),
 ("'samples': rows, 'cases': [asdict(c) for c in workloads()], 'measured': args.measure}", "'samples': rows, 'cases': [], 'measured': args.measure,\n                             'workload_source': 'chess' if args.chess else 'synthetic'}"),
 ("        cases = workloads()\n        for c in cases:", "        chess_traces = []\n        if args.chess:\n            from .chess_workloads import collect\n            cases, chess_traces, report['chess_corpus'] = collect()\n        else:\n            cases = workloads()\n        report['cases'] = [asdict(c) for c in cases]\n        for c in cases:"),
 ("        reference_cases = reference.fixtures()", "        reference_cases = reference.fixtures() + chess_traces\n        if args.chess:\n            report['chess_replay_cases'] = [\n                {'name': c.name, 'bits': c.bits, 'operations': len(c.ops),\n                 'input_sha256': hashlib.sha256(reference.encode(c).encode()).hexdigest(),\n                 'expected_sha256': hashlib.sha256(reference.expected(c).encode()).hexdigest()}\n                for c in chess_traces]")]
for old,new in changes:
 assert s.count(old)==1;s=s.replace(old,new)
assert hashlib.sha256(s.encode()).hexdigest()=='1935a842213561abfbb734253921918b91950b26a689f1d47a36acd30f3ba069'
block='''@dataclass(frozen=True)
class Workload:
    name: str
    bits: int
    initial: tuple[tuple[int, int], ...]
    ops: tuple[tuple[int, int, int], ...]  # get=0, put=1, remove=2
'''
assert s.count(block)==1
s=s.replace('from dataclasses import asdict, dataclass','from dataclasses import asdict').replace('from . import cpu_target','from . import cpu_target\nfrom .workload import Workload').replace(block+'\n\n','')
assert hashlib.sha256(s.encode()).hexdigest()=='ff3bbe0a5762da9f7a8bc4e15129d26b75a0cef2a556b58cc72ea3b866e2d4ae'
p.write_text(s); paths=[*files,str(p.relative_to(work))]
schema='''"""Shared immutable input schema for numeric-map operation drivers."""
from __future__ import annotations

from dataclasses import dataclass


'''+block
assert hashlib.sha256(schema.encode()).hexdigest()=='710ba181e2298a02e71e1f80377b4d7694c8a706419afca03dc5234657c5d558'
p=work/'native/bend_engine/u64_map_probe/workload.py';assert not p.exists();p.write_text(schema);paths.append(str(p.relative_to(work)))
p=work/'native/bend_engine/u64_map_probe/chess_workloads.py';s=p.read_text();assert s.count('from .benchmark import Workload')==1
s=s.replace('from .benchmark import Workload','from .workload import Workload')
assert hashlib.sha256(s.encode()).hexdigest()=='844f8c14e81b61c11e7b820e6a9ff79f551a8929bdd5870cf2322fb3226942ae'
p.write_text(s)
p=work/'docs/experiments/2026-09-24-chess-key-map-replay.md'
p.write_text(p.read_text()+'''\n\n### Preserved first static failure\n\nRun 36032177266 passed source checks, the locked build and Ruff, then Basedpyright rejected the circular import between the benchmark and the new chess fixture producer. No hosted Python or native replay had run. The unchanged frozen Workload schema now lives in workload.py and both producers import it, rather than suppressing the cycle diagnostic. The existing benchmark still exposes Workload through its import. Fixtures, native implementations, oracles and expected results are unchanged. Fourteen pure tests passed locally after this refactor; full hosted qualification is still a separate gate.\n''')
for name,text in {
 'docs/experiments/README.md':'\n- [CBoard-key map replay](2026-09-24-chess-key-map-replay.md): actual source-derived position keys, exact dictionary replay and explicit history/EP cache-identity limits; no performance or cache-adoption claim.\n',
 'native/bend_engine/u64_map_probe/README.md':'\n\n## Actual chess-key fixtures\n\nAdd `--chess` to the comparison driver to generate deterministic legal-position\nworkloads using the inspected checkout\'s actual CBoard transposition-key code.\nSee [the chess-key replay record](../../../docs/experiments/2026-09-24-chess-key-map-replay.md).\nIt includes full encounter replays and native EP/castling/history-key boundary\nchecks. These generated legal walks are not recorded production access traces.\nThe numeric map remains unaware of board fields, history and model identity;\ndo not treat a hit as permission to reuse a neural value. No performance panel\nis part of this correctness extension.\n'}.items():
 p=work/name;p.write_text(p.read_text()+text);paths.append(name)
# Fixed-budget amendment before native replay: sparse promotion walks can end early.
p=work/'native/bend_engine/u64_map_probe/chess_workloads.py';s=p.read_text()
s=s.replace('WALKS = 4','WALKS = 8')
s=s.replace("raise ValueError('chess corpus too small or exceeds bounded replay')", "raise ValueError(f'chess corpus {name}: {len(unique)} distinct / {len(keys)} observed; '\n                         f'requires at least {ENTRIES * 2} distinct and at most {WALKS * (PLIES + 1)} observed')")
s=s.replace('# 1,024 buckets / 512-entry limit exceeds this bounded corpus without eviction.', '# 2,048 buckets / 1,024-entry limit exceeds the 520-record bound without eviction.')
s=s.replace("Case('chess-' + name + '-encounters', 10, replay)", "Case('chess-' + name + '-encounters', 11, replay)")
assert hashlib.sha256(s.encode()).hexdigest()=='2b707b1093a524b6fa24245399a95f6de84b2672330ab8d27eeeb393dc4ed161'
p.write_text(s)
p=work/'tests/test_bend_map_chess.py';s=p.read_text()
s=s.replace('replay.bits == 10','replay.bits == 11').replace('list(range(261))','list(range(521))')
s=s.replace("corpus['observations'] <= 260", "corpus['observations'] <= 520").replace("startswith('begin 10\\n')", "startswith('begin 11\\n')")
assert hashlib.sha256(s.encode()).hexdigest()=='83d79871486089880a150ca3e0025a7f506108832bcc0348ad22003d0e4bb61f'
p.write_text(s)
p=work/'docs/experiments/2026-09-24-chess-key-map-replay.md'
p.write_text(p.read_text()+'''\n\n### Fixed corpus-budget amendment\n\nRun 36032729272 passed Ruff and Basedpyright, and 101 of 102 Python cases. All six native identity relationships and the raw-hash substitution negative passed. The corpus reproducibility case rejected the promotion seed because four bounded random legal walks did not supply the required 128 distinct keys. No native map replay or performance measurement ran. This is a fixture-coverage failure, not an observed map mismatch.\n\nThe fixed budget is now eight walks per seed, still capped at 64 plies each, for at most 520 observations per corpus. All four seeds use the same expanded budget and deterministic RNG sequence; no keys or seeds are selected by hash bucket. The minimum 128 distinct-key requirement, 64-entry operation workloads and all equality/identity expectations are unchanged. The encounter replay now reserves 2,048 buckets / 1,024 entries so every possible observed key fits without eviction. Oversize/replay tests use the corresponding 520-record bound. This supersedes the original four-walk/1,024-bucket plan above and is recorded before native replay. No performance result was rerolled.\n\nFailure artifact 10823596752 retains the 102-case JUnit report (101 passed, one failed), ZIP SHA-256 dbf0b580d095ff914e471a364776ddd035443d81ce3505640d876c55c9d08f43. Fourteen pure tests passed locally after the bound amendment; engine-dependent tests remain a hosted gate.\n''')
subprocess.run(['git','-C',str(work),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(work),'diff','--cached','--check'],check=True)
manifest={n:hashlib.sha256((work/n).read_bytes()).hexdigest() for n in paths}
(Path(os.environ['RUNNER_TEMP'])/'chess-map-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
