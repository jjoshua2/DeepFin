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
subprocess.run(['git','-C',str(work),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(work),'diff','--cached','--check'],check=True)
manifest={n:hashlib.sha256((work/n).read_bytes()).hexdigest() for n in paths}
(Path(os.environ['RUNNER_TEMP'])/'chess-map-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
