"""Opt-in actual Chess.slide checks; no production source is modified.

Usage: python review_native.py REPOSITORY PINNED_COMPILER --bun /path/to/bun
       --report /tmp/lookup-review.json [--cc clang]
Run from a checkout containing this promoted proof suite. The pinned compiler
is verified before and after execution. Native fixtures do not establish an
independent source-level geometry theorem.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('repository', type=Path)
    ap.add_argument('compiler', type=Path)
    ap.add_argument('--bun', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    ap.add_argument('--cc', default='clang')
    args = ap.parse_args()
    engine = args.repository.resolve() / 'native/bend_engine'
    cli = args.compiler.resolve() / 'bend2/main.ts'
    probe = engine / 'standalone/proofs/lookup/probe.bend'
    bun = args.bun.resolve()
    env = dict(os.environ, BEND_NO_TELEMETRY='1')

    def invoke(argv: list[str | Path], timeout: int = 180) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(v) for v in argv], capture_output=True, text=True,
                              env=env, timeout=timeout, check=False)

    def run(argv: list[str | Path]) -> str:
        p = invoke(argv)
        if p.returncode != 0 or p.stderr:
            raise RuntimeError(f'Unexpected process result {p.returncode}: {p.stderr[-2400:]} {p.stdout[-1200:]}')
        return p.stdout

    identity = run([bun, engine / 'standalone/verify_compiler.js', args.compiler.resolve()]).strip()
    assert identity == 'verified compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae (Bend 2.0.21 + U64), 84 files, source d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4'
    seed = 0x12FEDCAB89127345
    word_mask = (1 << 64) - 1
    cases: list[tuple[int, int]] = []
    for key in range(128):
        values = [0, word_mask, 1 << 63, 1 << (key % 64), 0xAAAAAAAAAAAAAAAA, 0x5555555555555555]
        for _ in range(2):
            seed ^= (seed << 13) & word_mask
            seed ^= seed >> 7
            seed ^= (seed << 17) & word_mask
            seed &= word_mask
            values.append(seed)
        cases.extend((key, value) for value in values)

    def attacks(key: int, occupancy: int) -> int:
        square = key % 64
        x, y = square % 8, square // 8
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if key >= 64 else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        result = 0
        for dx, dy in directions:
            file, rank = x + dx, y + dy
            while 0 <= file < 8 and 0 <= rank < 8:
                bit = 1 << (rank * 8 + file)
                result |= bit
                if occupancy & bit:
                    break
                file += dx
                rank += dy
        return result

    expected = [f'{key} {attacks(key, v) >> 32} {attacks(key, v) & 0xffffffff}' for key, v in cases]
    query_args = [s for key, v in cases for s in (str(key), str(v >> 32), str(v & 0xffffffff))]
    tracked = ['legal_probe/Chess.bend', 'bitboard_probe/Sliders.bend', 'standalone/Tables.bend',
               'standalone/Subsets.bend', 'standalone/Text.bend', 'standalone/proofs/lookup/probe.bend',
               'standalone/proofs/lookup/Route.bend']
    before = {name: digest((engine / name).read_bytes()) for name in tracked}
    results = []
    with tempfile.TemporaryDirectory(prefix='deepfin-lookup-review-') as work:
        workdir = Path(work)
        c = workdir / 'lookup.c'
        run([bun, cli, probe, '-o', c])
        for mode, flags in [('generic', []), ('portable', ['-DBEND_U64_PORTABLE']),
                            ('native', ['-march=native']),
                            ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
            exe = workdir / mode
            run([args.cc, '-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', *flags,
                 c, '-pthread', '-lm', '-o', exe])
            raw = run([exe, '--threads', '1', *query_args])
            lines = raw.splitlines()
            assert len(lines) == len(expected), (mode, len(lines))
            for i, (observed, wanted) in enumerate(zip(lines, expected)):
                assert observed == wanted, (mode, i, cases[i], observed, wanted)
            invalid = [['128', '0', '0'], ['0', '4294967296', '0'], ['x', '0', '0'],
                       ['0', '0'], ['0', '0', '-1'], ['0', '0', '0'] * 1025]
            for bad in invalid:
                p = invoke([exe, '--threads', '1', *bad])
                assert p.returncode == 2 and re.search(r'invalid lookup|lookup input budget', p.stdout + p.stderr)
            results.append({'mode': mode, 'rows': len(lines), 'invalid_rejections': len(invalid),
                            'output_sha256': digest(raw.encode())})
            print(f'PASS {mode}: {len(lines)} actual lookups and {len(invalid)} invalid requests', flush=True)

        mutant = workdir / 'mutant'
        shutil.copytree(engine, mutant)
        chess = mutant / 'legal_probe/Chess.bend'
        original = chess.read_bytes()
        needle = 'Sliders.lookup(table, U64.low(offset), index)'
        text = original.decode()
        assert text.count(needle) == 1
        chess.write_text(text.replace(needle, 'Sliders.lookup(table, U32.inc(U64.low(offset)), index)'))
        run([bun, cli, mutant / 'standalone/proofs/lookup/probe.bend', '-o', workdir / 'mutant.c'])
        run([args.cc, '-std=c11', '-O1', workdir / 'mutant.c', '-pthread', '-lm', '-o', workdir / 'mutant-bin'])
        bad = run([workdir / 'mutant-bin', '--threads', '1', *query_args]).splitlines()
        assert len(bad) == len(expected)
        first = next((i for i, (a, b) in enumerate(zip(bad, expected)) if a != b), None)
        assert first is not None, 'actual shifted lookup unexpectedly matched the independent reference'
        source_bad = invoke([bun, cli, mutant / 'standalone/proofs/lookup/Route.bend'])
        diagnostic = (source_bad.stdout + source_bad.stderr).strip()
        assert source_bad.returncode == 1 and re.search(r'expected[\s\S]*observed', diagnostic)
        assert 'Location: model' in diagnostic
        assert not re.search(r'RangeError|no such file|Segmentation fault|more than once', diagnostic)
        mutation = {'native_wrong_value_rejected': True, 'row': first,
                    'input_key': cases[first][0], 'input_occupancy': str(cases[first][1]),
                    'observed': bad[first], 'expected': expected[first],
                    'source_rejection_status': source_bad.returncode, 'source_rejection_location': 'Route.model',
                    'source_diagnostic_sha256': digest(diagnostic.encode()),
                    'source_diagnostic_bytes': len(diagnostic.encode()),
                    'original_Chess_sha256': digest(original), 'mutated_Chess_sha256': digest(chess.read_bytes())}
    after = {name: digest((engine / name).read_bytes()) for name in tracked}
    assert after == before
    assert run([bun, engine / 'standalone/verify_compiler.js', args.compiler.resolve()]).strip() == identity
    report = {'native_review': 'PASS', 'compiler_identity': identity, 'rows_per_mode': 1024,
              'all_keys': 128, 'distinct_key_occupancy_pairs': len(set(cases)), 'modes': results, 'actual_lookup_mutation': mutation,
              'source_sha256s': before, 'driver_sha256': digest(Path(__file__).read_bytes()),
              'scope': 'Actual Chess.slide on one retained Tables.build per execution; generic, portable, native-target and UBSan. Selected queries, not full buffers or an independent source ray theorem.',
              'cc': run([args.cc, '--version']).splitlines()[0]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
