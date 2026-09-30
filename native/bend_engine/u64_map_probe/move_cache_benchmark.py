"""Opt-in matched-work cache screen. Timings are descriptive, never CI speed gates."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import subprocess
from typing import Any

from ..legal_probe import run_probe as legal
from . import cpu_target
from .reservation_ownership import Commands

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ('direct', 'cached')
PAIRS = 6
MIN_MS = 50
MAX_ROUNDS = 65536
MASK = (1 << 32) - 1
MULT = 1664525


@dataclass(frozen=True)
class Case:
    name: str
    bits: int
    prime: tuple[str, ...]
    positions: tuple[str, ...]
    cold: bool = False


def canonical(fen: str) -> tuple[int, ...]:
    position = legal.fen_position(fen)
    side, _, ep = position[8:]
    pawns = position[0] & position[6 if side else 7]
    usable = ep < 64 and any(pawns & (1 << square)
        and ep // 8 - square // 8 == (1 if side else -1)
        and abs(ep % 8 - square % 8) == 1 for square in range(64))
    return (*position[:10], ep if usable else 64)


def cases() -> list[Case]:
    pool = (*(fen for _, fen, _ in legal.CANONICAL), *(fen for _, fen in legal.EDGES))
    target, filler = pool[:8], pool[8:16]
    if len({canonical(fen) for fen in (*target, *filler)}) != 16:
        raise ValueError('cache screen requires sixteen distinct structural boards')
    terminal = next(fen for name, fen in legal.EDGES if name == 'checkmate')
    return [*(Case(f'warm-{k * 100 // 8}pct-hits', 4, (*target[:k], *filler[:8-k]), target)
              for k in (0, 2, 4, 6, 8)),
            Case('cold-eight-once', 4, (), target, True),
            Case('warm-eight-large-arena', 16, target, target),
            Case('warm-empty-move-list', 1, (terminal,), (terminal,))]


def encode(case: Case, arm: str, rounds: int, diagnostic: bool = False) -> str:
    if arm not in ARMS or type(rounds) is not int or not 0 <= rounds <= MAX_ROUNDS:
        raise ValueError('invalid benchmark arm or cycle count')
    if type(case.bits) is not int or not 1 <= case.bits <= 16:
        raise ValueError('invalid benchmark capacity')
    if type(case.cold) is not bool or type(diagnostic) is not bool:
        raise ValueError('invalid benchmark mode')
    if not 1 <= len(case.positions) <= 128 or len(case.prime) > 128 or (case.cold and case.prime):
        raise ValueError('invalid benchmark corpus bounds')
    for fen in (*case.prime, *case.positions):
        if any(c in fen for c in '|;\n\r'):
            raise ValueError('invalid benchmark position delimiter')
        legal.fen_position(fen)
    prime = ';'.join('fen ' + fen for fen in case.prime) or '-'
    body = ';'.join('fen ' + fen for fen in case.positions)
    text = f'{ARMS.index(arm)} {case.bits} {rounds} {int(case.cold)} {int(diagnostic)}|{prime}|{body}'
    if len(text) > 100_000:
        raise ValueError('benchmark input exceeds transport budget')
    return text


def route_counts(case: Case, arm: str, rounds: int) -> tuple[int, int, int]:
    """Only two cycles are needed: an append-only repeated working set stabilizes."""
    if arm == 'direct':
        return 0, 0, len(case.positions) * rounds
    stored: set[tuple[int, ...]] = set()
    capacity = 1 << (case.bits - 1)
    for fen in case.prime:
        if len(stored) < capacity:
            stored.add(canonical(fen))
    def cycle() -> tuple[int, int, int]:
        counts = [0, 0, 0]
        if case.cold:
            stored.clear()
        for fen in case.positions:
            key = canonical(fen)
            if key in stored:
                counts[0] += 1
            elif len(stored) < capacity:
                stored.add(key)
                counts[1] += 1
            else:
                counts[2] += 1
        return counts[0], counts[1], counts[2]
    first = cycle() if rounds else (0, 0, 0)
    later = cycle() if rounds > 1 else (0, 0, 0)
    values = [a + max(0, rounds - 1) * b for a, b in zip(first, later, strict=True)]
    return values[0], values[1], values[2]


def pack(move: legal.Move) -> int:
    src, dst, promotion, flag = move
    return src | dst << 6 | promotion << 12 | flag << 15


def repeated_checksum(order: list[list[int]], rounds: int) -> int:
    if type(rounds) is not int or not 0 <= rounds <= MAX_ROUNDS:
        raise ValueError("invalid checksum cycle count")
    # Independently compose the state-preserving operation sequence modulo 2^32.
    a, b = 1, 0
    for moves in order:
        for token in [*(move + 1 for move in moves), 131072 + len(moves)]:
            a, b = a * MULT & MASK, (b * MULT + token) & MASK
    total = 0
    while rounds:
        if rounds & 1:
            total = (a * total + b) & MASK
        a, b = a * a & MASK, b * (a + 1) & MASK
        rounds >>= 1
    return total


def parse(text: str, case: Case, arm: str, rounds: int, diagnostic: bool,
          order: list[list[int]] | None, reference: list[set[int]]) -> dict[str, Any]:
    if not text.endswith('\n'):
        raise ValueError('incomplete benchmark output')
    lines = text.splitlines()
    header = f'configuration {ARMS.index(arm)} {case.bits} {rounds} {int(case.cold)} {int(diagnostic)}'
    wanted = rounds * len(case.positions) if diagnostic else 0
    if len(lines) != wanted + 2 or lines[0] != header:
        raise ValueError('benchmark configuration or output length differs')
    seen = []
    for ordinal, line in enumerate(lines[1:-1]):
        parts = line.split()
        if parts[:2] != ['trace', str(ordinal)] or any(not s.isascii() or not s.isdecimal() for s in parts[2:]):
            raise ValueError('invalid benchmark move trace')
        moves = list(map(int, parts[2:]))
        expected_moves = reference[ordinal % len(reference)]
        if len(moves) != len(set(moves)) or set(moves) != expected_moves:
            raise ValueError('benchmark moves differ from independent legal oracle')
        if order is not None and moves != order[ordinal % len(order)]:
            raise ValueError('benchmark changed native move ordering')
        seen.append(moves)
    if order is None:
        if not diagnostic or rounds != 1:
            raise ValueError('ordered reference requires a one-cycle diagnostic')
        order = seen
    parts = lines[-1].split()
    if len(parts) != 8 or parts[0] != 'sample' or any(not p.isascii() or not p.isdecimal() for p in parts[1:]):
        raise ValueError('invalid benchmark summary')
    ms, requests, count, checksum, hits, fills, bypasses = map(int, parts[1:])
    expected_summary = (len(case.positions) * rounds, sum(map(len, order)) * rounds,
                repeated_checksum(order, rounds), *route_counts(case, arm, rounds))
    if (requests, count, checksum, hits, fills, bypasses) != expected_summary or ms > 120000:
        raise ValueError('benchmark requested work, checksum, routes or timer differs')
    return {'milliseconds': ms, 'requests': requests, 'moves': count, 'checksum': checksum,
            'hits': hits, 'fills': fills, 'bypasses': bypasses, 'order': order}


def rss(text: str) -> int:
    words = text.split()
    if len(words) != 2 or any(not s.isascii() or not s.isdecimal() for s in words):
        raise ValueError('invalid child RSS record')
    peak, code = map(int, words)
    if not 0 < peak <= 1 << 40 or code != 0:
        raise ValueError('invalid child RSS or exit status')
    return peak


def summarize(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    measurements = [row for row in rows if row['phase'] == 'measurement']
    wanted = {(name, pair, arm) for name in names for pair in range(PAIRS) for arm in ARMS}
    keys = [(r['case'], r['pair'], r['arm']) for r in measurements]
    if len(keys) != len(wanted) or set(keys) != wanted:
        raise ValueError('incomplete or duplicate benchmark panel')
    summary = {}
    for name in names:
        panel = [r for r in measurements if r['case'] == name]
        if len({(r['rounds'], r['requests'], r['moves'], r['checksum'], r['workload_sha256']) for r in panel}) != 1:
            raise ValueError('unequal work between benchmark arms')
        for row in panel:
            if any(type(row[k]) is not int for k in ('pair', 'order', 'rounds', 'milliseconds', 'peak_rss_kib')):
                raise ValueError('invalid benchmark observation type')
            if not 1 <= row['rounds'] <= MAX_ROUNDS or not 0 <= row['milliseconds'] <= 120000 or row['peak_rss_kib'] <= 0:
                raise ValueError('invalid benchmark observation bounds')
            if row['order'] != (row['pair'] + ARMS.index(row['arm'])) % 2:
                raise ValueError('unbalanced benchmark execution order')
        reliable = min(r['milliseconds'] for r in panel) >= MIN_MS
        ratios = []
        for pair in range(PAIRS):
            times = {r['arm']: r['milliseconds'] for r in panel if r['pair'] == pair}
            ratios.append(times['direct'] / times['cached'] if times['cached'] else 0)
        decision = 'below_measurement_floor'
        if reliable:
            decision = ('cache_faster' if all(v > 1.05 for v in ratios) else
                        'cache_slower' if all(v < 1 / 1.05 for v in ratios) else 'inconclusive_at_5pct')
        summary[name] = {'cycles': panel[0]['rounds'], 'requests_per_sample': panel[0]['requests'],
            'reliable': reliable, 'decision': decision,
            'median_direct_over_cached': statistics.median(ratios) if reliable else None,
            'ratios': ratios if reliable else None,
            'median_ms': {a: statistics.median(r['milliseconds'] for r in panel if r['arm'] == a) for a in ARMS},
            'rss_kib': {a: {'median': statistics.median(r['peak_rss_kib'] for r in panel if r['arm'] == a),
                           'min': min(r['peak_rss_kib'] for r in panel if r['arm'] == a),
                           'max': max(r['peak_rss_kib'] for r in panel if r['arm'] == a)} for a in ARMS}}
    return summary



def invalid_inputs() -> tuple[str, ...]:
    """The recovered local screen's seven native admission checks, now reusable."""
    body = encode(cases()[4], 'cached', 1).partition('|')[2]
    headers = ('2 4 1 0 0', '1 0 1 0 0', '1 17 1 0 0', '1 4 65537 0 0',
               '1 4 4 0 1', '1 4 1 1 0')
    return (*(header + '|' + body for header in headers),
            '1 4 1 0 0|-|' + ';'.join(['fen ' + legal.START] * 129))


def check_invalid(result: subprocess.CompletedProcess[str]) -> None:
    if (result.returncode != 2 or result.stdout
            or result.stderr != 'invalid benchmark bounds\n'):
        raise ValueError('invalid benchmark input was not rejected as intended')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measure', action='store_true')
    args = parser.parse_args()
    if not args.bun or not args.cc or args.output.exists() or not Path('/usr/bin/time').is_file():
        parser.error('Bun, Clang, GNU time and a fresh output directory are required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    commands = Commands(output)
    samples: list[dict[str, Any]] = []
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'measured': args.measure,
                             'samples': samples, 'cases': [asdict(c) for c in cases()]}
    try:
        report['compiler'] = commands.clean([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                             str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = commands.clean([args.cc, '--version'], 'cc')
        report['target'] = cpu_target.qualify(args.cc, output, commands.clean)
        report['timer'] = commands.clean(['/usr/bin/time', '--version'], 'timer')
        if 'GNU' not in report['timer']:
            raise ValueError('GNU time is required for KiB RSS units')
        paths = [HERE / 'move_cache_benchmark.py', ROOT / 'native/bend_engine/legal_probe/run_probe.py',
                 *(HERE.glob('*.bend')), *(HERE.parent / 'standalone').glob('*.bend'),
                 *(HERE.parent / 'legal_probe').glob('*.[ch]'), HERE.parent / 'legal_probe/Chess.bend',
                 *(ROOT / 'chess_anti_engine/encoding').glob('*.h')]
        report['sources'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        generated = output / 'driver.c'
        commands.clean([*compiler, str(HERE / 'move_cache_benchmark.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in [('optimized', list(cpu_target.TARGET_FLAGS)),
                            ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
            commands.clean([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags, str(generated),
                            '-pthread', '-lm', '-o', str(output / mode)], 'build-' + mode)
        oracle = output / 'oracle'
        commands.clean([args.cc, '-std=c11', '-O2', '-DLEGAL_ORACLE', '-I', str(ROOT),
                        str(HERE.parent / 'legal_probe/support.c'), '-pthread', '-lm', '-o', str(oracle)], 'oracle-build')
        known = {}
        for case in cases():
            for fen in (*case.prime, *case.positions):
                if fen in known:
                    continue
                response = subprocess.run([str(oracle)], input=legal.request(legal.fen_position(fen), mode=1),
                    text=True, capture_output=True, check=True, timeout=30)
                if response.stderr:
                    raise ValueError('independent oracle emitted diagnostics')
                moves, _ = legal.parse_moves(response.stdout)
                if not isinstance(moves, dict):
                    raise ValueError('independent oracle returned a non-map result')
                known[fen] = {pack(m) for m in moves}
                (output / ('oracle-' + hashlib.sha256(fen.encode()).hexdigest() + '.stdout')).write_text(response.stdout)
        report['oracle_boards'] = len(known)
        orders: dict[str, list[list[int]]] = {}
        def sample(case: Case, arm: str, rounds: int, phase: str, pair: int = 0, order: int = 0,
                   mode: str = 'optimized', diagnostic: bool = False) -> dict[str, Any]:
            label = f'{len(samples):04}-{case.name}-{arm}-{phase}-{mode}'
            record = output / (label + '.rss')
            encoded = encode(case, arm, rounds, diagnostic)
            text = commands.clean(['env', 'DEEPFIN_MOVE_CACHE_BENCHMARK=' + encoded,
                '/usr/bin/time', '-f', '%M %x', '-o', str(record), str(output / mode), '--threads', '1'], label)
            parsed = parse(text, case, arm, rounds, diagnostic, orders.get(case.name), [known[fen] for fen in case.positions])
            orders[case.name] = parsed.pop('order')
            row = {'case': case.name, 'arm': arm, 'rounds': rounds, 'phase': phase, 'pair': pair,
                   'order': order, 'mode': mode, **parsed, 'peak_rss_kib': rss(record.read_text()),
                   'workload_sha256': hashlib.sha256(json.dumps(asdict(case), sort_keys=True).encode()).hexdigest(),
                   'input_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                   'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()}
            samples.append(row)
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            return row
        for mode in ('optimized', 'ubsan'):
            for case in cases():
                for arm in ARMS:
                    sample(case, arm, 1, 'diagnostic', mode=mode, diagnostic=True)
                    for rounds in (0, 3):
                        sample(case, arm, rounds, 'contract', mode=mode)
        report['contract_executions'] = len(samples)
        report['ordered_moves'] = orders
        # A reduced workload must not look like a faster implementation.
        with tempfile.TemporaryDirectory(prefix='cache-bench-negative-', dir=HERE.parent) as temp:
            folder = Path(temp)
            for name in ('MoveCache.bend', 'BoardIndex.bend', 'PositionIndex.bend', 'U64Map.bend'):
                shutil.copyfile(HERE / name, folder / name)
            original = (HERE / 'move_cache_benchmark.bend').read_text()
            old = 'repeat(U32.to_nat(rounds), arm, bits, cold, diag, boards,'
            if original.count(old) != 1:
                raise ValueError('benchmark negative-control site changed')
            changed = original.replace(old, 'repeat(U32.to_nat(U32.shr(rounds)), arm, bits, cold, diag, boards,')
            source = folder / 'driver.bend'
            source.write_text(changed)
            generated = output / 'shortened.c'
            commands.clean([*compiler, str(source), '-o', str(generated)], 'shortened-generate')
            binary = output / 'shortened'
            commands.clean([args.cc, '-std=c11', '-O2', str(generated), '-pthread', '-lm', '-o', str(binary)], 'shortened-build')
            case = cases()[4]
            response = commands.clean(['env', 'DEEPFIN_MOVE_CACHE_BENCHMARK=' + encode(case, 'cached', 3),
                                       str(binary), '--threads', '1'], 'shortened-run')
            try:
                parse(response, case, 'cached', 3, False, orders[case.name], [known[f] for f in case.positions])
            except ValueError as error:
                report['shortened_work_rejected'] = str(error)
            else:
                raise AssertionError('reduced benchmark work was accepted')
        for number, invalid in enumerate(invalid_inputs()):
            result = commands.run(['env', 'DEEPFIN_MOVE_CACHE_BENCHMARK=' + invalid,
                                   str(output / 'optimized'), '--threads', '1'], f'invalid-{number}')
            check_invalid(result)
        report['invalid_native_inputs'] = len(invalid_inputs())
        if args.measure:
            import time
            deadline = time.monotonic() + 240
            for case in cases():
                rounds = 32
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError('cache benchmark budget exhausted')
                    times = [sample(case, arm, rounds, 'calibration')['milliseconds'] for arm in ARMS]
                    if min(times) >= 150 or max(times) >= 2000 or rounds == MAX_ROUNDS:
                        break
                    rounds *= 2
                for pair in range(PAIRS):
                    for order, arm in enumerate(ARMS if pair % 2 == 0 else ARMS[::-1]):
                        if time.monotonic() > deadline:
                            raise TimeoutError('cache benchmark budget exhausted')
                        sample(case, arm, rounds, 'measurement', pair, order)
                print('Measured ' + case.name, flush=True)
            report['summary'] = summarize(samples, [case.name for case in cases()])
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report.get('summary', {'status': report['status']}), indent=2))


if __name__ == '__main__':
    main()
