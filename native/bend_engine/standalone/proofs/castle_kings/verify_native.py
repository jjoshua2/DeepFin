"""Actual start/transit/final Boards and checks versus an external coordinate/set oracle.

No proof-only Spec, law, expected answer or validity decision runs in the candidate.
All query sides are bounded; raw board metadata is preserved in diagnostic fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
REFERENCE = SUITE.parent / 'attack_witness/verify_native.py'
_spec = importlib.util.spec_from_file_location('accepted_forward_attack_reference', REFERENCE)
if _spec is None or _spec.loader is None:
    raise ImportError(str(REFERENCE))
Ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Ref)
MODES = Ref.MODES
MASK32 = 0xFFFFFFFF
CORNER = {0: 2, 7: 1, 56: 8, 63: 4}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                       env={**os.environ, 'BEND_NO_TELEMETRY': '1', 'TERM': 'dumb'})
    if r.returncode != 0 or r.stderr:
        raise AssertionError((args[:3], r.returncode, r.stderr[-3000:], r.stdout[-1000:]))
    return r


def consistent(board: list) -> bool:
    return all((not k and not c) or (len(k) == 1 and len(c) == 1) for k, c in board)


def copy(board: list) -> list:
    return [(set(k), set(c)) for k, c in board]


def move(board: list, meta: tuple[int, int, int], src: int, dst: int, castle: bool) -> tuple[list, tuple[int, int, int]]:
    # Exact raw update contract. Rights/EP arithmetic is not an independent rules proof.
    kind = next((k for k in range(5) if k in board[src][0]), 5)
    out = copy(board)
    rook_from = dst + 1 if dst % 8 == 6 else (dst - 2) & MASK32
    for sq in [src, dst] + ([rook_from] if castle else []):
        if 0 <= sq < 64:
            out[sq] = (set(), set())
    side = int(meta[0] == 1)
    Ref.put(out, dst, kind, side)
    if castle:
        middle = (src + dst) // 2
        out[middle][0].add(3)
        out[middle][1].add(side)
    lost = CORNER.get(src, 0) | CORNER.get(dst, 0) | ((3 if side else 12) if kind == 5 else 0)
    ep = (src + dst) // 2 if kind == 0 and (src ^ dst) == 16 else 64
    return out, (meta[0] ^ 1, meta[1] & (~lost & MASK32), ep)


def selected(board: list, side: int) -> list[int]:
    return [i for i, (k, c) in enumerate(board) if 5 in k and side in c]


def guard(board: list, meta: tuple[int, int, int], wing: int) -> bool:
    side = int(meta[0] == 1)
    home = 0 if side else 56
    src, rook = home + 4, home + (7 if wing else 0)
    path = [home + i for i in ([5, 6] if wing else [1, 2, 3])]
    right = (1 if side else 4) * (1 if wing else 2)
    return bool(meta[1] & right and 5 in board[src][0] and side in board[src][1]
                and 3 in board[rook][0] and side in board[rook][1]
                and all(not board[sq][1] for sq in path))


def expected(board: list, meta: tuple[int, int, int], sq: int, side: int) -> list[int]:
    kings = selected(board, side)
    plane = sum(1 << i for i in kings)
    index = min(kings) if kings else 64
    check = int(Ref.attacked(board, index, 1 - side)) if kings else 2
    return Ref.encode(board, meta) + [plane >> 32, plane & MASK32, index, check, int(Ref.attacked(board, sq, 1 - side))]


def fixtures() -> list[dict]:
    rng = random.Random(0xC451E2026)
    cases: list[dict] = []
    seen: set[tuple] = set()

    def add(category: str, b: list, meta: tuple[int, int, int], side: int, wing: int) -> None:
        inp = [side, wing, *Ref.encode(b, meta)]
        key = tuple(inp)
        if key in seen:
            return
        seen.add(key)
        src, mid, dst = (4, 5 if wing else 3, 6 if wing else 2) if side else (60, 61 if wing else 59, 62 if wing else 58)
        good = consistent(b)
        single = selected(b, side) == [src]
        valid_turn = meta[0] == side
        path_guard = guard(b, meta, wing)
        rows = [expected(b, meta, src, side)]
        mb, mm = move(b, meta, src, mid, False)
        fb, fm = move(b, meta, src, dst, True)
        rows += [expected(mb, mm, mid, side), expected(fb, fm, dst, side)]
        if good and single and valid_turn:
            assert selected(mb, side) == [mid]
        if good and single and valid_turn and path_guard:
            assert selected(fb, side) == [dst]
            assert all(r[-2] == r[-1] for r in rows)
        cases.append({'category': category, 'input': inp, 'expected': rows,
                      'transit_premises': good and single and valid_turn,
                      'castle_premises': good and single and valid_turn and path_guard})

    for side in (0, 1):
        home = 0 if side else 56
        src = home + 4
        for wing in (0, 1):
            rook = home + (7 if wing else 0)
            path = {home + i for i in ([5, 6] if wing else [1, 2, 3])}
            for i in range(64):
                b = Ref.empty()
                for sq in range(64):
                    if sq not in path | {src, rook} and rng.randrange(4) == 0:
                        Ref.put(b, sq, rng.randrange(5), rng.randrange(2))
                Ref.put(b, src, 5, side)
                Ref.put(b, rook, 3, side)
                allowed = sorted(set(range(64)) - path - {src, rook})
                Ref.put(b, allowed[i % len(allowed)], 5, 1 - side)
                add('guarded_singleton', b, (side, 15 | (rng.getrandbits(28) << 4), rng.getrandbits(32)), side, wing)
            for sq in range(64):
                b = Ref.empty()
                Ref.put(b, src, 5, side)
                Ref.put(b, rook, 3, side)
                Ref.put(b, sq, 5, 1 - side)
                add('opposing_king_all_squares', b, (side, 15, 64), side, wing)
                b2 = Ref.empty()
                Ref.put(b2, src, 5, side)
                Ref.put(b2, rook, 3, side)
                Ref.put(b2, sq, 5, side)
                add('additional_moving_king', b2, (side, 15, 64), side, wing)
            for old_side in [side, 1 - side, 2, 3, MASK32]:
                for kind in range(6):
                    b = Ref.empty()
                    Ref.put(b, src, kind, side)
                    Ref.put(b, rook, 3, side)
                    Ref.put(b, (56 if side else 0) + 4, 5, 1 - side)
                    add('kind_and_raw_side', b, (old_side, 15, 64), side, wing)
            for i in range(64):
                b = Ref.empty()
                for sq in range(64):
                    v = rng.randrange(13)
                    if v:
                        Ref.put(b, sq, (v - 1) % 6, int(v > 6))
                add('arbitrary_consistent', b, (side, rng.getrandbits(32), rng.getrandbits(32)), side, wing)
            b = Ref.empty()
            Ref.put(b, src, 5, side)
            Ref.put(b, rook, 3, side)
            Ref.put(b, home + (5 if wing else 3), 5, 1 - side)
            add('blocked_rook_king_counterexample', b, (side, 15, 32), side, wing)
            b[src][0].add(0)
            add('inconsistent_decoder_priority', b, (side, 15, 32), side, wing)
    return cases


def observe(binary: Path, cases: list[dict]) -> str:
    raw = ''
    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]
        text = run([str(binary), *map(str, [x for c in batch for x in c['input']])], 90).stdout
        lines = text.splitlines()
        if len(lines) != 3 * len(batch):
            raise AssertionError(('row count', start, len(lines)))
        for i, c in enumerate(batch):
            for stage, want in enumerate(c['expected']):
                got = list(map(int, lines[3 * i + stage].split()))
                if got != want:
                    field = next((j for j, pair in enumerate(zip(got, want)) if pair[0] != pair[1]), min(len(got), len(want)))
                    raise AssertionError(('wrong-value', start + i, c['category'], stage, field, got, want))
        raw += text
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('compiler', type=Path)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    compiler = args.compiler.resolve()
    bun, cc = os.environ.get('BUN', 'bun'), os.environ.get('CC', 'clang')
    pin = ENGINE / 'standalone/verify_compiler.js'
    identity = run([bun, str(pin), str(compiler)]).stdout
    paths = list(SUITE.glob('*')) + [ENGINE / p for p in ['legal_probe/Chess.bend', 'bitboard_probe/Sliders.bend', 'standalone/Tables.bend', 'standalone/Text.bend', 'standalone/verify_compiler.js', 'standalone/toolchain.json']] + [REFERENCE]
    hashes = {str(p.relative_to(ENGINE)): digest(p.read_bytes()) for p in paths if p.is_file()}
    cases = fixtures()
    if len(cases) != len({tuple(c['input']) for c in cases}):
        raise AssertionError('duplicate native input')
    malformed: list[list[str]] = []
    base = list(map(str, cases[0]['input']))
    for i, val in [(0, '2'), (1, '2'), (2, '-1'), (2, '4294967296'), (2, 'x')]:
        row = base.copy()
        row[i] = val
        malformed.append(row)
    malformed += [base[:-1], base + ['0'], base * 129]
    with tempfile.TemporaryDirectory(prefix='deepfin-castle-kings-') as temp:
        work = Path(temp)
        generated = work / 'probe.c'
        run([bun, str(compiler / 'bend2/main.ts'), str(SUITE / 'probe.bend'), '-o', str(generated)])
        modes = []
        for mode, flags in MODES.items():
            binary = work / mode
            run([cc, '-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', *flags, str(generated), '-pthread', '-lm', '-o', str(binary)])
            raw = observe(binary, cases)
            for row in malformed:
                r = subprocess.run([str(binary), *row], capture_output=True, text=True, timeout=90, check=False)
                if r.returncode != 2 or 'invalid castle king' not in r.stderr:
                    raise AssertionError(('invalid request not rejected', r.returncode, r.stderr[-1000:]))
            modes.append({'mode': mode, 'output_sha256': digest(raw.encode()), 'invalid_rejections': len(malformed)})
        mutations = []
        edits = [
            ('retains-old-king', 'update_piece(get_kings(b), remove, target, U32.is_eq(put, 5)),', 'U64.or(get_kings(b), select_u64(U32.is_eq(put, 5), target, U64.zero())),', cases),
            ('rook-added-to-king-plane', 'update_piece(get_kings(b), remove, target, U32.is_eq(put, 5)),', 'update_piece(get_kings(b), remove, U64.or(target, rook_to), U32.is_eq(put, 5)),', cases),
            ('turn-not-flipped', 'U32.xor(side, 1), U32.and(get_rights(b), U32.not(rights_lost)), ep}', 'side, U32.and(get_rights(b), U32.not(rights_lost)), ep}', cases),
        ]
        for name, before, after, selected_cases in edits:
            copy_engine = work / name
            shutil.copytree(ENGINE, copy_engine)
            chess = copy_engine / 'legal_probe/Chess.bend'
            txt = chess.read_text()
            if txt.count(before) != 1:
                raise AssertionError((name, 'nonunique mutation', txt.count(before)))
            chess.write_text(txt.replace(before, after))
            cfile, binary = work / (name + '.c'), work / (name + '.exe')
            run([bun, str(compiler / 'bend2/main.ts'), str(copy_engine / 'standalone/proofs/castle_kings/probe.bend'), '-o', str(cfile)])
            run([cc, '-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', str(cfile), '-pthread', '-lm', '-o', str(binary)])
            try:
                observe(binary, selected_cases)
            except AssertionError as exc:
                if not exc.args or not isinstance(exc.args[0], tuple) or exc.args[0][0] != 'wrong-value':
                    raise
                detail = exc.args[0]
                mutations.append({'name': name, 'compiled_and_executed': True, 'rejected': True, 'case': detail[1], 'category': detail[2], 'stage': detail[3], 'field': detail[4], 'observed': detail[5][detail[4]], 'expected': detail[6][detail[4]]})
            else:
                raise AssertionError((name, 'mutation accepted'))
    for name, h in hashes.items():
        if digest((ENGINE / name).read_bytes()) != h:
            raise AssertionError(('source drift', name))
    if run([bun, str(pin), str(compiler)]).stdout != identity:
        raise AssertionError('compiler drift')
    report = {'native_gate': 'PASS', 'cases': len(cases), 'distinct_inputs': len(cases), 'stage_boards_per_mode': 3 * len(cases), 'fields_per_mode': 72 * len(cases), 'transit_premise_cases': sum(c['transit_premises'] for c in cases), 'castle_premise_cases': sum(c['castle_premises'] for c in cases), 'categories': dict(Counter(c['category'] for c in cases)), 'fixture_sha256': digest(json.dumps(cases, sort_keys=True).encode()), 'modes': modes, 'mutations': mutations, 'source_sha256s': hashes, 'cc': run([cc, '--version']).stdout.splitlines()[0], 'scope': 'Complete actual start/transit/final Boards, mover king plane, bounded check index and check/attack values; not legal-move soundness or forward/reverse theorem.'}
    text = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
