"""Opt-in session cache qualification: identical PUCT work, wire and evaluator requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

from . import run_probe as sessions
from ..u64_map_probe.reservation_ownership import Commands

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETTING = 'DEEPFIN_SESSION_MOVE_CACHE_BITS'
INVALID = ('', '-1', '+1', ' 6', '6 ', '1.0', '17', '99', '000', '4294967296', 'x', '١')


def environment(value: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop(SETTING, None)
    env.pop('BEND_U64_CORRUPT', None)
    if value is not None:
        env[SETTING] = value
    return env


class Peer(sessions.Peer):
    def __init__(self, binary: Path, board: sessions.rules.Position, value: str | None) -> None:
        self.transcript: list[tuple[str, str]] = []
        super().__init__(binary, board, environment=environment(value))

    def write(self, text: str) -> None:
        self.transcript.append(('in', text))
        super().write(text)

    def line(self) -> str:
        line = super().line()
        self.transcript.append(('out', line))
        return line


def parse_stats(line: str) -> tuple[int, int, int, int]:
    bits, hits, fills, bypasses = sessions.numbers(line, 'cache', 4)
    if bits > 16 or (bits == 0 and (hits or fills or bypasses)):
        raise ValueError('invalid cache mode/counters')
    if bits and fills > 1 << (bits - 1):
        raise ValueError('cache fills exceed record capacity')
    return bits, hits, fills, bypasses


def stats(peer: Peer, clear: bool = False) -> tuple[int, int, int, int]:
    peer.write('clear_cache\n' if clear else 'cache\n')
    result = parse_stats(peer.line())
    peer.expect('ready')
    return result


def check_reuse(first: tuple[int, ...], second: tuple[int, ...], after_clear: tuple[int, ...]) -> None:
    if first != (6, 0, 16, 0) or second != (6, 16, 16, 0) or after_clear != first:
        raise ValueError('cache reuse/clear contract failed; accepted setting is not sufficient')


def check_invalid(result: subprocess.CompletedProcess[str], message: str) -> None:
    if result.returncode != 2 or result.stderr != message + '\n':
        raise ValueError('unexpected session-cache rejection')


def fingerprint(transcript: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(transcript, separators=(',', ':')).encode()).hexdigest()


def query(peer: Peer, oracle: sessions.Oracle, board: sessions.rules.Position, **kwargs: Any) -> tuple[dict[str, Any], str]:
    before = len(peer.transcript)
    result = sessions.session(peer, oracle, board, **kwargs)
    return result, fingerprint(peer.transcript[before:])


def advance(peer: Peer, oracle: sessions.Oracle, board: sessions.rules.Position,
            old: int, new: int, src: int, dst: int) -> sessions.rules.Position:
    legal = oracle.moves(board)
    key = next(k for k in legal if k & 63 == src and (k >> 6) & 63 == dst)
    peer.write(f'advance {old:x} {new:x} {key:x}\n')
    if sessions.numbers(peer.line(), 'advance_result', 5) != [old, new, key, 0, new]:
        raise AssertionError('root advance changed')
    child = sessions.position(sessions.numbers(peer.line(), 'board', 19))
    if child != legal[key]:
        raise AssertionError('advance returned a different board')
    peer.expect('ready')
    return child


def run_scenario(binary: Path, oracle: sessions.Oracle, value: str | None, output: Path) -> dict[str, Any]:
    start = sessions.rules.fen_position(sessions.rules.START)
    peer = Peer(binary, start, value)
    observations: list[dict[str, Any]] = []
    snapshots = []
    try:
        snapshots.append(stats(peer))
        for epoch in (1, 2):
            result, wire = query(peer, oracle, start, epoch=epoch, budget=16)
            observations.append({'label': f'repeat-{epoch}', 'result': result, 'wire_sha256': wire})
            snapshots.append(stats(peer))
        snapshots.append(stats(peer, clear=True))
        result, wire = query(peer, oracle, start, epoch=3, budget=16)
        observations.append({'label': 'after-clear', 'result': result, 'wire_sha256': wire})
        snapshots.append(stats(peer))
        # Root advancement keeps pure legal lists; its own validation remains uncached.
        current = start
        for old, new, src, dst in ((3, 4, 6, 21), (4, 5, 62, 45), (5, 6, 21, 6), (6, 7, 45, 62)):
            current = advance(peer, oracle, current, old, new, src, dst)
        if current != start:
            raise AssertionError('knight cycle failed to restore the structural board')
        result, wire = query(peer, oracle, current, epoch=8, budget=16)
        observations.append({'label': 'after-cycle', 'result': result, 'wire_sha256': wire})
        snapshots.append(stats(peer))
        for epoch, fault in enumerate(('cancel', 'backend', 'epoch', 'request', 'node', 'count',
                                       'nan', 'infinity', 'negative', 'zero_policy', 'bad_wdl', 'status'), 9):
            result, wire = query(peer, oracle, current, epoch=epoch, fault=fault)
            observations.append({'label': fault, 'result': result, 'wire_sha256': wire})
        for epoch, label, options in ((21, 'recovery', {'budget': 8}),
                                       (22, 'capacity', {'cap': 32}),
                                       (23, 'cutoff', {'depth': 1, 'budget': 64}),
                                       (24, 'changed-policy', {'variant': 1}),
                                       (25, 'no-room', {'cap': 1})):
            result, wire = query(peer, oracle, current, epoch=epoch, **options)
            observations.append({'label': label, 'result': result, 'wire_sha256': wire})
        peer.finish()
        peer.errors.seek(0)
        if peer.errors.read():
            raise AssertionError('unexpected session stderr')
    finally:
        output.write_text(json.dumps(peer.transcript, indent=2) + '\n')
        peer.close()
    if value == '6':
        check_reuse(snapshots[1], snapshots[2], snapshots[4])
        if snapshots[5] != snapshots[2]:
            raise AssertionError('legal lists did not survive root advance')
    elif value == '1':
        if snapshots[1] != (1, 0, 1, 15) or snapshots[2] != (1, 1, 1, 30):
            raise AssertionError('saturated cache changed admission/fallback')
    elif value in (None, '0') and any(s != (0, 0, 0, 0) for s in snapshots):
        raise AssertionError('disabled cache allocated or handled a lookup')
    return {'observations': observations, 'stats': snapshots}


def host_decisions(binary: Path, oracle: sessions.Oracle, value: str | None) -> list[str]:
    """A warm legal hit must not bypass a new host draw or optional-claim reply."""
    board = sessions.rules.fen_position(sessions.rules.START)
    peer = Peer(binary, board, value)
    fingerprints = []
    try:
        query(peer, oracle, board, epoch=1, budget=1)
        for epoch, claim in ((2, False), (3, True)):
            before = len(peer.transcript)
            ref = sessions.Reference(board, oracle, cap=4096, depth=4, budget=1)
            wanted = ref.next()
            assert wanted == 0
            peer.write(f'config {epoch:x} 1 1000 4\n')
            header = sessions.numbers(peer.line(), 'eval', 4)
            assert header[:3] == [epoch, 1, 0]
            assert sessions.position(sessions.numbers(peer.line(), 'board', 19)) == board
            assert sessions.parse_path(peer.line()) == []
            actions = [sessions.numbers(peer.line(), 'action', 1)[0] for _ in range(header[3])]
            peer.expect('end_eval')
            assert set(actions) == set(oracle.moves(board))
            if claim:
                wdl, policy = sessions.evaluation(board, actions)
                ref.accept(0, actions, wdl, policy, claim=True)
                fields = [epoch, 1, 0, 4, *(sessions.bits(v) for v in wdl), len(policy),
                          *(sessions.bits(v) for v in policy)]
            else:
                ref.accept_draw(0)
                fields = [epoch, 1, 0, 3, 0, sessions.bits(1.0), 0, 0]
            peer.write('reply ' + ' '.join(f'{v:x}' for v in fields) + '\n')
            assert ref.next() is None
            result = sessions.numbers(peer.line(), 'result', 6)
            rows = [sessions.numbers(peer.line(), 'node', 30) for _ in ref.nodes]
            best = sessions.numbers(peer.line(), 'best', 1)[0]
            ref.check_snapshot(rows, result, best, epoch)
            peer.expect('ready')
            fingerprints.append(fingerprint(peer.transcript[before:]))
        if value == '6' and stats(peer) != (6, 2, 1, 0):
            raise AssertionError('host-decision requests did not exercise warm legal hits')
        peer.finish()
    finally:
        peer.close()
    return fingerprints



def check_generation(result: subprocess.CompletedProcess[str], baseline: bool) -> None:
    # Existing foreign IO is expected; an extra warning or foreign definition is not.
    names = ['Job.load', 'Command.read', 'Reply.read', 'step_path_checked', 'step_traced',
             'step_read', 'step_prepared']
    if not baseline:
        names.append('cached_step')
    names.extend(('step', 'run', 'begin_valid', 'begin', 'command', 'dispatch', 'command_dispatch',
                  'serve', 'validated', 'start', 'main'))
    diagnostic = f'All terms check, but {len(names)} defs rely on unsafe or foreign code:\n'
    diagnostic += ''.join('- ' + name + '\n' for name in names)
    if result.returncode != 0 or result.stdout or result.stderr != diagnostic:
        raise ValueError('unexpected session compiler diagnostic or foreign boundary')


def build(root: Path, output: Path, compiler: list[str], cc: str, commands: Commands,
          label: str, flags: list[str]) -> Path:
    output.mkdir()
    generated = output / 'session.c'
    result = commands.run([*compiler, str(root / 'native/bend_engine/session_probe/main.bend'), '-o', str(generated)], label + '-generate')
    check_generation(result, label == 'baseline')
    support = output / 'support.o'
    commands.clean([cc, '-std=c11', '-O1', *flags, '-I', str(root), '-I', str(ROOT), '-c',
                    str(root / 'native/bend_engine/legal_probe/support.c'), '-o', str(support)], label + '-support')
    binary = output / 'session'
    commands.clean([cc, '-std=c11', '-O1', '-ffp-contract=off', *flags, '-I',
                    str(root / 'native/bend_engine/legal_probe'), str(generated), str(support),
                    '-pthread', '-lm', '-o', str(binary)], label + '-link')
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python-chess', action='store_true')
    args = parser.parse_args()
    if not args.bun or not args.cc or args.output.exists():
        parser.error('Bun, Clang and a fresh output directory are required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'modes': [],
                              'full_history_suites': args.python_chess}
    commands = Commands(output)
    try:
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        report['compiler'] = commands.clean([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                             str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = commands.clean([args.cc, '--version'], 'cc')
        # Exact previously published frontend is the disabled-wire compatibility control.
        baseline = args.baseline_root.resolve()
        expected = 'd0845b79fbae845eb7f24153c5017ead4adb319dbb0d3793bc0b7e9bac28bb8b'
        if hashlib.sha256((baseline / 'native/bend_engine/session_probe/main.bend').read_bytes()).hexdigest() != expected:
            raise ValueError('baseline is not the preregistered parent frontend')
        reference = output / 'oracle'
        commands.clean([args.cc, '-std=c11', '-O2', '-DLEGAL_ORACLE', '-I', str(ROOT),
                        str(sessions.rules.HERE / 'support.c'), '-pthread', '-lm', '-o', str(reference)], 'oracle-build')
        oracle = sessions.Oracle(reference, args.python_chess)
        old = build(baseline, output / 'baseline', compiler, args.cc, commands, 'baseline', [])
        start = sessions.rules.fen_position(sessions.rules.START)
        old_peer = Peer(old, start, None)
        try:
            control = query(old_peer, oracle, start, epoch=1, budget=16)
            old_peer.finish()
        finally:
            old_peer.close()
        for mode, flags in (('generic', []), ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])):
            binary = build(ROOT, output / mode, compiler, args.cc, commands, mode, flags)
            cases = {str(value): run_scenario(binary, oracle, value, output / f'{mode}-{value}.json')
                     for value in (None, '0', '6', '1')}
            default = cases['None']['observations']
            if (default[0]['result'], default[0]['wire_sha256']) != control:
                raise AssertionError('disabled frontend changed parent search output')
            for value in ('0', '6', '1'):
                if cases[value]['observations'] != default:
                    raise AssertionError('cache changed full search/evaluator transcript')
            decisions = [host_decisions(binary, oracle, value) for value in (None, '6')]
            if decisions[0] != decisions[1]:
                raise AssertionError('warm legal cache altered host draw/claim decisions')
            # Terminal empty lists, promotion/castling/en-passant: independent full tree oracle.
            edges = []
            edge_fixtures = [(name, fen) for name, fen, _ in sessions.rules.CANONICAL[:2]]
            edge_fixtures += [sessions.rules.EDGES[i] for i in (0, 3, 4, 5, 6, 14, 15, 16)]
            for label, fen in edge_fixtures:
                board = sessions.rules.fen_position(fen)
                pairs = []
                for value in (None, '6'):
                    peer = Peer(binary, board, value)
                    try:
                        pair = [query(peer, oracle, board, epoch=e, budget=8) for e in (1, 2)]
                        pairs.append(pair)
                        peer.finish()
                    finally:
                        peer.close()
                if pairs[0] != pairs[1]:
                    raise AssertionError('cache changed edge-position search')
                edges.append(label)
            for value in (*map(str, range(17)), '00', '06'):
                bits = int(value)
                reply = f'cache {bits} 0 0 0'
                result = subprocess.run([str(binary), '--threads', '1'],
                    input=sessions.rules.request(start, mode=1) + 'cache\nclear_cache\nconfig 0 0 0 0\n',
                    capture_output=True, text=True, env=environment(value), timeout=10, check=False)
                if (result.returncode or result.stderr
                        or result.stdout != f'ready\n{reply}\nready\n{reply}\nready\nbye\n'):
                    raise AssertionError('admitted cache setting/clear failed')
            for value in INVALID:
                result = subprocess.run([str(binary), '--threads', '1'], input='', capture_output=True,
                                        text=True, env=environment(value), timeout=10, check=False)
                check_invalid(result, 'invalid ' + SETTING)
                if result.stdout:
                    raise AssertionError('invalid cache configuration initialized a session')
            for command in ('cache 1\n', 'clear_cache 1\n', 'config 1 1 1000 4\nclear_cache\n'):
                result = subprocess.run([str(binary), '--threads', '1'],
                                        input=sessions.rules.request(start, mode=1) + command,
                                        capture_output=True, text=True, env=environment('6'), timeout=10, check=False)
                check_invalid(result, 'wrong transport record' if command.startswith('config') else 'invalid cache command')
            wrapper = output / ('enabled-' + mode)
            wrapper.write_text('#!/bin/sh\nexec env ' + SETTING + '=6 ' + shlex.quote(str(binary)) + ' "$@"\n')
            wrapper.chmod(0o755)
            existing = sessions.verify({'reference': reference, 'cached': wrapper}, args.python_chess)
            history: dict[str, Any] = {}
            if args.python_chess:
                from . import claim_probe, draw_probe
                history['draw'] = draw_probe.verify({'reference': reference, 'cached': wrapper})
                history['claims'] = claim_probe.verify({'reference': reference, 'cached': wrapper})
            report['modes'].append({'mode': mode, 'scenarios': cases, 'edge_positions': edges,
                                    'host_decision_wire': decisions[0], 'existing_sessions': existing,
                                    'history': history, 'invalid_settings': len(INVALID), 'invalid_commands': 3})
        # Detect a silently ineffective lifetime choice even when chess output matches.
        mutant_root = output / 'mutant-source'
        shutil.copytree(HERE.parent, mutant_root / 'native/bend_engine')
        path = mutant_root / 'native/bend_engine/session_probe/main.bend'
        source = path.read_text()
        old_site = '        done : Progress <- run('
        if source.count(old_site) != 1:
            raise ValueError('lifetime mutation site changed')
        path.write_text(source.replace(old_site, '        cache : Cache.State <- Cache.clear(cache)\n' + old_site))
        mutant = build(mutant_root, output / 'mutant', compiler, args.cc, commands, 'mutant', [])
        try:
            run_scenario(mutant, oracle, '6', output / 'mutant-wire.json')
        except ValueError as error:
            if 'cache reuse/clear contract' not in str(error):
                raise
            report['reset_each_epoch_rejected'] = True
        else:
            raise AssertionError('ineffective cache lifetime was not detected')
        report['sources'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(HERE.parent.rglob('*')) if p.is_file() and p.suffix in ('.bend', '.c', '.h')}
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'modes': len(report['modes'])}))


if __name__ == '__main__':
    main()
