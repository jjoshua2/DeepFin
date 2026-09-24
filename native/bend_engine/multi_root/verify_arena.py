"""Opt-in bounded arena qualification; storage boundaries and actual selected-leaf searches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from . import verify
from .ci_completion import Commands
from .verify_fifo import semantic_view

HERE = Path(__file__).resolve().parent
CAPACITIES = (0, 1, 20, 4095, 4096, 4097, 8191, 8192, 8193, 16384, 16385,
              32768, 32769, 65536, 65537, 4294967295)
ENV = 'DEEPFIN_COHORT_ARENA_NODES'
ERROR = 'DEEPFIN_COHORT_ARENA_NODES must be a decimal integer in 1..65536\n'


def expected_probe(capacity: int = 4096) -> str:
    rows = [f'config {capacity}']

    def node(label: str, index: int, visits: int) -> None:
        rows.append(f'{label} {index} {visits} 4294967295 305419896')

    for cap in CAPACITIES:
        if not 1 <= cap <= 65536:
            rows.append(f'arena {cap} 1 1 1 0 4096 4')
            continue
        size = max(4096, 1 << (cap - 1).bit_length())
        rows.append(f'arena {cap} {cap} {size} 1 0 {max(4096, cap)} 0')
        node('first', 0, 24 if cap == 1 else 101)
        if cap > 4096:
            node('cross', 4096, cap + 23 if cap == 4097 else 777)
        node('last', cap - 1, cap + 23)
    for label, index, used, cap, completed, pending, request, stop in (
        ('idle', 4096, 1, 4096, 0, 4096, 20, 4),
        ('unallocated', 4096, 4096, 8192, 0, 8192, 20, 4),
        ('outside', 8192, 8193, 8192, 0, 8192, 20, 4),
        ('active', 4096, 4097, 8192, 1, 8192, 21, 0),
        ('active_last', 65535, 65536, 65536, 1, 65536, 21, 0),
        ('halt', 0, 1, 8192, 0, 8192, 1, 2),
        ('cancel', 0, 1, 65536, 0, 65536, 1, 2),
        ('deadline', 0, 1, 65536, 0, 65536, 1, 2),
    ):
        rows.append(f'{label} {index} {used} {cap} {completed} {pending} {request} {stop}')
        node('root', 0, completed)
        node('target', min(index, cap - 1), completed)
    return '\n'.join(rows) + '\n'


def probe_result(text: str, capacity: int = 4096) -> None:
    if text != expected_probe(capacity):
        raise ValueError('arena probe differs from storage/transaction contract')


def outcome(base: dict[str, Any], larger: dict[str, Any]) -> dict[str, int]:
    """Require actual work past the old capacity, not merely an accepted setting."""
    before, after = base['roots'][1], larger['roots'][1]
    if not (before['stop_code'] == 1 and before['completed_simulations'] < 256
            and before['used_nodes'] <= 4096 and after['stop_code'] == 0
            and after['completed_simulations'] == 256 and 4096 < after['used_nodes'] <= 8192):
        raise ValueError('larger arena did not satisfy the actual search gate')
    return {'default_simulations': before['completed_simulations'],
            'default_nodes': before['used_nodes'], 'larger_simulations': after['completed_simulations'],
            'larger_nodes': after['used_nodes']}


def compact(parsed: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in semantic_view(parsed).items() if k not in ('events', 'nodes')}


def main() -> None:
    # Heavy chess/runtime imports are only needed by this opt-in native verifier.
    import chess
    import torch
    from chess_anti_engine.encoding import rep_fix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('output exists; use a fresh directory')
    out.mkdir(parents=True)
    commands = Commands(out)
    bun, cc = os.environ.get('BUN', 'bun'), os.environ.get('CC', 'clang')
    report: dict[str, Any] = {'status': 'failed', 'scope': 'fixed bounded arenas, not dynamic growth or performance',
                              'searches': [], 'source_sha256': {}}
    try:
        # Match verify.qualify before any reference CBoard is constructed.
        torch.set_num_threads(2)
        rep_fix.apply(True)
        report['reference_history_rep_fix'] = rep_fix.current()
        commands.run('compiler', [bun, str(HERE.parent / 'standalone/verify_compiler.js'), str(args.compiler_root)])
        for name in ('session_probe/Search.bend', 'multi_root/ArenaConfig.bend', 'multi_root/arena_probe.bend',
                     'multi_root/verify_arena.py', 'multi_root/main.bend', 'multi_root/AsyncRun.bend',
                     'multi_root/DeadlineWork.bend', 'multi_root/verify.py', 'session_probe/run_probe.py'):
            report['source_sha256'][name] = hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest()

        def build(source: Path, binary: Path, flags: list[str]) -> None:
            generated = binary.with_suffix('.c')
            commands.run('generate-' + binary.name, [bun, str(args.compiler_root / 'bend2/main.ts'),
                         str(source), '-o', str(generated)], expected_stderr='')
            commands.run('build-' + binary.name, [cc, '-std=c11', '-O1', '-ffp-contract=off', *flags,
                         str(generated), '-pthread', '-lm', '-o', str(binary)])

        traces = {}
        for mode, flags in (('normal', []), ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])):
            binary = out / ('arena-' + mode)
            build(HERE / 'arena_probe.bend', binary, flags)
            # The probe's output includes configuration. Explicit env -u avoids
            # inheriting a user override into the default oracle.
            text = commands.run('run-' + mode, ['env', '-u', ENV, str(binary), '--threads', '1'], expected_stderr='')
            probe_result(text)
            traces[mode] = hashlib.sha256(text.encode()).hexdigest()
        report['probe_rows_per_mode'] = len(expected_probe().splitlines())
        report['probe_sha256'] = traces
        for setting in ('1', '00021', '4097', '8192', '65536'):
            text = commands.run('config-' + setting, ['env', f'{ENV}={setting}', str(out / 'arena-normal'),
                                '--threads', '1'], expected_stderr='')
            probe_result(text, int(setting))
        bad_values = ('', '0', '65537', '4294967295', '-1', '+8192', '8192 ', ' 8192',
                      '8_192', '0x2000', 'NaN', '000001', '1\n', '\u0668\u0661\u0669\u0662')
        for index, setting in enumerate(bad_values):
            text = commands.run(f'invalid-{index}', ['env', f'{ENV}={setting}', str(out / 'arena-normal'),
                                '--threads', '1'], expected_exit=2, expected_stderr=ERROR)
            if text:
                raise ValueError('invalid arena setting produced search output')
        report['invalid_settings_rejected'] = len(bad_values)

        shadow = out / 'mutations'
        shutil.copytree(HERE.parent, shadow, ignore=shutil.ignore_patterns('__pycache__'))
        search = shadow / 'session_probe/Search.bend'
        original = search.read_text()
        rejected = []
        for label, old, new in (
            ('allocation', 'Array.new(Node, storage_depth(cap), blank(b))', 'Array.new(Node, 12n, blank(b))'),
            ('sentinel', 'U32.max(4096, cap)', '4096'),
            ('live-node', 'Bool.and(U32.is_lt(id, used), U32.is_lt(id, cap))', 'U32.is_eq(id, id)'),
        ):
            if original.count(old) != 1:
                raise ValueError('arena mutation site changed')
            search.write_text(original.replace(old, new))
            binary = out / ('mutant-' + label)
            build(shadow / 'multi_root/arena_probe.bend', binary, [])
            text = commands.run('mutant-' + label, ['env', '-u', ENV, str(binary), '--threads', '1'], expected_stderr='')
            try:
                probe_result(text)
            except ValueError:
                rejected.append(label)
            else:
                raise ValueError('incorrect arena mutation survived')
        report['executed_mutations_rejected'] = rejected
        shutil.rmtree(shadow)

        first = chess.Board()
        second = chess.Board()
        second.push_uci('e2e4')
        second.push_uci('e7e5')
        terminal = chess.Board('k7/1Q6/2K5/8/8/8/8/8 b - - 150 1')

        def search_case(binary: Path, channels: int, mode: bool, name: str,
                        capacity: int | None, roots: list[Any], sims: int) -> dict[str, Any]:
            folder = out / f'c{channels}-{int(mode)}-{name}'
            folder.mkdir()
            trace = folder / 'input.trace'
            env = {**verify.environment(), 'DEEPFIN_COHORT_ASYNC': str(int(mode))}
            if capacity is not None:
                env[ENV] = str(capacity)
            actual = 4096 if capacity is None else capacity
            run = verify.execute(binary, roots, {**env, 'DEEPFIN_BEND_MODEL_TRACE': str(trace)}, sims, 8)
            if run.returncode or run.stderr:
                raise RuntimeError(f'arena search failed: {name}: {run.returncode}: {run.stderr}')
            parsed = verify.parse(run.stdout, len(roots), 4, sims, 0, True,
                                  asynchronous=mode, arena_nodes=actual)
            checked = verify.oracle_check(parsed, roots, trace, args.oracle, channels, 4, sims, 8, 0,
                                          arena_nodes=actual)
            quiet = verify.execute(binary, roots, env, sims, 8, diagnostics=False)
            if quiet.returncode or quiet.stderr:
                raise RuntimeError(f'quiet arena search failed: {name}: {quiet.returncode}: {quiet.stderr}')
            plain = verify.parse(quiet.stdout, len(roots), 4, sims, 0, False,
                                 asynchronous=mode, arena_nodes=actual)
            if compact(parsed) != compact(plain):
                raise ValueError('arena diagnostics changed realized work')
            report['searches'].append({'channels': channels, 'async': mode, 'case': name,
                'arena_nodes': actual, 'oracle': checked, 'roots': parsed['roots'],
                'semantic_sha256': hashlib.sha256(json.dumps(semantic_view(parsed), sort_keys=True).encode()).hexdigest()})
            trace.unlink()  # Raw input tensors are temporary; retain verification, not bulky arrays.
            return parsed

        effects = []
        for channels in (146, 175):
            binary = args.matrix / f'c{channels}-b4/runner'
            for mode in (False, True):
                base = search_case(binary, channels, mode, 'default', None, [first], 256)
                explicit = search_case(binary, channels, mode, 'explicit-default', 4096, [first], 256)
                if semantic_view(base) != semantic_view(explicit):
                    raise ValueError('explicit default changed complete search state')
                large = search_case(binary, channels, mode, '8192', 8192, [first], 256)
                effects.append({'channels': channels, 'async': mode, **outcome(base, large)})
                for capacity, sims in ((1, 2), (20, 2), (21, 2), (4097, 256), (65536, 4)):
                    search_case(binary, channels, mode, str(capacity), capacity, [first], sims)
                search_case(binary, channels, mode, 'mixed', 16384, [first, second, terminal], 256)
        report['capacity_effects'] = effects
        # A complete larger-arena run through sanitizer-instrumented coordinator
        # C and the existing native worker, in addition to storage-only probes.
        search_case(args.matrix / 'c146-b4/callback-ubsan', 146, True, 'ubsan-8192', 8192, [first], 256)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('searches', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
