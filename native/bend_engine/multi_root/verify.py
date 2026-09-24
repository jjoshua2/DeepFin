"""Opt-in actual selected-leaf cohort oracle; ordinary tests exercise parsing only.

Deterministic callbacks and real model execution have explicitly different scopes.
No subprocess performs search on the runner's behalf; Python is an external oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np

from native.bend_engine.batch_backend.verify import trace_record

PREFIX = 'info string '
BATCHES = (1, 2, 4, 8, 16)
ZERO = ('failed_forward_rows', 'failed_rows', 'cancelled_rows', 'stale_rows',
        'rejected_rows', 'executed_wasted_rows', 'unresolved_rows', 'unconfirmed_forward_rows')


def integer(v: object, lo: int = 0, hi: int = 0xffffffff) -> int:
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError(f'invalid integer: {v}')
    return v


def obj(text: str) -> dict[str, Any]:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for k, v in items:
            if k in data:
                raise ValueError('duplicate JSON key')
            data[k] = v
        return data
    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value


def histogram(value: object, batch: int) -> dict[int, int]:
    if not isinstance(value, dict):
        raise ValueError('missing histogram')
    result = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.isascii() or not k.isdecimal() or str(int(k)) != k:
            raise ValueError('noncanonical batch size')
        result[integer(int(k), 1, batch)] = integer(v, 1)
    return result


def parse(stdout: str, root_count: int, batch: int, sims: int, budget: int,
          diagnostics: bool, *, asynchronous: bool = False, arena_nodes: int = 4096) -> dict[str, Any]:
    integer(root_count, 1, 16)
    integer(arena_nodes, 1, 65536)
    arena_seen = False
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported batch')
    roots, nodes, events, batches = {}, {}, [], []
    work = None
    deadlines: set[int] = set()
    expired: set[int] = set()
    for line in stdout.splitlines():
        if work is not None or not line.startswith(PREFIX):
            raise ValueError('extra/malformed output')
        content = line[len(PREFIX):]
        name, sep, tail = content.partition(' ')
        if name == 'cohort_arena':
            expected = f'{arena_nodes} {max(4096, 1 << (arena_nodes - 1).bit_length())}'
            if arena_nodes == 4096 or arena_seen or roots or nodes or events or tail != expected:
                raise ValueError('invalid or duplicate arena declaration')
            arena_seen = True
            continue
        if asynchronous and name == 'cohort_ready' and not sep:
            continue
        if asynchronous and name == 'cohort_control':
            deadline = re.fullmatch(r'deadline ([1-9]|1[0-6]) (0|[1-9][0-9]*)', tail)
            expiry = re.fullmatch(r'expired ([1-9]|1[0-6])', tail)
            if deadline:
                root = integer(int(deadline[1]), 1, root_count)
                integer(int(deadline[2]), 0, 3600000)
                if root in expired:
                    raise ValueError('deadline revived expired root')
                deadlines.add(root)
            elif expiry:
                root = integer(int(expiry[1]), 1, root_count)
                if root not in deadlines or root in expired:
                    raise ValueError('invalid or duplicate deadline expiry')
                expired.add(root)
            elif tail not in ('stop', 'quit', 'error invalid-root', 'error invalid-command',
                              'error malformed-line', 'error invalid-deadline',
                              'error inactive-root', 'error deadline-extension') and not re.fullmatch(r'cancel (?:[1-9]|1[0-6])', tail):
                raise ValueError('invalid control acknowledgment')
            if tail.startswith('cancel '):
                integer(int(tail.split()[1]), 1, root_count)
            continue
        if not sep:
            raise ValueError('missing record payload')
        if name == 'cohort_work':
            work = obj(tail)
        elif name == 'cohort_root':
            r = obj(tail)
            epoch = integer(r.get('root'), 1, root_count)
            if epoch in roots:
                raise ValueError('duplicate root')
            n = integer(r.get('completed_simulations'), 0, sims)
            accepted = integer(r.get('accepted_neural_rows'), 0, n)
            if asynchronous:
                sent = integer(r.get('dispatched_real_rows'))
                wasted = integer(r.get('cancelled_rows'), 0, 1)
                requested = r.get('cancel_requested')
                if type(requested) is not bool or (wasted and not requested):
                    raise ValueError('invalid root cancellation')
                if sent != accepted + wasted or integer(r.get('executed_real_rows')) != sent:
                    raise ValueError('root execution/acceptance/cancellation mismatch')
                if budget and sent > budget:
                    raise ValueError('cancelled admission was refunded')
            elif integer(r.get('executed_real_rows')) != accepted:
                raise ValueError('root execution/acceptance mismatch')
            integer(r.get('rule_draw_replies'), 0, n - accepted)
            integer(r.get('used_nodes'), 1, arena_nodes)
            integer(r.get('stop_code'), 0, 2 if asynchronous else 1)
            if integer(r.get('simulation_budget')) != sims or integer(r.get('neural_budget')) != budget:
                raise ValueError('root budget mismatch')
            met = r.get('neural_budget_met')
            if budget:
                if accepted > budget or type(met) is not bool or met != (accepted == budget):
                    raise ValueError('invalid neural budget result')
            elif met is not None:
                raise ValueError('absent budget is not measured')
            searched = r.get('searched_move')
            move = r.get('bestmove')
            if type(searched) is not bool or not isinstance(move, str):
                raise ValueError('invalid move result')
            if (not searched and move != '0000') or (searched and not re.fullmatch('[a-h][1-8][a-h][1-8][nbrq]?', move)):
                raise ValueError('invalid move syntax')
            roots[epoch] = r
        elif name == 'cohort_node' and diagnostics:
            ep, index, payload = tail.split(' ', 2)
            epoch, at = integer(int(ep), 1, root_count), integer(int(index), 0, arena_nodes - 1)
            row = json.loads(payload)
            if not isinstance(row, list) or len(row) != 29:
                raise ValueError('invalid node')
            row = [integer(v) for v in row]
            if at in nodes.setdefault(epoch, {}):
                raise ValueError('duplicate node')
            nodes[epoch][at] = row
        elif name in ('cohort_batch', 'cohort_rule', 'native_path', 'native_reply') and diagnostics:
            fields = [integer(int(s)) for s in tail.split()]
            if name == 'cohort_batch':
                if len(fields) != 3 or fields[0] != len(batches) + 1 or fields[2] != batch:
                    raise ValueError('invalid batch record')
                integer(fields[1], 1, batch)
                batches.append(fields[1])
            elif len(fields) < 3:
                raise ValueError('missing request identity')
            events.append((name, fields))
        else:
            raise ValueError('unexpected record: ' + name)
    if work is None or set(roots) != set(range(1, root_count + 1)):
        raise ValueError('missing roots or summary')
    schema = 'deepfin.multi-root-async-work.v1' if asynchronous else 'deepfin.multi-root-work.v1'
    scope = 'bounded_async_cohort' if asynchronous else 'bounded_cohort'
    schemas = (schema, 'deepfin.multi-root-async-work.v2') if asynchronous else (schema,)
    if work.get('schema') not in schemas or work.get('scope') != scope:
        raise ValueError('wrong accounting schema')
    if work.get('schema') == 'deepfin.multi-root-async-work.v2':
        observed_mask = 0
        for ep, r in roots.items():
            if 'deadline_offset_ms' not in r or type(r.get('deadline_expired')) is not bool:
                raise ValueError('missing/invalid deadline status')
            due = r['deadline_offset_ms']
            if due is not None:
                integer(due, 0, 2**63 - 1)
            if (due is not None) != (ep in deadlines) or r['deadline_expired'] != (ep in expired):
                raise ValueError('deadline acknowledgments/status mismatch')
            if r['deadline_expired']:
                if not r['cancel_requested'] or r['stop_code'] != 2:
                    raise ValueError('expired root not logically stopped')
                observed_mask |= 1 << ep
        if integer(work.get('deadline_expired_mask'), 0, (1 << (root_count + 1)) - 2) != observed_mask:
            raise ValueError('deadline expiry mask mismatch')
    elif deadlines or expired or 'deadline_expired_mask' in work or any(
            'deadline_offset_ms' in r or 'deadline_expired' in r for r in roots.values()):
        raise ValueError('deadline metadata requires async v2 schema')
    calls = integer(work.get('forward_calls'))
    real = integer(work.get('executed_real_rows'))
    physical = integer(work.get('physical_rows'))
    if integer(work.get('roots')) != root_count or integer(work.get('dispatched_real_rows')) != real:
        raise ValueError('root/dispatch mismatch')
    accepted_total = integer(work.get('accepted_neural_rows'), 0, real)
    wasted_total = integer(work.get('cancelled_rows'), 0, real) if asynchronous else 0
    if accepted_total + wasted_total != real or sum(r['accepted_neural_rows'] for r in roots.values()) != accepted_total:
        raise ValueError('accepted rows do not reconcile')
    if asynchronous and (sum(r['dispatched_real_rows'] for r in roots.values()) != real
                         or sum(r['cancelled_rows'] for r in roots.values()) != wasted_total
                         or integer(work.get('executed_wasted_rows')) != wasted_total):
        raise ValueError('cancelled rows do not reconcile')
    if sum(r['completed_simulations'] for r in roots.values()) != integer(work.get('completed_simulations')):
        raise ValueError('simulations do not reconcile')
    if physical != calls * batch or integer(work.get('padded_rows')) != physical - real:
        raise ValueError('padding/physical mismatch')
    for k in ZERO:
        if asynchronous and k in ('cancelled_rows', 'executed_wasted_rows'):
            continue
        if integer(work.get(k)) != 0:
            raise ValueError('unexpected incomplete/error work')
    rh = histogram(work.get('real_batch_histogram'), batch)
    ph = histogram(work.get('physical_batch_histogram'), batch)
    if sum(rh.values()) != calls or sum(k * v for k, v in rh.items()) != real or ph != ({batch: calls} if calls else {}):
        raise ValueError('histogram mismatch')
    if diagnostics:
        observed = {n: batches.count(n) for n in set(batches)}
        if observed != rh:
            raise ValueError('observed batch histogram mismatch')
        for ep, r in roots.items():
            if set(nodes.get(ep, {})) != set(range(r['used_nodes'])):
                raise ValueError('incomplete final tree')
    wall = work.get('wall_seconds')
    if (not isinstance(wall, (float, int)) or isinstance(wall, bool)) or not math.isfinite(wall) or wall < 0:
        raise ValueError('invalid clock')
    for k in ('useful_eps', 'executed_eps'):
        numerator = accepted_total if k == 'useful_eps' else real
        v = work.get(k)
        if not wall:
            if v is not None:
                raise ValueError('EPS at zero clock interval')
        elif (not isinstance(v, (float, int)) or isinstance(v, bool)) or not math.isfinite(v) or abs(v - numerator / wall) > 2e-6:
            raise ValueError('invalid EPS')
    if work.get('warmup_excluded') is not False or work.get('clock_resolution_seconds') != 0.001:
        raise ValueError('unsupported timing claims')
    for k in ('gathering_seconds', 'backend_and_transport_seconds', 'normalization_and_backup_seconds'):
        v = work.get(k)
        if asynchronous:
            if k not in work or v is not None:
                raise ValueError('async phase time is unmeasured')
            continue
        if (not isinstance(v, (float, int)) or isinstance(v, bool)) or not math.isfinite(v) or not 0 <= v <= wall:
            raise ValueError('invalid phase time')
    phases = work.get('phase_seconds')
    if phases != {'queue_wait': None, 'h2d': None, 'gpu': None, 'd2h': None}:
        raise ValueError('unmeasured phases must be null')
    if arena_nodes != 4096 and not arena_seen:
        raise ValueError('missing arena declaration')
    return {'roots': roots, 'nodes': nodes, 'events': events, 'work': work}


def expected_test_output(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    i = np.arange(1861)
    return (flat[(i * 7 + 11) % len(flat)] * np.float32(0.03125)
            + ((i % 31) - 15).astype(np.float32) * np.float32(0.015625))


def environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(('DEEPFIN_BEND_', 'DEEPFIN_MULTI_TEST_', 'DEEPFIN_COHORT_'))}


def execute(binary: Path, roots: list[Any], env: dict[str, str], sims: int = 4,
            depth: int = 2, budget: int = 0, diagnostics: bool = True) -> subprocess.CompletedProcess[str]:
    from native.bend_engine.standalone.verify_rules import position
    args = [str(binary.resolve()), '--threads', '1', '--', str(sims), str(depth), str(budget), str(int(diagnostics))]
    args += [position(r).removeprefix('position ') for r in roots]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=90, check=False)


def oracle_check(parsed: dict[str, Any], roots: list[Any], trace: Path, oracle_binary: Path,
                 channels: int, batch: int, sims: int, depth: int, budget: int,
                 eager: Any = None, history_encoding: str = 'lc0_root_legacy_meta',
                 arena_nodes: int = 4096) -> dict[str, Any]:
    import torch
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY, move_to_index
    from native.bend_engine.neural_probe.adapter import board_position, decode_key, probabilities
    from native.bend_engine.session_probe import run_probe as sessions
    from native.bend_engine.session_probe.search_draws import automatic_draw
    from native.bend_engine.standalone.verify_neural import descendant, floats, history_path
    from native.bend_engine.standalone.verify_policy import key

    oracle = sessions.Oracle(oracle_binary, with_python_chess=True)
    refs = {i: sessions.Reference(board_position(r), oracle, cap=arena_nodes, depth=depth, budget=sims)
            for i, r in enumerate(roots, 1)}
    paths, evaluated, draws = {}, dict.fromkeys(refs, 0), dict.fromkeys(refs, 0)
    current: list[tuple[np.ndarray, np.ndarray]] = []
    batch_roots: set[int] = set()
    sequence = priors = 0
    max_error = 0.0
    root_inputs: dict[int, np.ndarray] = {}
    with trace.open('rb') as stream:
        for name, values in parsed['events']:
            if name == 'cohort_batch':
                assert not current
                assert not paths
                sequence += 1
                seq, rows, size = values
                assert seq == sequence
                assert size == batch
                x, y = trace_record(stream, seq, batch, rows, channels)
                assert np.isfinite(x).all()
                assert np.isfinite(y).all()
                assert np.all(x[rows:].view(np.uint32) == 0), 'physical padding is not exact +0'
                current = list(zip(x[:rows], y, strict=True))
                batch_roots = set()
                continue
            ep, req, node = values[:3]
            assert ep in refs
            ref, root = refs[ep], roots[ep - 1]
            identity = ep, req, node
            if name == 'native_path':
                assert identity not in paths
                paths[identity] = values[3:]
                continue
            assert (req, node) == (ref.seq, ref.next()), (identity, ref.seq)
            path = history_path(ref, node)
            leaf = descendant(root, path)
            assert board_position(leaf) == ref.nodes[node].board
            if name == 'cohort_rule':
                assert len(values) == 4
                assert automatic_draw(leaf) is not None
                ref.accept_draw(node)
                draws[ep] += 1
                continue
            assert name == 'native_reply'
            assert paths.pop(identity) == path
            assert current, 'duplicate root within one physical batch'
            assert ep not in batch_roots, 'duplicate root within one physical batch'
            batch_roots.add(ep)
            assert automatic_draw(leaf) is None
            x, y = current.pop(0)
            cb = CBoard.from_board(leaf)
            reference_input = cb.encode_full(1 if history_encoding == 'lc0_root' else 2, channels - 112)
            np.testing.assert_array_equal(x.view(np.uint32), reference_input.view(np.uint32))
            if node == 0:
                root_inputs[ep] = x.copy()
            if eager is None:
                expected = expected_test_output(x)
                np.testing.assert_array_equal(y.view(np.uint32), expected.view(np.uint32))
            else:
                with torch.no_grad():
                    raw = eager(torch.from_numpy(x[None]))
                expected = np.concatenate((raw['policy'].numpy()[0], raw['wdl'].numpy()[0]))
                np.testing.assert_allclose(y, expected, atol=2e-6, rtol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(expected - y))))
            count = integer(values[6], 1, 256)
            assert len(values) == 7 + count * 2
            keys, ps, wdl = values[7::2], floats(values[8::2]), floats(values[3:6])
            assert len(set(keys)) == count
            assert set(keys) == {key(leaf, m) for m in leaf.legal_moves}
            full = np.asarray([move_to_index(decode_key(leaf, k), leaf) for k in keys], dtype=np.int64)
            assert len(set(FULL_TO_COMPACT_POLICY[full].tolist())) == count
            expected_wdl, expected_ps = probabilities(y[:1858][None], y[1858:][None], full)
            np.testing.assert_allclose(wdl, expected_wdl, atol=2e-7, rtol=3e-6)
            np.testing.assert_allclose(ps, expected_ps, atol=2e-7, rtol=3e-6)
            ref.accept(node, keys, wdl.tolist(), ps.tolist())
            evaluated[ep] += 1
            priors += count
        assert not paths
        assert not current
        assert not stream.read(1)
    assert sum(evaluated.values()) == parsed['work']['accepted_neural_rows']
    for ep, ref in refs.items():
        if not budget or evaluated[ep] < budget:
            assert ref.next() is None
        r = parsed['roots'][ep]
        assert (r['completed_simulations'], r['used_nodes'], r['stop_code'], r['rule_draw_replies']) == (
            ref.completed, len(ref.nodes), ref.stop, draws[ep])
        assert evaluated[ep] == r['executed_real_rows']
        children = [a for i, a in enumerate(ref.nodes) if i and a.parent == 0]
        best = min(children, key=lambda a: (-a.n, a.key)).key if children else sessions.SENTINEL
        move = decode_key(roots[ep - 1], best).uci() if children else '0000'
        assert r['bestmove'] == move, r
        assert r['searched_move'] == bool(children), r
        converted = []
        for at in range(len(ref.nodes)):
            row = parsed['nodes'][ep][at]
            converted.append([at, row[20], row[19], row[28], row[27], row[23],
                              row[24], row[25], row[26], row[21], row[22], *row[:19]])
        ref.check_snapshot(converted, [ep, ref.completed, len(ref.nodes), ref.stop, max(4096, arena_nodes), ref.seq], best, ep)
    if len(roots) >= 14:
        assert not np.array_equal(root_inputs[13][:104], root_inputs[14][:104])
    return {'roots': len(roots), 'real_rows': sum(evaluated.values()), 'forwards': sequence,
            'padding': parsed['work']['padded_rows'], 'rule_draw_replies': sum(draws.values()),
            'legal_priors_compared': priors, 'max_logit_absolute_error': max_error,
            'final_tree_nodes_compared': sum(len(r.nodes) for r in refs.values())}


def check_run(binary: Path, roots: list[Any], batch: int, channels: int, oracle: Path,
              directory: Path, env: dict[str, str], *, sims: int = 4, depth: int = 2,
              budget: int = 0, eager: Any = None, history_encoding: str = 'lc0_root_legacy_meta') -> tuple[dict[str, Any], dict[str, Any]]:
    directory.mkdir()
    trace = directory / 'raw.trace'
    result = execute(binary, roots, {**env, 'DEEPFIN_BEND_MODEL_TRACE': str(trace)}, sims, depth, budget)
    assert result.returncode == 0, (result.returncode, result.stderr, result.stdout[-1000:])
    assert not result.stderr, (result.returncode, result.stderr, result.stdout[-1000:])
    parsed = parse(result.stdout, len(roots), batch, sims, budget, True,
                   asynchronous=env.get('DEEPFIN_COHORT_ASYNC') == '1')
    checked = oracle_check(parsed, roots, trace, oracle, channels, batch, sims, depth, budget, eager, history_encoding)
    quiet = execute(binary, roots, env, sims, depth, budget, False)
    assert quiet.returncode == 0, quiet.stderr
    assert not quiet.stderr, quiet.stderr
    plain = parse(quiet.stdout, len(roots), batch, sims, budget, False,
                  asynchronous=env.get('DEEPFIN_COHORT_ASYNC') == '1')
    assert plain['roots'] == parsed['roots']
    for k in ('forward_calls', 'executed_real_rows', 'padded_rows', 'real_batch_histogram'):
        assert plain['work'][k] == parsed['work'][k]
    return checked, parsed


def invalid_controls(binary: Path, env: dict[str, str], directory: Path) -> int:
    valid = ['4', '2', '0', '0', 'startpos']
    cases = [[], ['4','2','0','0'], [*valid[:4], *(['startpos'] * 17)],
             [*valid[:4], 'fen nonsense'], [*valid[:4], 'startpos moves e2e5'],
             [*valid[:4], 'startpos ' + 'x' * 16385]]
    for at, bads in ((0, ('0','257','-1','4294967296')), (1, ('0','33')), (2, ('257','-1')), (3, ('2','-1'))):
        for bad in bads:
            args = valid.copy()
            args[at] = bad
            cases.append(args)
    for i, args in enumerate(cases):
        marker = directory / f'bad-{i}.trace'
        result = subprocess.run([str(binary.resolve()), '--threads', '1', '--', *args],
                                env={**env, 'DEEPFIN_BEND_MODEL_TRACE': str(marker)},
                                text=True, capture_output=True, timeout=30, check=False)
        assert result.returncode == 2, (args, result)
        assert not result.stdout, (args, result)
        assert not marker.exists(), (args, result)
    return len(cases)


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from chess_anti_engine.encoding import rep_fix
    from native.bend_engine.neural_probe.checkpoint import EagerReference, load_checkpoint
    from native.bend_engine.standalone.verify_neural import fixtures
    torch.set_num_threads(2)
    rep_fix.apply(True)
    roots = fixtures()
    env = environment()
    if getattr(args, 'asynchronous', False):
        env['DEEPFIN_COHORT_ASYNC'] = '1'
    loaded, history, eager = None, 'lc0_root_legacy_meta', None
    if args.package is not None:
        manifest = json.loads(args.package.with_suffix('.json').read_text())
        assert manifest['device'] == 'cpu'
        assert manifest['dtype'] == 'float32'
        assert manifest['batch'] == args.batch
        assert manifest['channels'] == args.channels
        assert hashlib.sha256(args.package.read_bytes()).hexdigest() == manifest['sha256']
        loaded = load_checkpoint(args.checkpoint, weights_key=manifest['weights_key'])
        assert loaded.identity['checkpoint_sha256'] == manifest['checkpoint_sha256']
        eager = EagerReference(loaded.model, torch.device('cpu'), torch.float32)
        history = loaded.encoding.input_history_encoding
        env['DEEPFIN_BEND_MODEL_PACKAGE'] = str(args.package.resolve())
    reports = []
    with tempfile.TemporaryDirectory(prefix='deepfin-multi-root-') as tmp:
        folder = Path(tmp)
        checked, main = check_run(args.binary, roots, args.batch, args.channels, args.oracle, folder / 'main', env,
                                  eager=eager, history_encoding=history)
        reports.append(checked)
        # Terminal roots must never dispatch, while explicit neural caps remain visibly underfilled.
        terminal = [roots[7], roots[8], roots[9], roots[10]]
        checked, _ = check_run(args.binary, terminal, args.batch, args.channels, args.oracle, folder / 'terminal', env,
                               budget=3, eager=eager, history_encoding=history)
        assert checked['real_rows'] == checked['forwards'] == 0
        reports.append(checked)
        checked, limited = check_run(args.binary, roots[:3], args.batch, args.channels, args.oracle, folder / 'limited', env,
                                     sims=256, depth=32, budget=3, eager=eager, history_encoding=history)
        assert all(r['neural_budget_met'] for r in limited['roots'].values())
        reports.append(checked)
        bad_inputs = invalid_controls(args.binary, env, folder)
        faults = 0
        if loaded is None and args.batch > 1:
            for fault in ('fail', 'nan'):
                run = execute(args.binary, roots[:args.batch], {**env, 'DEEPFIN_MULTI_TEST_FAULT': fault})
                assert run.returncode == 2
                assert 'cohort_work' not in run.stdout
                assert 'native_reply' not in run.stdout
                faults += 1
            try:
                check_run(args.binary, roots, args.batch, args.channels, args.oracle, folder / 'swapped',
                          {**env, 'DEEPFIN_MULTI_TEST_FAULT': 'swap'})
            except AssertionError:
                faults += 1
            else:
                raise AssertionError('swapped rows escaped the independent oracle')
    tree_hash = hashlib.sha256(json.dumps(main['nodes'], sort_keys=True).encode()).hexdigest()
    report = {'status': 'passed', 'scope': 'CPU selected-leaf model' if loaded else 'deterministic selected-leaf callback',
              'asynchronous': getattr(args, 'asynchronous', False),
              'batch': args.batch, 'channels': args.channels, 'cases': reports,
              'invalid_input_rejections': bad_inputs, 'fault_controls': faults,
              'root_results': main['roots'], 'complete_tree_sha256': tree_hash,
              'counters': {k: main['work'][k] for k in ('forward_calls','executed_real_rows','padded_rows','real_batch_histogram')},
              'warmup_excluded': False, 'speed_or_strength_qualified': False}
    if loaded:
        assert args.package is not None
        report['checkpoint_sha256'] = loaded.identity['checkpoint_sha256']
        report['package_sha256'] = hashlib.sha256(args.package.read_bytes()).hexdigest()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--batch', type=int, choices=BATCHES, required=True)
    p.add_argument('--channels', type=int, choices=(146, 175), required=True)
    p.add_argument('--asynchronous', action='store_true', help='require the async cohort schema and enable its worker')
    p.add_argument('--package', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if (args.package is None) != (args.checkpoint is None):
        p.error('--package and --checkpoint must be supplied together')
    report = qualify(args)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
